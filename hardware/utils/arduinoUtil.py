"""
arduinoUtil.py  —  Arduino Serial Utility  v2.0
================================================
Giao tiếp Serial với Arduino: điều khiển servo + đọc cảm biến siêu âm.

Commands gửi đến Arduino:
    '0' → servo RIGHT       (ORGANIC  / Biological)
    '1' → servo LEFT        (RECYCLABLE)
    '2' → servo DOWN-RIGHT  (HAZARDOUS / Battery)
    '3' → servo DOWN-LEFT   (OTHER / General_Waste)

Commands đọc mức đầy (4 cảm biến siêu âm, gửi khi tích hợp phần cứng):
    'F' → Arduino trả về JSON:  {"org":12,"rec":34,"haz":5,"oth":67}
          (đơn vị: cm — khoảng cách từ nắp xuống rác)
    Hoặc tuỳ chỉnh protocol khi bạn thêm cảm biến vào file .ino

Yêu cầu:
    pip install pyserial
"""

import time
import threading
import random
from typing import Optional, Callable

import serial
import serial.tools.list_ports

# ============================================================
# CẤU HÌNH
# ============================================================

ARDUINO_BAUDRATE   = 9600
ARDUINO_TIMEOUT    = 8.0           # giây chờ ACK lệnh servo
ARDUINO_ACK_PREFIX = "Hoan thanh"  # prefix phản hồi từ Arduino

# Các khoảng đệm để Serial/servo/cảm biến có thời gian ổn định.
# Có thể tăng nhẹ nếu phần cứng phản hồi chập chờn.
ARDUINO_BOOT_DELAY_SEC          = 2.5
ARDUINO_READY_TIMEOUT_SEC       = 6.0
ARDUINO_BEFORE_WRITE_DELAY_SEC  = 0.08
ARDUINO_AFTER_WRITE_DELAY_SEC   = 0.08
ARDUINO_AFTER_ACK_DELAY_SEC     = 0.35
ARDUINO_BEFORE_SENSOR_DELAY_SEC = 0.50
ARDUINO_SENSOR_RETRY_DELAY_SEC  = 0.20

# Khoảng cách từ cảm biến siêu âm tới đáy/ngưỡng rỗng của từng ngăn.
# Khi chưa có rác, cảm biến đọc khoảng 41cm.
BIN_EMPTY_DISTANCE_CM = {
    "ORGANIC":    41.0,
    "RECYCLABLE": 41.0,
    "HAZARDOUS":  41.0,
    "OTHER":      41.0,
}

# Chiều cao vùng chứa rác thực tế. Khi rác cao 27cm thì xem là đầy 100%.
BIN_TRASH_HEIGHT_CM = 27.0
SIMULATED_DROP_MIN_RATIO = 0.05
SIMULATED_DROP_MAX_RATIO = 0.10

# Backward-compatible alias nếu module khác còn import BIN_DEPTH_CM.
BIN_DEPTH_CM = BIN_EMPTY_DISTANCE_CM

# Map ngăn → Arduino command
BIN_TO_ARDUINO_CMD = {
    "ORGANIC":    '0',
    "RECYCLABLE": '1',
    "HAZARDOUS":  '2',
    "OTHER":      '3',
}

# Map ngăn → key trong JSON trả về của Arduino (khi có cảm biến siêu âm)
BIN_TO_SENSOR_KEY = {
    "ORGANIC":    "org",
    "RECYCLABLE": "rec",
    "HAZARDOUS":  "haz",
    "OTHER":      "oth",
}

_simulated_fill_lock = threading.Lock()
_simulated_distance_cm = dict(BIN_EMPTY_DISTANCE_CM)


def _distance_to_fill_pct(bin_name: str, distance_cm: float) -> float:
    """Tính % đầy từ khoảng cách siêu âm theo mô hình 41cm rỗng, 27cm chiều cao thùng."""
    empty_distance = BIN_EMPTY_DISTANCE_CM.get(bin_name, 41.0)
    filled_height = empty_distance - distance_cm
    fill_pct = (filled_height / BIN_TRASH_HEIGHT_CM) * 100.0
    return round(max(0.0, min(100.0, fill_pct)), 1)


def _min_distance_for_full(bin_name: str) -> float:
    return BIN_EMPTY_DISTANCE_CM.get(bin_name, 41.0) - BIN_TRASH_HEIGHT_CM


def _build_fill_result(distance_by_bin: dict) -> dict:
    result = {}
    for bin_name in BIN_TO_SENSOR_KEY.keys():
        dist = float(distance_by_bin.get(bin_name, BIN_EMPTY_DISTANCE_CM.get(bin_name, 41.0)))
        if dist < 0:
            fill_pct = None
        else:
            fill_pct = _distance_to_fill_pct(bin_name, dist)
        result[bin_name] = {
            "distance_cm": round(dist, 1),
            "fill_pct":    fill_pct,
        }
    return result

# ============================================================
# KHỞI TẠO
# ============================================================

def list_available_ports():
    """In danh sách các cổng Serial đang kết nối."""
    print("--- Các cổng Serial đang kết nối ---")
    ports = serial.tools.list_ports.comports()
    for i, p in enumerate(ports):
        print(f"[{i}] {p.device} - {p.description}")
    return ports


def find_arduino_port() -> Optional[str]:
    """Liệt kê cổng Serial và yêu cầu người dùng chọn."""
    ports = list_available_ports()
    if not ports:
        print("Không tìm thấy cổng Serial nào!")
        return None
    try:
        idx = int(input("\nNhập số thứ tự cổng muốn kết nối: "))
        return ports[idx].device
    except (ValueError, IndexError):
        print("Lựa chọn không hợp lệ.")
        return None


def init_arduino(port: Optional[str] = None):
    """
    Khởi tạo kết nối Serial với Arduino.

    Args:
        port: Tên cổng (vd: 'COM3', '/dev/ttyUSB0').
              Nếu None → tự động hỏi người dùng chọn.

    Returns:
        serial.Serial nếu thành công, None nếu thất bại.
    """
    selected_port = port or find_arduino_port()
    if selected_port is None:
        print("[WARN] Không tìm thấy cổng Arduino — chạy không có Arduino.")
        return None
    try:
        ser = serial.Serial(selected_port, ARDUINO_BAUDRATE, timeout=1)
        time.sleep(ARDUINO_BOOT_DELAY_SEC)   # Chờ Arduino reset sau khi mở serial
        ser.reset_input_buffer()

        # Chờ "Ready" từ Arduino
        deadline = time.time() + ARDUINO_READY_TIMEOUT_SEC
        while time.time() < deadline:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if line == "Ready":
                print(f"[ARDUINO] Sẵn sàng trên {selected_port} @ {ARDUINO_BAUDRATE} baud")
                return ser
            elif line:
                print(f"[ARDUINO] Startup: {line}")
        print("[ARDUINO] Không nhận được 'Ready' — tiếp tục dù sao.")
        return ser
    except Exception as e:
        print(f"[ARDUINO ERROR] init: {e}")
        return None


# ============================================================
# GỬI LỆNH SERVO
# ============================================================

_arduino_lock       = threading.Lock()
_arduino_done_event = threading.Event()
_arduino_done_event.set()   # Khởi đầu: free


def arduino_send_command(
    arduino_serial,
    bin_type: str,
    on_done_callback: Optional[Callable] = None,
):
    """
    Gửi lệnh servo Arduino trong thread riêng, chờ ACK "Hoan thanh".

    Args:
        arduino_serial:   serial.Serial từ init_arduino(). None → giả lập.
        bin_type:         Tên ngăn (ORGANIC / RECYCLABLE / HAZARDOUS / OTHER).
        on_done_callback: Hàm gọi sau khi ACK nhận được (hoặc timeout).
    """
    cmd = BIN_TO_ARDUINO_CMD.get(bin_type, '3')

    def _send():
        _arduino_done_event.clear()
        try:
            if arduino_serial is None:
                print(f"[ARDUINO] (offline) Lệnh: '{cmd}' → {bin_type}")
                time.sleep(2.0)
            else:
                with _arduino_lock:
                    arduino_serial.reset_input_buffer()
                    time.sleep(ARDUINO_BEFORE_WRITE_DELAY_SEC)
                    arduino_serial.write(cmd.encode())
                    arduino_serial.flush()
                    time.sleep(ARDUINO_AFTER_WRITE_DELAY_SEC)
                    print(f"[ARDUINO] Gửi lệnh '{cmd}' → {bin_type}")

                    deadline = time.time() + ARDUINO_TIMEOUT
                    while time.time() < deadline:
                        line = arduino_serial.readline().decode(
                            'utf-8', errors='ignore').strip()
                        if line:
                            print(f"[ARDUINO] ← {line}")
                        if ARDUINO_ACK_PREFIX in line:
                            print(f"[ARDUINO] ACK nhận được: {line}")
                            break
                    else:
                        print(f"[ARDUINO] TIMEOUT chờ ACK ({ARDUINO_TIMEOUT}s)")
                    time.sleep(ARDUINO_AFTER_ACK_DELAY_SEC)
        except Exception as e:
            print(f"[ARDUINO ERROR] send_command: {e}")
        finally:
            _arduino_done_event.set()
            if on_done_callback:
                on_done_callback()

    threading.Thread(target=_send, daemon=True).start()


# ============================================================
# ĐỌC MỨC ĐẦY TỪ CẢM BIẾN SIÊU ÂM
# ============================================================

def read_fill_levels(
    arduino_serial,
    timeout: float = 3.0,
) -> Optional[dict]:
    """
    Gửi lệnh 'F' đến Arduino để lấy khoảng cách từ 4 cảm biến siêu âm,
    sau đó tính phần trăm đầy cho từng ngăn.

    Protocol Arduino (.ino đã tích hợp):
        - Nhận ký tự 'F'
        - Đo 4 cảm biến, trả về 1 dòng dạng:
          DIST:12.3,8.5,25.0,-1   ← khoảng cách (cm), thứ tự: org,rec,haz,oth
        - Nếu cảm biến lỗi, trả giá trị -1 cho ngăn đó.

    Args:
        arduino_serial: serial.Serial từ init_arduino(). None → trả None.
        timeout:        Số giây chờ phản hồi từ Arduino.

    Returns:
        dict dạng:
        {
            "ORGANIC":    {"distance_cm": 12.0, "fill_pct": 60.0},
            "RECYCLABLE": {"distance_cm": 34.0, "fill_pct": -13.3},  # vượt 100% → lỗi
            "HAZARDOUS":  {"distance_cm":  5.0, "fill_pct": 75.0},
            "OTHER":      {"distance_cm": -1.0, "fill_pct": None},   # lỗi cảm biến
        }
        Hoặc None nếu không đọc được.

    Ghi chú:
        Hàm này CHƯA HOẠT ĐỘNG cho đến khi bạn thêm cảm biến siêu âm
        và viết handler lệnh 'F' vào file .ino. Khi phần cứng chưa sẵn,
        hàm sẽ trả về None và in cảnh báo.
    """
    if arduino_serial is None:
        print("[ARDUINO] (offline) read_fill_levels → None")
        return None

    try:
        with _arduino_lock:
            arduino_serial.reset_input_buffer()
            time.sleep(ARDUINO_BEFORE_SENSOR_DELAY_SEC)
            arduino_serial.write(b'F')
            arduino_serial.flush()
            time.sleep(ARDUINO_AFTER_WRITE_DELAY_SEC)
            print("[ARDUINO] Gửi lệnh 'F' — đọc mức đầy cảm biến siêu âm")

            deadline = time.time() + timeout
            dist_values = None
            while time.time() < deadline:
                raw = arduino_serial.readline().decode('utf-8', errors='ignore').strip()
                if not raw:
                    time.sleep(ARDUINO_SENSOR_RETRY_DELAY_SEC)
                    continue
                print(f"[ARDUINO] ← {raw}")
                # Định dạng: DIST:12.3,8.5,25.0,-1  (thứ tự: org, rec, haz, oth)
                if raw.startswith("DIST:"):
                    try:
                        parts = raw[5:].split(",")  # bỏ "DIST:"
                        if len(parts) == 4:
                            dist_values = [float(p) for p in parts]
                            break
                    except ValueError:
                        pass  # format lỗi → đọc tiếp
                # Bỏ qua các dòng debug khác
            else:
                print("[ARDUINO] TIMEOUT đọc mức đầy cảm biến")
                return None

        # Ghép dist_values theo thứ tự: ORGANIC, RECYCLABLE, HAZARDOUS, OTHER
        bin_order = list(BIN_TO_SENSOR_KEY.keys())
        distance_by_bin = {}
        for i, bin_name in enumerate(bin_order):
            distance_by_bin[bin_name] = dist_values[i]

        result = _build_fill_result(distance_by_bin)
        print(f"[ARDUINO] Mức đầy: {result}")
        return result

    except Exception as e:
        print(f"[ARDUINO ERROR] read_fill_levels: {e}")
        return None


def read_fill_levels_simulated(added_bin: Optional[str] = None) -> dict:
    """
    Giả lập dữ liệu cảm biến siêu âm — dùng để test phần mềm
    khi chưa có phần cứng thực tế.

    Args:
        added_bin: Ngăn vừa nhận thêm rác. Nếu truyền vào, khoảng cách siêu âm
                   của ngăn đó giảm 5-10% chiều cao thùng rác.

    Returns:
        Cùng định dạng với read_fill_levels().
    """
    with _simulated_fill_lock:
        if added_bin in _simulated_distance_cm:
            delta = random.uniform(
                SIMULATED_DROP_MIN_RATIO,
                SIMULATED_DROP_MAX_RATIO,
            ) * BIN_TRASH_HEIGHT_CM
            old_dist = _simulated_distance_cm[added_bin]
            new_dist = max(_min_distance_for_full(added_bin), old_dist - delta)
            _simulated_distance_cm[added_bin] = new_dist
            print(
                f"[ARDUINO] (simulated) {added_bin}: "
                f"distance {old_dist:.1f}cm → {new_dist:.1f}cm "
                f"(+{delta:.1f}cm rác)"
            )

        result = _build_fill_result(_simulated_distance_cm)

    print(f"[ARDUINO] (simulated) Mức đầy: {result}")
    return result
