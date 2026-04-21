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
import json
import threading
from typing import Optional, Callable

import serial
import serial.tools.list_ports

# ============================================================
# CẤU HÌNH
# ============================================================

ARDUINO_BAUDRATE   = 9600
ARDUINO_TIMEOUT    = 8.0           # giây chờ ACK lệnh servo
ARDUINO_ACK_PREFIX = "Hoan thanh"  # prefix phản hồi từ Arduino

# Độ sâu (cm) khi ngăn rác CÒN TRỐNG hoàn toàn — dùng để tính % đầy.
# Chỉnh lại theo kích thước thực tế của từng ngăn khi lắp cảm biến.
BIN_DEPTH_CM = {
    "ORGANIC":    30.0,
    "RECYCLABLE": 30.0,
    "HAZARDOUS":  20.0,
    "OTHER":      30.0,
}

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
        time.sleep(2.0)   # Chờ Arduino reset sau khi mở serial
        ser.reset_input_buffer()

        # Chờ "Ready" từ Arduino
        deadline = time.time() + 5.0
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
                    arduino_serial.write(cmd.encode())
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

    Protocol Arduino (cần thêm vào file .ino):
        - Nhận ký tự 'F'
        - Đo 4 cảm biến, trả về JSON 1 dòng:
          {"org":12,"rec":34,"haz":5,"oth":67}   ← khoảng cách (cm)
        - Nếu cảm biến lỗi, trả giá trị -1 cho key đó.

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
            arduino_serial.write(b'F')
            print("[ARDUINO] Gửi lệnh 'F' — đọc mức đầy cảm biến siêu âm")

            deadline = time.time() + timeout
            while time.time() < deadline:
                raw = arduino_serial.readline().decode('utf-8', errors='ignore').strip()
                if not raw:
                    continue
                print(f"[ARDUINO] ← {raw}")
                try:
                    sensor_data = json.loads(raw)
                    break
                except json.JSONDecodeError:
                    # Bỏ qua dòng không phải JSON (vd: dòng debug)
                    continue
            else:
                print("[ARDUINO] TIMEOUT đọc mức đầy cảm biến")
                return None

        result = {}
        for bin_name, sensor_key in BIN_TO_SENSOR_KEY.items():
            dist = float(sensor_data.get(sensor_key, -1))
            depth = BIN_DEPTH_CM.get(bin_name, 30.0)

            if dist < 0:
                fill_pct = None          # cảm biến báo lỗi
            else:
                # Khi dist = 0 → đầy 100%, khi dist = depth → đầy 0%
                fill_pct = round((1.0 - dist / depth) * 100.0, 1)
                fill_pct = max(0.0, min(100.0, fill_pct))

            result[bin_name] = {
                "distance_cm": dist,
                "fill_pct":    fill_pct,
            }

        print(f"[ARDUINO] Mức đầy: {result}")
        return result

    except Exception as e:
        print(f"[ARDUINO ERROR] read_fill_levels: {e}")
        return None


def read_fill_levels_simulated() -> dict:
    """
    Giả lập dữ liệu cảm biến siêu âm — dùng để test phần mềm
    khi chưa có phần cứng thực tế.

    Returns:
        Cùng định dạng với read_fill_levels().
    """
    import random
    result = {}
    for bin_name, depth in BIN_DEPTH_CM.items():
        dist     = round(random.uniform(2.0, depth), 1)
        fill_pct = round((1.0 - dist / depth) * 100.0, 1)
        result[bin_name] = {
            "distance_cm": dist,
            "fill_pct":    fill_pct,
        }
    print(f"[ARDUINO] (simulated) Mức đầy: {result}")
    return result