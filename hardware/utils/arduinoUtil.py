"""
arduinoUtil.py  —  Arduino Serial Utility
==========================================
Chứa toàn bộ cấu hình, khởi tạo và logic giao tiếp Serial
với Arduino để điều khiển servo trong hệ thống smart trash bin.

Commands:
    '0' → servo RIGHT       (ORGANIC / Biological)
    '1' → servo LEFT        (RECYCLABLE)
    '2' → servo DOWN-RIGHT  (HAZARDOUS / Battery)
    '3' → servo DOWN-LEFT   (OTHER / General_Waste)

Yêu cầu:
    pip install pyserial
"""

import time
import threading

import serial
import serial.tools.list_ports

# ============================================================
# CẤU HÌNH ARDUINO
# ============================================================

ARDUINO_PORT       = None          # None = tự động tìm cổng Arduino
ARDUINO_BAUDRATE   = 9600
ARDUINO_TIMEOUT    = 8.0           # giây chờ ACK tối đa
ARDUINO_ACK_PREFIX = "Hoan thanh"  # prefix phản hồi từ Arduino

# Map ngăn → Arduino command
BIN_TO_ARDUINO_CMD = {
    "ORGANIC":    '0',   # servo RIGHT
    "RECYCLABLE": '1',   # servo LEFT
    "HAZARDOUS":  '2',   # servo DOWN-RIGHT
    "OTHER":      '3',   # servo DOWN-LEFT
}

# ============================================================
# KHỞI TẠO ARDUINO
# ============================================================

def list_available_ports():
    """In danh sách các cổng Serial đang kết nối."""
    print("--- Các cổng Serial đang kết nối ---")
    ports = serial.tools.list_ports.comports()
    for i, p in enumerate(ports):
        print(f"[{i}] {p.device} - {p.description}")
    return ports


def find_arduino_port() -> str | None:
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


def init_arduino():
    """
    Khởi tạo kết nối Serial với Arduino.
    Trả về đối tượng serial.Serial nếu thành công, None nếu thất bại.
    """
    port = ARDUINO_PORT or find_arduino_port()
    if port is None:
        print("[WARN] Không tìm thấy cổng Arduino — chạy không có Arduino.")
        return None
    try:
        ser = serial.Serial(port, ARDUINO_BAUDRATE, timeout=1)
        time.sleep(2.0)   # Chờ Arduino reset sau khi mở serial
        ser.reset_input_buffer()
        # Chờ "Ready" từ Arduino
        deadline = time.time() + 5.0
        while time.time() < deadline:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if line == "Ready":
                print(f"[ARDUINO] Sẵn sàng trên {port} @ {ARDUINO_BAUDRATE} baud")
                return ser
            elif line:
                print(f"[ARDUINO] Startup: {line}")
        print(f"[ARDUINO] Không nhận được 'Ready' — tiếp tục dù sao.")
        return ser
    except Exception as e:
        print(f"[ARDUINO ERROR] {e}")
        return None


# ============================================================
# GỬI LỆNH ARDUINO
# ============================================================

# Lock để tránh gửi nhiều lệnh cùng lúc
_arduino_lock = threading.Lock()

# Event báo hiệu Arduino đã xong
_arduino_done_event = threading.Event()
_arduino_done_event.set()   # Khởi đầu: free


def arduino_send_command(arduino_serial, bin_type: str, on_done_callback=None):
    """
    Gửi lệnh Arduino trong thread riêng.
    Chờ ACK "Hoan thanh X" rồi gọi callback.

    Args:
        arduino_serial:     Đối tượng serial.Serial từ init_arduino().
                            Nếu None thì giả lập delay servo.
        bin_type:           Tên ngăn rác (ORGANIC / RECYCLABLE / HAZARDOUS / OTHER).
        on_done_callback:   Hàm gọi sau khi Arduino xác nhận hoàn thành (hoặc timeout).
    """
    cmd = BIN_TO_ARDUINO_CMD.get(bin_type, '3')

    def _send():
        _arduino_done_event.clear()
        try:
            if arduino_serial is None:
                print(f"[ARDUINO] (offline) Lệnh: {cmd} → {bin_type}")
                time.sleep(2.0)   # Giả lập delay servo
            else:
                with _arduino_lock:
                    arduino_serial.reset_input_buffer()
                    arduino_serial.write(cmd.encode())
                    print(f"[ARDUINO] Gửi lệnh '{cmd}' → {bin_type}")

                    # Chờ ACK
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
            print(f"[ARDUINO ERROR] send failed: {e}")
        finally:
            _arduino_done_event.set()
            if on_done_callback:
                on_done_callback()

    threading.Thread(target=_send, daemon=True).start()