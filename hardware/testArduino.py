import serial
import serial.tools.list_ports
import time

def list_available_ports():
    print("--- Các cổng Serial đang kết nối ---")
    ports = serial.tools.list_ports.comports()
    for i, p in enumerate(ports):
        print(f"[{i}] {p.device} - {p.description}")
    return ports

def main():
    ports = list_available_ports()
    if not ports:
        print("Không tìm thấy cổng Serial nào!")
        return

    # Chọn cổng
    try:
        idx = int(input("\nNhập số thứ tự cổng muốn kết nối: "))
        selected_port = ports[idx].device
    except (ValueError, IndexError):
        print("Lựa chọn không hợp lệ.")
        return

    # Khởi tạo kết nối
    try:
        ser = serial.Serial(port=selected_port, baudrate=9600, timeout=1)
        print(f"Đang kết nối tới {selected_port}...")
        time.sleep(2) # Chờ Arduino Reset
        
        print("\n--- ĐIỀU KHIỂN THÙNG RÁC THÔNG MINH ---")
        print("Nhập lệnh:")
        print("  0: ORGANIC (Phải)")
        print("  1: RECYCLABLE (Trái)")
        print("  2: HAZARDOUS (Dưới-Phải)")
        print("  3: OTHER (Dưới-Trái)")
        print("  q: Thoát")
        print("--------------------------------------")

        while True:
            cmd = input("Nhập lệnh (0,1,2,3): ").strip().lower()

            if cmd == 'q':
                break
            
            if cmd in ['0', '1', '2', '3']:
                # Gửi lệnh xuống Arduino
                ser.write(cmd.encode())
                print(f">> Đã gửi lệnh: {cmd}")
                
                # Đợi phản hồi (ACK) từ Arduino (Nếu code Arduino có Serial.println)
                # Script sẽ đợi tối đa 5 giây để nhận "Hoan thanh"
                start_time = time.time()
                while (time.time() - start_time) < 5:
                    if ser.in_waiting > 0:
                        line = ser.readline().decode('utf-8', errors='ignore').strip()
                        print(f"<< Arduino phản hồi: {line}")
                        if "Hoan thanh" in line:
                            break
            else:
                print("Lệnh không hợp lệ! Vui lòng nhập 0, 1, 2, 3 hoặc q.")

    except Exception as e:
        print(f"Lỗi: {e}")
    finally:
        if 'ser' in locals() and ser.is_open:
            ser.close()
            print("\nĐã đóng kết nối Serial.")

if __name__ == "__main__":
    main()