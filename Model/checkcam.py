import cv2

def check_cameras():
    for i in range(5): # Thử từ 0 đến 4
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"Camera ID {i} đang hoạt động!")
                cv2.imshow(f"Test Camera {i}", frame)
                cv2.waitKey(2000) # Hiện 2 giây rồi tắt
                cv2.destroyAllWindows()
        cap.release()

check_cameras()