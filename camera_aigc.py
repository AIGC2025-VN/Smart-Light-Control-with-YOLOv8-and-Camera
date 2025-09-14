import threading
import time
import cv2
import requests
from ultralytics import YOLO

# Biến dùng chung để lưu frame mới nhất
latest_frame = None
frame_lock = threading.Lock()

# Biến trạng thái hiện tại của đèn (None, 'on' hoặc 'off')
current_light_state = None


def camera_thread():
    """Luồng đọc camera và cập nhật latest_frame."""
    global latest_frame
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Không mở được camera!")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        # Cập nhật frame mới nhất một cách thread-safe
        with frame_lock:
            latest_frame = frame.copy()
        time.sleep(0.01)  # nghỉ ngơi một chút để tránh chiếm CPU quá mức

    cap.release()


def processing_thread():
    """Luồng xử lý frame với YOLOv8 và điều khiển đèn."""
    global latest_frame, current_light_state
    # Khởi tạo mô hình YOLOv8 (sử dụng phiên bản yolov8n.pt nhẹ)
    model = YOLO('yolov8n.pt')

    while True:
        # Lấy frame mới nhất
        with frame_lock:
            if latest_frame is None:
                frame = None
            else:
                frame = latest_frame.copy()

        if frame is None:
            time.sleep(0.01)
            continue

        # Chạy dự đoán với YOLOv8
        results = model(frame)

        # Đếm số người phát hiện (trong COCO, "person" có id = 0)
        person_count = 0
        if results[0].boxes is not None:
            for box in results[0].boxes:
                if int(box.cls.item()) == 0:
                    person_count += 1

        # Điều khiển đèn dựa vào số người phát hiện
        if person_count > 0:
            if current_light_state != 'on':
                try:
                    # Gửi yêu cầu bật đèn
                    requests.get("http://192.168.1.111/update?relay=1&state=1")
                    current_light_state = 'on'
                    print("Đã bật đèn (phát hiện người)")
                except Exception as e:
                    print("Lỗi bật đèn:", e)
        else:
            if current_light_state != 'off':
                try:
                    # Gửi yêu cầu tắt đèn
                    requests.get("http://192.168.1.111/update?relay=1&state=0")
                    current_light_state = 'off'
                    print("Đã tắt đèn (không phát hiện người)")
                except Exception as e:
                    print("Lỗi tắt đèn:", e)

        # Vẽ kết quả phát hiện (bounding box) lên frame
        annotated_frame = results[0].plot()
        cv2.imshow("YOLOv8 Detection", annotated_frame)

        # Nhấn 'q' để thoát
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


def main():
    # Tạo và khởi chạy 2 luồng: một luồng đọc camera, một luồng xử lý YOLO
    t1 = threading.Thread(target=camera_thread)
    t2 = threading.Thread(target=processing_thread)
    t1.start()
    t2.start()
    t1.join()
    t2.join()


if __name__ == '__main__':
    main()
