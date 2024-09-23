import cv2
from ultralytics import YOLO

# Загрузка модели YOLOv8
model = YOLO('models/yolov8s.pt')


# Функция для обнаружения автомобилей
def detect_cars(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Ошибка при открытии видеофайла")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Уменьшение размера изображения
        scale_percent = 0.5  # Множитель уменьшения - 50%
        width = int(frame.shape[1] * scale_percent)
        height = int(frame.shape[0] * scale_percent)
        dim = (width, height)
        frame_resized = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

        # Обнаружение объектов с использованием YOLOv8
        results = model(frame_resized)

        # Анализ результатов
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])

                if confidence > 0.5 and class_id == 2:  # 2 - класс автомобиля
                    cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    print(f"Обнаружен автомобиль в координатах: ({x1}, {y1}) - ({x2}, {y2})")

        cv2.imshow("frame", frame_resized)

        # Остановка программы по нажатию Esc
        if cv2.waitKey(40) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    video_path = 'data/traffic_videos/video1.mp4'
    detect_cars(video_path)
