import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

# Загрузка модели YOLOv8
model = YOLO('models/yolov8s.pt')


# Функция для определения направления движения
def determine_direction(motion):
    if motion[0] > 0 and abs(motion[0]) > abs(motion[1]):
        if motion[1] > 0:
            return "Down-Right"
        elif motion[1] < 0:
            return "Up-Right"
        else:
            return "Right"
    elif motion[0] < 0 and abs(motion[0]) > abs(motion[1]):
        if motion[1] > 0:
            return "Down-Left"
        elif motion[1] < 0:
            return "Up-Left"
        else:
            return "Left"
    elif motion[1] > 0 and abs(motion[1]) > abs(motion[0]):
        return "Down"
    elif motion[1] < 0 and abs(motion[1]) > abs(motion[0]):
        return "Up"
    else:
        return "Not moving"


# Функция для добавления текста на изображение
def add_text_to_image(image, text, position, color):
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)
    font = ImageFont.truetype("arial.ttf", 20)
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


# Функция для обнаружения автомобилей
def detect_cars(input_path, output_path=None, skip_frames=0, scale_percent=0.5, show_frames=True, motion_threshold=1.2):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Ошибка при открытии видеофайла")
        return

    ret, frame2 = cap.read()
    if not ret:
        print("Ошибка при чтении первого кадра")
        return

    frame_count = 0

    # Уменьшение размера изображения
    width = int(frame2.shape[1] * scale_percent)
    height = int(frame2.shape[0] * scale_percent)
    dim = (width, height)
    frame2_resized = cv2.resize(frame2, dim, interpolation=cv2.INTER_AREA)

    if output_path is not None:
        # Инициалзиция VideoWriter для записи результата работы
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS)
        output_video = cv2.VideoWriter(output_path, fourcc, fps, (frame2_resized.shape[1], frame2_resized.shape[0]))

    while cap.isOpened():
        frame_count += 1

        # Условие для пропуска кадров
        if skip_frames > 0:
            if frame_count % skip_frames != 0:
                continue

        frame1 = frame2
        frame1_resized = frame2_resized

        ret, frame2 = cap.read()
        # Проверка на наличие кадра
        if not ret:
            break

        frame2_resized = cv2.resize(frame2, dim, interpolation=cv2.INTER_AREA)

        # Обнаружение объектов с использованием YOLOv8
        results = model(frame2_resized)

        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])

                if confidence > 0.5 and class_id == 2:  # 2 - класс автомобиля
                    # Вычисление оптического потока только для области автомобиля
                    gray1 = cv2.cvtColor(frame1_resized[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
                    gray2 = cv2.cvtColor(frame2_resized[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
                    flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)

                    # Вычисление среднего движения в области объекта
                    motion = np.mean(flow, axis=(0, 1))
                    motion_magnitude = np.linalg.norm(motion)

                    # Определение, движется ли объект
                    if motion_magnitude > motion_threshold:  # Порог для определения движения
                        color = (0, 255, 0)  # Зеленый цвет рамки для движущихся машин
                        direction = determine_direction(motion)
                        print(
                            f"Обнаружен движущийся автомобиль в координатах: ({x1}, {y1}) - ({x2}, {y2}), направление: {direction}")
                        frame2_resized = add_text_to_image(frame2_resized, direction, (x1 + 5, y1 + 5), color)
                    else:
                        color = (0, 0, 255)  # Зеленый цвет рамки для стоящих машин
                        # print(f"Обнаружен стоящий автомобиль в координатах: ({x1}, {y1}) - ({x2}, {y2})")

                    # Добавление рамки
                    cv2.rectangle(frame2_resized, (x1, y1), (x2, y2), color, 2)

        if output_path is not None:
            # Записываем текущий кадр в видеофайл MP4
            output_video.write(frame2_resized)

        # Отображение кадра
        if show_frames:
            cv2.imshow("frame", frame2_resized)

            # Прерывание программы по нажатию Esc
            if cv2.waitKey(40) == 27:
                break

    cap.release()
    if output_path is not None:
        output_video.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    video_input_path = 'data/traffic_videos/video1.mp4'
    video_output_path = 'data/output_videos/video1.mp4'
    detect_cars(video_input_path)
