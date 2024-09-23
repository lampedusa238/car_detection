# Детекция автомобилей с использованием YOLOv8 и оптического потока

## Описание
Этот проект использует модель YOLOv8 для детекции автомобилей на видео, а также вычисляет движение объектов с помощью оптического потока. 

Результаты обработки видео отображаются с визуальными рамками и информацией о движении автомобилей:
- красная рамка - автомобиль неподвижен;
- зеленая рамка - автомобиль движется.

Направление движения автомобилей указывается в консоль, а также выводится внутри каждой рамки.

![Результат работы](data/gif/output_demo.gif)

## Установка и настройка с использованием Poetry

Для управления зависимостями и виртуальной средой в проекте используется [Poetry](https://python-poetry.org/).

### Установка Poetry (если еще не установлен)

Следуйте инструкциям на [официальном сайте](https://python-poetry.org/docs/#installation).

### Установка проекта

1. Клонируйте репозиторий:
    ```bash
    git clone https://github.com/lampedusa238/car_detection.git
    cd car_detection
    ```

2. Установите зависимости с помощью Poetry:
    ```bash
    poetry install
    ```

3. Активируйте виртуальную среду:
    ```bash
    poetry shell
    ```
   
### Использование

Для запуска скрипта с помощью Poetry используйте команду:
   ```bash
   poetry run python main.py 
   ```

Для обнаружения автомобилей на видео, достаточно указать только входной путь для видео в `main.py`:
```Python
if __name__ == '__main__':
   input_video_path = 'YOUR_INPUT_VIDEO_PATH'
   detect_cars('input_video_path', {output_path='YOUR_OUTPUT_VIDEO_PATH'}, {skip_frames=0}, {scale_percent=0.5}, {show_frames=True}, {motion_threshold=1.2})
```
Однако, также можно указать следующие параметры:
- output_path (`string`, default: None) — путь для созранения выходного видеофайла в формате mp4 (например, `output_video.mp4`)
- skip_frames (`int`, default: 0) — номер кадра который будем пропускать (например 0 - ничего не пропускаем, 1 - пропускаем все, 2 - пропускаем кадый 2 кадр, 3 - пропускаем каждый 3-ий кадр и тд)
- scale_percent (`float`, default: 0.5) — масштаб изображения для ускорения работы (1 - 100%, 0.5 - 50% и тд)
- show_frames (`boolean`, default: True) — отображать ли предпросмотр разметки
- motion_threshold (`float`, default:1.2) — пороговое значение для определения движения


При плохом обнаружении движущихся автомобилей можно изменить параметры skip_frames, scale_percent, motion_threshold или использовать более точную модель YOLOv8 (например, yolov8m.pt).