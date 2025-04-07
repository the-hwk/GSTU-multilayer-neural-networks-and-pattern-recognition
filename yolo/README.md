## Использование модели YOLO для детекции

Для написания и выполнения кода лучше использовать свою среду Python, т.к. требуется доустановить дополнительные библиотеки.

Необходимые библиотеки:
- `ultralytics`
- `cv2`

Установка:

```
pip install <название>
```

#### Общий алгоритм использования модели YOLO

1. Подготовить набор данных. Т.к. набор данных представляет собой картинки, то лучше найти уже готовый датасет. Они, как правило, уже подготовлены по структуре для YOLO
2. Загрузить базовую модель YOLO `yolov8n.pt`
3. Обучить модель
4. Протестировать на тестовых изображениях

#### Написание кода

```Python
# Импорт библиотек
from ultralytics import YOLO
from matplotlib import pyplot as plt
import cv2
```

```Python
# Здесь указываем путь до .yaml файла с описанием датасета
data_yaml = "path_to_yaml"
```

```Python
# Загрузка модели
model = YOLO("yolov8n.pt")
```

```Python
# Дообучение базовой модели
model.train(
    data=data_yaml,
    epochs=5,
    imgsz=320,
    batch=8,
    project="runs/train",        # папка, в которую модель будет сохранена
    name="custom_yolo_model",    # название модели
    exist_ok=True
)
```

```Python
# Тестируем дообученную модель
metrics = model.val()
print(metrics)
```

```Python
# Указываем путь к весам модели
trained_model_path = "./runs/train/custom_yolo_model/weights/best.pt"
# И загружаем ее
model = YOLO(trained_model_path)
```

Теперь посмотрим на результат.

```Python
# Путь к тестовому изображению
img_path = "path_to_img"
# Даем картинку модели
results = model(img_path, save=False, show=False)
```

```Python
# Отобразим box (прямоугольник) на картинке
for result in results:
    boxes = result.boxes
    for box in boxes:
        # Координаты прямоугольника
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Название класса и подпись
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = f"{model.names[cls_id]} {conf:.2f}"

        # Нарисуем прямоугольник
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Подпись к рамке
        cv2.putText(img, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()
```