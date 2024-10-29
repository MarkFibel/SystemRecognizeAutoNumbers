import torch
import cv2
import os
import warnings
warnings.filterwarnings("ignore")

# Загрузка модели YOLOv5
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Параметры сохранения
output_folder = "Resources/Scanned"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Параметры для обработки изображений
input_folder = "Resources/Images"  # Папка с входными изображениями
image_files = os.listdir(input_folder)  # Получаем список файлов в папке

count = 0  # Счетчик сохраненных изображений

for image_file in image_files:
    if count >= 5:  # Ограничиваем до 5 изображений
        print("Image limit reached. Stopping capture.")
        break

    # Загружаем изображение
    img_path = os.path.join(input_folder, image_file)
    img = cv2.imread(img_path)

    if img is None:
        print(f"Error: Could not read image {image_file}.")
        continue

    # Выполнение предсказания с помощью YOLOv5
    results = model(img)

    # Проверка на наличие результатов
    if results.xyxy[0].size(0) == 0:
        print(f"No objects detected in {image_file}.")
    else:
        # Извлечение информации о каждом обнаруженном объекте
        for *box, conf, cls in results.xyxy[0]:
            x1, y1, x2, y2 = map(int, box)  # Координаты рамки
            label = model.names[int(cls)]  # Метка класса
            if label == "license plate" and conf > 0.5:  # Фильтрация только номерных знаков
                imgRoi = img[y1:y2, x1:x2]
                filename = os.path.join(output_folder, f"NoPlate_{count}.jpg")
                cv2.imwrite(filename, imgRoi)
                print(f"Image saved as {filename}")
                count += 1
                if count >= 5:
                    print("Image limit reached. Stopping capture.")
                    break

# Завершение работы
print("Program finished.")