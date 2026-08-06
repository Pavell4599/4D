import os
import torch
import numpy as np
import gradio as gr

# Защита от конфликта библиотек OpenMP
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Импортируем вашу модель
from model import PointNet 

CLASSES = ['bathtub', 'bed', 'chair', 'desk', 'dresser', 'monitor', 'night', 'sofa', 'table', 'toilet']
WEIGHTS_PATH = "best_pointnet_modelnet10.pth" 

# =============================================================================
# ИНИЦИАЛИЗАЦИЯ МОДЕЛИ И ВЕСОВ
# =============================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = len(CLASSES)
model = PointNet(num_classes=num_classes)

if not os.path.exists(WEIGHTS_PATH):
    raise FileNotFoundError(f"Файл весов '{WEIGHTS_PATH}' не найден!")

model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device, weights_only=True))
model.eval()
model.to(device)
print(f"Модель успешно загружена на {device}. Родной парсер .off готов!")

# =============================================================================
# ВАША СТАНДАРТНАЯ ФУНКЦИЯ ЧТЕНИЯ .OFF ФАЙЛА (из скрипта обучения)
# =============================================================================
def read_off(file_path):
    with open(file_path, 'r') as f:
        # Проверяем заголовок файла
        header = f.readline().strip()
        if 'OFF' not in header:
            raise ValueError("Некорректный заголовок .OFF файла")
        
        # Если заголовок склеен со значениями (например, OFF 400 120 0)
        if len(header) > 3:
            line = header[3:].strip().split()
        else:
            line = f.readline().strip().split()
            
        if not line:
            line = f.readline().strip().split()
            
        n_verts = int(line[0])
        
        # Читаем координаты всех вершин
        verts = []
        for _ in range(n_verts):
            verts.append([float(x) for x in f.readline().strip().split()])
            
        return np.array(verts)

# =============================================================================
# ФУНКЦИЯ ОБРАБОТКИ И ИНФЕРЕНСА
# =============================================================================
def predict_only(file_obj):
    if file_obj is None:
        return {"Файл не выбран": 1.0}
    
    try:
        # 1. Читаем облако точек с помощью проверенного метода
        verts = read_off(file_obj)
        
        # 2. Если точек больше или меньше 2048 — делаем сэмплирование/выборку
        num_points = 2048
        if len(verts) >= num_points:
            # Случайный выбор 2048 точек без повторений
            choice = np.random.choice(len(verts), num_points, replace=False)
        else:
            # Если точек не хватает, выбираем с повторениями
            choice = np.random.choice(len(verts), num_points, replace=True)
        points = verts[choice, :]
        
        # 3. Детерминированная нормализация (как при обучении)
        points = points - np.mean(points, axis=0)
        max_dist = np.max(np.linalg.norm(points, axis=1))
        if max_dist > 0:
            points = points / max_dist
            
        # Формируем тензор для вашей PointNet: [Batch=1, Num_Points=2048, Channels=3]
        points_tensor = torch.tensor(points, dtype=torch.float32).unsqueeze(0).to(device)
        
        # 4. Прогон через нейросеть
        with torch.no_grad():
            outputs = model(points_tensor)
            probabilities = torch.softmax(outputs, dim=1).cpu().squeeze(0).numpy()
            
        # Возвращаем результаты в формате {Класс: Вероятность}
        results = {CLASSES[i]: float(probabilities[i]) for i in range(num_classes)}
        return results
        
    except Exception as e:
        return {f"Ошибка при чтении или обработке .off файла: {str(e)}": 1.0}

# =============================================================================
# ЛЕГКИЙ ИНТЕРФЕЙС GRADIO
# =============================================================================
with gr.Blocks(title="PointNet Fast Classifier") as demo:
    gr.Markdown("# 🏢 Быстрый классификатор 3D-файлов (PointNet)")
    gr.Markdown("Загрузите любой файл формата **.off** из вашего датасета для мгновенного распознавания класса.")
    
    with gr.Row():
        with gr.Column():
            # Обычное поле загрузки файлов
            file_input = gr.File(label="Загрузите .off файл")
            submit_btn = gr.Button("Угадать класс", variant="primary")
            
        with gr.Column():
            # Результаты предсказания
            output_labels = gr.Label(label="Результат (Top-3 вероятных класса)", num_top_classes=3)
            
    # Привязываем обработку к кнопке
    submit_btn.click(
        fn=predict_only, 
        inputs=file_input, 
        outputs=output_labels
    )

if __name__ == "__main__":
    demo.launch()

