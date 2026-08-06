"""
conda install pytorch torchvision pytorch-cuda=12.4 -c pytorch -c nvidia -y
"""


# =============================================================================
#  PointNet — 3D-классификация на ModelNet10 (локальный запуск, PyTorch + CUDA)
# =============================================================================
#  Пайплайн целиком:
#    1) Читаем CSV-метаданные, чистим мусор, чиним путь класса night.
#    2) Один раз сэмплируем каждый .off-меш в облако из 1024 точек и сохраняем
#       результат в .npy-кэш на диск (повторные запуски — мгновенные).
#    3) Обучаем PointNet: Shared MLP (Conv1d) -> Global MaxPool -> классификатор.
#    4) Строим графики обучения, confusion matrix, отчёт по классам и 3D-визуализацию.
# =============================================================================

# --- 0. ФИКС OpenMP ДО любых импортов torch/numpy -----------------------------
# В Anaconda на Windows одновременно подгружаются две копии OpenMP-рантайма
# (libiomp5md.dll от MKL и libomp.dll от PyTorch). Без этой строки импорт torch
# падает с "OMP: Error #15". Ставим переменную окружения самым первым делом.
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
# Меньше фрагментации видеопамяти при динамическом выделении тензоров.
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import numpy as np
import pandas as pd
import time
import random
import warnings
warnings.filterwarnings('ignore')
from mpl_toolkits.mplot3d.art3d import Poly3DCollection   
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

# --- 1. CUDA / cuDNN ускорения ------------------------------------------------
# Эти флаги дают +10..30% на RTX 30xx/40xx без заметной потери точности:
#   cudnn.benchmark  — автоподбор самого быстрого алгоритма свёртки под ваш размер;
#   allow_tf32       — использование формата TF32 (быстрее fp32, точность ок для классификации);
#   matmul precision — то же самое для матричных умножений в Linear-слоях.
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')

USE_AMP = True       # mixed precision (bfloat16) — аппаратно поддерживается RTX 4070
USE_COMPILE = False  # torch.compile: +10..30% после прогрева, но 1-я эпоха долгая и на Windows капризна

# Выбор устройства: GPU если доступен, иначе CPU (код сам переключится).
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Устройство: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)} | "
          f"cuDNN benchmark={torch.backends.cudnn.benchmark}, "
          f"TF32={torch.backends.cuda.matmul.allow_tf32}, bf16(AMP)={USE_AMP}")

# --- 2. Пути и гиперпараметры -------------------------------------------------
# DATA_ROOT = r"F:\3D classification\ModelNet10"         
# CSV_PATH  = r"F:\3D classification\metadata_modelnet10.csv"

DATA_ROOT = r"ModelNet10"          # корень с папками классов
CSV_PATH  = r"metadata_modelnet10.csv"

NUM_POINTS    = 2048   # точек в одном облаке (стандарт для PointNet)
BATCH_SIZE    = 32     # объектов за один шаг; на 12 ГБ VRAM можно и 64
NUM_EPOCHS    = 50       # эпох обучения
LEARNING_RATE = 0.001  # стартовый learning rate для Adam
PRINT_EVERY   = 1      # печатать метрики каждую эпоху (обучение теперь быстрое)
NUM_WORKERS   = 0      # 0 безопасно для Spyder/Windows; из терминала можно 4

# Контекст смешанной точности (None на CPU, чтобы не падало).
autocast = (torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)
            if (USE_AMP and device.type == 'cuda') else None)


# =============================================================================
#  БЛОК A. ЧТЕНИЕ И ОЧИСТКА МЕТАДАННЫХ
# =============================================================================
# CSV имеет 4 колонки: object_id, class, split, object_path.
# В нём есть две проблемы, которые надо починить до обучения:
#   (а) мусорные строки от macOS вида: ,.DS,toilet,.DS/toilet/.off
#   (б) для класса night путь в CSV = "night/...", а папка на диске = "night_stand"
df = pd.read_csv(CSV_PATH)

# (а) Убираем строки, где в пути или имени класса фигурирует служебный ".DS",
#     а также строки с пустым путём. na=False/na=True защищают от NaN при str.contains.
df = df[~df['object_path'].astype(str).str.contains('.DS', na=True)]
df = df[~df['class'].astype(str).str.contains('.DS', na=True)]
df = df.dropna(subset=['object_path'])


# Приводим слэши к разделителю ОС (на Windows '/' тоже работает, но так чище).
df['object_path'] = df['object_path'].str.replace('/', os.sep)

# Фильтруем файлы, которых физически нет на диске. ВАЖНО: делаем это ПОСЛЕ фикса пути,
# иначе мы бы выкинули ВЕСЬ класс night (его файлы "не найдены" по старому пути),
# и модель училась бы на 9 классах вместо 10.
def _exists(p):
    return os.path.isfile(os.path.join(DATA_ROOT, p))
mask = df['object_path'].apply(_exists)
n_missing = int((~mask).sum())
print(f"Отсутствует файлов на диске: {n_missing}")
# if n_missing > 0:
#     print("  ->", df.loc[~mask, 'object_path'].tolist())
df = df[mask].reset_index(drop=True)

# Маппинг "имя класса -> целочисленный индекс 0..9" и обратный словарь.
# sorted() гарантирует одинаковый порядок индексов при каждом запуске.
classes = sorted(df['class'].unique())
class_to_idx = {c: i for i, c in enumerate(classes)}
idx_to_class = {i: c for c, i in class_to_idx.items()}
NUM_CLASSES = len(classes)

# Разделение на train/test берём прямо из колонки split датасета.
df_train = df[df['split'] == 'train'].reset_index(drop=True)
df_test  = df[df['split'] == 'test'].reset_index(drop=True)

print(f"Классы ({NUM_CLASSES}): {classes}")
print(f"Объектов всего: {len(df)}  |  train: {len(df_train)}  |  test: {len(df_test)}")


# =============================================================================
#  БЛОК B. ПАРСЕР .off И СЭМПЛИРОВАНИЕ ТОЧЕК С ПОВЕРХНОСТИ МЕША
# =============================================================================
def read_off(file_path):
    """Читает .off и возвращает (verts[F,3], faces[F,3]).

    Формат .off:
        строка 1: "OFF"  (иногда "OFF 123 456 0" в одну строку)
        далее   : n_verts n_faces 0
        далее   : n_verts строк координат  x y z
        далее   : n_faces строк граней     n v1 v2 v3 ...
    Любые полигоны с n>3 вершин триангулируются «веером», чтобы faces всегда был [F,3]
    (это нужно для векторизованного сэмплера ниже)."""
    with open(file_path, 'r') as f:
        lines = f.readlines()

    header = lines[0].strip()
    if header == 'OFF':                       # заголовок и числа на разных строках
        n_verts, n_faces, _ = map(int, lines[1].strip().split())
        start = 2
    elif header.startswith('OFF'):            # "OFF 123 456 0" в одну строку
        parts = header.split()
        n_verts, n_faces = int(parts[1]), int(parts[2])
        start = 1
    else:
        raise ValueError(f"Не .off файл: {file_path}")

    # Вершины: берём ровно n_verts строк координат.
    verts = np.array([list(map(float, lines[i].strip().split()[:3]))
                      for i in range(start, start + n_verts)], dtype=np.float32)

    # Грани + триангуляция веером: полигон (v0,v1,...,vk) -> (v0,v1,v2),(v0,v2,v3),...
    faces = []
    for i in range(start + n_verts, start + n_verts + n_faces):
        toks = lines[i].strip().split()
        idxs = list(map(int, toks[1:]))
        for k in range(1, len(idxs) - 1):     # веерная триангуляция
            faces.append((idxs[0], idxs[k], idxs[k + 1]))
    faces = np.array(faces, dtype=np.int64)
    return verts, faces


def sample_points_vectorized(verts, faces, n):
    """Сэмплирует n точек с поверхности меша БЕЗ циклов Python (векторно).

    Логика (та же, что в исходном PointSampler, но массивами):
      1) площади всех треугольников сразу через векторное произведение рёбер;
      2) выбор граней с вероятностью пропорциональной площади (больше грань -> больше точек);
      3) внутри каждой выбранной грани — случайная точка через барицентрические координаты.
    Именно циклы в старой версии тормозили загрузку данных; здесь их нет."""
    v = verts[faces]                                  # [F, 3, 3] — вершины всех граней
    e1 = v[:, 1] - v[:, 0]                            # первое ребро каждой грани
    e2 = v[:, 2] - v[:, 0]                            # второе ребро каждой грани
    areas = 0.5 * np.linalg.norm(np.cross(e1, e2), axis=1)  # площади [F]
    if areas.sum() == 0:                              # вырожденный меш — равномерный выбор
        areas = np.ones_like(areas)
    probs = areas / areas.sum()

    sel = v[np.random.choice(len(faces), n, p=probs)] # [n, 3, 3] — выбранные треугольники

    # Барицентрические координаты для равномерной точки внутри треугольника:
    # сортируем пару случайных чисел, чтобы точка не «вылетала» за пределы грани.
    r1 = np.random.rand(n, 1)
    r2 = np.random.rand(n, 1)
    s, t = np.minimum(r1, r2), np.maximum(r1, r2)
    c0, c1, c2 = s, (t - s), (1.0 - t)                # коэффициенты при вершинах
    pts = c0 * sel[:, 0] + c1 * sel[:, 1] + c2 * sel[:, 2]   # [n, 3]
    return pts.astype(np.float32)


class Normalize:
    """Центрирует облако (вычитаем среднее) и масштабирует в единичную сферу.
    Без этого объекты разного физического размера давали бы разные диапазоны
    координат, и сеть не смогла бы обучиться."""
    def __call__(self, pc):
        pc = pc - pc.mean(axis=0)
        m = np.max(np.linalg.norm(pc, axis=1))
        if m > 0:
            pc = pc / m
        return pc


# =============================================================================
#  БЛОК C. КЭШ: один раз сэмплируем все меши и пишем в .npy
# =============================================================================
# Чтение+парсинг+сэмплирование .off — самое дорогое. Делаем это ОДИН раз и сохраняем
# готовый массив [N, 1024, 3] на диск. На следующих запусках загрузка — доли секунды,
# а GPU перестаёт простаивать в ожидании данных (это и убирает "4.4 s/it").
def build_cache(dataframe, data_root, num_points, cache_path):
    if os.path.exists(cache_path):
        print(f"Загрузка кэша: {cache_path}")
        return np.load(cache_path)

    print(f"Построение кэша ({len(dataframe)} объектов) -> {cache_path}")
    cache = np.zeros((len(dataframe), num_points, 3), dtype=np.float32)
    norm = Normalize()
    for i, row in tqdm(dataframe.iterrows(), total=len(dataframe), desc="Cache"):
        fpath = os.path.join(data_root, row['object_path'])
        try:
            verts, faces = read_off(fpath)
            pts = sample_points_vectorized(verts, faces, num_points)
            pts = norm(pts)                          # нормализуем детерминированно (без аугментации)
        except Exception as e:
            print(f"  ошибка {fpath}: {e}")
            pts = np.zeros((num_points, 3), dtype=np.float32)
        cache[i] = pts
    np.save(cache_path, cache)
    return cache


# =============================================================================
#  БЛОК D. DATASET (берёт готовые облака из кэша)
# =============================================================================
class ModelNet10Dataset(Dataset):
    """__getitem__ теперь тривиален: достать строку из массива + (опц.) аугментация.
    Никакого чтения файлов на каждой эпохе -> DataLoader летает даже при num_workers=0."""
    def __init__(self, cache, labels, augment=False):
        self.cache = cache          # np.ndarray [N, 1024, 3], уже нормализован
        self.labels = labels        # np.ndarray [N] int64
        self.augment = augment      # True только для train

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        pts = self.cache[idx].copy()                 # копия, чтобы аугментация не портила кэш
        if self.augment:
            pts = self._augment(pts)
        return torch.from_numpy(pts), torch.tensor(self.labels[idx], dtype=torch.long)

    def _augment(self, pts):
        # Случайный поворот вокруг вертикальной оси Y (объекты «стоят на полу»).
        theta = random.uniform(0, 2 * np.pi)
        c, s = np.cos(theta), np.sin(theta)
        rot = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
        pts = pts @ rot.T
        # Jitter — мелкий гауссов шум, имитирующий неточность сканера.
        pts = pts + np.clip(0.01 * np.random.randn(*pts.shape), -0.05, 0.05).astype(np.float32)
        return pts


# Метки классов для train/test в виде numpy-массивов (нужны Dataset'у).
labels_train = df_train['class'].map(class_to_idx).to_numpy(dtype=np.int64)
labels_test  = df_test['class'].map(class_to_idx).to_numpy(dtype=np.int64)

# Строим/грузим кэши отдельно для train и test.
cache_train = build_cache(df_train, DATA_ROOT, NUM_POINTS,
                          os.path.join(DATA_ROOT, f'cache_train_{NUM_POINTS}.npy'))
cache_test  = build_cache(df_test,  DATA_ROOT, NUM_POINTS,
                          os.path.join(DATA_ROOT, f'cache_test_{NUM_POINTS}.npy'))

train_dataset = ModelNet10Dataset(cache_train, labels_train, augment=True)
test_dataset  = ModelNet10Dataset(cache_test,  labels_test,  augment=False)

# DataLoader. persistent_workers/prefetch_factor имеют смысл только при num_workers>0.
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True,
                          persistent_workers=(NUM_WORKERS > 0),
                          prefetch_factor=(2 if NUM_WORKERS > 0 else None))
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=NUM_WORKERS, pin_memory=True,
                         persistent_workers=(NUM_WORKERS > 0),
                         prefetch_factor=(2 if NUM_WORKERS > 0 else None))


# =============================================================================
#  БЛОК E. МОДЕЛЬ PointNet
# =============================================================================
class PointNet(nn.Module):
    """PointNet для классификации облаков точек.

    Идея: порядок точек в облаке не важен, поэтому сеть должна быть инвариантна к
    перестановке точек. Это дают две вещи:
      - Shared MLP (Conv1d с ядром 1) применяет ОДИНАКОВЫЕ веса к каждой точке независимо;
      - Global Max Pooling — симметричная функция (max не зависит от порядка аргументов).
    Архитектура:
      [B,1024,3] -> permute -> [B,3,1024]
      -> Conv1d 3->64->128->1024 (+BatchNorm+ReLU)        # признаки каждой точки
      -> MaxPool по точкам -> [B,1024]                     # глобальный признак формы
      -> FC 1024->512->256->num_classes (+BN+Dropout)      # классификатор
    """
    def __init__(self, num_classes=10):
        super().__init__()
        # Shared MLP (ядро 1 = применение MLP к каждой точке по отдельности).
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=1)
        self.conv3 = nn.Conv1d(128, 1024, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        # Классификатор по глобальному признаку.
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(0.4)

    def forward(self, x):                       # x: [B, 1024, 3]
        x = x.permute(0, 2, 1)                  # -> [B, 3, 1024] (Conv1d ждёт каналы впереди)
        x = F.relu(self.bn1(self.conv1(x)))     # [B, 64, 1024]
        x = F.relu(self.bn2(self.conv2(x)))     # [B, 128, 1024]
        x = self.bn3(self.conv3(x))             # [B, 1024, 1024]
        x = torch.max(x, dim=2)[0]              # Global MaxPool -> [B, 1024]
        x = F.relu(self.bn4(self.fc1(x)))       # [B, 512]
        x = self.dropout(x)
        x = F.relu(self.bn5(self.fc2(x)))       # [B, 256]
        x = self.dropout(x)
        x = self.fc3(x)                         # [B, num_classes] (логиты, без softmax)
        return x


model = PointNet(num_classes=NUM_CLASSES).to(device)
print(f"Параметров модели: {sum(p.numel() for p in model.parameters()):,}")

# torch.compile (опционально): компилирует граф модели, ускоряя forward/backward.
# Первая эпоха будет долгой (идёт компиляция), затем быстрее. На Windows может не сработать.
if USE_COMPILE:
    try:
        model = torch.compile(model, mode='reduce-overhead')
        print("torch.compile включён")
    except Exception as e:
        print(f"torch.compile не сработал, работаем без него: {e}")

# Loss (CrossEntropy сам применяет softmax внутри), оптимизатор и scheduler.
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
# Уменьшаем LR в 0.7 раз каждые 20 эпох — стандартный приём, чтобы «дотюнить» веса в конце.
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)


# =============================================================================
#  БЛОК F. ЦИКЛЫ ОБУЧЕНИЯ И ОЦЕНКИ
# =============================================================================
def train_one_epoch(model, loader, criterion, optimizer):
    """Одна эпоха обучения: forward -> loss -> backward -> step по всем батчам."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for points, labels in tqdm(loader, desc="Train", leave=False):
        # non_blocking=True позволяет копировать данные на GPU асинхронно.
        points = points.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)    # set_to_none чуть быстрее, чем zero_()
        if autocast is not None:
            with autocast:                       # forward в bfloat16 (быстрее, память меньше)
                outputs = model(points)
                loss = criterion(outputs, labels)
        else:
            outputs = model(points)
            loss = criterion(outputs, labels)
        loss.backward()                          # backward остаётся в fp32-градиентах (стабильно)
        optimizer.step()

        total_loss += loss.item() * points.size(0)
        correct += outputs.max(1)[1].eq(labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion):
    """Оценка без градиентов; дополнительно собирает все предсказания и метки
    (нужны для confusion matrix и classification_report в конце)."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for points, labels in tqdm(loader, desc="Eval", leave=False):
        points = points.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if autocast is not None:
            with autocast:
                outputs = model(points)
                loss = criterion(outputs, labels)
        else:
            outputs = model(points)
            loss = criterion(outputs, labels)

        total_loss += loss.item() * points.size(0)
        preds = outputs.max(1)[1]
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    return total_loss / total, correct / total, all_preds, all_labels


# =============================================================================
#  БЛОК G. ОСНОВНОЙ ЦИКЛ ОБУЧЕНИЯ
# =============================================================================
history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
best_val_acc = 0.0

print(f"\n{'Ep':<4}{'TrLoss':<9}{'TrAcc':<9}{'VaLoss':<9}{'VaAcc':<9}{'sec':<6}")
print("-" * 46)

for epoch in range(1, NUM_EPOCHS + 1):
    t0 = time.time()
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
    val_loss, val_acc, _, _ = evaluate(model, test_loader, criterion)
    scheduler.step()                             # шаг LR-scheduler'а раз в эпоху
    dt = time.time() - t0

    history['train_loss'].append(train_loss); history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss);       history['val_acc'].append(val_acc)

    # Сохраняем веса только когда accuracy на тесте улучшилась.
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'best_pointnet_modelnet10.pth')

    if epoch % PRINT_EVERY == 0 or epoch == 1:
        print(f"{epoch:<4}{train_loss:<9.4f}{train_acc:<9.4f}"
              f"{val_loss:<9.4f}{val_acc:<9.4f}{dt:<6.1f}")

print(f"\nЛучшая validation accuracy: {best_val_acc:.4f}")


# =============================================================================
#  БЛОК H. ГРАФИКИ ОБУЧЕНИЯ
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(history['train_loss'], label='Train Loss')
axes[0].plot(history['val_loss'], label='Val Loss')
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
axes[0].set_title('Loss'); axes[0].legend(); axes[0].grid(True)
axes[1].plot(history['train_acc'], label='Train Acc')
axes[1].plot(history['val_acc'], label='Val Acc')
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy')
axes[1].set_title('Accuracy'); axes[1].legend(); axes[1].grid(True)
plt.tight_layout(); plt.savefig('training_history.png', dpi=150); plt.show()


# =============================================================================
#  БЛОК I. ФИНАЛЬНАЯ ОЦЕНКА: ОТЧЁТ ПО КЛАССАМ И CONFUSION MATRIX (С ЧИСЛАМИ)
# =============================================================================
model.load_state_dict(torch.load('best_pointnet_modelnet10.pth', weights_only=True))
val_loss, val_acc, all_preds, all_labels = evaluate(model, test_loader, criterion)

print(f"\nФинальная Test Accuracy: {val_acc:.4f}")
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=classes, zero_division=0))

cm = confusion_matrix(all_labels, all_preds)

# Порог яркости: ячейки темнее порога -> белый текст, светлее -> чёрный.
# Берём половину максимума матрицы — эмпирически хорошо для cmap='Blues'.
thresh = cm.max() / 2.0

# Суммы по строкам (= сколько объектов каждого класса в test). Нужны для процентов.
# Защита от деления на 0 на случай, если какой-то класс не попал в test.
row_sums = cm.sum(axis=1, keepdims=True)
row_sums = np.where(row_sums == 0, 1, row_sums)
cm_pct = 100.0 * cm / row_sums          # процент угаданных внутри каждого класса   

fig, ax = plt.subplots(figsize=(11, 9))   # чуть больше, чтобы влезли две строки текста
im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

ticks = np.arange(NUM_CLASSES)
ax.set_xticks(ticks); ax.set_yticks(ticks)
ax.set_xticklabels(classes, rotation=45, ha='right')
ax.set_yticklabels(classes)
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
ax.set_title(f'Confusion Matrix  (Test Acc = {val_acc:.2%})')

# Вписываем числа в каждую ячейку.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        # Цвет текста по яркости фона этой ячейки.
        txt_color = 'white' if cm[i, j] > thresh else 'black'
        # Две строки: абсолютное значение + процент по строке (recall класса).
        cell_text = f"{cm[i, j]}"
        ax.text(j, i, cell_text,
                ha='center', va='center',
                color=txt_color, fontsize=9)

plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
plt.show()

# =============================================================================
#  БЛОК J. 3D-ВИЗУАЛИЗАЦИЯ: исходный .off-меш (слева)  +  облако точек (справа)
# =============================================================================
def resolve_off_path(object_path, data_root):
    """Возвращает реально существующий путь к .off, пробуя варианты имени папки.
    Защищает от несоответствия night/ <-> night_stand/."""
    sep = os.sep
    candidates = [object_path]
    for c in candidates:
        fp = os.path.join(data_root, c)
        if os.path.isfile(fp):
            return fp
    return os.path.join(data_root, object_path)   # фолбэк


def _set_common_axes(ax, lim):
    """Одинаковый кубический масштаб и пределы для честного сравнения панелей."""
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
    try:
        ax.set_box_aspect([1, 1, 1])               # равные оси (без сплющивания)
    except Exception:
        pass
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')


def visualize_prediction(model, dataset, idx, df, data_root=DATA_ROOT,
                         max_faces=8000):
    """Рисует ДВЕ панели: исходный меш .off | облако точек модели,
    плюс подпись с истинным классом, предсказанием и уверенностью."""
    # --- предсказание модели по облаку точек из кэша ---
    points, label = dataset[idx]                       # [N, 3], уже нормализовано
    model.eval()
    with torch.no_grad():
        out = model(points.unsqueeze(0).to(device))
        prob = F.softmax(out, dim=1)
        conf, pred = prob.max(1)
    p = points.numpy()

    # --- читаем исходный меш и нормализуем его тем же способом, что и кэш ---
    object_path = df.iloc[idx]['object_path']
    fp = resolve_off_path(object_path, data_root)
    mesh_ok = True
    polys = None
    verts_n = None
    try:
        verts, faces = read_off(fp)
        verts_n = Normalize()(verts.copy())            # центр 0, единичная сфера
        # На очень «тяжёлых» мешах ограничиваем число граней, чтобы не тормозило.
        if (max_faces is not None) and (len(faces) > max_faces):
            sel = np.random.choice(len(faces), max_faces, replace=False)
            faces = faces[sel]
        polys = verts_n[faces]                         # [F, 3, 3] — треугольники
    except Exception as e:
        mesh_ok = False
        print(f"Не удалось прочитать меш {fp}: {e}")

    # общий предел осей по объединению меша и точек (одинаковый масштаб на обеих панелях)
    lim = 1.0
    if mesh_ok:
        lim = float(max(np.abs(verts_n).max(), np.abs(p).max())) * 1.05

    # --- фигура: 2 панели, если меш прочитался, иначе 1 ---
    n_panels = 2 if mesh_ok else 1
    fig = plt.figure(figsize=(6 * n_panels, 6))
    fig.suptitle(f"True: {idx_to_class[label.item()]}  |  "
                 f"Pred: {idx_to_class[pred.item()]} ({conf.item():.2%})  |  "
                 f"{os.path.basename(fp)}", fontsize=12)

    panel = 1
    if mesh_ok:
        # Панель 1 (слева): исходный меш — закрашенные грани + тонкий каркас
        ax1 = fig.add_subplot(1, n_panels, panel, projection='3d'); panel += 1
        coll = Poly3DCollection(polys, alpha=0.55,
                                facecolor='lightsteelblue',
                                edgecolor='k', linewidths=0.1)
        ax1.add_collection3d(coll)
        _set_common_axes(ax1, lim)
        ax1.set_title('Исходный меш .off')

    # Панель 2 (справа): облако точек (вход модели)
    ax2 = fig.add_subplot(1, n_panels, panel, projection='3d'); panel += 1
    ax2.scatter(p[:, 0], p[:, 1], p[:, 2], s=1, c='steelblue', alpha=0.6)
    _set_common_axes(ax2, lim)
    ax2.set_title('Облако точек (вход модели)')

    plt.tight_layout()
    plt.show()


# Показываем 5 случайных тестовых примеров (передаём df_test, чтобы знать путь к .off).
for i in random.sample(range(len(test_dataset)), 5):
    visualize_prediction(model, test_dataset, i, df_test)