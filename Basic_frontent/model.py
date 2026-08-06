import torch
import torch.nn as nn
import torch.nn.functional as F

class PointNet(nn.Module):
    """
    PointNet для классификации облаков точек.
    Принимает тензор размерности [B, N, 3] (Батч, Количество точек, Координаты x/y/z).
    Возвращает логиты для каждого класса размерности [B, num_classes].
    """
    def __init__(self, num_classes=10):
        super().__init__()
        # Shared MLP (применение одинаковых весов к каждой точке отдельно)
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=1)
        self.conv3 = nn.Conv1d(128, 1024, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        
        # Классификатор по глобальному признаку формы
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(0.4)

    def forward(self, x): # На вход ожидается: [B, N, 3]
        # Изменяем размерность, так как Conv1d ждет каналы (координаты) на втором месте
        x = x.permute(0, 2, 1) # -> [B, 3, N]
        
        x = F.relu(self.bn1(self.conv1(x))) # [B, 64, N]
        x = F.relu(self.bn2(self.conv2(x))) # [B, 128, N]
        x = self.bn3(self.conv3(x))         # [B, 1024, N]
        
        # Симметричная функция: Max Pooling по измерению точек (N)
        x = torch.max(x, dim=2)[0]          # Global MaxPool -> [B, 1024]
        
        # Полносвязные слои классификации
        x = F.relu(self.bn4(self.fc1(x)))   # [B, 512]
        x = self.dropout(x)
        x = F.relu(self.bn5(self.fc2(x)))   # [B, 256]
        x = self.dropout(x)
        x = self.fc3(x)                     # [B, num_classes] (Логиты)
        return x
