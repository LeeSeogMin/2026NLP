"""
4장 실습: 모델 학습 루프
- Training Loop 구현
- Validation Loop 구현
- 과적합 방지 기법
- Early Stopping
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt


class SyntheticDataset(Dataset):
    """합성 분류 데이터셋"""

    def __init__(self, n_samples=1000, n_features=20, n_classes=3):
        np.random.seed(42)

        # 클래스별 중심점 생성
        centers = np.random.randn(n_classes, n_features) * 3

        # 데이터 생성
        X = []
        y = []
        for i in range(n_samples):
            label = i % n_classes
            sample = centers[label] + np.random.randn(n_features) * 0.5
            X.append(sample)
            y.append(label)

        self.X = torch.FloatTensor(np.array(X))
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class MLPClassifier(nn.Module):
    """MLP 분류 모델 (정규화 기법 포함)"""

    def __init__(self, input_size, hidden_size, num_classes, dropout=0.2):
        super(MLPClassifier, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.BatchNorm1d(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes),
        )

    def forward(self, x):
        return self.network(x)


def train_epoch(model, dataloader, criterion, optimizer, device):
    """한 에폭 학습"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_X, batch_y in dataloader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)

        # 순전파
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)

        # 역전파
        loss.backward()
        optimizer.step()

        # 통계
        total_loss += loss.item() * batch_X.size(0)
        _, predicted = outputs.max(1)
        total += batch_y.size(0)
        correct += predicted.eq(batch_y).sum().item()

    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def validate(model, dataloader, criterion, device):
    """검증"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_X, batch_y in dataloader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            total_loss += loss.item() * batch_X.size(0)
            _, predicted = outputs.max(1)
            total += batch_y.size(0)
            correct += predicted.eq(batch_y).sum().item()

    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


class EarlyStopping:
    """Early Stopping 클래스"""

    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

        return self.early_stop


def main():
    print("=" * 50)
    print("모델 학습 루프 실습")
    print("=" * 50)

    # 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 데이터 준비
    dataset = SyntheticDataset(n_samples=1000, n_features=20, n_classes=3)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    print(f"\n학습 데이터: {len(train_dataset)}")
    print(f"검증 데이터: {len(val_dataset)}")

    # 모델 초기화
    model = MLPClassifier(input_size=20, hidden_size=64, num_classes=3, dropout=0.2)
    model = model.to(device)

    # 손실 함수 및 옵티마이저
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    # Early Stopping
    early_stopping = EarlyStopping(patience=10)

    # 학습 기록
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    print("\n" + "=" * 50)
    print("학습 시작")
    print("=" * 50)

    num_epochs = 50
    best_val_acc = 0

    for epoch in range(num_epochs):
        # 학습
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)

        # 검증
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # 스케줄러 업데이트
        scheduler.step()

        # 기록
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        # 최고 성능 저장
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")

        # 로그 출력 (10 에폭마다)
        if (epoch + 1) % 10 == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch [{epoch+1:2d}/{num_epochs}] "
                f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.1f}% | "
                f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.1f}% | "
                f"LR: {current_lr:.6f}"
            )

        # Early Stopping 체크
        if early_stopping(val_loss):
            print(f"\nEarly Stopping at epoch {epoch + 1}")
            break

    print(f"\n최고 검증 정확도: {best_val_acc:.2f}%")

    # 학습 과정 시각화
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # 손실 그래프
    axes[0].plot(history["train_loss"], label="Train")
    axes[0].plot(history["val_loss"], label="Validation")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss Curve")
    axes[0].legend()
    axes[0].grid(True)

    # 정확도 그래프
    axes[1].plot(history["train_acc"], label="Train")
    axes[1].plot(history["val_acc"], label="Validation")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("Accuracy Curve")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("\n그래프 저장: training_curves.png")

    # 정리
    import os

    if os.path.exists("best_model.pth"):
        os.remove("best_model.pth")

    print("\n" + "=" * 50)
    print("핵심 정리")
    print("=" * 50)
    print("""
    Training Loop:
    1. model.train() - 학습 모드 설정
    2. optimizer.zero_grad() - 그래디언트 초기화
    3. outputs = model(inputs) - 순전파
    4. loss = criterion(outputs, targets) - 손실 계산
    5. loss.backward() - 역전파
    6. optimizer.step() - 파라미터 업데이트

    Validation Loop:
    1. model.eval() - 평가 모드 설정
    2. with torch.no_grad(): - 그래디언트 계산 비활성화
    3. 순전파 및 손실/정확도 계산

    과적합 방지:
    - Dropout: 무작위 뉴런 비활성화
    - BatchNorm: 배치 정규화
    - weight_decay: L2 정규화
    - Early Stopping: 검증 손실 정체 시 중단
    """)


if __name__ == "__main__":
    main()
