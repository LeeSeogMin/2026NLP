"""
2장 실습 (2): PyTorch 모델 개발과 평가
- nn.Module 모델 정의
- Dataset/DataLoader 데이터 파이프라인
- Optimizer 비교 (SGD, Adam, AdamW)
- Training/Validation Loop
- Early Stopping + 과적합 방지
- 학습 과정 시각화
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


# ─── 1. nn.Module 모델 정의 ───────────────────────────────

class SimpleMLP(nn.Module):
    """간단한 다층 퍼셉트론"""

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class MLPClassifier(nn.Module):
    """MLP 분류 모델 (BatchNorm + Dropout)"""

    def __init__(self, input_size, hidden_size, num_classes, dropout=0.2):
        super().__init__()
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


# ─── 2. Dataset ───────────────────────────────────────────

class SyntheticDataset(Dataset):
    """합성 분류 데이터셋"""

    def __init__(self, n_samples=1000, n_features=20, n_classes=3):
        np.random.seed(42)
        centers = np.random.randn(n_classes, n_features) * 3

        X, y = [], []
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


# ─── 3. Early Stopping ───────────────────────────────────

class EarlyStopping:
    """검증 손실이 개선되지 않으면 학습 중단"""

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


# ─── 4. Training / Validation ─────────────────────────────

def train_epoch(model, dataloader, criterion, optimizer, device):
    """한 에폭 학습"""
    model.train()
    total_loss, correct, total = 0, 0, 0

    for batch_X, batch_y in dataloader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)

        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_X.size(0)
        _, predicted = outputs.max(1)
        total += batch_y.size(0)
        correct += predicted.eq(batch_y).sum().item()

    return total_loss / total, 100.0 * correct / total


def validate(model, dataloader, criterion, device):
    """검증"""
    model.eval()
    total_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for batch_X, batch_y in dataloader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            total_loss += loss.item() * batch_X.size(0)
            _, predicted = outputs.max(1)
            total += batch_y.size(0)
            correct += predicted.eq(batch_y).sum().item()

    return total_loss / total, 100.0 * correct / total


# ─── 실행 ─────────────────────────────────────────────────

def main():
    # ── Part 1: nn.Module 기본 ──
    print("=" * 60)
    print("Part 1: nn.Module 모델 정의")
    print("=" * 60)

    model = SimpleMLP(input_size=10, hidden_size=32, output_size=2)
    print(f"모델 구조:\n{model}\n")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"총 파라미터 수: {total_params}")

    x = torch.randn(5, 10)
    output = model(x)
    print(f"입력 shape: {x.shape}")
    print(f"출력 shape: {output.shape}")

    # 파라미터 초기화
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    model.apply(init_weights)
    print(f"\nXavier 초기화 후:")
    print(f"  fc1.weight mean={model.fc1.weight.mean().item():.4f}, std={model.fc1.weight.std().item():.4f}")

    # ── Part 2: Dataset / DataLoader ──
    print("\n" + "=" * 60)
    print("Part 2: Dataset / DataLoader")
    print("=" * 60)

    dataset = SyntheticDataset(n_samples=1000, n_features=20, n_classes=3)
    print(f"데이터셋 크기: {len(dataset)}")
    print(f"샘플 shape: X={dataset[0][0].shape}, y={dataset[0][1].item()}")

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    print(f"학습: {len(train_dataset)}, 검증: {len(val_dataset)}")

    batch_X, batch_y = next(iter(train_loader))
    print(f"배치 shape: X={batch_X.shape}, y={batch_y.shape}")

    # ── Part 3: Optimizer 비교 ──
    print("\n" + "=" * 60)
    print("Part 3: Optimizer 비교 (50 에폭)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    optimizer_configs = {
        "SGD": lambda p: optim.SGD(p, lr=0.01),
        "SGD+Momentum": lambda p: optim.SGD(p, lr=0.01, momentum=0.9),
        "Adam": lambda p: optim.Adam(p, lr=0.001),
        "AdamW": lambda p: optim.AdamW(p, lr=0.001, weight_decay=0.01),
    }

    criterion = nn.CrossEntropyLoss()
    opt_results = {}

    for name, opt_fn in optimizer_configs.items():
        torch.manual_seed(42)
        m = MLPClassifier(20, 64, 3, dropout=0.2).to(device)
        opt = opt_fn(m.parameters())

        losses = []
        for epoch in range(50):
            loss, _ = train_epoch(m, train_loader, criterion, opt, device)
            losses.append(loss)

        opt_results[name] = losses
        print(f"  {name:15s}: 최종 손실 = {losses[-1]:.4f}")

    # ── Part 4: Training Loop (전체) ──
    print("\n" + "=" * 60)
    print("Part 4: Training Loop + Early Stopping")
    print("=" * 60)

    torch.manual_seed(42)
    model_full = MLPClassifier(20, 64, 3, dropout=0.2).to(device)
    optimizer_full = optim.AdamW(model_full.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer_full, T_max=50)
    early_stopping = EarlyStopping(patience=10)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0
    num_epochs = 50

    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(
            model_full, train_loader, criterion, optimizer_full, device
        )
        val_loss, val_acc = validate(model_full, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc

        if (epoch + 1) % 10 == 0:
            lr = optimizer_full.param_groups[0]["lr"]
            print(
                f"Epoch [{epoch+1:2d}/{num_epochs}] "
                f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.1f}% | "
                f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.1f}% | "
                f"LR: {lr:.6f}"
            )

        if early_stopping(val_loss):
            print(f"\nEarly Stopping at epoch {epoch + 1}")
            break

    print(f"\n최고 검증 정확도: {best_val_acc:.2f}%")

    # ── 시각화 ──
    output_dir = Path(__file__).parent.parent / "data" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Loss Curve
    axes[0].plot(history["train_loss"], label="Train")
    axes[0].plot(history["val_loss"], label="Validation")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss Curve")
    axes[0].legend()
    axes[0].grid(True)

    # Accuracy Curve
    axes[1].plot(history["train_acc"], label="Train")
    axes[1].plot(history["val_acc"], label="Validation")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("Accuracy Curve")
    axes[1].legend()
    axes[1].grid(True)

    # Optimizer 비교
    for name, losses in opt_results.items():
        axes[2].plot(losses, label=name)
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Loss")
    axes[2].set_title("Optimizer Comparison")
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    save_path = output_dir / "ch2_training_curves.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n그래프 저장: {save_path}")

    print("\n" + "=" * 60)
    print("핵심 정리")
    print("=" * 60)
    print("""
    nn.Module:
    - __init__: 층 정의, forward: 순전파 정의
    - nn.Sequential로 간결하게 구성 가능

    Dataset / DataLoader:
    - Dataset: __init__, __len__, __getitem__ 구현
    - DataLoader: batch_size, shuffle, num_workers

    Optimizer:
    - SGD: 기본, 느린 수렴
    - Adam: 적응적 학습률, 빠른 수렴
    - AdamW: Weight Decay 분리, 대규모 모델 권장

    Training Loop:
    - model.train() → zero_grad → forward → loss → backward → step
    - model.eval() + torch.no_grad() → Validation

    과적합 방지:
    - Dropout, BatchNorm, Weight Decay, Early Stopping
    """)


if __name__ == "__main__":
    main()
