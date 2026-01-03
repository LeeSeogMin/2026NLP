"""
4장 실습: 옵티마이저와 학습률 스케줄러
- 다양한 옵티마이저 비교
- 학습률 스케줄러 활용
"""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np


class SimpleModel(nn.Module):
    """테스트용 간단한 모델"""

    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(10, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


def compare_optimizers():
    """다양한 옵티마이저 비교"""
    print("=" * 50)
    print("1. 옵티마이저 비교")
    print("=" * 50)

    # 데이터 생성
    torch.manual_seed(42)
    X = torch.randn(100, 10)
    y = torch.randn(100, 1)

    optimizers_config = {
        "SGD": lambda params: optim.SGD(params, lr=0.01),
        "SGD+Momentum": lambda params: optim.SGD(params, lr=0.01, momentum=0.9),
        "Adam": lambda params: optim.Adam(params, lr=0.01),
        "AdamW": lambda params: optim.AdamW(params, lr=0.01, weight_decay=0.01),
    }

    results = {}

    for name, opt_fn in optimizers_config.items():
        # 모델 초기화 (동일한 초기 가중치)
        torch.manual_seed(42)
        model = SimpleModel()
        optimizer = opt_fn(model.parameters())
        criterion = nn.MSELoss()

        losses = []
        for epoch in range(100):
            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        results[name] = losses
        print(f"{name}: 최종 손실 = {losses[-1]:.4f}")

    # 시각화
    plt.figure(figsize=(10, 6))
    for name, losses in results.items():
        plt.plot(losses, label=name)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Optimizer Comparison")
    plt.legend()
    plt.grid(True)
    plt.savefig("optimizer_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("\n그래프 저장: optimizer_comparison.png")


def demonstrate_schedulers():
    """학습률 스케줄러 비교"""
    print("\n" + "=" * 50)
    print("2. 학습률 스케줄러")
    print("=" * 50)

    model = SimpleModel()
    optimizer = optim.Adam(model.parameters(), lr=0.1)

    # 다양한 스케줄러
    schedulers = {
        "StepLR": optim.lr_scheduler.StepLR(
            optim.Adam(SimpleModel().parameters(), lr=0.1), step_size=10, gamma=0.5
        ),
        "ExponentialLR": optim.lr_scheduler.ExponentialLR(
            optim.Adam(SimpleModel().parameters(), lr=0.1), gamma=0.95
        ),
        "CosineAnnealingLR": optim.lr_scheduler.CosineAnnealingLR(
            optim.Adam(SimpleModel().parameters(), lr=0.1), T_max=50
        ),
    }

    print("\n[스케줄러별 학습률 변화]")
    lr_histories = {}

    for name, scheduler in schedulers.items():
        lrs = []
        for epoch in range(50):
            lrs.append(scheduler.get_last_lr()[0])
            scheduler.step()
        lr_histories[name] = lrs
        print(f"{name}: 초기={lrs[0]:.4f}, 최종={lrs[-1]:.4f}")

    # 시각화
    plt.figure(figsize=(10, 6))
    for name, lrs in lr_histories.items():
        plt.plot(lrs, label=name)
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate Scheduler Comparison")
    plt.legend()
    plt.grid(True)
    plt.savefig("scheduler_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("\n그래프 저장: scheduler_comparison.png")


def demonstrate_reduce_on_plateau():
    """ReduceLROnPlateau 스케줄러"""
    print("\n" + "=" * 50)
    print("3. ReduceLROnPlateau")
    print("=" * 50)

    torch.manual_seed(42)
    model = SimpleModel()
    optimizer = optim.Adam(model.parameters(), lr=0.1)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # 가상의 손실값 시뮬레이션
    simulated_losses = [1.0, 0.9, 0.85, 0.83, 0.82, 0.81, 0.80, 0.80, 0.80, 0.80, 0.80, 0.80, 0.60, 0.55]

    print("\n[ReduceLROnPlateau 동작]")
    for epoch, val_loss in enumerate(simulated_losses):
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]["lr"]

        if new_lr != current_lr:
            print(f"  Epoch {epoch}: val_loss={val_loss:.2f}, LR: {current_lr:.4f} → {new_lr:.4f}")


def warmup_cosine_scheduler():
    """Warmup + Cosine Annealing 조합"""
    print("\n" + "=" * 50)
    print("4. Warmup + Cosine Annealing")
    print("=" * 50)

    def get_lr_with_warmup(epoch, warmup_epochs, max_epochs, base_lr):
        if epoch < warmup_epochs:
            return base_lr * (epoch + 1) / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / (max_epochs - warmup_epochs)
            return base_lr * 0.5 * (1 + np.cos(np.pi * progress))

    warmup_epochs = 5
    max_epochs = 50
    base_lr = 0.1

    lrs = [get_lr_with_warmup(e, warmup_epochs, max_epochs, base_lr) for e in range(max_epochs)]

    print(f"Warmup 기간: {warmup_epochs} 에폭")
    print(f"최대 학습률: {max(lrs):.4f} (에폭 {warmup_epochs})")
    print(f"최종 학습률: {lrs[-1]:.6f}")

    plt.figure(figsize=(10, 6))
    plt.plot(lrs, "b-", linewidth=2)
    plt.axvline(x=warmup_epochs, color="r", linestyle="--", label=f"Warmup End (epoch {warmup_epochs})")
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.title("Warmup + Cosine Annealing Schedule")
    plt.legend()
    plt.grid(True)
    plt.savefig("warmup_cosine.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("\n그래프 저장: warmup_cosine.png")


def main():
    compare_optimizers()
    demonstrate_schedulers()
    demonstrate_reduce_on_plateau()
    warmup_cosine_scheduler()

    print("\n" + "=" * 50)
    print("핵심 정리")
    print("=" * 50)
    print("""
    옵티마이저 선택:
    - SGD: 기본, 수렴 느림
    - SGD+Momentum: 수렴 가속
    - Adam: 적응적 학습률, 빠른 수렴
    - AdamW: 가중치 감쇠 분리, 일반화 향상

    학습률 스케줄러:
    - StepLR: 일정 간격 감소
    - ExponentialLR: 지수 감소
    - CosineAnnealingLR: 코사인 곡선
    - ReduceLROnPlateau: 성능 정체 시 감소

    Warmup:
    - 초기 불안정성 방지
    - 대규모 모델에 효과적
    """)


if __name__ == "__main__":
    main()
