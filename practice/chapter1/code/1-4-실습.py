"""
1-4-실습.py
제1장 실습: Copilot 활용 실습 — Autograd로 선형 회귀

이 스크립트는 PyTorch의 Autograd를 활용하여 선형 회귀 모델을
밑바닥부터 구현한다. y = 2x + 1 관계를 학습하는 과정을 통해
경사 하강법의 원리를 체감한다.

실행 방법:
    python 1-4-실습.py
"""

import torch
import numpy as np
from pathlib import Path

# matplotlib 비표시 백엔드 설정 (서버/CI 호환)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def generate_data(n_samples=50, seed=42):
    """y = 2x + 1 + 노이즈 형태의 학습 데이터를 생성한다."""
    torch.manual_seed(seed)
    x = torch.linspace(-3, 3, n_samples).unsqueeze(1)  # (50, 1)
    noise = torch.randn_like(x) * 0.3
    y = 2 * x + 1 + noise  # 정답: w=2, b=1
    return x, y


def train_linear_regression():
    """Autograd를 사용한 선형 회귀 학습."""
    print("=" * 50)
    print("1. 선형 회귀: Autograd로 밑바닥 구현")
    print("=" * 50)

    # 데이터 생성
    x, y = generate_data()
    print(f"데이터: {len(x)}개 샘플")
    print(f"정답 파라미터: w = 2.0, b = 1.0\n")

    # 학습 가능한 파라미터 초기화 (랜덤)
    torch.manual_seed(0)
    w = torch.randn(1, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    lr = 0.01  # 학습률
    epochs = 200

    print(f"초기값: w = {w.item():.4f}, b = {b.item():.4f}")
    print(f"학습률: {lr}, 에포크: {epochs}\n")

    # 학습 기록
    losses = []

    for epoch in range(1, epochs + 1):
        # 순전파: 예측값 계산
        y_pred = w * x + b

        # 손실 함수: MSE (Mean Squared Error)
        loss = ((y_pred - y) ** 2).mean()
        losses.append(loss.item())

        # 역전파: 기울기 계산
        loss.backward()

        # 파라미터 업데이트 (경사 하강법)
        with torch.no_grad():
            w -= lr * w.grad
            b -= lr * b.grad

        # 기울기 초기화
        w.grad.zero_()
        b.grad.zero_()

        # 진행 상황 출력
        if epoch <= 5 or epoch % 50 == 0:
            print(f"  Epoch {epoch:3d}: Loss = {loss.item():.4f}, "
                  f"w = {w.item():.4f}, b = {b.item():.4f}")

    print(f"\n학습 완료!")
    print(f"  학습된 w = {w.item():.4f} (정답: 2.0)")
    print(f"  학습된 b = {b.item():.4f} (정답: 1.0)")
    print()

    return x, y, w, b, losses


def train_with_nn_module():
    """nn.Module을 사용한 선형 회귀 (2장 예고)."""
    print("=" * 50)
    print("2. nn.Module 방식 (2장 미리보기)")
    print("=" * 50)

    x, y = generate_data()

    # nn.Linear 사용
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()

    print(f"초기 파라미터: w = {model.weight.item():.4f}, "
          f"b = {model.bias.item():.4f}")

    for epoch in range(1, 201):
        # 순전파
        y_pred = model(x)
        loss = criterion(y_pred, y)

        # 역전파 + 업데이트
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch <= 3 or epoch % 50 == 0:
            print(f"  Epoch {epoch:3d}: Loss = {loss.item():.4f}")

    w_learned = model.weight.item()
    b_learned = model.bias.item()
    print(f"\n학습 완료!")
    print(f"  학습된 w = {w_learned:.4f} (정답: 2.0)")
    print(f"  학습된 b = {b_learned:.4f} (정답: 1.0)")
    print()

    return model


def visualize_results(x, y, w, b, losses):
    """학습 결과를 시각화한다."""
    print("=" * 50)
    print("3. 결과 시각화")
    print("=" * 50)

    output_dir = Path(__file__).parent.parent / "data" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 왼쪽: 데이터와 회귀선
    axes[0].scatter(x.numpy(), y.numpy(), alpha=0.5, label="Data", s=20)
    x_line = torch.linspace(-3, 3, 100).unsqueeze(1)
    with torch.no_grad():
        y_line = w * x_line + b
    axes[0].plot(x_line.numpy(), y_line.numpy(), "r-", linewidth=2,
                 label=f"y = {w.item():.2f}x + {b.item():.2f}")
    axes[0].plot(x_line.numpy(), (2 * x_line + 1).numpy(), "g--",
                 linewidth=1, alpha=0.7, label="y = 2x + 1 (정답)")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].set_title("Linear Regression Result")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 오른쪽: 손실 곡선
    axes[1].plot(losses, linewidth=1.5)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MSE Loss")
    axes[1].set_title("Training Loss Curve")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = output_dir / "linear_regression.png"
    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f"시각화 저장: {save_path}")
    print()


def main():
    """선형 회귀 실습을 단계별로 수행한다."""
    print()
    print("╔" + "═" * 48 + "╗")
    print("║    Autograd 활용 선형 회귀 실습                ║")
    print("╚" + "═" * 48 + "╝")
    print()

    # 1. Autograd 밑바닥 구현
    x, y, w, b, losses = train_linear_regression()

    # 2. nn.Module 방식 (미리보기)
    train_with_nn_module()

    # 3. 시각화
    visualize_results(x, y, w, b, losses)

    print("=" * 50)
    print("실습 완료!")
    print("=" * 50)
    print()
    print("핵심 포인트:")
    print("  1. requires_grad=True로 기울기 추적 활성화")
    print("  2. loss.backward()로 역전파 수행")
    print("  3. with torch.no_grad()로 파라미터 업데이트")
    print("  4. grad.zero_()로 기울기 초기화 (매 스텝)")
    print()


if __name__ == "__main__":
    main()
