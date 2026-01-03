"""
3장 실습: PyTorch로 선형 회귀 구현
- 수동 구현 (forward, backward)
- nn.Linear 사용
- 학습 루프 이해
"""

import torch
import torch.nn as nn
import torch.optim as optim


def manual_linear_regression():
    """수동으로 선형 회귀 구현"""
    print("=" * 50)
    print("1. 수동 선형 회귀 구현")
    print("=" * 50)

    # 데이터 생성: y = 2x + 1 + noise
    torch.manual_seed(42)
    X = torch.linspace(0, 10, 100).reshape(-1, 1)
    y = 2 * X + 1 + torch.randn(100, 1) * 0.5

    print(f"데이터: X shape = {X.shape}, y shape = {y.shape}")
    print(f"실제 관계: y = 2x + 1")

    # 파라미터 초기화
    w = torch.randn(1, requires_grad=True)
    b = torch.randn(1, requires_grad=True)

    print(f"\n초기 파라미터: w = {w.item():.4f}, b = {b.item():.4f}")

    # 학습 설정
    learning_rate = 0.01
    epochs = 100

    print(f"\n학습 시작 (lr={learning_rate}, epochs={epochs})")
    print("-" * 40)

    for epoch in range(epochs):
        # 순전파 (Forward)
        y_pred = X * w + b

        # 손실 계산 (MSE)
        loss = ((y_pred - y) ** 2).mean()

        # 역전파 (Backward)
        loss.backward()

        # 파라미터 업데이트 (no_grad 내에서)
        with torch.no_grad():
            w -= learning_rate * w.grad
            b -= learning_rate * b.grad

        # 기울기 초기화
        w.grad.zero_()
        b.grad.zero_()

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch + 1:3d}: Loss = {loss.item():.4f}, "
                  f"w = {w.item():.4f}, b = {b.item():.4f}")

    print("-" * 40)
    print(f"최종 결과: y = {w.item():.4f}x + {b.item():.4f}")
    print(f"실제 관계: y = 2.0000x + 1.0000")


def nn_linear_regression():
    """nn.Module을 사용한 선형 회귀"""
    print("\n" + "=" * 50)
    print("2. nn.Module 사용 선형 회귀")
    print("=" * 50)

    # 데이터 생성
    torch.manual_seed(42)
    X = torch.linspace(0, 10, 100).reshape(-1, 1)
    y = 2 * X + 1 + torch.randn(100, 1) * 0.5

    # 모델 정의
    model = nn.Linear(in_features=1, out_features=1)
    print(f"모델 구조: {model}")
    print(f"초기 파라미터:")
    print(f"  weight = {model.weight.data.item():.4f}")
    print(f"  bias = {model.bias.data.item():.4f}")

    # 손실 함수와 옵티마이저
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    print(f"\n학습 시작")
    print("-" * 40)

    # 학습 루프
    epochs = 100
    for epoch in range(epochs):
        # 순전파
        y_pred = model(X)
        loss = criterion(y_pred, y)

        # 역전파
        optimizer.zero_grad()  # 기울기 초기화
        loss.backward()        # 기울기 계산
        optimizer.step()       # 파라미터 업데이트

        if (epoch + 1) % 20 == 0:
            w = model.weight.data.item()
            b = model.bias.data.item()
            print(f"Epoch {epoch + 1:3d}: Loss = {loss.item():.4f}, "
                  f"w = {w:.4f}, b = {b:.4f}")

    print("-" * 40)
    w_final = model.weight.data.item()
    b_final = model.bias.data.item()
    print(f"최종 결과: y = {w_final:.4f}x + {b_final:.4f}")
    print(f"실제 관계: y = 2.0000x + 1.0000")


def compare_optimizers():
    """다양한 옵티마이저 비교"""
    print("\n" + "=" * 50)
    print("3. 옵티마이저 비교")
    print("=" * 50)

    # 데이터 생성
    torch.manual_seed(42)
    X = torch.linspace(0, 10, 100).reshape(-1, 1)
    y = 2 * X + 1 + torch.randn(100, 1) * 0.5

    optimizers_config = [
        ("SGD", lambda params: optim.SGD(params, lr=0.01)),
        ("SGD+Momentum", lambda params: optim.SGD(params, lr=0.01, momentum=0.9)),
        ("Adam", lambda params: optim.Adam(params, lr=0.01)),
    ]

    results = []

    for name, opt_fn in optimizers_config:
        torch.manual_seed(42)
        model = nn.Linear(1, 1)
        criterion = nn.MSELoss()
        optimizer = opt_fn(model.parameters())

        # 학습
        for epoch in range(50):
            y_pred = model(X)
            loss = criterion(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        final_loss = loss.item()
        w = model.weight.data.item()
        b = model.bias.data.item()
        results.append((name, final_loss, w, b))

    print("\n[50 에폭 후 결과]")
    print(f"{'옵티마이저':<15} {'Loss':<10} {'w':<10} {'b':<10}")
    print("-" * 45)
    for name, loss, w, b in results:
        print(f"{name:<15} {loss:<10.4f} {w:<10.4f} {b:<10.4f}")
    print("-" * 45)
    print(f"{'실제값':<15} {'-':<10} {'2.0000':<10} {'1.0000':<10}")


def main():
    manual_linear_regression()
    nn_linear_regression()
    compare_optimizers()

    print("\n" + "=" * 50)
    print("핵심 정리")
    print("=" * 50)
    print("""
    선형 회귀 학습 루프:
    1. 순전파 (Forward): y_pred = model(X)
    2. 손실 계산: loss = criterion(y_pred, y)
    3. 기울기 초기화: optimizer.zero_grad()
    4. 역전파: loss.backward()
    5. 파라미터 업데이트: optimizer.step()

    주요 옵티마이저:
    - SGD: 기본 경사 하강법
    - SGD + Momentum: 관성 추가로 수렴 가속
    - Adam: 적응적 학습률, 가장 널리 사용
    """)


if __name__ == "__main__":
    main()
