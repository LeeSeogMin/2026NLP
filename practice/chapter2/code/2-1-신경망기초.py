"""
2장 실습 (1): 신경망 기본 구조
- 퍼셉트론과 다층 퍼셉트론(MLP)
- 활성화 함수 비교 (ReLU, GELU, Softmax)
- 손실 함수 (Cross-Entropy, MSE)
- 역전파와 경사 하강법 데모
"""

import torch
import torch.nn as nn
import numpy as np


def demo_perceptron():
    """퍼셉트론 기초 — 단순 AND 게이트"""
    print("=" * 50)
    print("1. 퍼셉트론: AND 게이트 구현")
    print("=" * 50)

    # AND 게이트 데이터
    X = torch.FloatTensor([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = torch.FloatTensor([[0], [0], [0], [1]])

    # 단층 퍼셉트론 (선형 + 시그모이드)
    torch.manual_seed(42)
    perceptron = nn.Sequential(
        nn.Linear(2, 1),
        nn.Sigmoid()
    )

    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(perceptron.parameters(), lr=1.0)

    # 학습
    for epoch in range(100):
        output = perceptron(X)
        loss = criterion(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 결과
    print("\n[AND 게이트 학습 결과]")
    with torch.no_grad():
        predictions = perceptron(X)
        for i in range(4):
            x1, x2 = int(X[i][0]), int(X[i][1])
            pred = predictions[i].item()
            print(f"  {x1} AND {x2} = {pred:.3f} (기대값: {int(y[i])})")

    print("\n[XOR 문제 — 단층 퍼셉트론의 한계]")
    # XOR 데이터
    X_xor = torch.FloatTensor([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_xor = torch.FloatTensor([[0], [1], [1], [0]])

    torch.manual_seed(42)
    perceptron_xor = nn.Sequential(nn.Linear(2, 1), nn.Sigmoid())
    optimizer_xor = torch.optim.SGD(perceptron_xor.parameters(), lr=1.0)

    for epoch in range(1000):
        output = perceptron_xor(X_xor)
        loss = criterion(output, y_xor)
        optimizer_xor.zero_grad()
        loss.backward()
        optimizer_xor.step()

    with torch.no_grad():
        preds = perceptron_xor(X_xor)
        for i in range(4):
            x1, x2 = int(X_xor[i][0]), int(X_xor[i][1])
            pred = preds[i].item()
            print(f"  {x1} XOR {x2} = {pred:.3f} (기대값: {int(y_xor[i])}) {'✗' if abs(pred - y_xor[i].item()) > 0.3 else '✓'}")

    print("  → 단층 퍼셉트론은 XOR을 해결할 수 없다 (비선형 분리 불가)")


def demo_mlp():
    """다층 퍼셉트론(MLP)으로 XOR 해결"""
    print("\n" + "=" * 50)
    print("2. 다층 퍼셉트론(MLP): XOR 문제 해결")
    print("=" * 50)

    X = torch.FloatTensor([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = torch.FloatTensor([[0], [1], [1], [0]])

    torch.manual_seed(42)
    mlp = nn.Sequential(
        nn.Linear(2, 4),     # 은닉층 (뉴런 4개)
        nn.ReLU(),
        nn.Linear(4, 1),     # 출력층
        nn.Sigmoid()
    )

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(mlp.parameters(), lr=0.05)

    # 학습
    for epoch in range(500):
        output = mlp(X)
        loss = criterion(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"\n모델 구조:\n{mlp}")
    total_params = sum(p.numel() for p in mlp.parameters())
    print(f"총 파라미터 수: {total_params}")

    print("\n[XOR 학습 결과]")
    with torch.no_grad():
        preds = mlp(X)
        for i in range(4):
            x1, x2 = int(X[i][0]), int(X[i][1])
            pred = preds[i].item()
            label = "✓" if abs(pred - y[i].item()) < 0.3 else "✗"
            print(f"  {x1} XOR {x2} = {pred:.3f} (기대값: {int(y[i])}) {label}")
    print("  → 은닉층 추가로 XOR 문제 해결!")


def demo_activations():
    """활성화 함수 비교"""
    print("\n" + "=" * 50)
    print("3. 활성화 함수 비교")
    print("=" * 50)

    x = torch.linspace(-3, 3, 7)

    # ReLU
    relu = nn.ReLU()
    print(f"\n[ReLU] — 음수는 0, 양수는 그대로")
    print(f"  입력:  {[f'{v:.1f}' for v in x.tolist()]}")
    print(f"  출력:  {[f'{v:.1f}' for v in relu(x).tolist()]}")

    # GELU
    gelu = nn.GELU()
    print(f"\n[GELU] — ReLU의 부드러운 버전 (Transformer에서 사용)")
    print(f"  입력:  {[f'{v:.1f}' for v in x.tolist()]}")
    print(f"  출력:  {[f'{v:.2f}' for v in gelu(x).tolist()]}")

    # Sigmoid
    sigmoid = nn.Sigmoid()
    print(f"\n[Sigmoid] — 출력을 0~1로 압축")
    print(f"  입력:  {[f'{v:.1f}' for v in x.tolist()]}")
    print(f"  출력:  {[f'{v:.3f}' for v in sigmoid(x).tolist()]}")

    # Softmax
    logits = torch.tensor([2.0, 1.0, 0.5])
    softmax = nn.Softmax(dim=0)
    probs = softmax(logits)
    print(f"\n[Softmax] — 출력을 확률 분포로 변환 (합 = 1)")
    print(f"  입력 (로짓):  {logits.tolist()}")
    print(f"  출력 (확률):  {[f'{v:.3f}' for v in probs.tolist()]}")
    print(f"  합계: {probs.sum().item():.3f}")

    # 비선형성이 없으면?
    print(f"\n[왜 비선형 활성화가 필요한가?]")
    torch.manual_seed(42)
    w1 = torch.randn(3, 3)
    w2 = torch.randn(3, 3)
    combined = w2 @ w1
    print(f"  Linear(3→3) × Linear(3→3) = Linear(3→3)")
    print(f"  W2 × W1 = 하나의 행렬 W'와 동일 → 층을 아무리 쌓아도 선형 변환")
    print(f"  → 활성화 함수가 비선형성을 추가해야 깊은 네트워크가 의미 있다")


def demo_loss_functions():
    """손실 함수 비교"""
    print("\n" + "=" * 50)
    print("4. 손실 함수 비교")
    print("=" * 50)

    # Cross-Entropy Loss (분류)
    print("\n[Cross-Entropy Loss] — 분류 문제의 표준 손실 함수")
    ce_loss = nn.CrossEntropyLoss()

    # 좋은 예측 vs 나쁜 예측
    logits_good = torch.tensor([[2.5, -1.0, -0.5]])  # 클래스 0에 확신
    logits_bad = torch.tensor([[-1.0, 2.5, -0.5]])    # 클래스 1에 확신
    target = torch.tensor([0])  # 정답은 클래스 0

    loss_good = ce_loss(logits_good, target)
    loss_bad = ce_loss(logits_bad, target)
    print(f"  정답: 클래스 0")
    print(f"  좋은 예측 [2.5, -1.0, -0.5] → 손실: {loss_good.item():.4f}")
    print(f"  나쁜 예측 [-1.0, 2.5, -0.5] → 손실: {loss_bad.item():.4f}")
    print(f"  → 틀릴수록 손실이 커진다")

    # MSE Loss (회귀)
    print(f"\n[MSE Loss] — 회귀 문제의 표준 손실 함수")
    mse_loss = nn.MSELoss()

    pred1 = torch.tensor([2.8])
    pred2 = torch.tensor([5.0])
    target_reg = torch.tensor([3.0])

    loss1 = mse_loss(pred1, target_reg)
    loss2 = mse_loss(pred2, target_reg)
    print(f"  정답: 3.0")
    print(f"  예측 2.8 → MSE: {loss1.item():.4f}")
    print(f"  예측 5.0 → MSE: {loss2.item():.4f}")
    print(f"  → 정답에서 멀수록 손실이 (제곱으로) 커진다")


def demo_backpropagation():
    """역전파 + 경사 하강법 시연"""
    print("\n" + "=" * 50)
    print("5. 역전파와 경사 하강법")
    print("=" * 50)

    # 간단한 2층 네트워크에서 역전파 과정 추적
    torch.manual_seed(42)

    # y = 2x + 1 학습 (단순 선형 회귀)
    X = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    y = torch.tensor([[3.0], [5.0], [7.0], [9.0]])  # y = 2x + 1

    model = nn.Linear(1, 1)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    print(f"\n[학습 전 파라미터]")
    print(f"  가중치 (w): {model.weight.item():.4f}")
    print(f"  편향 (b):   {model.bias.item():.4f}")

    # 1 스텝 역전파 과정 추적
    print(f"\n[역전파 1 스텝 추적]")
    output = model(X)
    loss = criterion(output, y)
    print(f"  1) 순전파: 예측값 = w×x + b")
    print(f"     예측: {output.detach().flatten().tolist()}")
    print(f"  2) 손실 계산: MSE = {loss.item():.4f}")

    loss.backward()
    print(f"  3) 역전파: 그래디언트 계산")
    print(f"     ∂L/∂w = {model.weight.grad.item():.4f}")
    print(f"     ∂L/∂b = {model.bias.grad.item():.4f}")

    optimizer.step()
    print(f"  4) 파라미터 업데이트 (lr=0.01)")
    print(f"     w: {model.weight.item():.4f}")
    print(f"     b: {model.bias.item():.4f}")

    # 전체 학습
    print(f"\n[100 에폭 학습]")
    torch.manual_seed(42)
    model2 = nn.Linear(1, 1)
    optimizer2 = torch.optim.SGD(model2.parameters(), lr=0.01)

    for epoch in range(100):
        output = model2(X)
        loss = criterion(output, y)
        optimizer2.zero_grad()
        loss.backward()
        optimizer2.step()

        if (epoch + 1) % 25 == 0:
            print(
                f"  Epoch {epoch+1:3d}: "
                f"Loss={loss.item():.4f}, "
                f"w={model2.weight.item():.4f}, "
                f"b={model2.bias.item():.4f}"
            )

    print(f"\n  목표: y = 2x + 1")
    print(f"  학습 결과: y = {model2.weight.item():.2f}x + {model2.bias.item():.2f}")


def main():
    demo_perceptron()
    demo_mlp()
    demo_activations()
    demo_loss_functions()
    demo_backpropagation()

    print("\n" + "=" * 50)
    print("핵심 정리")
    print("=" * 50)
    print("""
    신경망 구조:
    - 퍼셉트론: 입력 × 가중치 + 편향 → 활성화 (단순 판단기)
    - MLP: 퍼셉트론을 여러 층 쌓음 → 비선형 문제 해결

    활성화 함수:
    - ReLU: max(0, x) — 가장 널리 사용
    - GELU: Transformer 계열에서 표준
    - Softmax: 출력을 확률 분포로 변환 (분류 출력층)

    손실 함수:
    - Cross-Entropy: 분류 문제
    - MSE: 회귀 문제

    역전파 (Backpropagation):
    1. 순전파 → 예측값 계산
    2. 손실 계산 → 정답과 비교
    3. 역전파 → 그래디언트 계산 (chain rule)
    4. 파라미터 업데이트 → 손실 줄이는 방향으로
    """)


if __name__ == "__main__":
    main()
