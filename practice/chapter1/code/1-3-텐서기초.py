"""
1-3-텐서기초.py
제1장 실습: Python 딥러닝 기초 — Tensor 연산과 Autograd

이 스크립트는 PyTorch Tensor의 기본 조작(생성, 연산, 인덱싱, GPU 이동)과
Autograd 자동 미분을 단계적으로 실습한다.

실행 방법:
    python 1-3-텐서기초.py
"""

import torch
import numpy as np


def tensor_creation():
    """다양한 방법으로 Tensor를 생성한다."""
    print("=" * 50)
    print("1. Tensor 생성")
    print("=" * 50)

    # Python 리스트에서 생성
    t1 = torch.tensor([1, 2, 3])
    print(f"리스트에서 생성: {t1}")

    # 2차원 텐서 (행렬)
    t2 = torch.tensor([[1, 2, 3], [4, 5, 6]])
    print(f"2D 텐서:\n{t2}")
    print(f"  shape: {t2.shape}, dtype: {t2.dtype}")

    # 특수 텐서
    zeros = torch.zeros(2, 3)
    ones = torch.ones(2, 3)
    rand = torch.randn(2, 3)  # 표준정규분포
    print(f"영 텐서:\n{zeros}")
    print(f"일 텐서:\n{ones}")
    print(f"랜덤 텐서:\n{rand}")

    # NumPy 배열에서 변환
    np_arr = np.array([10, 20, 30])
    t_from_np = torch.from_numpy(np_arr)
    print(f"NumPy → Tensor: {t_from_np}")

    # arange, linspace
    t_range = torch.arange(0, 10, 2)
    t_lin = torch.linspace(0, 1, 5)
    print(f"arange(0, 10, 2): {t_range}")
    print(f"linspace(0, 1, 5): {t_lin}")
    print()


def tensor_operations():
    """Tensor의 기본 산술 연산을 수행한다."""
    print("=" * 50)
    print("2. Tensor 연산")
    print("=" * 50)

    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([4.0, 5.0, 6.0])

    # 요소별 연산
    print(f"a = {a}")
    print(f"b = {b}")
    print(f"a + b = {a + b}")
    print(f"a * b = {a * b}")   # 요소별 곱
    print(f"a ** 2 = {a ** 2}")

    # 행렬 연산
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    y = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
    print(f"\n행렬 x:\n{x}")
    print(f"행렬 y:\n{y}")
    print(f"행렬 곱 (x @ y):\n{x @ y}")
    print(f"전치 (x.T):\n{x.T}")

    # 통계 연산
    data = torch.tensor([2.0, 4.0, 6.0, 8.0, 10.0])
    print(f"\ndata = {data}")
    print(f"평균: {data.mean():.1f}")
    print(f"표준편차: {data.std():.2f}")
    print(f"합계: {data.sum():.1f}")
    print(f"최대값: {data.max():.1f}, 최소값: {data.min():.1f}")
    print()


def tensor_indexing():
    """Tensor의 인덱싱과 슬라이싱을 실습한다."""
    print("=" * 50)
    print("3. Tensor 인덱싱과 슬라이싱")
    print("=" * 50)

    t = torch.tensor([[1, 2, 3, 4],
                       [5, 6, 7, 8],
                       [9, 10, 11, 12]])
    print(f"원본 텐서:\n{t}")
    print(f"shape: {t.shape}")

    # 기본 인덱싱
    print(f"\nt[0]: {t[0]}")        # 첫 번째 행
    print(f"t[1, 2]: {t[1, 2]}")    # 2행 3열 요소
    print(f"t[:, 0]: {t[:, 0]}")    # 첫 번째 열
    print(f"t[0:2, 1:3]:\n{t[0:2, 1:3]}")  # 부분 행렬

    # 조건부 인덱싱
    mask = t > 5
    print(f"\nt > 5 마스크:\n{mask}")
    print(f"5보다 큰 요소: {t[mask]}")

    # reshape
    reshaped = t.reshape(2, 6)
    print(f"\nreshape(2, 6):\n{reshaped}")
    print()


def tensor_device():
    """Tensor를 GPU(또는 MPS)로 이동하는 방법을 보여준다."""
    print("=" * 50)
    print("4. Tensor 디바이스 이동")
    print("=" * 50)

    # 사용 가능한 디바이스 확인
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = f"CUDA ({torch.cuda.get_device_name(0)})"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        device_name = "Apple MPS"
    else:
        device = torch.device("cpu")
        device_name = "CPU"

    print(f"사용 디바이스: {device_name}")

    # 디바이스 이동
    x = torch.randn(3, 3)
    print(f"x 디바이스 (이동 전): {x.device}")

    x_dev = x.to(device)
    print(f"x 디바이스 (이동 후): {x_dev.device}")

    # GPU에서 연산
    y_dev = torch.randn(3, 3, device=device)
    z_dev = x_dev @ y_dev
    print(f"GPU 연산 결과 디바이스: {z_dev.device}")

    # CPU로 복원 (NumPy 변환 시 필요)
    z_cpu = z_dev.cpu().numpy()
    print(f"CPU로 복원 후 NumPy:\n{z_cpu}")
    print()


def autograd_basics():
    """Autograd 자동 미분의 기초를 실습한다."""
    print("=" * 50)
    print("5. Autograd 자동 미분")
    print("=" * 50)

    # requires_grad=True → 이 텐서에 대한 기울기를 추적한다
    x = torch.tensor(3.0, requires_grad=True)
    print(f"x = {x.item()}")

    # 함수: y = x² + 2x + 1
    y = x ** 2 + 2 * x + 1
    print(f"y = x² + 2x + 1 = {y.item()}")

    # 역전파: dy/dx 계산
    y.backward()
    print(f"dy/dx = 2x + 2 = {x.grad.item()}")
    print(f"검증: 2 * {x.item()} + 2 = {2 * x.item() + 2}")
    print()

    # 벡터에 대한 자동 미분
    print("-" * 30)
    print("벡터 자동 미분")
    print("-" * 30)

    w = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    print(f"w = {w.data}")

    # L = sum(w²) — 벡터 각 원소의 제곱합
    loss = (w ** 2).sum()
    print(f"L = sum(w²) = {loss.item()}")

    loss.backward()
    print(f"dL/dw = 2w = {w.grad}")
    print(f"검증: 2 * [1, 2, 3] = {2 * w.data}")
    print()

    # 연쇄 법칙 (Chain Rule)
    print("-" * 30)
    print("연쇄 법칙 (Chain Rule)")
    print("-" * 30)

    a = torch.tensor(2.0, requires_grad=True)
    b = a * 3        # b = 3a
    c = b ** 2        # c = (3a)² = 9a²
    d = c + 5         # d = 9a² + 5

    d.backward()
    print(f"a = {a.item()}")
    print(f"d = 9a² + 5 = {d.item()}")
    print(f"dd/da = 18a = {a.grad.item()}")
    print(f"검증: 18 * {a.item()} = {18 * a.item()}")
    print()


def gradient_descent_demo():
    """경사 하강법의 직관적 시연."""
    print("=" * 50)
    print("6. 미니 경사 하강법 시연")
    print("=" * 50)

    # 목표: f(x) = (x - 3)² 의 최솟값을 찾아라
    x = torch.tensor(0.0, requires_grad=True)
    lr = 0.1  # 학습률

    print(f"목표 함수: f(x) = (x - 3)²")
    print(f"최솟값 위치: x = 3.0")
    print(f"시작점: x = {x.item():.4f}")
    print(f"학습률: {lr}\n")

    for step in range(1, 21):
        # 순전파: 함수값 계산
        loss = (x - 3) ** 2

        # 역전파: 기울기 계산
        loss.backward()

        # 기울기 하강 (수동 업데이트)
        with torch.no_grad():
            x -= lr * x.grad

        # 기울기 초기화 (중요!)
        x.grad.zero_()

        if step <= 5 or step % 5 == 0:
            print(f"  Step {step:2d}: x = {x.item():.4f}, f(x) = {(x.item() - 3) ** 2:.6f}")

    print(f"\n최종 결과: x = {x.item():.4f} (목표: 3.0)")
    print()


def main():
    """Tensor 기초와 Autograd를 단계별로 실습한다."""
    print()
    print("╔" + "═" * 48 + "╗")
    print("║     PyTorch Tensor 기초 + Autograd 실습       ║")
    print("╚" + "═" * 48 + "╝")
    print()

    tensor_creation()
    tensor_operations()
    tensor_indexing()
    tensor_device()
    autograd_basics()
    gradient_descent_demo()

    print("=" * 50)
    print("Tensor 기초 실습 완료!")
    print("=" * 50)
    print("다음 단계: 선형 회귀 실습 (1-4-실습.py)")
    print()


if __name__ == "__main__":
    main()
