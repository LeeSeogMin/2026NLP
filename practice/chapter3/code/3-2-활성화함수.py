"""
3장 실습: 활성화 함수 비교
- Sigmoid, Tanh, ReLU, Leaky ReLU, GELU 시각화 및 비교
"""

import numpy as np
import math


def sigmoid(x):
    """Sigmoid: σ(x) = 1 / (1 + e^(-x))"""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


def sigmoid_derivative(x):
    """Sigmoid의 도함수: σ'(x) = σ(x)(1 - σ(x))"""
    s = sigmoid(x)
    return s * (1 - s)


def tanh(x):
    """Tanh: (e^x - e^(-x)) / (e^x + e^(-x))"""
    return np.tanh(x)


def tanh_derivative(x):
    """Tanh의 도함수: 1 - tanh²(x)"""
    return 1 - np.tanh(x) ** 2


def relu(x):
    """ReLU: max(0, x)"""
    return np.maximum(0, x)


def relu_derivative(x):
    """ReLU의 도함수: x > 0이면 1, 아니면 0"""
    return np.where(x > 0, 1.0, 0.0)


def leaky_relu(x, alpha=0.01):
    """Leaky ReLU: max(αx, x)"""
    return np.where(x > 0, x, alpha * x)


def leaky_relu_derivative(x, alpha=0.01):
    """Leaky ReLU의 도함수"""
    return np.where(x > 0, 1.0, alpha)


def gelu(x):
    """GELU: x * Φ(x) 근사식 사용"""
    # 근사식: 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))


def gelu_derivative(x):
    """GELU의 도함수 (근사)"""
    # 수치적 근사 사용
    h = 1e-5
    return (gelu(x + h) - gelu(x - h)) / (2 * h)


def softmax(x):
    """Softmax: 다중 클래스 확률 출력"""
    exp_x = np.exp(x - np.max(x))  # 수치 안정성
    return exp_x / np.sum(exp_x)


def main():
    print("=" * 60)
    print("활성화 함수 비교")
    print("=" * 60)

    # 테스트 입력값
    x = np.array([-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0])

    print("\n[입력값]")
    print(f"  x = {x}")

    # 1. Sigmoid
    print("\n" + "-" * 40)
    print("[Sigmoid]")
    print("-" * 40)
    print(f"  수식: σ(x) = 1 / (1 + e^(-x))")
    print(f"  출력 범위: (0, 1)")
    y_sigmoid = sigmoid(x)
    print(f"  결과: {np.round(y_sigmoid, 4)}")
    print(f"  특징:")
    print(f"    - 확률 해석 가능 (0~1 사이)")
    print(f"    - 기울기 소실 문제: 양 끝에서 기울기 ≈ 0")
    print(f"    - σ(-3) = {sigmoid(-3):.4f}, σ'(-3) = {sigmoid_derivative(-3):.4f}")

    # 2. Tanh
    print("\n" + "-" * 40)
    print("[Tanh]")
    print("-" * 40)
    print(f"  수식: tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))")
    print(f"  출력 범위: (-1, 1)")
    y_tanh = tanh(x)
    print(f"  결과: {np.round(y_tanh, 4)}")
    print(f"  특징:")
    print(f"    - 0 중심 출력 (Sigmoid보다 학습 안정)")
    print(f"    - 여전히 기울기 소실 문제 존재")

    # 3. ReLU
    print("\n" + "-" * 40)
    print("[ReLU]")
    print("-" * 40)
    print(f"  수식: f(x) = max(0, x)")
    print(f"  출력 범위: [0, ∞)")
    y_relu = relu(x)
    print(f"  결과: {np.round(y_relu, 4)}")
    print(f"  특징:")
    print(f"    - 계산 매우 효율적 (비교 연산만)")
    print(f"    - 기울기 소실 문제 완화")
    print(f"    - Dying ReLU 문제: 음수 입력에서 뉴런 비활성화")
    print(f"    - x < 0에서 기울기 = 0 (학습 불가)")

    # 4. Leaky ReLU
    print("\n" + "-" * 40)
    print("[Leaky ReLU]")
    print("-" * 40)
    alpha = 0.01
    print(f"  수식: f(x) = max(αx, x), α = {alpha}")
    print(f"  출력 범위: (-∞, ∞)")
    y_leaky = leaky_relu(x, alpha)
    print(f"  결과: {np.round(y_leaky, 4)}")
    print(f"  특징:")
    print(f"    - Dying ReLU 문제 완화")
    print(f"    - 음수 영역에서도 작은 기울기 유지")

    # 5. GELU
    print("\n" + "-" * 40)
    print("[GELU]")
    print("-" * 40)
    print(f"  수식: GELU(x) = x × Φ(x) (Φ: 표준정규분포 CDF)")
    print(f"  근사: 0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))")
    y_gelu = gelu(x)
    print(f"  결과: {np.round(y_gelu, 4)}")
    print(f"  특징:")
    print(f"    - 부드러운 곡선 (x=0에서도 미분 가능)")
    print(f"    - 음수 입력에서도 0이 아닌 기울기")
    print(f"    - GPT, BERT 등 Transformer 모델 표준")
    print(f"    - 계산 비용이 ReLU보다 높음")

    # 6. Softmax
    print("\n" + "-" * 40)
    print("[Softmax]")
    print("-" * 40)
    print(f"  수식: softmax(xᵢ) = e^xᵢ / Σe^xⱼ")
    logits = np.array([2.0, 1.0, 0.1])
    y_softmax = softmax(logits)
    print(f"  입력 (logits): {logits}")
    print(f"  결과 (확률): {np.round(y_softmax, 4)}")
    print(f"  합계: {np.sum(y_softmax):.4f}")
    print(f"  특징:")
    print(f"    - 다중 클래스 분류 출력층에 사용")
    print(f"    - 모든 출력의 합 = 1 (확률 분포)")

    # 비교 요약
    print("\n" + "=" * 60)
    print("활성화 함수 선택 가이드")
    print("=" * 60)
    print("""
    | 용도                  | 권장 함수      |
    |-----------------------|----------------|
    | CNN 은닉층            | ReLU           |
    | Transformer 은닉층    | GELU           |
    | RNN 은닉층            | Tanh           |
    | 이진 분류 출력층      | Sigmoid        |
    | 다중 분류 출력층      | Softmax        |
    | Dying ReLU 문제 시    | Leaky ReLU     |
    """)

    # 기울기 비교 (x = 0 근처)
    print("\n[기울기 비교 (x = 0)]")
    x_zero = 0.0
    print(f"  Sigmoid:    f'(0) = {sigmoid_derivative(x_zero):.4f}")
    print(f"  Tanh:       f'(0) = {tanh_derivative(x_zero):.4f}")
    print(f"  ReLU:       f'(0) = {relu_derivative(x_zero):.4f} (정의에 따라 0 또는 1)")
    print(f"  Leaky ReLU: f'(0) = {leaky_relu_derivative(x_zero):.4f}")
    print(f"  GELU:       f'(0) = {gelu_derivative(x_zero):.4f}")


if __name__ == "__main__":
    main()
