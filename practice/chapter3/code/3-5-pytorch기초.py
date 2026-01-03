"""
3장 실습: PyTorch 기초 - Tensor와 Autograd
- Tensor 생성 및 연산
- Autograd를 이용한 자동 미분
"""

import torch


def tensor_basics():
    """텐서 기본 조작"""
    print("=" * 50)
    print("1. 텐서 생성")
    print("=" * 50)

    # 1.1 리스트에서 생성
    data = [[1, 2], [3, 4]]
    t1 = torch.tensor(data)
    print(f"리스트에서 생성:\n{t1}")
    print(f"  shape: {t1.shape}, dtype: {t1.dtype}")

    # 1.2 특수 텐서 생성
    zeros = torch.zeros(2, 3)
    ones = torch.ones(2, 3)
    randn = torch.randn(2, 3)  # 표준정규분포
    arange = torch.arange(0, 10, 2)

    print(f"\nzeros(2,3):\n{zeros}")
    print(f"\nones(2,3):\n{ones}")
    print(f"\nrandn(2,3):\n{randn}")
    print(f"\narange(0, 10, 2): {arange}")

    # 1.3 텐서 속성
    print("\n[텐서 속성]")
    t = torch.randn(3, 4)
    print(f"  shape: {t.shape}")
    print(f"  dtype: {t.dtype}")
    print(f"  device: {t.device}")

    print("\n" + "=" * 50)
    print("2. 텐서 연산")
    print("=" * 50)

    # 2.1 기본 연산
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([4.0, 5.0, 6.0])

    print(f"a = {a}")
    print(f"b = {b}")
    print(f"a + b = {a + b}")
    print(f"a * b = {a * b}")  # 요소별 곱셈
    print(f"a @ b = {a @ b}")  # 내적 (dot product)

    # 2.2 행렬 연산
    print("\n[행렬 연산]")
    A = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
    B = torch.tensor([[5, 6], [7, 8]], dtype=torch.float32)

    print(f"A =\n{A}")
    print(f"B =\n{B}")
    print(f"A @ B (행렬 곱) =\n{A @ B}")
    print(f"A * B (요소별 곱) =\n{A * B}")
    print(f"A.T (전치) =\n{A.T}")

    # 2.3 인덱싱과 슬라이싱
    print("\n[인덱싱과 슬라이싱]")
    t = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print(f"t =\n{t}")
    print(f"t[0] = {t[0]}")
    print(f"t[:, 0] = {t[:, 0]}")
    print(f"t[1:, 1:] =\n{t[1:, 1:]}")

    # 2.4 차원 변환
    print("\n[차원 변환]")
    t = torch.arange(12)
    print(f"원본: {t}, shape: {t.shape}")
    t_reshaped = t.reshape(3, 4)
    print(f"reshape(3,4):\n{t_reshaped}")
    t_view = t.view(4, 3)
    print(f"view(4,3):\n{t_view}")


def autograd_basics():
    """Autograd를 이용한 자동 미분"""
    print("\n" + "=" * 50)
    print("3. Autograd - 자동 미분")
    print("=" * 50)

    # 3.1 기본 자동 미분
    print("\n[기본 예제: y = x²]")
    x = torch.tensor(3.0, requires_grad=True)
    y = x ** 2
    print(f"x = {x.item()}")
    print(f"y = x² = {y.item()}")

    y.backward()  # dy/dx 계산
    print(f"dy/dx = 2x = {x.grad.item()}")
    print(f"검증: 2 × 3 = 6 ✓")

    # 3.2 복잡한 함수
    print("\n[복잡한 함수: y = 3x² + 2x + 1]")
    x = torch.tensor(2.0, requires_grad=True)
    y = 3 * x ** 2 + 2 * x + 1
    print(f"x = {x.item()}")
    print(f"y = 3x² + 2x + 1 = {y.item()}")

    y.backward()
    print(f"dy/dx = 6x + 2 = {x.grad.item()}")
    print(f"검증: 6×2 + 2 = 14 ✓")

    # 3.3 벡터 입력
    print("\n[벡터 입력]")
    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = x ** 2
    z = y.sum()  # 스칼라로 변환 (backward는 스칼라에서 호출)

    print(f"x = {x.tolist()}")
    print(f"y = x² = {y.tolist()}")
    print(f"z = sum(y) = {z.item()}")

    z.backward()
    print(f"dz/dx = 2x = {x.grad.tolist()}")

    # 3.4 연산 그래프
    print("\n[연산 그래프 예제]")
    x = torch.tensor(2.0, requires_grad=True)
    w = torch.tensor(3.0, requires_grad=True)
    b = torch.tensor(1.0, requires_grad=True)

    # y = wx + b
    y = w * x + b
    print(f"y = w*x + b = {w.item()}*{x.item()} + {b.item()} = {y.item()}")

    y.backward()
    print(f"dy/dw = x = {w.grad.item()}")
    print(f"dy/dx = w = {x.grad.item()}")
    print(f"dy/db = 1 = {b.grad.item()}")

    # 3.5 기울기 누적과 초기화
    print("\n[기울기 누적 주의사항]")
    x = torch.tensor(2.0, requires_grad=True)

    for i in range(3):
        y = x ** 2
        y.backward()
        print(f"  반복 {i + 1}: x.grad = {x.grad.item()}")

    print("  → 기울기가 누적됨! 매 반복마다 zero_grad() 필요")

    # 올바른 방법
    print("\n[올바른 방법: zero_grad()]")
    x = torch.tensor(2.0, requires_grad=True)

    for i in range(3):
        if x.grad is not None:
            x.grad.zero_()  # 기울기 초기화
        y = x ** 2
        y.backward()
        print(f"  반복 {i + 1}: x.grad = {x.grad.item()}")

    # 3.6 no_grad - 추론 모드
    print("\n[torch.no_grad() - 추론 모드]")
    x = torch.tensor(2.0, requires_grad=True)

    with torch.no_grad():
        y = x ** 2
        print(f"no_grad 내부: y.requires_grad = {y.requires_grad}")

    y_normal = x ** 2
    print(f"일반 모드: y.requires_grad = {y_normal.requires_grad}")
    print("→ 추론 시 no_grad() 사용으로 메모리 절약")


def main():
    print("PyTorch 기초 실습")
    print("=" * 50)

    # 버전 확인
    print(f"PyTorch 버전: {torch.__version__}")
    print(f"CUDA 사용 가능: {torch.cuda.is_available()}")

    # 텐서 기초
    tensor_basics()

    # Autograd
    autograd_basics()

    print("\n" + "=" * 50)
    print("핵심 정리")
    print("=" * 50)
    print("""
    1. Tensor: 다차원 배열 (NumPy와 유사)
       - torch.tensor(), torch.zeros(), torch.randn()
       - 연산: +, *, @(행렬곱), reshape(), view()

    2. Autograd: 자동 미분 엔진
       - requires_grad=True로 기울기 추적
       - backward()로 역전파 실행
       - .grad로 기울기 접근
       - 기울기 누적에 주의 (zero_grad() 사용)
       - 추론 시 torch.no_grad() 사용
    """)


if __name__ == "__main__":
    main()
