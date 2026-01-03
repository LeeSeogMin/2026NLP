"""
1-4-pytorch확인.py
제1장 실습: 개발 환경 구축 - PyTorch 설치 및 GPU 확인

이 스크립트는 PyTorch가 올바르게 설치되었는지 확인하고,
GPU(CUDA) 사용 가능 여부를 점검한다.

실행 방법:
    python 1-4-pytorch확인.py
"""

import sys


def check_pytorch_installation():
    """PyTorch 설치 상태를 확인한다."""
    print("=" * 50)
    print("1. PyTorch 설치 확인")
    print("=" * 50)

    try:
        import torch
        print(f"PyTorch 버전: {torch.__version__}")
        print("✓ PyTorch 정상 설치됨")
        return True
    except ImportError:
        print("✗ PyTorch가 설치되지 않았습니다.")
        print()
        print("설치 방법:")
        print("  CPU 버전: pip install torch")
        print("  GPU 버전: https://pytorch.org/get-started/locally/ 참조")
        return False
    finally:
        print()


def check_cuda_availability():
    """CUDA(GPU) 사용 가능 여부를 확인한다."""
    print("=" * 50)
    print("2. GPU(CUDA) 확인")
    print("=" * 50)

    try:
        import torch

        if torch.cuda.is_available():
            print("✓ CUDA 사용 가능")
            print(f"  CUDA 버전: {torch.version.cuda}")
            print(f"  cuDNN 버전: {torch.backends.cudnn.version()}")
            print(f"  GPU 개수: {torch.cuda.device_count()}")

            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory
                gpu_memory_gb = gpu_memory / (1024 ** 3)
                print(f"  GPU {i}: {gpu_name} ({gpu_memory_gb:.1f} GB)")
        else:
            print("△ CUDA를 사용할 수 없습니다.")
            print("  (CPU 모드로 실행됩니다)")
            print()
            print("GPU 사용을 원한다면:")
            print("  1. NVIDIA GPU가 설치되어 있는지 확인")
            print("  2. CUDA Toolkit 설치")
            print("  3. GPU 버전 PyTorch 재설치")
    except Exception as e:
        print(f"✗ CUDA 확인 중 오류: {e}")
    print()


def check_mps_availability():
    """Apple Silicon MPS 사용 가능 여부를 확인한다."""
    print("=" * 50)
    print("3. Apple Silicon (MPS) 확인")
    print("=" * 50)

    try:
        import torch

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            print("✓ MPS(Metal Performance Shaders) 사용 가능")
            print("  Apple Silicon GPU 가속을 사용할 수 있습니다.")
        else:
            print("△ MPS를 사용할 수 없습니다.")
            print("  (Apple Silicon Mac이 아니거나 지원되지 않는 환경)")
    except Exception as e:
        print(f"△ MPS 확인 불가: {e}")
    print()


def run_simple_tensor_test():
    """간단한 텐서 연산 테스트를 수행한다."""
    print("=" * 50)
    print("4. 텐서 연산 테스트")
    print("=" * 50)

    try:
        import torch

        # 텐서 생성
        x = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
        y = torch.tensor([[5, 6], [7, 8]], dtype=torch.float32)

        print("텐서 x:")
        print(x)
        print()
        print("텐서 y:")
        print(y)
        print()

        # 행렬 곱
        z = torch.matmul(x, y)
        print("행렬 곱 (x @ y):")
        print(z)
        print()

        # 사용 가능한 디바이스 선택
        if torch.cuda.is_available():
            device = torch.device("cuda")
            device_name = "CUDA GPU"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
            device_name = "Apple MPS"
        else:
            device = torch.device("cpu")
            device_name = "CPU"

        # 디바이스에서 연산
        x_device = x.to(device)
        y_device = y.to(device)
        z_device = torch.matmul(x_device, y_device)

        print(f"연산 디바이스: {device_name}")
        print(f"결과 텐서 위치: {z_device.device}")
        print("✓ 텐서 연산 정상 작동")

    except Exception as e:
        print(f"✗ 텐서 연산 오류: {e}")
    print()


def check_autograd():
    """자동 미분(Autograd) 기능을 테스트한다."""
    print("=" * 50)
    print("5. 자동 미분 (Autograd) 테스트")
    print("=" * 50)

    try:
        import torch

        # requires_grad=True로 텐서 생성
        x = torch.tensor([2.0, 3.0], requires_grad=True)
        print(f"입력 텐서 x: {x}")

        # 연산 수행
        y = x ** 2
        z = y.sum()
        print(f"연산: z = sum(x^2) = {z.item()}")

        # 역전파
        z.backward()
        print(f"기울기 dz/dx = 2x: {x.grad}")

        # 검증: dz/dx = 2x 이므로 [4.0, 6.0]이어야 함
        expected = 2 * x.detach()
        if torch.allclose(x.grad, expected):
            print("✓ 자동 미분 정상 작동")
        else:
            print("✗ 자동 미분 결과가 예상과 다릅니다.")

    except Exception as e:
        print(f"✗ 자동 미분 오류: {e}")
    print()


def main():
    """PyTorch 환경을 종합적으로 확인한다."""
    print()
    print("╔" + "═" * 48 + "╗")
    print("║         PyTorch 환경 확인 도구               ║")
    print("╚" + "═" * 48 + "╝")
    print()

    # PyTorch 설치 확인
    if not check_pytorch_installation():
        print("PyTorch를 먼저 설치해주세요.")
        return

    # GPU 확인
    check_cuda_availability()
    check_mps_availability()

    # 연산 테스트
    run_simple_tensor_test()
    check_autograd()

    print("=" * 50)
    print("PyTorch 환경 확인 완료")
    print("=" * 50)
    print()
    print("요약:")

    import torch
    print(f"  - PyTorch 버전: {torch.__version__}")

    if torch.cuda.is_available():
        print(f"  - 연산 디바이스: CUDA GPU ({torch.cuda.get_device_name(0)})")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("  - 연산 디바이스: Apple MPS")
    else:
        print("  - 연산 디바이스: CPU")

    print()
    print("이제 딥러닝 실습을 시작할 준비가 되었습니다!")
    print()


if __name__ == "__main__":
    main()
