"""
1-2-환경설정.py
제1장 실습: 개발 환경 구축 — 라이브러리 및 PyTorch 환경 확인

이 스크립트는 딥러닝 개발에 필요한 핵심 라이브러리와 PyTorch가
올바르게 설치되었는지 확인하고, GPU 사용 가능 여부를 점검한다.

실행 방법:
    python 1-2-환경설정.py
"""

import sys
from pathlib import Path


def check_python_version():
    """Python 버전을 확인한다."""
    print("=" * 50)
    print("1. Python 버전 확인")
    print("=" * 50)

    version = sys.version_info
    print(f"Python 버전: {version.major}.{version.minor}.{version.micro}")
    print(f"실행 경로: {sys.executable}")

    if version.major >= 3 and version.minor >= 10:
        print("✓ Python 3.10 이상 확인됨 (권장 버전)")
    elif version.major >= 3 and version.minor >= 9:
        print("△ Python 3.9 확인됨 (최소 요구사항 충족)")
    else:
        print("✗ Python 3.9 이상이 필요합니다.")
    print()


def check_numpy():
    """NumPy 설치를 확인한다."""
    print("=" * 50)
    print("2. NumPy 확인")
    print("=" * 50)

    try:
        import numpy as np
        print(f"NumPy 버전: {np.__version__}")
        arr = np.array([1, 2, 3, 4, 5])
        print(f"배열 생성 테스트: {arr}")
        print(f"배열 평균: {np.mean(arr)}")
        print("✓ NumPy 정상 작동")
    except ImportError:
        print("✗ NumPy가 설치되지 않았습니다.")
        print("  설치: pip install numpy")
    print()


def check_pandas():
    """Pandas 설치를 확인한다."""
    print("=" * 50)
    print("3. Pandas 확인")
    print("=" * 50)

    try:
        import pandas as pd
        print(f"Pandas 버전: {pd.__version__}")
        df = pd.DataFrame({
            "이름": ["홍길동", "김철수", "이영희"],
            "나이": [25, 30, 28]
        })
        print("DataFrame 생성 테스트:")
        print(df.to_string())
        print("✓ Pandas 정상 작동")
    except ImportError:
        print("✗ Pandas가 설치되지 않았습니다.")
        print("  설치: pip install pandas")
    print()


def check_matplotlib():
    """Matplotlib 설치를 확인한다."""
    print("=" * 50)
    print("4. Matplotlib 확인")
    print("=" * 50)

    try:
        import matplotlib
        matplotlib.use("Agg")  # 비표시 백엔드 (서버/CI 호환)
        import matplotlib.pyplot as plt
        print(f"Matplotlib 버전: {matplotlib.__version__}")

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot([1, 2, 3, 4], [1, 4, 2, 3])
        ax.set_title("Test Plot")

        output_dir = Path(__file__).parent.parent / "data" / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "test_plot.png"
        fig.savefig(output_path)
        plt.close()

        print(f"테스트 그래프 저장: {output_path}")
        print("✓ Matplotlib 정상 작동")
    except ImportError:
        print("✗ Matplotlib가 설치되지 않았습니다.")
        print("  설치: pip install matplotlib")
    print()


def check_pytorch():
    """PyTorch 설치 및 GPU 환경을 확인한다."""
    print("=" * 50)
    print("5. PyTorch 확인")
    print("=" * 50)

    try:
        import torch
        print(f"PyTorch 버전: {torch.__version__}")
    except ImportError:
        print("✗ PyTorch가 설치되지 않았습니다.")
        print("  설치: pip install torch")
        print()
        return
    print()

    # GPU 확인
    print("=" * 50)
    print("6. GPU 환경 확인")
    print("=" * 50)

    if torch.cuda.is_available():
        print("✓ CUDA 사용 가능")
        print(f"  CUDA 버전: {torch.version.cuda}")
        print(f"  GPU 개수: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(i)
            mem = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
            print(f"  GPU {i}: {name} ({mem:.1f} GB)")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("✓ Apple MPS 사용 가능")
        print("  Apple Silicon GPU 가속을 사용할 수 있습니다.")
    else:
        print("△ GPU를 사용할 수 없습니다 (CPU 모드)")
    print()

    # 텐서 연산 테스트
    print("=" * 50)
    print("7. 텐서 연산 테스트")
    print("=" * 50)

    x = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
    y = torch.tensor([[5, 6], [7, 8]], dtype=torch.float32)
    z = torch.matmul(x, y)
    print(f"행렬 곱 결과:\n{z}")

    # 디바이스 이동 테스트
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    x_dev = x.to(device)
    print(f"텐서 디바이스: {x_dev.device}")
    print("✓ 텐서 연산 정상 작동")
    print()


def check_transformers():
    """Hugging Face Transformers 설치를 확인한다 (선택)."""
    print("=" * 50)
    print("8. Hugging Face Transformers 확인 (선택)")
    print("=" * 50)

    try:
        import transformers
        print(f"Transformers 버전: {transformers.__version__}")
        print("✓ Transformers 정상 설치됨")
    except ImportError:
        print("△ Transformers 미설치 (나중에 설치해도 됩니다)")
        print("  설치: pip install transformers")
    print()


def main():
    """모든 라이브러리 설치 상태를 종합 확인한다."""
    print()
    print("╔" + "═" * 48 + "╗")
    print("║   딥러닝 자연어처리 — 개발 환경 종합 확인 도구  ║")
    print("╚" + "═" * 48 + "╝")
    print()

    check_python_version()
    check_numpy()
    check_pandas()
    check_matplotlib()
    check_pytorch()
    check_transformers()

    print("=" * 50)
    print("환경 확인 완료!")
    print("=" * 50)
    print("다음 단계: 텐서 기초 실습 (1-3-텐서기초.py)")
    print()


if __name__ == "__main__":
    main()
