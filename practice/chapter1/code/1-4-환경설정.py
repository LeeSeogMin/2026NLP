"""
1-4-환경설정.py
제1장 실습: 개발 환경 구축 - 라이브러리 설치 확인

이 스크립트는 딥러닝 개발에 필요한 핵심 라이브러리가
올바르게 설치되었는지 확인한다.

실행 방법:
    python 1-4-환경설정.py
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

        # 간단한 연산 테스트
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

        # 간단한 DataFrame 테스트
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
        import matplotlib.pyplot as plt
        print(f"Matplotlib 버전: {matplotlib.__version__}")

        # 그래프 생성 테스트 (저장만, 표시하지 않음)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot([1, 2, 3, 4], [1, 4, 2, 3])
        ax.set_title("Test Plot")

        # 출력 폴더에 저장
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


def check_transformers():
    """Hugging Face Transformers 설치를 확인한다 (선택)."""
    print("=" * 50)
    print("5. Hugging Face Transformers 확인 (선택)")
    print("=" * 50)

    try:
        import transformers
        print(f"Transformers 버전: {transformers.__version__}")
        print("✓ Transformers 정상 설치됨")
    except ImportError:
        print("△ Transformers가 설치되지 않았습니다.")
        print("  (선택 사항) 설치: pip install transformers")
    print()


def main():
    """모든 라이브러리 설치 상태를 확인한다."""
    print()
    print("╔" + "═" * 48 + "╗")
    print("║     딥러닝 자연어처리 개발 환경 확인 도구      ║")
    print("╚" + "═" * 48 + "╝")
    print()

    check_python_version()
    check_numpy()
    check_pandas()
    check_matplotlib()
    check_transformers()

    print("=" * 50)
    print("환경 확인 완료")
    print("=" * 50)
    print("다음 단계: PyTorch 설치 확인 (1-4-pytorch확인.py)")
    print()


if __name__ == "__main__":
    main()
