"""
환경 구축 스크립트
- Python 버전 확인
- GPU 감지 (CUDA / MPS / CPU)
- 필수 패키지 설치
- 설치 검증 및 벤치마크
"""

import os
import sys
import subprocess
import platform
import importlib.util
import shutil
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
VENV_DIR = ROOT_DIR / ".venv"


def venv_python_path():
    """루트 가상환경의 Python 실행 파일 경로"""
    if platform.system() == "Windows":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def is_compatible_python(executable):
    """NLP 패키지 휠 호환성을 고려한 Python 버전 확인"""
    try:
        result = subprocess.run(
            [
                str(executable),
                "-c",
                "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')",
            ],
            check=True,
            text=True,
            capture_output=True,
        )
    except Exception:
        return False

    major, minor = map(int, result.stdout.strip().split("."))
    return major == 3 and 8 <= minor <= 12


def choose_python_for_venv():
    """가상환경 생성에 사용할 Python 선택"""
    if is_compatible_python(sys.executable):
        return Path(sys.executable)

    for name in ("python3.12", "python3.11", "python3.10", "python3.9", "python3.8"):
        path = shutil.which(name)
        if path and is_compatible_python(path):
            return Path(path)

    return None


def running_inside_root_venv():
    """현재 Python이 루트 .venv 안에서 실행 중인지 확인"""
    try:
        return Path(sys.prefix).resolve() == VENV_DIR.resolve()
    except Exception:
        return False


def ensure_root_virtualenv():
    """루트 .venv를 만들고, 현재 프로세스를 그 Python으로 재실행"""
    if running_inside_root_venv():
        return

    print_section("Step 0: 루트 가상환경 준비")

    python_for_venv = choose_python_for_venv()
    if python_for_venv is None:
        print("  ✗ Python 3.8~3.12 실행 파일을 찾지 못했습니다.")
        print("    BERTopic/HDBSCAN 등 일부 NLP 패키지는 최신 Python에서 휠 지원이 늦을 수 있습니다.")
        print("    python3.12 또는 python3.11 설치 후 다시 실행하세요.")
        sys.exit(1)

    if VENV_DIR.exists():
        print(f"  ✓ 기존 가상환경 사용: {VENV_DIR}")
    else:
        print(f"  - 가상환경 생성: {VENV_DIR}")
        print(f"    Python: {python_for_venv}")
        subprocess.check_call([str(python_for_venv), "-m", "venv", str(VENV_DIR)])

    venv_python = venv_python_path()
    if not venv_python.exists():
        print(f"  ✗ 가상환경 Python을 찾지 못했습니다: {venv_python}")
        sys.exit(1)

    print("  - pip 업그레이드 중...")
    subprocess.check_call([str(venv_python), "-m", "pip", "install", "--upgrade", "pip"])

    print("  - 루트 가상환경 안에서 setup_env.py 재실행")
    os.execv(str(venv_python), [str(venv_python), str(Path(__file__).resolve())])


def print_section(title):
    print(f"\n{'='*60}")
    print(f"[{title}]")
    print(f"{'='*60}")


def check_python_version():
    """Python 버전 확인 (3.8~3.12 권장/필수)"""
    v = sys.version_info
    version_str = f"{v.major}.{v.minor}.{v.micro}"

    if v.major == 3 and 8 <= v.minor <= 12:
        print(f"  ✓ Python {version_str}")
        return True
    else:
        print(f"  ✗ Python {version_str} (3.8~3.12 필요)")
        return False


def detect_gpu():
    """GPU 자동 감지 (CUDA / MPS / CPU)"""
    try:
        import torch
    except ImportError:
        print("  ! PyTorch 미설치 — GPU 감지 후 설치 진행")
        # PyTorch 없이 간단한 감지
        if platform.system() == "Darwin" and platform.machine() == "arm64":
            print("  → Apple Silicon 감지됨 (MPS 가능)")
            return "mps"
        print("  → CPU 사용")
        return "cpu"

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  ✓ NVIDIA GPU: {gpu_name} ({memory:.1f} GB)")
        print(f"    CUDA 버전: {torch.version.cuda}")
        return "cuda"

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print(f"  ✓ Apple Silicon GPU (MPS)")
        return "mps"

    print(f"  ! GPU 미감지 — CPU만 사용")
    return "cpu"


def install_pytorch(device_type):
    """PyTorch 설치 (GPU 유형에 따라 자동 선택)"""
    try:
        import torch
        print(f"  ✓ PyTorch {torch.__version__} 이미 설치됨")
        return True
    except ImportError:
        pass

    print(f"  - PyTorch 설치 중...")

    if device_type == "cuda":
        cmd = [
            sys.executable, "-m", "pip", "install",
            "torch", "torchvision", "torchaudio",
            "--index-url", "https://download.pytorch.org/whl/cu121",
        ]
        print(f"    (NVIDIA GPU — CUDA 12.1)")
    else:
        cmd = [
            sys.executable, "-m", "pip", "install",
            "torch", "torchvision", "torchaudio",
        ]
        label = "Apple Silicon MPS" if device_type == "mps" else "CPU"
        print(f"    ({label})")

    try:
        subprocess.check_call(cmd, stdout=subprocess.DEVNULL)
        import torch
        print(f"  ✓ PyTorch {torch.__version__} 설치 완료")
        return True
    except Exception as e:
        print(f"  ✗ PyTorch 설치 실패: {e}")
        return False


def install_dependencies():
    """필수 패키지 설치"""
    packages = [
        "numpy",
        "matplotlib",
        "transformers",
        "tqdm",
        "ipykernel",
    ]

    print(f"  - 필수 패키지 확인/설치 중...")
    all_ok = True

    for package in packages:
        spec = importlib.util.find_spec(package)
        if spec is not None:
            print(f"    ✓ {package}")
        else:
            try:
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", package],
                    stdout=subprocess.DEVNULL,
                )
                print(f"    ✓ {package} (새로 설치)")
            except Exception as e:
                print(f"    ✗ {package} 설치 실패: {e}")
                all_ok = False

    return all_ok


def verify_installation():
    """설치 검증: import 테스트"""
    checks = {
        "torch": "PyTorch",
        "numpy": "NumPy",
        "matplotlib": "Matplotlib",
        "transformers": "Transformers",
        "ipykernel": "IPython Kernel",
    }

    all_ok = True
    for module, name in checks.items():
        try:
            mod = importlib.import_module(module)
            version = getattr(mod, "__version__", "OK")
            print(f"    ✓ {name} {version}")
        except ImportError:
            print(f"    ✗ {name} import 실패")
            all_ok = False

    return all_ok


def register_jupyter_kernel():
    """루트 가상환경을 Jupyter 커널로 등록"""
    try:
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "ipykernel",
                "install",
                "--user",
                "--name",
                "2026nlp",
                "--display-name",
                "2026NLP",
            ],
            stdout=subprocess.DEVNULL,
        )
        print("    ✓ Jupyter 커널 등록: 2026NLP")
        return True
    except Exception as e:
        print(f"    ! Jupyter 커널 등록 스킵: {e}")
        return False


def benchmark(device_type):
    """CPU vs GPU 행렬 곱 벤치마크"""
    import torch
    import time

    size = 1000
    A = torch.randn(size, size)

    # CPU
    start = time.time()
    for _ in range(3):
        torch.matmul(A, A)
    cpu_time = (time.time() - start) / 3
    print(f"    CPU  ({size}×{size} 행렬 곱): {cpu_time:.4f}초")

    # GPU
    if device_type == "cuda":
        A_gpu = A.cuda()
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(3):
            torch.matmul(A_gpu, A_gpu)
        torch.cuda.synchronize()
        gpu_time = (time.time() - start) / 3
        print(f"    CUDA ({size}×{size} 행렬 곱): {gpu_time:.4f}초")
        print(f"    → 속도 향상: {cpu_time / gpu_time:.1f}배")

    elif device_type == "mps":
        A_mps = A.to("mps")
        start = time.time()
        for _ in range(3):
            torch.matmul(A_mps, A_mps)
        mps_time = (time.time() - start) / 3
        print(f"    MPS  ({size}×{size} 행렬 곱): {mps_time:.4f}초")
        print(f"    → 속도 향상: {cpu_time / mps_time:.1f}배")


def main():
    ensure_root_virtualenv()

    print(f"{'='*60}")
    print(f"딥러닝 자연어처리 — 환경 구축")
    print(f"{'='*60}")
    print(f"  OS: {platform.system()} {platform.machine()}")
    print(f"  Python: {sys.version.split()[0]}")
    print(f"  가상환경: {VENV_DIR}")

    # Step 1
    print_section("Step 1: Python 버전 확인")
    if not check_python_version():
        print("\n✗ Python 3.8 이상으로 업그레이드하세요.")
        return False

    # Step 2
    print_section("Step 2: GPU 감지")
    device_type = detect_gpu()

    # Step 3
    print_section("Step 3: PyTorch 설치")
    if not install_pytorch(device_type):
        print("\n✗ PyTorch 설치에 실패했습니다.")
        return False

    # Step 4
    print_section("Step 4: 필수 패키지 설치")
    if not install_dependencies():
        print("\n✗ 일부 패키지 설치에 실패했습니다.")
        return False

    # Step 5
    print_section("Step 5: 설치 검증")
    if not verify_installation():
        print("\n✗ 일부 패키지 import에 실패했습니다.")
        return False

    # Step 6
    print_section("Step 6: Jupyter 커널 등록")
    register_jupyter_kernel()

    # Step 7
    print_section("Step 7: 벤치마크")
    try:
        benchmark(device_type)
    except Exception as e:
        print(f"    ! 벤치마크 스킵: {e}")

    # 완료
    print_section("완료")
    print("  ✓ 모든 설정이 완료되었습니다!")
    device_map = {"cuda": "cuda", "mps": "mps", "cpu": "cpu"}
    print(f"  디바이스: torch.device('{device_map[device_type]}')")
    print(f"\n  다음 단계:")
    print(f"    source .venv/bin/activate")
    print(f"    jupyter notebook")
    print(f"    # 노트북에서 커널을 '2026NLP'로 선택")
    print(f"    python practice/chapter1/code/1-2-환경설정.py")
    print(f"{'='*60}")
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
