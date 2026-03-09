import os
import sys
import subprocess
import platform
import torch
import importlib.util

def print_section(title):
    """섹션 제목 출력"""
    print(f"\n{'='*60}")
    print(f"[{title}]")
    print(f"{'='*60}")

def check_python_version():
    """Python 버전 확인 (3.8 이상 필요)"""
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"

    if version.major >= 3 and version.minor >= 8:
        print(f"✓ Python {version_str} (OK)")
        return True
    else:
        print(f"✗ Python {version_str} (3.8+ 필요)")
        return False

def detect_gpu():
    """GPU 자동 감지 및 정보 출력 (CUDA/MPS 지원)"""
    # NVIDIA GPU (CUDA) 감지
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        cuda_version = torch.version.cuda
        gpu_name = torch.cuda.get_device_name(0)
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9

        print(f"✓ NVIDIA GPU 감지됨")
        print(f"  CUDA 버전: {cuda_version}")
        print(f"  GPU 개수: {gpu_count}")
        print(f"  GPU 이름: {gpu_name}")
        print(f"  GPU 메모리: {total_memory:.1f} GB")
        return "cuda"
    
    # Apple Silicon GPU (MPS) 감지
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print(f"✓ Apple Silicon GPU (MPS) 감지됨")
        print(f"  디바이스: Metal Performance Shaders")
        print(f"  플랫폼: macOS (Apple Silicon)")
        return "mps"
    
    else:
        print(f"! GPU 미감지 (CPU만 사용 가능)")
        return "cpu"

def create_virtual_env():
    """가상환경 생성 (이미 있으면 스킵)"""
    venv_path = "venv"

    if os.path.exists(venv_path):
        print(f"✓ 가상환경 이미 존재 ({venv_path})")
        return True

    try:
        print(f"- 가상환경 생성 중...")
        subprocess.check_call([sys.executable, "-m", "venv", venv_path])

        # 활성화 명령어 안내
        if platform.system() == "Windows":
            activate_cmd = f"{venv_path}\\Scripts\\activate"
        else:
            activate_cmd = f"source {venv_path}/bin/activate"

        print(f"✓ 가상환경 생성 완료")
        print(f"  활성화 명령어: {activate_cmd}")
        return True
    except Exception as e:
        print(f"✗ 가상환경 생성 실패: {e}")
        return False

def install_pytorch(device_type):
    """PyTorch 설치 (CUDA/MPS/CPU 자동 선택)"""
    try:
        # 이미 설치되었으면 스킵
        if torch.__version__:
            print(f"✓ PyTorch {torch.__version__} 이미 설치됨")
            return True
    except:
        pass

    print(f"- PyTorch 설치 중...")

    # PyTorch 설치 명령어
    if device_type == "cuda":
        # NVIDIA GPU - CUDA 12.1 지원 버전
        cmd = [
            sys.executable, "-m", "pip", "install",
            "torch", "torchvision", "torchaudio",
            "--index-url", "https://download.pytorch.org/whl/cu121"
        ]
        print(f"  (NVIDIA GPU 버전 - CUDA 12.1)")
    elif device_type == "mps":
        # Apple Silicon - 기본 버전 (MPS 포함)
        cmd = [
            sys.executable, "-m", "pip", "install",
            "torch", "torchvision", "torchaudio"
        ]
        print(f"  (Apple Silicon 버전 - MPS 지원)")
    else:
        # CPU 전용
        cmd = [
            sys.executable, "-m", "pip", "install",
            "torch", "torchvision", "torchaudio"
        ]
        print(f"  (CPU 버전)")

    try:
        subprocess.check_call(cmd)
        print(f"✓ PyTorch {torch.__version__} 설치 완료")
        return True
    except Exception as e:
        print(f"✗ PyTorch 설치 실패: {e}")
        return False

def install_dependencies():
    """필수 패키지 설치"""
    packages = [
        "numpy",
        "pandas",
        "matplotlib",
        "jupyter",
        "transformers",
        "tqdm"
    ]

    print(f"- 필수 패키지 설치 중...")

    try:
        for package in packages:
            # 이미 설치되었는지 확인
            spec = importlib.util.find_spec(package)
            if spec is not None:
                print(f"  ✓ {package}")
            else:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                print(f"  ✓ {package}")

        return True
    except Exception as e:
        print(f"✗ 패키지 설치 실패: {e}")
        return False

def verify_installation():
    """설치 검증: import 테스트"""
    packages = {
        "torch": "PyTorch",
        "numpy": "NumPy",
        "pandas": "Pandas",
        "matplotlib": "Matplotlib",
        "transformers": "Transformers"
    }

    print(f"- 설치 검증 중...")

    all_ok = True
    for module, name in packages.items():
        try:
            importlib.import_module(module)
            print(f"  ✓ {name} import OK")
        except ImportErrCUDA/MPS 지원)"""
    import time

    print(f"- 벤치마크 수행 중...")

    size = 1000
    A_cpu = torch.randn(size, size)
    B_cpu = torch.randn(size, size)

    # CPU 벤치마크
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.time()
    for _ in range(3):
        C_cpu = torch.matmul(A_cpu, A_cpu)
    cpu_time = (time.time() - start) / 3

    print(f"  CPU 행렬 곱 (1000×1000): {cpu_time:.4f}초")

    # NVIDIA GPU (CUDA) 벤치마크
    if torch.cuda.is_available():
        A_gpu = A_cpu.cuda()
        B_gpu = B_cpu.cuda()

        torch.cuda.synchronize()
        start = time.time()
        for _ in range(3):
            C_gpu = torch.matmul(A_gpu, A_gpu)
        torch.cuda.synchronize()
        gpu_time = (time.time() - start) / 3

        print(f"  NVIDIA GPU 행렬 곱 (1000×1000): {gpu_time:.4f}초")
        speedup = cpu_time / gpu_time
        print(f"  속도 향상: {speedup:.1f}배 ✓")
    
    # Apple Silicon GPU (MPS) 벤치마크
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        A_mps = A_cpu.to("mps")
        B_mps = B_cpu.to("mps")

        start = time.time()
        for _ in range(3):
            C_mps = torch.matmul(A_mps, A_mps)
        mps_time = (time.time() - start) / 3

        print(f"  Apple MPS 행렬 곱 (1000×1000): {mps_time:.4f}초")
        speedup = cpu_time / mps_time
        print(f"  속도 향상: {speedup:.1f}배 ✓")
    
            C_gpu = torch.matmul(A_gpu, A_gpu)
        torch.cuda.synchronize()
        gpu_time = (time.time() - start) / 3

        print(f"  GPU 행렬 곱 (1000×1000): {gpu_time:.4f}초")
        speedup = cpu_time / gpu_time
        print(f"  속도 향상: {speedup:.1f}배 ✓")
    else:
        print(f"  GPU 미감지: GPU 벤치마크 스킵")

def main():
    device_type = detect_gpu()

    print_section("Step 3: PyTorch 설치")
    if not install_pytorch(device_type

    print(f"시스템 정보:")
    print(f"  OS: {platform.system()}")
    print(f"  Python: {sys.version.split()[0]}")

    # 단계별 실행
    print_section("Step 1: Python 버전 확인")
    if not check_python_version():
        print("✗ Python 버전이 낮습니다. 업그레이드하세요.")
        return False

    print_section("Step 2: GPU 감지")
    has_gpu = detect_gpu()

    print_section("Step 3: PyTorch 설치")
    if not install_pytorch(has_gpu):
        print("✗ PyTorch 설치에 실패했습니다.")
        return False
if device_type == "cuda":
        print(f"  감지된 디바이스: NVIDIA GPU (CUDA)")
        print(f"  사용 방법: device = torch.device('cuda')")
    elif device_type == "mps":
        print(f"  감지된 디바이스: Apple Silicon (MPS)")
        print(f"  사용 방법: device = torch.device('mps')")
    else:
        print(f"  감지된 디바이스: CPU")
        print(f"  사용 방법: device = torch.device('cpu')
    print_section("Step 4: 필수 패키지 설치")
    if not install_dependencies():
        print("✗ 일부 패키지 설치에 실패했습니다.")
        return False

    print_section("Step 5: 설치 검증")
    if not verify_installation():
        print("✗ 일부 패키지 import에 실패했습니다.")
        return False

    print_section("Step 6: 벤치마크")
    benchmark_cpu_gpu()

    print_section("완료")
    print("✓ 모든 설정이 완료되었습니다!")
    print(f"  다음 단계: GPU가 있다면 'cuda', 없다면 'cpu'를 활용하세요")
    print("="*60)

    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
