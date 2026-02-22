#!/usr/bin/env python3
"""
setup_env.py — 딥러닝 자연어처리 실습 환경 자동 설정 스크립트

이 스크립트 하나로 15주차 전체 실습 환경을 자동 구축한다:
  1. Python 가상환경 생성
  2. GPU 사양 자동 감지 (NVIDIA CUDA / Apple MPS / CPU)
  3. GPU에 맞는 PyTorch + CUDA 버전 자동 설치
  4. 전체 실습 패키지 일괄 설치
  5. 설치 결과 검증 및 GPU 벤치마크

실행 방법:
    python scripts/setup_env.py              # 자동 감지
    python scripts/setup_env.py --cpu        # 강제 CPU 모드
    python scripts/setup_env.py --cuda 12.1  # CUDA 버전 수동 지정

요구사항:
    - Python 3.10 이상
    - (선택) NVIDIA GPU + 드라이버 설치 완료
"""

import subprocess
import sys
import os
import platform
import argparse
import shutil
import time
import json
from pathlib import Path


# ── 색상 출력 ─────────────────────────────────────────────────────────
class Colors:
    """터미널 색상 코드 (Windows 호환)."""

    HEADER = "\033[95m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    END = "\033[0m"

    @staticmethod
    def init():
        """Windows에서 ANSI 이스케이프 시퀀스를 활성화한다."""
        if platform.system() == "Windows":
            os.system("")  # Windows 10+ ANSI 활성화


def print_header(msg):
    print(f"\n{Colors.BOLD}{Colors.HEADER}{'═' * 60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.HEADER}  {msg}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.HEADER}{'═' * 60}{Colors.END}")


def print_ok(msg):
    print(f"  {Colors.GREEN}✓{Colors.END} {msg}")


def print_warn(msg):
    print(f"  {Colors.YELLOW}△{Colors.END} {msg}")


def print_fail(msg):
    print(f"  {Colors.RED}✗{Colors.END} {msg}")


def print_info(msg):
    print(f"  {Colors.BLUE}ℹ{Colors.END} {msg}")


# ── 프로젝트 경로 ────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
VENV_DIR = PROJECT_ROOT / "venv"
REQUIREMENTS = PROJECT_ROOT / "requirements.txt"

# ── PyTorch CUDA 매핑 테이블 ─────────────────────────────────────────
# NVIDIA 드라이버가 지원하는 CUDA 버전 → PyTorch pip index URL
PYTORCH_CUDA_MAP = {
    "12.6": "cu124",
    "12.5": "cu124",
    "12.4": "cu124",
    "12.3": "cu121",
    "12.2": "cu121",
    "12.1": "cu121",
    "12.0": "cu121",
    "11.8": "cu118",
    "11.7": "cu118",
    "11.6": "cu118",
}


# ── 1단계: 시스템 정보 확인 ───────────────────────────────────────────
def check_system():
    """운영체제, Python 버전 등 시스템 정보를 확인한다."""
    print_header("1단계: 시스템 정보 확인")

    info = {
        "os": platform.system(),
        "os_version": platform.version(),
        "arch": platform.machine(),
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "python_path": sys.executable,
    }

    print_info(f"운영체제: {info['os']} {info['os_version']}")
    print_info(f"아키텍처: {info['arch']}")
    print_info(f"Python: {info['python']} ({info['python_path']})")

    if sys.version_info < (3, 10):
        print_fail("Python 3.10 이상이 필요합니다.")
        print_fail(f"현재 버전: {info['python']}")
        sys.exit(1)

    print_ok(f"Python {info['python']} 확인됨")
    return info


# ── 2단계: GPU 자동 감지 ──────────────────────────────────────────────
def detect_gpu():
    """시스템의 GPU를 자동 감지한다.

    Returns:
        dict: GPU 정보. keys: type, name, vram_gb, cuda_version, driver_version
              type은 'cuda', 'mps', 'cpu' 중 하나
    """
    print_header("2단계: GPU 자동 감지")

    gpu_info = {"type": "cpu", "name": None, "vram_gb": 0,
                "cuda_version": None, "driver_version": None}

    # NVIDIA GPU 감지 (nvidia-smi)
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0 and result.stdout.strip():
                line = result.stdout.strip().split("\n")[0]
                parts = [p.strip() for p in line.split(",")]
                gpu_info["type"] = "cuda"
                gpu_info["name"] = parts[0]
                gpu_info["vram_gb"] = round(float(parts[1]) / 1024, 1)
                gpu_info["driver_version"] = parts[2]

                # CUDA 버전 감지
                cuda_result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=compute_cap",
                     "--format=csv,noheader"],
                    capture_output=True, text=True, timeout=10
                )

                # nvidia-smi 헤더에서 CUDA Version 파싱
                smi_full = subprocess.run(
                    ["nvidia-smi"], capture_output=True, text=True, timeout=10
                )
                for smi_line in smi_full.stdout.split("\n"):
                    if "CUDA Version" in smi_line:
                        # "CUDA Version: 12.4" 같은 패턴
                        cuda_ver = smi_line.split("CUDA Version:")[1].strip().split()[0]
                        gpu_info["cuda_version"] = cuda_ver
                        break

                print_ok(f"NVIDIA GPU 감지: {gpu_info['name']}")
                print_info(f"VRAM: {gpu_info['vram_gb']} GB")
                print_info(f"드라이버: {gpu_info['driver_version']}")
                print_info(f"CUDA 지원 버전: {gpu_info['cuda_version']}")
                return gpu_info
        except (subprocess.TimeoutExpired, FileNotFoundError, IndexError, ValueError):
            pass

    # Apple MPS 감지
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        gpu_info["type"] = "mps"
        gpu_info["name"] = "Apple Silicon"
        print_ok("Apple Silicon GPU 감지 (MPS 가속 지원)")
        return gpu_info

    # CPU 폴백
    print_warn("GPU를 감지하지 못했습니다. CPU 모드로 설치합니다.")
    print_info("나중에 GPU를 사용하려면 이 스크립트를 다시 실행하세요.")
    return gpu_info


def determine_pytorch_install(gpu_info, force_cpu=False, manual_cuda=None):
    """GPU 정보를 바탕으로 PyTorch 설치 명령을 결정한다.

    Args:
        gpu_info: detect_gpu()의 반환값
        force_cpu: True이면 CPU 버전 강제 설치
        manual_cuda: 수동 지정 CUDA 버전 (예: "12.1")

    Returns:
        tuple: (pip_args, description)
    """
    if force_cpu:
        return (["torch", "torchvision", "torchaudio"], "CPU 버전")

    if gpu_info["type"] == "cuda":
        cuda_ver = manual_cuda or gpu_info["cuda_version"]
        if cuda_ver:
            # 주 버전.부 버전에서 매핑 찾기
            major_minor = ".".join(cuda_ver.split(".")[:2])
            cuda_tag = PYTORCH_CUDA_MAP.get(major_minor)
            if cuda_tag:
                index_url = f"https://download.pytorch.org/whl/{cuda_tag}"
                return (
                    ["torch", "torchvision", "torchaudio",
                     "--index-url", index_url],
                    f"CUDA {major_minor} ({cuda_tag})"
                )
        # CUDA 버전을 특정하지 못한 경우, 최신 안정 CUDA 사용
        return (
            ["torch", "torchvision", "torchaudio",
             "--index-url", "https://download.pytorch.org/whl/cu124"],
            "CUDA 12.4 (기본값)"
        )

    if gpu_info["type"] == "mps":
        # macOS는 기본 pip install로 MPS 지원
        return (["torch", "torchvision", "torchaudio"], "MPS (Apple Silicon)")

    return (["torch", "torchvision", "torchaudio"], "CPU 버전")


# ── 3단계: 가상환경 생성 ──────────────────────────────────────────────
def create_venv():
    """Python 가상환경을 생성한다."""
    print_header("3단계: 가상환경 생성")

    # 이미 가상환경 안에 있는지 확인
    in_venv = (
        hasattr(sys, "real_prefix")
        or (hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix)
    )

    if in_venv:
        print_ok("이미 가상환경 안에 있습니다.")
        print_info(f"경로: {sys.prefix}")
        return sys.executable

    if VENV_DIR.exists():
        print_ok(f"기존 가상환경 발견: {VENV_DIR}")
    else:
        print_info(f"가상환경 생성 중: {VENV_DIR}")
        subprocess.run([sys.executable, "-m", "venv", str(VENV_DIR)], check=True)
        print_ok("가상환경 생성 완료")

    # 가상환경의 Python 경로 반환
    if platform.system() == "Windows":
        venv_python = VENV_DIR / "Scripts" / "python.exe"
    else:
        venv_python = VENV_DIR / "bin" / "python"

    if not venv_python.exists():
        print_fail(f"가상환경 Python을 찾을 수 없습니다: {venv_python}")
        sys.exit(1)

    print_ok(f"가상환경 Python: {venv_python}")

    # pip 업그레이드
    print_info("pip 업그레이드 중...")
    subprocess.run(
        [str(venv_python), "-m", "pip", "install", "--upgrade", "pip"],
        capture_output=True
    )
    print_ok("pip 업그레이드 완료")

    return str(venv_python)


# ── 4단계: PyTorch 설치 ───────────────────────────────────────────────
def install_pytorch(venv_python, gpu_info, force_cpu=False, manual_cuda=None):
    """GPU에 맞는 PyTorch를 설치한다."""
    print_header("4단계: PyTorch 설치")

    pip_args, description = determine_pytorch_install(
        gpu_info, force_cpu, manual_cuda
    )

    print_info(f"설치 버전: PyTorch ({description})")
    print_info(f"명령: pip install {' '.join(pip_args)}")
    print_info("설치 중... (네트워크 속도에 따라 수 분 소요)")

    result = subprocess.run(
        [venv_python, "-m", "pip", "install"] + pip_args,
        capture_output=True, text=True
    )

    if result.returncode != 0:
        print_fail("PyTorch 설치 실패!")
        print(result.stderr[-500:] if result.stderr else "")
        sys.exit(1)

    print_ok(f"PyTorch ({description}) 설치 완료")


# ── 5단계: 전체 패키지 설치 ───────────────────────────────────────────
def install_requirements(venv_python):
    """requirements.txt의 모든 패키지를 설치한다."""
    print_header("5단계: 전체 실습 패키지 설치")

    if not REQUIREMENTS.exists():
        print_fail(f"requirements.txt를 찾을 수 없습니다: {REQUIREMENTS}")
        sys.exit(1)

    print_info(f"패키지 목록: {REQUIREMENTS}")
    print_info("설치 중... (첫 실행 시 약 5-10분 소요)")

    result = subprocess.run(
        [venv_python, "-m", "pip", "install", "-r", str(REQUIREMENTS)],
        capture_output=True, text=True
    )

    if result.returncode != 0:
        print_warn("일부 패키지 설치 실패 (선택 패키지일 수 있음)")
        # 실패 패키지 출력
        for line in result.stderr.split("\n"):
            if "ERROR" in line:
                print_fail(f"  {line.strip()}")
    else:
        print_ok("전체 패키지 설치 완료")


# ── 6단계: 설치 검증 ─────────────────────────────────────────────────
def verify_installation(venv_python, gpu_info):
    """설치된 패키지와 GPU 설정을 검증한다."""
    print_header("6단계: 설치 검증")

    verify_script = '''
import sys
import json

results = {"packages": {}, "gpu": {}}

# 핵심 패키지 확인
packages = [
    ("torch", "PyTorch"),
    ("numpy", "NumPy"),
    ("pandas", "Pandas"),
    ("matplotlib", "Matplotlib"),
    ("sklearn", "scikit-learn"),
    ("transformers", "Transformers"),
    ("datasets", "Datasets"),
    ("peft", "PEFT"),
    ("openai", "OpenAI"),
    ("anthropic", "Anthropic"),
    ("langchain", "LangChain"),
    ("fastapi", "FastAPI"),
    ("gradio", "Gradio"),
]

for module, name in packages:
    try:
        mod = __import__(module)
        ver = getattr(mod, "__version__", "OK")
        results["packages"][name] = {"status": "ok", "version": ver}
    except ImportError:
        results["packages"][name] = {"status": "fail", "version": None}

# GPU 확인
import torch
results["gpu"]["cuda_available"] = torch.cuda.is_available()
results["gpu"]["mps_available"] = (
    hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
)

if torch.cuda.is_available():
    results["gpu"]["device"] = "cuda"
    results["gpu"]["device_name"] = torch.cuda.get_device_name(0)
    results["gpu"]["cuda_version"] = torch.version.cuda
    props = torch.cuda.get_device_properties(0)
    results["gpu"]["vram_gb"] = round(props.total_memory / (1024**3), 1)
elif results["gpu"]["mps_available"]:
    results["gpu"]["device"] = "mps"
    results["gpu"]["device_name"] = "Apple Silicon"
else:
    results["gpu"]["device"] = "cpu"
    results["gpu"]["device_name"] = "CPU"

print(json.dumps(results))
'''

    result = subprocess.run(
        [venv_python, "-c", verify_script],
        capture_output=True, text=True
    )

    if result.returncode != 0:
        print_fail("검증 스크립트 실행 실패")
        print(result.stderr[-500:] if result.stderr else "")
        return

    try:
        results = json.loads(result.stdout.strip())
    except json.JSONDecodeError:
        print_fail("검증 결과 파싱 실패")
        return

    # 패키지 결과 출력
    print_info("패키지 설치 상태:")
    for name, info in results["packages"].items():
        if info["status"] == "ok":
            print_ok(f"{name}: {info['version']}")
        else:
            print_warn(f"{name}: 미설치")

    # GPU 결과 출력
    print()
    gpu = results["gpu"]
    print_info("GPU 설정:")
    if gpu["device"] == "cuda":
        print_ok(f"CUDA 사용 가능: {gpu['device_name']}")
        print_info(f"CUDA 버전: {gpu['cuda_version']}")
        print_info(f"VRAM: {gpu['vram_gb']} GB")
    elif gpu["device"] == "mps":
        print_ok("Apple MPS 사용 가능")
    else:
        print_warn("GPU 없음 — CPU 모드")

    return results


# ── 7단계: GPU 벤치마크 ───────────────────────────────────────────────
def run_benchmark(venv_python):
    """CPU vs GPU 행렬 연산 속도를 비교한다."""
    print_header("7단계: GPU 벤치마크 (CPU vs GPU)")

    benchmark_script = '''
import torch
import time
import json

results = {}
size = 4096  # 행렬 크기

# CPU 벤치마크
a_cpu = torch.randn(size, size)
b_cpu = torch.randn(size, size)

# 워밍업
_ = torch.matmul(a_cpu, b_cpu)

start = time.perf_counter()
for _ in range(3):
    _ = torch.matmul(a_cpu, b_cpu)
cpu_time = (time.perf_counter() - start) / 3
results["cpu_time"] = round(cpu_time, 4)

# GPU 벤치마크
if torch.cuda.is_available():
    device = torch.device("cuda")
    device_name = torch.cuda.get_device_name(0)
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
    device_name = "Apple Silicon (MPS)"
else:
    device = None
    device_name = None

if device:
    a_gpu = a_cpu.to(device)
    b_gpu = b_cpu.to(device)

    # 워밍업
    _ = torch.matmul(a_gpu, b_gpu)
    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(3):
        _ = torch.matmul(a_gpu, b_gpu)
        if device.type == "cuda":
            torch.cuda.synchronize()
    gpu_time = (time.perf_counter() - start) / 3

    results["gpu_time"] = round(gpu_time, 4)
    results["device_name"] = device_name
    results["speedup"] = round(cpu_time / gpu_time, 1)
else:
    results["gpu_time"] = None
    results["device_name"] = None
    results["speedup"] = None

results["matrix_size"] = size
print(json.dumps(results))
'''

    print_info("4096×4096 행렬 곱 속도 비교 중...")

    result = subprocess.run(
        [venv_python, "-c", benchmark_script],
        capture_output=True, text=True, timeout=120
    )

    if result.returncode != 0:
        print_warn("벤치마크 실행 실패 (무시 가능)")
        return

    try:
        bench = json.loads(result.stdout.strip())
    except json.JSONDecodeError:
        print_warn("벤치마크 결과 파싱 실패")
        return

    print_ok(f"행렬 크기: {bench['matrix_size']}×{bench['matrix_size']}")
    print_info(f"CPU 소요 시간: {bench['cpu_time']:.4f}초")

    if bench["gpu_time"] is not None:
        print_info(f"GPU 소요 시간: {bench['gpu_time']:.4f}초 ({bench['device_name']})")
        print_ok(f"GPU 가속: {bench['speedup']}배 빠름")
    else:
        print_warn("GPU 없음 — CPU 단독 결과")


# ── 결과 요약 ─────────────────────────────────────────────────────────
def print_summary(gpu_info, venv_python):
    """설치 결과를 요약 출력한다."""
    print_header("설치 완료!")

    print_ok("15주차 전체 실습 환경이 준비되었습니다.")
    print()

    # 활성화 명령 안내
    if platform.system() == "Windows":
        activate_cmd = r"venv\Scripts\activate"
    else:
        activate_cmd = "source venv/bin/activate"

    # 현재 가상환경 안에 있는지 확인
    in_venv = (
        hasattr(sys, "real_prefix")
        or (hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix)
    )

    if not in_venv:
        print_info("가상환경 활성화:")
        print(f"    {activate_cmd}")
        print()

    print_info("환경 확인:")
    print("    python practice/chapter1/code/1-2-환경설정.py")
    print()
    print_info("실습 시작:")
    print("    python practice/chapter1/code/1-3-텐서기초.py")
    print()

    if gpu_info["type"] == "cuda":
        print_info(f"GPU: {gpu_info['name']} ({gpu_info['vram_gb']} GB)")
    elif gpu_info["type"] == "mps":
        print_info("GPU: Apple Silicon (MPS)")
    else:
        print_info("GPU: 없음 (CPU 모드)")
        print_info("파인튜닝 실습(9-10주차)은 Google Colab GPU를 사용하세요.")


# ── 메인 ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="딥러닝 자연어처리 — 실습 환경 자동 설정"
    )
    parser.add_argument(
        "--cpu", action="store_true",
        help="GPU 무시하고 CPU 버전 PyTorch 설치"
    )
    parser.add_argument(
        "--cuda", type=str, default=None,
        help="CUDA 버전 수동 지정 (예: 12.1, 11.8)"
    )
    parser.add_argument(
        "--skip-venv", action="store_true",
        help="가상환경 생성 건너뛰기 (이미 가상환경 안에 있을 때)"
    )
    args = parser.parse_args()

    Colors.init()

    print()
    print(f"{Colors.BOLD}╔{'═' * 58}╗{Colors.END}")
    print(f"{Colors.BOLD}║  딥러닝 자연어처리 — 실습 환경 자동 설정 스크립트       ║{Colors.END}")
    print(f"{Colors.BOLD}║  LLM 시대의 NLP 엔지니어링: 원리부터 배포까지          ║{Colors.END}")
    print(f"{Colors.BOLD}╚{'═' * 58}╝{Colors.END}")

    # 1단계: 시스템 확인
    sys_info = check_system()

    # 2단계: GPU 감지
    if args.cpu:
        gpu_info = {"type": "cpu", "name": None, "vram_gb": 0,
                    "cuda_version": None, "driver_version": None}
        print_header("2단계: GPU 감지 (건너뜀 — CPU 모드)")
        print_info("--cpu 옵션으로 CPU 버전을 설치합니다.")
    else:
        gpu_info = detect_gpu()

    # CUDA 수동 지정
    if args.cuda:
        gpu_info["cuda_version"] = args.cuda
        print_info(f"CUDA 버전 수동 지정: {args.cuda}")

    # 3단계: 가상환경
    if args.skip_venv:
        venv_python = sys.executable
        print_header("3단계: 가상환경 (건너뜀)")
    else:
        venv_python = create_venv()

    # 4단계: PyTorch
    install_pytorch(venv_python, gpu_info, args.cpu, args.cuda)

    # 5단계: 전체 패키지
    install_requirements(venv_python)

    # 6단계: 검증
    verify_installation(venv_python, gpu_info)

    # 7단계: 벤치마크
    run_benchmark(venv_python)

    # 결과 요약
    print_summary(gpu_info, venv_python)


if __name__ == "__main__":
    main()
