"""
10-1-qlora파인튜닝.py
제10장 실습: PEFT와 QLoRA 파인튜닝

이 스크립트는 환경 자동 구성부터 QLoRA 파인튜닝까지 전체 파이프라인을 실행한다:

  Phase 1 — 자동 환경 구성 (처음 1회만 필요)
    1. 하드웨어 감지  : NVIDIA GPU / Intel NPU / AMD GPU 자동 탐지
    2. CUDA 자동 설치  : 드라이버 버전에 맞는 PyTorch+CUDA 자동 pip 설치
    3. 라이브러리 자동 설치 : peft, bitsandbytes, accelerate, datasets 자동 설치

  Phase 2 — QLoRA 실습
    4. 양자화 수학 시연  (GPU 불필요)
    5. OPT-1.3B 4-bit 모델 로딩 + 메모리 측정
    6. LoRA 설정 및 파인튜닝 (20 step)
    7. Full FT / LoRA / QLoRA 메모리 비교 차트
    8. 파인튜닝 모델 추론 테스트

대상 환경: Windows 11, NVIDIA GPU (RTX 4090 권장, 8GB+ VRAM)
모델: facebook/opt-1.3b (fp32 ~2.6GB, 4-bit ~800MB)

실행 방법:
    cd practice/chapter10
    python -m venv venv
    venv\\Scripts\\activate          # Windows
    source venv/bin/activate        # macOS / Linux
    pip install torch               # PyTorch 먼저 설치 (스크립트가 CUDA 버전 자동 재설치)
    python code/10-1-qlora파인튜닝.py
"""

import importlib
import os
import platform
import re
import subprocess
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import torch

torch.manual_seed(42)

# ── 경로 설정 ──────────────────────────────────────────
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── 기본 모델 (교실 실습용 — 소형 모델) ────────────────
DEFAULT_MODEL = "facebook/opt-1.3b"


# ──────────────────────────────────────────────────────
# 유틸리티 함수
# ──────────────────────────────────────────────────────
def _run_cmd(cmd, timeout=30):
    """외부 명령을 실행하고 (returncode, stdout) 를 반환한다."""
    try:
        r = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
        )
        return r.returncode, r.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return -1, ""


def _pip_install(*packages, index_url=None, timeout=600):
    """pip install 을 실행한다. 성공 여부를 반환한다."""
    cmd = [sys.executable, "-m", "pip", "install", "--quiet"]
    if index_url:
        cmd += ["--index-url", index_url]
    cmd += list(packages)
    print(f"    실행: {' '.join(cmd[-len(packages):])}")
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0 and r.stderr:
        # 핵심 에러 메시지만 출력
        for line in r.stderr.strip().split("\n")[-3:]:
            print(f"    ! {line}")
    return r.returncode == 0


def _pip_uninstall(*packages):
    """pip uninstall 을 실행한다."""
    cmd = [sys.executable, "-m", "pip", "uninstall", "-y", "--quiet"]
    cmd += list(packages)
    subprocess.run(cmd, capture_output=True, text=True, timeout=120)


def _get_nvidia_driver_version():
    """nvidia-smi 에서 드라이버 버전 문자열을 반환한다. 실패 시 None."""
    code, out = _run_cmd([
        "nvidia-smi", "--query-gpu=driver_version",
        "--format=csv,noheader",
    ])
    if code == 0 and out:
        # 여러 GPU 가 있으면 첫 줄만 사용
        return out.split("\n")[0].strip()
    return None


def _driver_to_cuda_tag(driver_ver):
    """NVIDIA 드라이버 버전 → 최적 PyTorch CUDA index URL 태그를 반환한다.

    드라이버-CUDA 호환 테이블 (NVIDIA 공식 기준):
      Driver >= 550  →  CUDA 12.4  →  cu124
      Driver >= 530  →  CUDA 12.1  →  cu121
      Driver >= 522  →  CUDA 11.8  →  cu118
      Driver <  522  →  드라이버 업데이트 필요
    """
    try:
        major = int(driver_ver.split(".")[0])
    except (ValueError, IndexError):
        return None

    if major >= 550:
        return "cu124"
    if major >= 530:
        return "cu121"
    if major >= 522:
        return "cu118"
    return None  # 드라이버가 너무 오래됨


def _detect_npu_windows():
    """Windows 에서 NPU (Intel AI Boost 등) 를 감지한다.
    반환: 감지된 NPU 이름 리스트.
    """
    if platform.system() != "Windows":
        return []

    # PowerShell 로 PnP 장치에서 NPU / Neural 키워드 검색
    ps_cmd = (
        "Get-PnpDevice -PresentOnly -ErrorAction SilentlyContinue "
        "| Where-Object { $_.FriendlyName -match 'NPU|Neural|AI Boost' } "
        "| Select-Object -ExpandProperty FriendlyName"
    )
    code, out = _run_cmd(["powershell", "-Command", ps_cmd], timeout=15)
    if code == 0 and out:
        return [line.strip() for line in out.split("\n") if line.strip()]
    return []


def _detect_npu_linux():
    """Linux 에서 NPU 장치를 감지한다."""
    code, out = _run_cmd(["lspci"], timeout=10)
    if code != 0:
        return []
    npus = []
    for line in out.split("\n"):
        lower = line.lower()
        if "npu" in lower or "neural" in lower or "ai accelerator" in lower:
            npus.append(line.strip())
    return npus


# ──────────────────────────────────────────────────────
# 1. 하드웨어 자동 감지
# ──────────────────────────────────────────────────────
def detect_hardware():
    """NVIDIA GPU, NPU, 기타 가속기를 자동 감지하고 결과를 반환한다."""
    print("=" * 60)
    print("[1] 하드웨어 자동 감지")
    print("=" * 60)

    info = {
        "os": platform.system(),
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "nvidia_driver": None,
        "nvidia_gpus": [],
        "npus": [],
        "cuda_available": torch.cuda.is_available(),
        "pytorch_cuda": torch.version.cuda,
    }

    # ── OS / Python ──
    print(f"\n  OS:     {info['os']} {platform.release()}")
    print(f"  Python: {info['python']}  ({sys.executable})")
    v = sys.version_info
    if not (v.major == 3 and v.minor >= 10):
        print("          [!] Python 3.10+ 권장")

    # ── NVIDIA GPU 감지 (nvidia-smi) ──
    print(f"\n  --- NVIDIA GPU 탐지 ---")
    code, out = _run_cmd([
        "nvidia-smi",
        "--query-gpu=name,driver_version,memory.total",
        "--format=csv,noheader,nounits",
    ])
    if code == 0 and out:
        for line in out.split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                gpu_name, drv, mem_mb = parts[0], parts[1], parts[2]
                mem_gb = int(mem_mb) / 1024
                info["nvidia_gpus"].append({
                    "name": gpu_name,
                    "driver": drv,
                    "vram_gb": round(mem_gb, 1),
                })
                info["nvidia_driver"] = drv
                print(f"  [GPU] {gpu_name}")
                print(f"        드라이버: {drv}  |  VRAM: {mem_gb:.1f} GB")
    else:
        print("  NVIDIA GPU 미감지 (nvidia-smi 실행 불가)")

    # ── NPU 감지 ──
    print(f"\n  --- NPU 탐지 ---")
    if info["os"] == "Windows":
        npus = _detect_npu_windows()
    elif info["os"] == "Linux":
        npus = _detect_npu_linux()
    else:
        npus = []

    if npus:
        info["npus"] = npus
        for npu_name in npus:
            print(f"  [NPU] {npu_name}")
        print("        ※ NPU는 현재 bitsandbytes/QLoRA를 지원하지 않습니다.")
        print("          NVIDIA CUDA GPU가 필요합니다.")
    else:
        print("  NPU 미감지")

    # ── PyTorch / CUDA 현황 ──
    print(f"\n  --- PyTorch 현황 ---")
    print(f"  PyTorch:     {torch.__version__}")
    print(f"  CUDA 빌드:   {info['pytorch_cuda'] or '없음 (CPU 전용)'}")
    print(f"  CUDA 사용:   {'가능' if info['cuda_available'] else '불가'}")

    if info["cuda_available"]:
        for i in range(torch.cuda.device_count()):
            prop = torch.cuda.get_device_properties(i)
            vram = prop.total_memory / (1024 ** 3)
            print(f"  torch GPU {i}: {prop.name}  "
                  f"({vram:.1f}GB, SM={prop.multi_processor_count})")

    # ── 요약 ──
    print(f"\n  --- 요약 ---")
    if info["nvidia_gpus"]:
        gpu0 = info["nvidia_gpus"][0]
        cuda_tag = _driver_to_cuda_tag(gpu0["driver"])
        if cuda_tag:
            print(f"  NVIDIA GPU 감지됨: {gpu0['name']} ({gpu0['vram_gb']}GB)")
            print(f"  드라이버 {gpu0['driver']} -> 호환 CUDA: {cuda_tag}")
        else:
            print(f"  NVIDIA GPU 감지됨: {gpu0['name']}")
            print(f"  [!] 드라이버 {gpu0['driver']}이 너무 오래되었습니다.")
            print(f"      https://www.nvidia.com/drivers 에서 업데이트하세요.")
    elif info["npus"]:
        print("  NPU만 감지됨. QLoRA 실습에는 NVIDIA GPU가 필요합니다.")
    else:
        print("  가속기 미감지. CPU 모드로 제한 실행됩니다.")

    print()
    return info


# ──────────────────────────────────────────────────────
# 2. CUDA + PyTorch 자동 설치
# ──────────────────────────────────────────────────────
def setup_cuda_and_pytorch(hw_info):
    """드라이버 버전에 맞는 PyTorch+CUDA 를 자동으로 설치한다.

    이미 CUDA 가 정상 작동 중이면 건너뛴다.
    반환: True (CUDA 사용 가능), False (불가).
    """
    print("=" * 60)
    print("[2] CUDA + PyTorch 자동 설치")
    print("=" * 60)

    # ── 이미 CUDA 사용 가능한 경우 ──
    if torch.cuda.is_available():
        cuda_ver = torch.version.cuda
        print(f"\n  PyTorch CUDA {cuda_ver} 이미 활성 상태 — 설치 불필요")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print()
        return True

    # ── NVIDIA GPU 가 없는 경우 → CPU fallback ──
    if not hw_info.get("nvidia_gpus"):
        print("\n  NVIDIA GPU 미감지 — CUDA 설치 불가")
        if hw_info.get("npus"):
            print("  NPU가 감지되었으나 QLoRA는 현재 NVIDIA CUDA만 지원합니다.")
        print()
        print("  -> CPU fallback 모드로 전환합니다.")
        print("     - 섹션 4 (양자화 수학): 정상 실행")
        print("     - 섹션 5 (모델 로딩):   CPU fp32 로딩 (4-bit 양자화 불가)")
        print("     - 섹션 6 (파인튜닝):    CPU 모드 학습 (느리지만 동작)")
        print("     - 섹션 7 (메모리 비교): 정상 실행 (이론값 계산)")
        print("     - 섹션 8 (추론):        CPU 모드 추론")
        print()

        if not hw_info.get("npus"):
            print("  [더 나은 실습을 위한 권장사항]")
            print("  1. NVIDIA GPU가 장착된 PC 사용 (RTX 4060 이상 권장)")
            print("  2. Google Colab (무료 T4 GPU):")
            print("     https://colab.research.google.com")
            print()

        return False

    # ── NVIDIA GPU 는 있으나 PyTorch CUDA 미활성 ──
    driver_ver = hw_info["nvidia_driver"]
    cuda_tag = _driver_to_cuda_tag(driver_ver)

    if cuda_tag is None:
        print(f"\n  [!] 드라이버 {driver_ver} 이 CUDA 11.8 미만과 대응됩니다.")
        print("  드라이버를 먼저 업데이트하세요:")
        print("    https://www.nvidia.com/drivers")
        print("  업데이트 후 이 스크립트를 다시 실행하세요.")
        print()
        return False

    index_url = f"https://download.pytorch.org/whl/{cuda_tag}"
    print(f"\n  NVIDIA GPU 감지: {hw_info['nvidia_gpus'][0]['name']}")
    print(f"  드라이버: {driver_ver} -> 호환 CUDA: {cuda_tag}")
    print(f"\n  PyTorch 를 CUDA {cuda_tag} 버전으로 자동 재설치합니다.")

    # 기존 PyTorch 제거
    print("\n  [단계 1/3] 기존 PyTorch 제거...")
    _pip_uninstall("torch", "torchvision", "torchaudio")

    # CUDA 버전 PyTorch 설치
    print(f"  [단계 2/3] PyTorch + CUDA ({cuda_tag}) 설치 중... (수 분 소요)")
    ok = _pip_install(
        "torch", "torchvision", "torchaudio",
        index_url=index_url,
        timeout=900,
    )
    if not ok:
        print("\n  [!] PyTorch CUDA 설치 실패.")
        print("  수동 설치를 시도하세요:")
        print(f"    pip install torch torchvision torchaudio "
              f"--index-url {index_url}")
        print()
        return False

    # 설치 검증: torch 재임포트
    print("  [단계 3/3] 설치 검증...")
    # 현재 프로세스의 torch 모듈은 이미 로드되어 있으므로
    # 새 프로세스에서 검증한다
    verify_code = (
        "import torch; "
        "print(f'cuda={torch.cuda.is_available()},"
        "ver={torch.version.cuda},"
        "gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"
    )
    code, out = _run_cmd(
        [sys.executable, "-c", verify_code], timeout=30,
    )

    if code == 0 and "cuda=True" in out:
        print(f"\n  PyTorch CUDA 설치 성공!")
        print(f"  검증 결과: {out}")
        print("\n  ※ 현재 프로세스는 이전 PyTorch를 사용 중입니다.")
        print("    스크립트를 재실행하면 CUDA가 활성화됩니다.")
        print()
        return "RESTART_NEEDED"
    else:
        print(f"\n  [!] CUDA 활성화 실패 (검증 출력: {out})")
        print("  NVIDIA 드라이버가 정상 설치되었는지 확인하세요.")
        print()
        return False


# ──────────────────────────────────────────────────────
# 3. 필수 라이브러리 자동 설치
# ──────────────────────────────────────────────────────
def install_required_libraries(cuda_available=True):
    """QLoRA 실습에 필요한 라이브러리를 점검하고, 미설치 시 자동 설치한다.

    CUDA 미사용 환경에서는 bitsandbytes 를 선택사항으로 처리하여
    CPU 모드로 실습을 계속할 수 있게 한다.

    반환: True (모두 사용 가능), False (필수 실패), "RESTART_NEEDED" (재실행 필요).
    """
    print("=" * 60)
    print("[3] 필수 라이브러리 자동 설치")
    print("=" * 60)

    # (패키지명, 최소버전, 설명, CUDA 전용 여부)
    required = [
        ("transformers", "4.36.0", "Hugging Face Transformers",         False),
        ("peft",         "0.7.0",  "Parameter-Efficient Fine-Tuning",   False),
        ("bitsandbytes", "0.43.0", "4-bit/8-bit 양자화 (CUDA 전용)",    True),
        ("accelerate",   "0.24.0", "분산학습 / 메모리 최적화",          False),
        ("datasets",     "2.14.0", "Hugging Face Datasets",             False),
        ("numpy",        "1.24.0", "수치 계산",                         False),
        ("matplotlib",   "3.7.0",  "시각화",                            False),
    ]

    installed = []
    missing = []

    print()
    for pkg, min_ver, desc, cuda_only in required:
        try:
            mod = importlib.import_module(pkg)
            ver = getattr(mod, "__version__", "?")
            print(f"  [OK] {pkg:<18} {ver:<12} ({desc})")
            installed.append(pkg)
        except ImportError:
            tag = "  [--]" if (not cuda_only or cuda_available) else "  [  ]"
            suffix = " (CUDA 없으므로 건너뜀)" if (cuda_only and not cuda_available) else ""
            print(f"{tag} {pkg:<18} 미설치        ({desc}){suffix}")
            if cuda_only and not cuda_available:
                continue  # CUDA 없으면 bitsandbytes 설치 시도하지 않음
            missing.append((pkg, min_ver, desc))

    if not missing:
        print("\n  모든 라이브러리 정상 설치됨")
        if not cuda_available:
            print("  (bitsandbytes 미설치 — CPU 모드에서는 불필요)")
        print()
        return True

    # ── 미설치 라이브러리 자동 설치 ──
    print(f"\n  미설치 {len(missing)}개 라이브러리를 자동 설치합니다...")
    any_installed = False

    for pkg, min_ver, desc in missing:
        print(f"\n  설치 중: {pkg}>={min_ver} ({desc})")
        ok = _pip_install(f"{pkg}>={min_ver}")
        if ok:
            print(f"    -> 설치 완료")
            any_installed = True
        else:
            # bitsandbytes Windows 특수 케이스
            if pkg == "bitsandbytes" and platform.system() == "Windows":
                print("    bitsandbytes Windows 설치 재시도 (--prefer-binary)...")
                ok = _pip_install(
                    f"{pkg}>={min_ver}", "--prefer-binary",
                )
                if ok:
                    print("    -> 재시도 설치 완료")
                    any_installed = True
                else:
                    print(f"    [!] {pkg} 설치 실패.")
                    print("    수동 설치를 시도하세요:")
                    print(f"      pip install {pkg}>={min_ver}")
            else:
                print(f"    [!] {pkg} 설치 실패.")

    # ── 설치 후 검증 ──
    print(f"\n  --- 설치 후 검증 ---")
    still_missing = []
    for pkg, min_ver, desc in missing:
        verify_code = f"import {pkg}; print({pkg}.__version__)"
        code, out = _run_cmd([sys.executable, "-c", verify_code], timeout=15)
        if code == 0:
            print(f"  [OK] {pkg} {out}")
        else:
            print(f"  [!!] {pkg} 여전히 import 불가")
            still_missing.append(pkg)

    if still_missing:
        # bitsandbytes만 실패한 경우 CUDA 없으면 허용
        non_bnb = [p for p in still_missing if p != "bitsandbytes"]
        if non_bnb:
            print(f"\n  [!] 설치 실패 라이브러리: {', '.join(still_missing)}")
            print("  수동 설치 후 재실행하세요:")
            print(f"    pip install {' '.join(still_missing)}")
            print()
            return False
        else:
            print("\n  bitsandbytes 설치 실패 — CPU 모드로 실습을 계속합니다.")
            print("  (4-bit 양자화 기능은 비활성)")

    if any_installed:
        print("\n  라이브러리가 새로 설치되었습니다.")
        print("  모듈 로딩을 위해 스크립트를 재실행하세요.")
        print()
        return "RESTART_NEEDED"

    print()
    return True


# ──────────────────────────────────────────────────────
# 4. 양자화 수학 시연 (GPU 불필요)
# ──────────────────────────────────────────────────────
def demonstrate_quantization_math():
    """양자화 수학 원리를 수치로 시연한다 (GPU 불필요)."""
    print("=" * 60)
    print("[4] 양자화 수학 시연")
    print("=" * 60)

    import numpy as np
    np.random.seed(42)

    # ── 4-1. int4 양자화·역양자화 ──
    print("\n[4-1] int4 양자화 원리")
    print("-" * 40)
    weights = np.array([0.5, 1.2, -0.8, 1.9, 0.1, -0.3, 0.77, -1.1],
                       dtype=np.float32)
    print(f"  원본 가중치 (float32): {weights}")

    min_val, max_val = weights.min(), weights.max()
    quantized = np.round(
        (weights - min_val) / (max_val - min_val) * 15
    ).astype(np.int32)
    print(f"  4-bit 양자화 (0~15):   {quantized}")

    recovered = (quantized / 15.0) * (max_val - min_val) + min_val
    error = np.abs(weights - recovered)
    print(f"  역양자화 (복원):       {np.round(recovered, 4)}")
    print(f"  평균 절대 오차 (MAE):  {error.mean():.4f}")

    # ── 4-2. NF4 vs 균등 분할 비교 ──
    print(f"\n[4-2] NF4 vs 균등 분할 비교")
    print("-" * 40)

    NF4_CODEBOOK = np.array([
        -1.0, -0.6962, -0.5251, -0.3949,
        -0.2844, -0.1848, -0.0911, 0.0,
        0.0796, 0.1609, 0.2461, 0.3379,
        0.4407, 0.5626, 0.7230, 1.0,
    ], dtype=np.float32)

    sample = np.random.randn(1000).astype(np.float32)
    sample = sample / np.abs(sample).max()

    nf4_indices = np.array(
        [np.argmin(np.abs(NF4_CODEBOOK - w)) for w in sample]
    )
    nf4_recovered = NF4_CODEBOOK[nf4_indices]
    nf4_mae = np.abs(sample - nf4_recovered).mean()

    uniform_q = np.round((sample + 1.0) / 2.0 * 15).clip(0, 15).astype(int)
    uniform_recovered = uniform_q / 15.0 * 2.0 - 1.0
    uniform_mae = np.abs(sample - uniform_recovered).mean()

    print(f"  NF4 양자화 MAE:    {nf4_mae:.5f}")
    print(f"  균등 분할 MAE:     {uniform_mae:.5f}")
    improvement = (uniform_mae - nf4_mae) / uniform_mae * 100
    print(f"  NF4 오차 감소율:   {improvement:.1f}%")
    print("  -> 정규분포를 따르는 가중치에서 NF4가 더 정밀")

    # ── 4-3. LoRA 파라미터 절감 계산 ──
    print(f"\n[4-3] LoRA 파라미터 절감 계산")
    print("-" * 40)
    configs = [
        ("OPT-1.3B FFN",         2048, 8192),
        ("Llama-7B Attention",   4096, 4096),
        ("Llama-70B Attention",  8192, 8192),
    ]
    ranks = [8, 16, 32]

    header = f"  {'층':<25} {'원본':>12}"
    for r in ranks:
        header += f"  {'r=' + str(r):>14}"
    print(header)
    print("  " + "-" * 75)

    for name, m, n in configs:
        orig = m * n
        row = f"  {name:<25} {orig:>12,}"
        for r in ranks:
            lora = r * (m + n)
            ratio = lora / orig * 100
            row += f"  {lora:>8,} ({ratio:4.1f}%)"
        print(row)

    # ── 4-4. 파인튜닝 방법별 메모리 비교 ──
    print(f"\n[4-4] 파인튜닝 방법별 메모리 비교 (OPT-1.3B)")
    print("-" * 40)
    param_count = 1.3e9
    lora_params = 16 * (2048 + 2048) * 2 * 24

    strategies = [
        ("Full Fine-tuning (fp32)", param_count * 4,
         param_count * 4 * 2, param_count * 4),
        ("LoRA (bf16, r=16)",       param_count * 2,
         lora_params * 4 * 2, lora_params * 4),
        ("QLoRA (4-bit NF4)",       param_count * 0.5,
         lora_params * 4 * 2, lora_params * 4),
    ]

    print(f"\n  {'방법':<24} {'모델':>8} {'옵티마이저':>10} "
          f"{'그래디언트':>10} {'합계':>8}")
    print("  " + "-" * 66)
    for method, model_mem, opt_mem, grad_mem in strategies:
        total = model_mem + opt_mem + grad_mem
        print(f"  {method:<24} {model_mem/1e9:>7.1f}G "
              f"{opt_mem/1e9:>9.1f}G "
              f"{grad_mem/1e9:>9.1f}G "
              f"{total/1e9:>7.1f}G")

    print()


# ──────────────────────────────────────────────────────
# 5. 4-bit 양자화 모델 로딩
# ──────────────────────────────────────────────────────
def load_model_with_qlora(model_name=DEFAULT_MODEL):
    """4-bit NF4 양자화로 모델을 로딩하고 메모리를 측정한다."""
    print("=" * 60)
    print("[5] 4-bit 양자화 모델 로딩")
    print("=" * 60)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    cuda_ok = torch.cuda.is_available()

    if cuda_ok:
        torch.cuda.empty_cache()
        mem_before = torch.cuda.memory_allocated(0) / (1024 ** 3)
        print(f"\n  GPU 메모리 (로딩 전): {mem_before:.2f} GB")
    else:
        mem_before = 0.0

    print(f"\n  모델: {model_name}")
    print("  (첫 실행 시 HuggingFace에서 다운로드 — 수 분 소요 가능)")

    try:
        if cuda_ok:
            from transformers import BitsAndBytesConfig

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            print("\n  양자화 설정:")
            print("    - NF4 양자화 사용")
            print("    - Double Quantization 활성화")
            print("    - 연산 정밀도: bfloat16")
            print("\n  모델 로딩 중...")

            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto",
            )
            print("  4-bit NF4 + Double Quantization 로딩 완료")

            mem_after = torch.cuda.memory_allocated(0) / (1024 ** 3)
            mem_diff = mem_after - mem_before
            total_mem = (
                torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            )
            print(f"\n  GPU 메모리 (로딩 후): {mem_after:.2f} GB")
            print(f"  모델 점유 메모리:     {mem_diff:.2f} GB")
            print(f"  남은 여유 메모리:     {total_mem - mem_after:.2f} GB")

            total_params = sum(p.numel() for p in model.parameters())
            fp32_est_gb = total_params * 4 / (1024 ** 3)
            print(f"\n  fp32 이론 크기:  {fp32_est_gb:.2f} GB")
            print(f"  4-bit 실제 크기: {mem_diff:.2f} GB")
            if mem_diff > 0:
                print(f"  압축률:          {fp32_est_gb / mem_diff:.1f}x")
        else:
            print("\n  CUDA 미사용 — CPU 모드로 로딩 (양자화 비활성)")
            print("  모델 로딩 중...")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
            )
            total_params = sum(p.numel() for p in model.parameters())
            print(f"\n  CPU 모드 로딩 완료")
            print(f"  총 파라미터: {total_params:,}")
            print("  ※ 양자화 실습은 CUDA GPU 환경에서 실행하세요")

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print(f"\n  토크나이저 로딩 완료 (어휘 크기: {tokenizer.vocab_size:,})")

        total = sum(p.numel() for p in model.parameters())
        trainable = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        print(f"\n  전체 파라미터:      {total:>15,}")
        print(f"  학습 가능 파라미터: {trainable:>15,}")
        print(f"  학습 비율:          {100 * trainable / total:.4f}%")

    except Exception as e:
        print(f"\n  모델 로딩 실패: {e}")
        print("  bitsandbytes 설치 상태와 CUDA 환경을 확인하세요")
        return None, None

    print()
    return model, tokenizer


# ──────────────────────────────────────────────────────
# 6. LoRA 설정 및 파인튜닝
# ──────────────────────────────────────────────────────
def configure_lora_and_train(model, tokenizer):
    """LoRA 설정을 적용하고 단기 파인튜닝을 실행한다."""
    print("=" * 60)
    print("[6] LoRA 설정 및 파인튜닝")
    print("=" * 60)

    if model is None:
        print("  모델이 없습니다. 섹션 5를 먼저 실행하세요.")
        return None, None

    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )
    from datasets import load_dataset

    # ── 6-1. kbit 학습 준비 ──
    if torch.cuda.is_available():
        model = prepare_model_for_kbit_training(model)
        print("\n  kbit 학습 준비 완료")
        print("  (LayerNorm float32 업캐스팅, 입력 임베딩 gradient 활성화)")

    # ── 6-2. LoRA 설정 ──
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    hidden_size = getattr(model.config, "hidden_size", "?")
    print(f"\n  [LoRA 하이퍼파라미터]")
    print(f"    r (rank):       {lora_config.r}")
    print(f"      -> dW = A*B,  A: {hidden_size}x{lora_config.r},  "
          f"B: {lora_config.r}x{hidden_size}")
    print(f"    lora_alpha:     {lora_config.lora_alpha}")
    print(f"      -> 스케일 = alpha/r = "
          f"{lora_config.lora_alpha}/{lora_config.r} = "
          f"{lora_config.lora_alpha / lora_config.r}")
    print(f"    target_modules: {lora_config.target_modules}")
    print(f"    lora_dropout:   {lora_config.lora_dropout}")

    model = get_peft_model(model, lora_config)
    print(f"\n  [학습 가능 파라미터]")
    model.print_trainable_parameters()

    # ── 6-3. Gradient Checkpointing ──
    if torch.cuda.is_available():
        mem_before_gc = torch.cuda.memory_allocated(0) / (1024 ** 3)
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.enable_input_require_grads()
        mem_after_gc = torch.cuda.memory_allocated(0) / (1024 ** 3)
        print(f"\n  Gradient Checkpointing 활성화")
        print(f"    메모리: {mem_before_gc:.2f} GB -> {mem_after_gc:.2f} GB")
        print("    (역전파 시 중간 활성화 재계산 -> 메모리 30~50% 절감)")

    # ── 6-4. 데이터셋 로딩 ──
    print(f"\n  데이터셋 로딩: wikitext-2 (5% 샘플)")
    dataset = load_dataset(
        "wikitext", "wikitext-2-raw-v1", split="train[:5%]"
    )
    dataset = dataset.filter(lambda x: len(x["text"].strip()) > 0)

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=256,
            padding="max_length",
        )

    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    print(f"  데이터 크기: {len(tokenized)} 샘플")
    print(f"  시퀀스 길이: 256 토큰 (교실 실습용 단축)")

    # ── 6-5. 학습 설정 ──
    output_path = OUTPUT_DIR / "lora_checkpoint"
    cuda_ok = torch.cuda.is_available()

    training_args = TrainingArguments(
        output_dir=str(output_path),
        max_steps=20,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_steps=5,
        logging_steps=5,
        save_steps=20,
        bf16=cuda_ok,
        fp16=False,
        gradient_checkpointing=True,
        seed=42,
        optim="paged_adamw_8bit" if cuda_ok else "adamw_torch",
        report_to="none",
    )

    print(f"\n  [학습 설정]")
    print(f"    최대 스텝:       {training_args.max_steps} (교실용 단축)")
    print(f"    배치 크기:       {training_args.per_device_train_batch_size}")
    print(f"    그래디언트 누적: {training_args.gradient_accumulation_steps} "
          f"(유효 배치 = "
          f"{training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps})")
    print(f"    학습률:          {training_args.learning_rate}")
    print(f"    옵티마이저:      {training_args.optim}")

    # ── 6-6. 학습 실행 ──
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    print(f"\n  학습 시작 (최대 {training_args.max_steps} 스텝)...")
    train_result = trainer.train()

    runtime = train_result.metrics.get("train_runtime", 0)
    samples_per_sec = train_result.metrics.get("train_samples_per_second", 0)
    print(f"\n  학습 완료")
    print(f"    총 학습 시간: {runtime:.1f}초")
    print(f"    초당 샘플:    {samples_per_sec:.1f}")

    if cuda_ok:
        final_mem = torch.cuda.memory_allocated(0) / (1024 ** 3)
        print(f"    최종 GPU 메모리: {final_mem:.2f} GB")

    adapter_path = OUTPUT_DIR / "lora_adapter"
    model.save_pretrained(str(adapter_path))

    adapter_size = sum(
        f.stat().st_size for f in Path(adapter_path).rglob("*") if f.is_file()
    )
    print(f"\n  LoRA 어댑터 저장: {adapter_path}")
    print(f"    어댑터 크기: {adapter_size / 1e6:.2f} MB (원본 모델 대비 극소)")

    print()
    return model, tokenizer


# ──────────────────────────────────────────────────────
# 7. 메모리 사용량 비교 차트
# ──────────────────────────────────────────────────────
def compare_memory_usage():
    """Full FT / LoRA / QLoRA 메모리를 이론값으로 비교하고 차트를 저장한다."""
    print("=" * 60)
    print("[7] 메모리 사용량 비교 (이론값)")
    print("=" * 60)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    system = platform.system()
    if system == "Windows":
        plt.rcParams["font.family"] = "Malgun Gothic"
    elif system == "Darwin":
        plt.rcParams["font.family"] = "AppleGothic"
    else:
        plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False

    models = {
        "OPT-1.3B":  1.3e9,
        "Llama-7B":  7e9,
        "Llama-70B": 70e9,
    }

    lora_r = 16
    lora_params = {
        "OPT-1.3B":  lora_r * (2048 + 2048) * 2 * 24,
        "Llama-7B":  lora_r * (4096 + 4096) * 2 * 32,
        "Llama-70B": lora_r * (8192 + 8192) * 2 * 80,
    }

    methods = ["Full FT (fp32)", "LoRA (bf16)", "QLoRA (4-bit)"]
    data = {m: [] for m in methods}

    for mname, params in models.items():
        lp = lora_params[mname]
        data["Full FT (fp32)"].append(params * 4 * 4 / (1024 ** 3))
        data["LoRA (bf16)"].append(
            (params * 2 + lp * 4 * 3) / (1024 ** 3)
        )
        data["QLoRA (4-bit)"].append(
            (params * 0.5 + lp * 4 * 3) / (1024 ** 3)
        )

    model_names = list(models.keys())
    print(f"\n  {'방법':<18}", end="")
    for mn in model_names:
        print(f" {mn:>12}", end="")
    print()
    print("  " + "-" * 56)

    for method in methods:
        print(f"  {method:<18}", end="")
        for val in data[method]:
            print(f" {val:>10.1f}GB", end="")
        print()

    print("  " + "-" * 56)
    print("  주: 이론 추정치. 실제 값은 배치 크기, 시퀀스 길이에 따라 다름")
    print(f"\n  참고 GPU VRAM:")
    print(f"    RTX 4090: 24 GB  |  RTX 3080: 10 GB  |  RTX 4060: 8 GB")

    fig, ax = plt.subplots(figsize=(10, 6))
    x = range(len(model_names))
    width = 0.25
    colors = ["#e74c3c", "#3498db", "#2ecc71"]

    for idx, method in enumerate(methods):
        offset = (idx - 1) * width
        bars = ax.bar(
            [xi + offset for xi in x],
            data[method],
            width=width,
            label=method,
            color=colors[idx],
            alpha=0.85,
        )
        for bar, val in zip(bars, data[method]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.05,
                f"{val:.0f}GB",
                ha="center", va="bottom", fontsize=8,
            )

    ax.axhline(y=24, color="gray", linestyle="--", linewidth=1.5,
               label="RTX 4090 (24GB)")
    ax.axhline(y=8, color="gray", linestyle=":", linewidth=1.5,
               label="RTX 4060 (8GB)")

    ax.set_xticks(list(x))
    ax.set_xticklabels(model_names)
    ax.set_ylabel("Memory (GB)")
    ax.set_title("Fine-tuning Memory: Full FT vs LoRA vs QLoRA")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.grid(True, alpha=0.3, axis="y")

    chart_path = OUTPUT_DIR / "memory_comparison.png"
    fig.tight_layout()
    fig.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  차트 저장: {chart_path}")

    print()


# ──────────────────────────────────────────────────────
# 8. 파인튜닝 모델 추론 테스트
# ──────────────────────────────────────────────────────
def inference_with_finetuned_model(model, tokenizer):
    """파인튜닝된 모델로 추론을 실행하고 결과를 저장한다."""
    print("=" * 60)
    print("[8] 파인튜닝 모델 추론 테스트")
    print("=" * 60)

    if model is None or tokenizer is None:
        print("  모델/토크나이저가 없습니다. 이전 섹션을 확인하세요.")
        return

    device = next(model.parameters()).device
    model.eval()

    prompts = [
        "The transformer architecture is",
        "Natural language processing allows",
        "Deep learning models can",
    ]

    results = []
    print()
    for i, prompt in enumerate(prompts, 1):
        print(f"  [프롬프트 {i}] \"{prompt}\"")
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            torch.manual_seed(42)
            output_ids = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        print(f"  생성 텍스트: {generated[:120]}")
        print()
        results.append({"prompt": prompt, "generated": generated})

    result_path = OUTPUT_DIR / "inference_results.txt"
    with open(result_path, "w", encoding="utf-8") as f:
        f.write("QLoRA 파인튜닝 모델 추론 결과\n")
        f.write("=" * 60 + "\n\n")
        for r in results:
            f.write(f"프롬프트: {r['prompt']}\n")
            f.write(f"생성:     {r['generated']}\n")
            f.write("-" * 40 + "\n")
    print(f"  추론 결과 저장: {result_path}")

    print()


# ──────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────
def main():
    print()
    print("=" * 60)
    print("  제10장 실습 — PEFT와 QLoRA 파인튜닝")
    print("  딥러닝 자연어처리 (2026)")
    print("=" * 60)
    print()

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Phase 1: 자동 환경 구성
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print("=" * 60)
    print("  Phase 1: 자동 환경 구성")
    print("=" * 60)
    print()

    # 섹션 1: 하드웨어 감지
    hw_info = detect_hardware()

    # 섹션 2: CUDA + PyTorch 자동 설치
    cuda_result = setup_cuda_and_pytorch(hw_info)
    if cuda_result == "RESTART_NEEDED":
        print("=" * 60)
        print("  PyTorch 가 CUDA 버전으로 재설치되었습니다.")
        print("  이 스크립트를 다시 실행하세요:")
        print(f"    python {Path(__file__).name}")
        print("=" * 60)
        return

    # GPU 없으면 CPU fallback 모드로 계속 진행
    cuda_ok = (cuda_result is True)
    if not cuda_ok:
        print("  -> CPU fallback 모드: 양자화 수학(섹션 4)과 메모리 비교(섹션 7)는")
        print("     GPU 없이 실행됩니다. 모델 로딩(섹션 5~6, 8)은 CPU로 수행합니다.")
        print()

    # 섹션 3: 필수 라이브러리 자동 설치 (CUDA 여부에 따라 bitsandbytes 처리)
    libs_result = install_required_libraries(cuda_available=cuda_ok)
    if libs_result == "RESTART_NEEDED":
        print("=" * 60)
        print("  라이브러리가 새로 설치되었습니다.")
        print("  이 스크립트를 다시 실행하세요:")
        print(f"    python {Path(__file__).name}")
        print("=" * 60)
        return
    if libs_result is False:
        print("=" * 60)
        print("  필수 라이브러리 설치에 실패했습니다.")
        print("  위의 에러 메시지를 확인하고 수동 설치 후 재실행하세요.")
        print("=" * 60)
        return

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Phase 2: QLoRA 실습
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print("=" * 60)
    print("  Phase 2: QLoRA 실습")
    print("=" * 60)
    print()

    # 섹션 4: 양자화 수학 (항상 실행 — GPU 불필요)
    demonstrate_quantization_math()

    # 섹션 5: 모델 로딩
    model, tokenizer = load_model_with_qlora(DEFAULT_MODEL)

    # 섹션 6: LoRA 설정 및 파인튜닝
    if model is not None:
        model, tokenizer = configure_lora_and_train(model, tokenizer)

    # 섹션 7: 메모리 비교 차트 (항상 실행 — 이론값)
    compare_memory_usage()

    # 섹션 8: 추론 테스트
    if model is not None:
        inference_with_finetuned_model(model, tokenizer)

    print("=" * 60)
    print("  제10장 실습 완료")
    print(f"  출력 파일 위치: {OUTPUT_DIR}")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
