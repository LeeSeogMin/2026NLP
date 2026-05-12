"""
10-1-qlora파인튜닝.py
제10장 실습: QLoRA로 한국어 Instruction Tuning

이 스크립트는 환경 자동 구성부터 한국어 Instruction Tuning까지 전체 파이프라인을 실행한다:

  Phase 1 — 자동 환경 구성 (처음 1회만 필요)
    1. 하드웨어 감지  : NVIDIA GPU / Apple MPS / CPU 자동 탐지
    2. CUDA 자동 설치  : NVIDIA GPU 사용 시, 드라이버에 맞는 PyTorch+CUDA pip 설치
    3. 라이브러리 자동 설치 : peft, bitsandbytes, accelerate, datasets 자동 설치

  Phase 2 — 한국어 Instruction Tuning
    4. 양자화 수학 시연  (GPU 불필요)
    5. polyglot-ko-1.3b 모델 로딩 (플랫폼별 최적화)
    6. Before: 파인튜닝 전 한국어 질문 응답 테스트
    7. KoAlpaca 데이터로 QLoRA/LoRA 파인튜닝 (20 step)
    8. After: 파인튜닝 후 동일 질문 응답 + Before/After 비교
    9. Full FT / LoRA / QLoRA 메모리 비교 차트

대상 환경:
  - Windows + NVIDIA GPU  → QLoRA (4-bit NF4 + LoRA)
  - macOS + Apple Silicon  → LoRA (float16, 양자화 없음)
  - CPU only              → LoRA (float32, 데모용)

모델: EleutherAI/polyglot-ko-1.3b (한국어 전용 GPT-NeoX, 1.3B 파라미터)
데이터: beomi/KoAlpaca-v1.1a (한국어 Instruction 21K, Apache-2.0)

실행 방법:
    cd practice/chapter10
    python -m venv venv
    venv\\Scripts\\activate          # Windows
    source venv/bin/activate        # macOS / Linux
    pip install torch               # PyTorch 먼저 설치
    python code/10-1-qlora파인튜닝.py
"""

import importlib
import os
import platform
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

# ── 모델 및 데이터 설정 ────────────────────────────────
DEFAULT_MODEL = "EleutherAI/polyglot-ko-1.3b"
DATASET_NAME = "beomi/KoAlpaca-v1.1a"

# ── 테스트 프롬프트 (Before/After 비교용) ───────────────
TEST_PROMPTS = [
    "인공지능이 우리 생활에 미치는 영향을 설명해주세요.",
    "건강한 식습관을 유지하는 팁 3가지를 알려주세요.",
    "파이썬 프로그래밍의 장점은 무엇인가요?",
]


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
        return out.split("\n")[0].strip()
    return None


def _driver_to_cuda_tag(driver_ver):
    """NVIDIA 드라이버 버전 → 최적 PyTorch CUDA index URL 태그."""
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
    return None


def _detect_npu_windows():
    """Windows 에서 NPU를 감지한다."""
    if platform.system() != "Windows":
        return []
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


def _get_device_type():
    """현재 환경의 가속기 유형을 반환한다: 'cuda', 'mps', 'cpu'."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ──────────────────────────────────────────────────────
# 1. 하드웨어 자동 감지
# ──────────────────────────────────────────────────────
def detect_hardware():
    """NVIDIA GPU, Apple MPS, NPU 등을 자동 감지하고 결과를 반환한다."""
    print("=" * 60)
    print("[1] 하드웨어 자동 감지")
    print("=" * 60)

    info = {
        "os": platform.system(),
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "nvidia_driver": None,
        "nvidia_gpus": [],
        "mps_available": False,
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

    # ── NVIDIA GPU 감지 ──
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
                    "name": gpu_name, "driver": drv, "vram_gb": round(mem_gb, 1),
                })
                info["nvidia_driver"] = drv
                print(f"  [GPU] {gpu_name}")
                print(f"        드라이버: {drv}  |  VRAM: {mem_gb:.1f} GB")
    else:
        print("  NVIDIA GPU 미감지 (nvidia-smi 실행 불가)")

    # ── Apple MPS 감지 ──
    print(f"\n  --- Apple MPS 탐지 ---")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        info["mps_available"] = True
        print("  [MPS] Apple Silicon GPU 사용 가능")
        print("        LoRA 파인튜닝 지원 (양자화 없이 float16)")
    else:
        print("  Apple MPS 미감지")

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
    else:
        print("  NPU 미감지")

    # ── PyTorch 현황 ──
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
            print(f"  NVIDIA GPU 감지: {gpu0['name']} ({gpu0['vram_gb']}GB)")
            print(f"  -> QLoRA 모드 (4-bit 양자화 + LoRA)")
        else:
            print(f"  [!] 드라이버 {gpu0['driver']}이 너무 오래되었습니다.")
    elif info["mps_available"]:
        print("  Apple Silicon MPS 감지")
        print("  -> LoRA 모드 (float16, 양자화 없음)")
    else:
        print("  가속기 미감지. CPU 모드로 제한 실행됩니다.")

    print()
    return info


# ──────────────────────────────────────────────────────
# 2. CUDA + PyTorch 자동 설치
# ──────────────────────────────────────────────────────
def setup_cuda_and_pytorch(hw_info):
    """드라이버 버전에 맞는 PyTorch+CUDA 를 자동으로 설치한다.
    MPS 환경에서는 설치가 불필요하므로 건너뛴다.
    반환: 'cuda', 'mps', 'cpu', 또는 'RESTART_NEEDED'.
    """
    print("=" * 60)
    print("[2] CUDA + PyTorch 자동 설치")
    print("=" * 60)

    # ── 이미 CUDA 사용 가능 ──
    if torch.cuda.is_available():
        cuda_ver = torch.version.cuda
        print(f"\n  PyTorch CUDA {cuda_ver} 이미 활성 상태 — 설치 불필요")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print()
        return "cuda"

    # ── Apple MPS 사용 가능 ──
    if hw_info.get("mps_available"):
        print("\n  Apple Silicon MPS 감지 — CUDA 설치 불필요")
        print("  MPS 백엔드로 LoRA 파인튜닝을 진행합니다.")
        print("  (bitsandbytes 양자화는 CUDA 전용이므로 비활성)")
        print()
        return "mps"

    # ── NVIDIA GPU 없는 경우 → CPU fallback ──
    if not hw_info.get("nvidia_gpus"):
        print("\n  NVIDIA GPU / Apple MPS 미감지 — CPU 모드로 전환")
        print("  -> CPU fallback 모드:")
        print("     - 양자화 수학(섹션 4): 정상 실행")
        print("     - 모델 로딩(섹션 5):   CPU fp32 로딩")
        print("     - 파인튜닝(섹션 7):    CPU 모드 학습 (느림)")
        print()
        return "cpu"

    # ── NVIDIA GPU 있으나 PyTorch CUDA 미활성 ──
    driver_ver = hw_info["nvidia_driver"]
    cuda_tag = _driver_to_cuda_tag(driver_ver)

    if cuda_tag is None:
        print(f"\n  [!] 드라이버 {driver_ver}이 너무 오래되었습니다.")
        print("  https://www.nvidia.com/drivers 에서 업데이트하세요.")
        print()
        return "cpu"

    index_url = f"https://download.pytorch.org/whl/{cuda_tag}"
    print(f"\n  NVIDIA GPU 감지: {hw_info['nvidia_gpus'][0]['name']}")
    print(f"  드라이버: {driver_ver} -> 호환 CUDA: {cuda_tag}")
    print(f"\n  PyTorch 를 CUDA {cuda_tag} 버전으로 자동 재설치합니다.")

    print("\n  [단계 1/3] 기존 PyTorch 제거...")
    _pip_uninstall("torch", "torchvision", "torchaudio")

    print(f"  [단계 2/3] PyTorch + CUDA ({cuda_tag}) 설치 중... (수 분 소요)")
    ok = _pip_install(
        "torch", "torchvision", "torchaudio",
        index_url=index_url, timeout=900,
    )
    if not ok:
        print("\n  [!] PyTorch CUDA 설치 실패. 수동 설치를 시도하세요.")
        print()
        return "cpu"

    print("  [단계 3/3] 설치 검증...")
    verify_code = (
        "import torch; "
        "print(f'cuda={torch.cuda.is_available()},"
        "ver={torch.version.cuda},"
        "gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"
    )
    code, out = _run_cmd([sys.executable, "-c", verify_code], timeout=30)

    if code == 0 and "cuda=True" in out:
        print(f"\n  PyTorch CUDA 설치 성공! ({out})")
        print("  스크립트를 재실행하면 CUDA가 활성화됩니다.")
        print()
        return "RESTART_NEEDED"
    else:
        print(f"\n  [!] CUDA 활성화 실패. CPU 모드로 계속합니다.")
        print()
        return "cpu"


# ──────────────────────────────────────────────────────
# 3. 필수 라이브러리 자동 설치
# ──────────────────────────────────────────────────────
def install_required_libraries(device_type="cpu"):
    """QLoRA/LoRA 실습에 필요한 라이브러리를 점검·설치한다.
    반환: True, False, 또는 'RESTART_NEEDED'.
    """
    print("=" * 60)
    print("[3] 필수 라이브러리 자동 설치")
    print("=" * 60)

    cuda_available = (device_type == "cuda")

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
            if cuda_only and not cuda_available:
                print(f"  [  ] {pkg:<18} 건너뜀        ({desc}) — CUDA 전용")
                continue
            print(f"  [--] {pkg:<18} 미설치        ({desc})")
            missing.append((pkg, min_ver, desc))

    if not missing:
        print("\n  모든 라이브러리 정상 설치됨")
        print()
        return True

    print(f"\n  미설치 {len(missing)}개 라이브러리를 자동 설치합니다...")
    any_installed = False

    for pkg, min_ver, desc in missing:
        print(f"\n  설치 중: {pkg}>={min_ver} ({desc})")
        ok = _pip_install(f"{pkg}>={min_ver}")
        if ok:
            print(f"    -> 설치 완료")
            any_installed = True
        elif pkg == "bitsandbytes" and platform.system() == "Windows":
            print("    bitsandbytes Windows 재시도 (--prefer-binary)...")
            ok = _pip_install(f"{pkg}>={min_ver}", "--prefer-binary")
            if ok:
                any_installed = True
            else:
                print(f"    [!] {pkg} 설치 실패.")
        else:
            print(f"    [!] {pkg} 설치 실패.")

    # 설치 후 검증
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
        non_bnb = [p for p in still_missing if p != "bitsandbytes"]
        if non_bnb:
            print(f"\n  [!] 설치 실패: {', '.join(still_missing)}")
            return False
        else:
            print("\n  bitsandbytes 설치 실패 — 양자화 없이 LoRA 모드로 계속합니다.")

    if any_installed:
        print("\n  라이브러리가 새로 설치되었습니다.")
        print("  스크립트를 재실행하세요.")
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
        ("polyglot-ko-1.3B QKV", 2048, 6144),
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
    print(f"\n[4-4] 파인튜닝 방법별 메모리 비교 (polyglot-ko-1.3B)")
    print("-" * 40)
    param_count = 1.3e9
    lora_params = 16 * (2048 + 6144) * 24

    strategies = [
        ("Full Fine-tuning (fp32)", param_count * 4,
         param_count * 4 * 2, param_count * 4),
        ("LoRA (fp16, r=16)",       param_count * 2,
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
# 5. 모델 로딩 (플랫폼 적응)
# ──────────────────────────────────────────────────────
def load_model(device_type, model_name=DEFAULT_MODEL):
    """polyglot-ko-1.3b를 플랫폼에 맞게 로딩한다.
    - cuda: 4-bit NF4 양자화 (QLoRA)
    - mps:  float16 (LoRA)
    - cpu:  float32 (LoRA)
    """
    print("=" * 60)
    print("[5] 모델 로딩")
    print("=" * 60)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\n  모델: {model_name}")
    print(f"  모드: {device_type.upper()}")
    print("  (첫 실행 시 HuggingFace에서 다운로드 — 수 분 소요 가능)")

    try:
        if device_type == "cuda":
            from transformers import BitsAndBytesConfig

            torch.cuda.empty_cache()
            mem_before = torch.cuda.memory_allocated(0) / (1024 ** 3)

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            print("\n  양자화 설정: NF4 + Double Quantization + bfloat16")
            print("  모델 로딩 중...")

            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto",
            )

            mem_after = torch.cuda.memory_allocated(0) / (1024 ** 3)
            print(f"  4-bit NF4 로딩 완료")
            print(f"  GPU 메모리: {mem_after:.2f} GB (모델: {mem_after - mem_before:.2f} GB)")

        elif device_type == "mps":
            print("\n  Apple MPS: float16으로 로딩 (양자화 없음)")
            print("  모델 로딩 중...")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
            ).to("mps")
            print("  float16 MPS 로딩 완료")

        else:
            print("\n  CPU: float32로 로딩 (양자화 없음)")
            print("  모델 로딩 중...")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
            )
            print("  CPU 로딩 완료")

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        total = sum(p.numel() for p in model.parameters())
        print(f"\n  파라미터: {total:,}")
        print(f"  어휘 크기: {tokenizer.vocab_size:,}")

    except Exception as e:
        print(f"\n  모델 로딩 실패: {e}")
        return None, None

    print()
    return model, tokenizer


# ──────────────────────────────────────────────────────
# 6. 추론 실행 (Before/After 공용)
# ──────────────────────────────────────────────────────
def run_inference(model, tokenizer, prompts, device_type):
    """Instruction 형식의 프롬프트로 추론을 실행한다."""
    model.eval()
    results = []

    for instruction in prompts:
        prompt = f"### 질문: {instruction}\n\n### 답변: "

        if device_type == "cuda":
            device = next(model.parameters()).device
        elif device_type == "mps":
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            torch.manual_seed(42)
            output_ids = model.generate(
                **inputs,
                max_new_tokens=80,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.2,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated = tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        results.append({
            "instruction": instruction,
            "generated": generated.strip(),
        })

    return results


def test_before_finetuning(model, tokenizer, device_type):
    """파인튜닝 전 모델의 응답을 테스트한다."""
    print("=" * 60)
    print("[6] Before: 파인튜닝 전 추론 테스트")
    print("=" * 60)

    if model is None:
        print("  모델이 없습니다.")
        return []

    results = run_inference(model, tokenizer, TEST_PROMPTS, device_type)

    print()
    for i, r in enumerate(results, 1):
        print(f"  [질문 {i}] {r['instruction']}")
        answer = r['generated'][:150]
        print(f"  [응답]   {answer}")
        if len(r['generated']) > 150:
            print(f"           ...")
        print()

    print("  -> 베이스 모델은 Instruction 형식을 이해하지 못한다.")
    print("     KoAlpaca 데이터로 파인튜닝 후 비교할 예정.")
    print()
    return results


# ──────────────────────────────────────────────────────
# 7. KoAlpaca 데이터로 LoRA/QLoRA 파인튜닝
# ──────────────────────────────────────────────────────
def finetune_with_koalpaca(model, tokenizer, device_type):
    """KoAlpaca 데이터로 Instruction Tuning을 수행한다."""
    print("=" * 60)
    print("[7] KoAlpaca 데이터로 파인튜닝")
    print("=" * 60)

    if model is None:
        print("  모델이 없습니다.")
        return None, None

    from peft import LoraConfig, get_peft_model
    from transformers import (
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )
    from datasets import load_dataset

    cuda_ok = (device_type == "cuda")

    # ── 7-1. kbit 학습 준비 (CUDA QLoRA 전용) ──
    if cuda_ok:
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model)
        print("\n  kbit 학습 준비 완료 (LayerNorm fp32 업캐스팅)")

    # ── 7-2. LoRA 설정 ──
    # GPT-NeoX 아키텍처: QKV가 query_key_value로 융합되어 있음
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["query_key_value"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    print(f"\n  [LoRA 설정]")
    print(f"    rank: {lora_config.r}")
    print(f"    alpha: {lora_config.lora_alpha}")
    print(f"    target: {lora_config.target_modules}")
    print(f"    ※ GPT-NeoX는 QKV가 융합 — query_key_value 하나로 적용")

    model = get_peft_model(model, lora_config)
    print(f"\n  [학습 가능 파라미터]")
    model.print_trainable_parameters()

    # ── 7-3. Gradient Checkpointing (CUDA 전용) ──
    if cuda_ok:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.enable_input_require_grads()
        print("  Gradient Checkpointing 활성화")

    # ── 7-4. KoAlpaca 데이터 로딩 + Instruction 포맷팅 ──
    print(f"\n  데이터 로딩: {DATASET_NAME}")
    dataset = load_dataset(DATASET_NAME, split="train")

    # 500개 샘플만 사용 (교실 실습용)
    dataset = dataset.shuffle(seed=42).select(range(min(500, len(dataset))))

    def format_and_tokenize(example):
        instruction = example.get("instruction", "")
        output = example.get("output", "")
        text = f"### 질문: {instruction}\n\n### 답변: {output}{tokenizer.eos_token}"
        return tokenizer(text, truncation=True, max_length=256, padding="max_length")

    tokenized = dataset.map(
        format_and_tokenize,
        remove_columns=dataset.column_names,
    )
    print(f"  데이터: {len(tokenized)} 샘플, 최대 256 토큰")

    # 샘플 출력
    sample = dataset[0]
    print(f"\n  [데이터 샘플]")
    print(f"    질문: {sample.get('instruction', '')[:80]}")
    print(f"    답변: {sample.get('output', '')[:80]}")

    # ── 7-5. 학습 설정 ──
    output_path = OUTPUT_DIR / "lora_checkpoint"
    max_steps = 10 if device_type == "cpu" else 20

    training_args = TrainingArguments(
        output_dir=str(output_path),
        max_steps=max_steps,
        per_device_train_batch_size=2 if device_type != "cpu" else 1,
        gradient_accumulation_steps=4,
        learning_rate=5e-4,
        lr_scheduler_type="cosine",
        warmup_steps=5,
        logging_steps=5,
        save_steps=max_steps,
        bf16=cuda_ok,
        fp16=False,
        gradient_checkpointing=cuda_ok,
        seed=42,
        optim="paged_adamw_8bit" if cuda_ok else "adamw_torch",
        report_to="none",
    )

    print(f"\n  [학습 설정]")
    print(f"    스텝: {max_steps}  |  배치: {training_args.per_device_train_batch_size}")
    print(f"    학습률: {training_args.learning_rate}  |  옵티마이저: {training_args.optim}")
    print(f"    모드: {'QLoRA (4-bit)' if cuda_ok else 'LoRA (' + device_type + ')'}")

    # ── 7-6. 학습 실행 ──
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    print(f"\n  학습 시작 ({max_steps} 스텝)...")
    train_result = trainer.train()

    runtime = train_result.metrics.get("train_runtime", 0)
    print(f"\n  학습 완료 ({runtime:.1f}초)")

    if cuda_ok:
        final_mem = torch.cuda.memory_allocated(0) / (1024 ** 3)
        print(f"  최종 GPU 메모리: {final_mem:.2f} GB")

    # LoRA 어댑터 저장
    adapter_path = OUTPUT_DIR / "lora_adapter"
    model.save_pretrained(str(adapter_path))
    adapter_size = sum(
        f.stat().st_size for f in Path(adapter_path).rglob("*") if f.is_file()
    )
    print(f"  LoRA 어댑터 저장: {adapter_path}")
    print(f"  어댑터 크기: {adapter_size / 1e6:.2f} MB")

    print()
    return model, tokenizer


# ──────────────────────────────────────────────────────
# 8. Before/After 비교
# ──────────────────────────────────────────────────────
def test_after_and_compare(model, tokenizer, before_results, device_type):
    """파인튜닝 후 동일 질문으로 추론하고 Before/After를 비교한다."""
    print("=" * 60)
    print("[8] After: 파인튜닝 후 추론 + Before/After 비교")
    print("=" * 60)

    if model is None or not before_results:
        print("  모델 또는 Before 결과가 없습니다.")
        return

    after_results = run_inference(model, tokenizer, TEST_PROMPTS, device_type)

    print()
    for i, (before, after) in enumerate(zip(before_results, after_results), 1):
        print(f"  {'=' * 50}")
        print(f"  [질문 {i}] {before['instruction']}")
        print(f"  {'─' * 50}")
        b_text = before['generated'][:120]
        a_text = after['generated'][:120]
        print(f"  Before: {b_text}")
        print(f"  After:  {a_text}")
        print()

    # 결과 파일 저장
    result_path = OUTPUT_DIR / "before_after_comparison.txt"
    with open(result_path, "w", encoding="utf-8") as f:
        f.write("QLoRA/LoRA Instruction Tuning — Before/After 비교\n")
        f.write(f"모델: {DEFAULT_MODEL}\n")
        f.write(f"데이터: {DATASET_NAME}\n")
        f.write("=" * 60 + "\n\n")
        for i, (before, after) in enumerate(zip(before_results, after_results), 1):
            f.write(f"[질문 {i}] {before['instruction']}\n")
            f.write(f"Before: {before['generated']}\n")
            f.write(f"After:  {after['generated']}\n")
            f.write("-" * 40 + "\n\n")
    print(f"  비교 결과 저장: {result_path}")
    print()


# ──────────────────────────────────────────────────────
# 9. 메모리 사용량 비교 차트
# ──────────────────────────────────────────────────────
def compare_memory_usage():
    """Full FT / LoRA / QLoRA 메모리를 이론값으로 비교하고 차트를 저장한다."""
    print("=" * 60)
    print("[9] 메모리 사용량 비교 (이론값)")
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
        "polyglot-ko\n1.3B":  1.3e9,
        "Llama-7B":           7e9,
        "Llama-70B":          70e9,
    }

    lora_r = 16
    lora_params = {
        "polyglot-ko\n1.3B":  lora_r * (2048 + 6144) * 24,
        "Llama-7B":           lora_r * (4096 + 4096) * 2 * 32,
        "Llama-70B":          lora_r * (8192 + 8192) * 2 * 80,
    }

    methods = ["Full FT (fp32)", "LoRA (fp16)", "QLoRA (4-bit)"]
    data = {m: [] for m in methods}

    for mname, params in models.items():
        lp = lora_params[mname]
        data["Full FT (fp32)"].append(params * 4 * 4 / (1024 ** 3))
        data["LoRA (fp16)"].append(
            (params * 2 + lp * 4 * 3) / (1024 ** 3)
        )
        data["QLoRA (4-bit)"].append(
            (params * 0.5 + lp * 4 * 3) / (1024 ** 3)
        )

    model_names = list(models.keys())
    display_names = [n.replace("\n", " ") for n in model_names]
    print(f"\n  {'방법':<18}", end="")
    for mn in display_names:
        print(f" {mn:>14}", end="")
    print()
    print("  " + "-" * 62)

    for method in methods:
        print(f"  {method:<18}", end="")
        for val in data[method]:
            print(f" {val:>12.1f}GB", end="")
        print()

    print("  " + "-" * 62)
    print("  참고 GPU VRAM:")
    print("    RTX 4090: 24 GB  |  RTX 4060: 8 GB  |  Apple M1: 8-16 GB")

    fig, ax = plt.subplots(figsize=(10, 6))
    x = range(len(model_names))
    width = 0.25
    colors = ["#e74c3c", "#3498db", "#2ecc71"]

    for idx, method in enumerate(methods):
        offset = (idx - 1) * width
        bars = ax.bar(
            [xi + offset for xi in x],
            data[method], width=width,
            label=method, color=colors[idx], alpha=0.85,
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
               label="RTX 4060 / M1 (8GB)")

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
# main
# ──────────────────────────────────────────────────────
def main():
    print()
    print("=" * 60)
    print("  제10장 실습 — QLoRA로 한국어 Instruction Tuning")
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

    hw_info = detect_hardware()

    device_type = setup_cuda_and_pytorch(hw_info)
    if device_type == "RESTART_NEEDED":
        print("=" * 60)
        print("  PyTorch가 CUDA 버전으로 재설치되었습니다.")
        print(f"  스크립트를 다시 실행하세요: python {Path(__file__).name}")
        print("=" * 60)
        return

    libs_result = install_required_libraries(device_type=device_type)
    if libs_result == "RESTART_NEEDED":
        print("=" * 60)
        print("  라이브러리가 새로 설치되었습니다.")
        print(f"  스크립트를 다시 실행하세요: python {Path(__file__).name}")
        print("=" * 60)
        return
    if libs_result is False:
        print("=" * 60)
        print("  필수 라이브러리 설치에 실패했습니다.")
        print("  위의 에러 메시지를 확인하고 수동 설치 후 재실행하세요.")
        print("=" * 60)
        return

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Phase 2: 한국어 Instruction Tuning
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print("=" * 60)
    print("  Phase 2: 한국어 Instruction Tuning")
    print(f"  모드: {device_type.upper()}"
          f" ({'QLoRA' if device_type == 'cuda' else 'LoRA'})")
    print("=" * 60)
    print()

    # 섹션 4: 양자화 수학 (항상 실행)
    demonstrate_quantization_math()

    # 섹션 5: 모델 로딩
    model, tokenizer = load_model(device_type)

    # 섹션 6: Before — 파인튜닝 전 추론
    before_results = []
    if model is not None:
        before_results = test_before_finetuning(model, tokenizer, device_type)

    # 섹션 7: KoAlpaca 파인튜닝
    if model is not None:
        model, tokenizer = finetune_with_koalpaca(model, tokenizer, device_type)

    # 섹션 8: After — Before/After 비교
    if model is not None and before_results:
        test_after_and_compare(model, tokenizer, before_results, device_type)

    # 섹션 9: 메모리 비교 차트 (항상 실행)
    compare_memory_usage()

    print("=" * 60)
    print("  제10장 실습 완료")
    print(f"  출력 파일 위치: {OUTPUT_DIR}")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
