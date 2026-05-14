"""
01-양자화수학.py
제10장 실습 ① — 양자화 수학 시연 (GPU 불필요)

이 스크립트는 QLoRA에서 사용하는 양자화 기법의 수학적 원리를
직접 숫자로 확인한다. PyTorch나 GPU가 필요 없다.

실행 결과 (모두 콘솔 출력):
  4-1. int4 양자화·역양자화 — 값이 어떻게 16칸으로 매핑되고 복원되는지
  4-2. NF4 vs 균등 분할  — 정규분포 가중치에서 NF4가 왜 더 정밀한지
  4-3. LoRA 파라미터 절감 — rank별로 학습 파라미터가 몇 % 줄어드는지
  4-4. 파인튜닝 방법별 메모리 — Full FT / LoRA / QLoRA 메모리 견적

의존성:
    pip install numpy

실행:
    python code/01-양자화수학.py

연계:
    02-qlora파인튜닝.py  — 실제 한국어 모델에 QLoRA 적용
    03-메모리비교.py     — 모델 크기별 메모리 차트
"""

import numpy as np

np.random.seed(42)


def demo_int4_quantization():
    """[4-1] int4 양자화·역양자화: 16칸 매핑과 복원 오차."""
    print("=" * 60)
    print("[4-1] int4 양자화 원리")
    print("=" * 60)

    weights = np.array(
        [0.5, 1.2, -0.8, 1.9, 0.1, -0.3, 0.77, -1.1],
        dtype=np.float32,
    )
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
    print()
    print("  → 16개 값으로 줄였지만 평균 오차는 ~0.1 수준")
    print("    신경망은 이 정도 잡음에 강하므로 성능 손실은 최소")
    print()


def demo_nf4_vs_uniform():
    """[4-2] NF4 vs 균등 분할: 정규분포에서 NF4가 더 정밀한 이유."""
    print("=" * 60)
    print("[4-2] NF4 vs 균등 분할 비교")
    print("=" * 60)

    # bitsandbytes에서 사용하는 NF4 코드북 (정규분포 분위수 기반)
    NF4_CODEBOOK = np.array([
        -1.0, -0.6962, -0.5251, -0.3949,
        -0.2844, -0.1848, -0.0911, 0.0,
        0.0796, 0.1609, 0.2461, 0.3379,
        0.4407, 0.5626, 0.7230, 1.0,
    ], dtype=np.float32)

    # 신경망 가중치를 모사하는 정규분포 샘플 (정규화하여 [-1, 1])
    sample = np.random.randn(1000).astype(np.float32)
    sample = sample / np.abs(sample).max()

    # NF4 양자화: 각 값을 가장 가까운 코드북 항목으로
    nf4_indices = np.array(
        [np.argmin(np.abs(NF4_CODEBOOK - w)) for w in sample]
    )
    nf4_recovered = NF4_CODEBOOK[nf4_indices]
    nf4_mae = np.abs(sample - nf4_recovered).mean()

    # 균등 분할 양자화: [-1, 1]을 16칸으로 등간격
    uniform_q = np.round((sample + 1.0) / 2.0 * 15).clip(0, 15).astype(int)
    uniform_recovered = uniform_q / 15.0 * 2.0 - 1.0
    uniform_mae = np.abs(sample - uniform_recovered).mean()

    print(f"  표본 수: {len(sample)} (정규분포)")
    print(f"  NF4 양자화 MAE:    {nf4_mae:.5f}")
    print(f"  균등 분할 MAE:     {uniform_mae:.5f}")
    improvement = (uniform_mae - nf4_mae) / uniform_mae * 100
    print(f"  NF4 오차 감소율:   {improvement:.1f}%")
    print()
    print("  → 가중치가 0 근처에 몰려 있으므로,")
    print("    0 근처에 칸을 더 촘촘히 둔 NF4가 평균 오차를 줄인다")
    print()


def demo_lora_param_savings():
    """[4-3] LoRA 파라미터 절감: rank별 학습 파라미터 비율."""
    print("=" * 60)
    print("[4-3] LoRA 파라미터 절감 계산")
    print("=" * 60)

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
    print()
    print("  → r=16 정도면 원본 대비 1% 미만의 파라미터만 학습")
    print("    학습 속도·메모리가 그만큼 절약된다")
    print()


def demo_memory_strategies():
    """[4-4] 파인튜닝 방법별 메모리 비교 (1.3B 모델 기준)."""
    print("=" * 60)
    print("[4-4] 파인튜닝 방법별 메모리 비교 (polyglot-ko-1.3B)")
    print("=" * 60)

    param_count = 1.3e9
    # LoRA가 학습하는 파라미터 수: rank × (in + out) × 층 수
    lora_params = 16 * (2048 + 6144) * 24

    strategies = [
        ("Full Fine-tuning (fp32)",
         param_count * 4,           # 모델: 4 bytes/param
         param_count * 4 * 2,       # Adam 옵티마이저: m, v 두 텐서
         param_count * 4),          # 그래디언트
        ("LoRA (fp16, r=16)",
         param_count * 2,           # 모델: 2 bytes/param
         lora_params * 4 * 2,       # 옵티마이저는 LoRA 파라미터만
         lora_params * 4),
        ("QLoRA (4-bit NF4)",
         param_count * 0.5,         # 모델: 0.5 bytes/param (4-bit)
         lora_params * 4 * 2,
         lora_params * 4),
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
    print("  → 같은 1.3B 모델인데도 메모리 차이가 한 자릿수")
    print("    QLoRA는 모델 본체를 4-bit로 줄이고 옵티마이저는 어댑터만")
    print()


def main():
    print()
    print("=" * 60)
    print("  제10장 실습 ① — 양자화 수학 시연")
    print("  딥러닝 자연어처리 (2026)")
    print("=" * 60)
    print()

    demo_int4_quantization()
    demo_nf4_vs_uniform()
    demo_lora_param_savings()
    demo_memory_strategies()

    print("=" * 60)
    print("  완료. 다음 실습:")
    print("    python code/02-qlora파인튜닝.py")
    print("    python code/03-메모리비교.py")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
