"""
4장 실습 1: Transformer 전체 구조와 Positional Encoding
- Sinusoidal Positional Encoding 구현
- Learned Positional Encoding 구현
- 위치 간 유사도 분석
- Pre-LN vs Post-LN 비교
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os

# 재현성을 위한 시드 설정
torch.manual_seed(42)


# ============================================================
# 1. Sinusoidal Positional Encoding
# ============================================================
class SinusoidalPositionalEncoding(nn.Module):
    """
    원본 Transformer 논문의 Sinusoidal Positional Encoding

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """

    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # (1, max_len, d_model)로 저장
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        """x: (batch, seq_len, d_model)"""
        return x + self.pe[:, : x.size(1), :]


# ============================================================
# 2. Learned Positional Encoding
# ============================================================
class LearnedPositionalEncoding(nn.Module):
    """
    BERT, GPT-2 등에서 사용하는 학습 가능한 Positional Encoding
    """

    def __init__(self, d_model, max_len=512):
        super().__init__()
        self.pe = nn.Embedding(max_len, d_model)

    def forward(self, x):
        """x: (batch, seq_len, d_model)"""
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device)
        return x + self.pe(positions)


def main():
    print("=" * 60)
    print("4장 실습 1: Transformer 구조와 Positional Encoding")
    print("=" * 60)

    d_model = 128
    max_len = 100

    # --------------------------------------------------------
    # 1. Sinusoidal Positional Encoding
    # --------------------------------------------------------
    print("\n" + "-" * 40)
    print("1. Sinusoidal Positional Encoding")
    print("-" * 40)

    sin_pe = SinusoidalPositionalEncoding(d_model, max_len)
    sin_params = sum(p.numel() for p in sin_pe.parameters())
    print(f"  학습 가능한 파라미터 수: {sin_params} (파라미터 없음)")

    # PE 값 확인
    pe_values = sin_pe.pe[0]  # (max_len, d_model)
    print(f"\n  [첫 3개 위치의 PE 값 (처음 8차원)]")
    for pos in range(3):
        vals = [f"{pe_values[pos, d]:.3f}" for d in range(8)]
        print(f"    위치 {pos}: {vals}")

    # 위치 간 코사인 유사도
    print(f"\n  [위치 간 코사인 유사도]")
    for target_pos in [1, 5, 10, 50]:
        sim = F.cosine_similarity(
            pe_values[0].unsqueeze(0), pe_values[target_pos].unsqueeze(0)
        )
        print(f"    위치 0 ↔ 위치 {target_pos:>2}: {sim.item():.4f}")

    # --------------------------------------------------------
    # 2. Learned Positional Encoding
    # --------------------------------------------------------
    print("\n" + "-" * 40)
    print("2. Learned Positional Encoding")
    print("-" * 40)

    learned_pe = LearnedPositionalEncoding(d_model, max_len)
    learned_params = sum(p.numel() for p in learned_pe.parameters())
    print(f"  학습 가능한 파라미터 수: {learned_params:,}")
    print(f"  = max_len × d_model = {max_len} × {d_model}")

    # --------------------------------------------------------
    # 3. 비교 표
    # --------------------------------------------------------
    print("\n" + "-" * 40)
    print("3. Sinusoidal vs Learned 비교")
    print("-" * 40)

    print("""
  | 특성             | Sinusoidal | Learned         |
  |------------------|------------|-----------------|
  | 파라미터 수      | 0          | {0:,} |
  | 긴 시퀀스 일반화 | 가능       | 불가능          |
  | 표현력           | 고정       | 유연 (학습)     |
  | 사용 모델        | 원본 Transformer | GPT-2, BERT    |""".format(
        learned_params
    ))

    # --------------------------------------------------------
    # 4. Transformer 구성 요소 크기 분석
    # --------------------------------------------------------
    print("\n" + "-" * 40)
    print("4. Transformer 구성 요소 크기 분석")
    print("-" * 40)

    d_model_sizes = [256, 512, 768, 1024]
    for dm in d_model_sizes:
        d_ff = 4 * dm
        # Encoder Block 파라미터: MHA + FFN + 2 LayerNorm
        mha_params = 4 * dm * dm  # W_q, W_k, W_v, W_o (bias 제외 간략 계산)
        ffn_params = dm * d_ff + d_ff * dm  # 2 linear layers
        ln_params = 2 * 2 * dm  # 2 LayerNorm (weight + bias)
        total = mha_params + ffn_params + ln_params
        print(f"  d_model={dm:>4}, d_ff={d_ff:>4}: "
              f"MHA={mha_params:>10,} + FFN={ffn_params:>10,} + LN={ln_params:>6,} "
              f"= {total:>10,} params/block")

    # --------------------------------------------------------
    # 5. Residual Connection 효과 시연
    # --------------------------------------------------------
    print("\n" + "-" * 40)
    print("5. Residual Connection 효과")
    print("-" * 40)

    # 깊은 네트워크에서 Residual 유무에 따른 신호 크기 변화
    x = torch.randn(1, 10, 128)  # (batch, seq, d_model)

    # Residual 없는 경우
    signal_no_res = x.clone()
    for i in range(20):
        layer = nn.Linear(128, 128)
        with torch.no_grad():
            signal_no_res = F.relu(layer(signal_no_res))

    # Residual 있는 경우
    signal_with_res = x.clone()
    for i in range(20):
        layer = nn.Linear(128, 128)
        with torch.no_grad():
            signal_with_res = signal_with_res + F.relu(layer(signal_with_res))

    print(f"  입력 신호 크기 (L2 norm): {x.norm().item():.4f}")
    print(f"  20층 후 (Residual 없음):  {signal_no_res.norm().item():.4f}")
    print(f"  20층 후 (Residual 있음):  {signal_with_res.norm().item():.4f}")
    print(f"  → Residual Connection은 깊은 네트워크에서 신호가 사라지는 것을 방지한다")

    # --------------------------------------------------------
    # 6. Layer Normalization vs Batch Normalization
    # --------------------------------------------------------
    print("\n" + "-" * 40)
    print("6. Layer Normalization 동작 확인")
    print("-" * 40)

    x = torch.randn(2, 5, 128)  # (batch, seq, d_model)
    layer_norm = nn.LayerNorm(128)

    output = layer_norm(x)
    # 각 위치에서의 평균과 표준편차
    mean_before = x[0, 0].mean().item()
    std_before = x[0, 0].std().item()
    mean_after = output[0, 0].mean().item()
    std_after = output[0, 0].std().item()

    print(f"  정규화 전 — 평균: {mean_before:.4f}, 표준편차: {std_before:.4f}")
    print(f"  정규화 후 — 평균: {mean_after:.4f}, 표준편차: {std_after:.4f}")
    print(f"  → LayerNorm은 각 위치에서 평균=0, 표준편차=1로 정규화한다")

    print("\n" + "=" * 60)
    print("실습 1 완료")
    print("=" * 60)


if __name__ == "__main__":
    main()
