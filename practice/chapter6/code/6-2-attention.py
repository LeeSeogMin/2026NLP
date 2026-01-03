"""
6장 실습: Attention 메커니즘 구현
- Scaled Dot-Product Attention
- Self-Attention 계산 과정
- Attention Weights 시각화
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
import numpy as np

# 재현성을 위한 시드 설정
torch.manual_seed(42)


def scaled_dot_product_attention(query, key, value, mask=None):
    """
    Scaled Dot-Product Attention 구현

    Args:
        query: (batch, seq_len, d_k)
        key: (batch, seq_len, d_k)
        value: (batch, seq_len, d_v)
        mask: (batch, seq_len, seq_len) or None

    Returns:
        output: (batch, seq_len, d_v)
        attention_weights: (batch, seq_len, seq_len)
    """
    d_k = query.size(-1)

    # 1단계: Q와 K의 dot product
    # (batch, seq_len, d_k) @ (batch, d_k, seq_len) = (batch, seq_len, seq_len)
    scores = torch.matmul(query, key.transpose(-2, -1))

    # 2단계: 스케일링 (√d_k로 나누기)
    scores = scores / math.sqrt(d_k)

    # 3단계: 마스킹 (선택적)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    # 4단계: Softmax로 정규화 → Attention Weights
    attention_weights = F.softmax(scores, dim=-1)

    # 5단계: Value와 가중합
    output = torch.matmul(attention_weights, value)

    return output, attention_weights


class SelfAttention(nn.Module):
    """Self-Attention 레이어"""

    def __init__(self, d_model, d_k=None, d_v=None):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_k if d_k else d_model
        self.d_v = d_v if d_v else d_model

        # Q, K, V를 생성하는 선형 변환
        self.W_q = nn.Linear(d_model, self.d_k, bias=False)
        self.W_k = nn.Linear(d_model, self.d_k, bias=False)
        self.W_v = nn.Linear(d_model, self.d_v, bias=False)

    def forward(self, x, mask=None):
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            output: (batch, seq_len, d_v)
            attention_weights: (batch, seq_len, seq_len)
        """
        # 1단계: Q, K, V 행렬 생성
        Q = self.W_q(x)  # (batch, seq_len, d_k)
        K = self.W_k(x)  # (batch, seq_len, d_k)
        V = self.W_v(x)  # (batch, seq_len, d_v)

        # 2단계: Scaled Dot-Product Attention
        output, attention_weights = scaled_dot_product_attention(Q, K, V, mask)

        return output, attention_weights


def visualize_attention(attention_weights, tokens, title="Attention Weights"):
    """Attention Weights 시각화"""
    import os
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'output')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'attention_weights.png')

    plt.figure(figsize=(8, 6))

    # numpy로 변환
    weights = attention_weights.squeeze().detach().numpy()

    plt.imshow(weights, cmap='Blues')
    plt.colorbar(label='Attention Weight')

    # 토큰 라벨 설정 (영문으로 대체하여 폰트 문제 회피)
    tokens_en = [f"Token_{i}" for i in range(len(tokens))]
    plt.xticks(range(len(tokens_en)), tokens_en, rotation=45, ha='right')
    plt.yticks(range(len(tokens_en)), tokens_en)

    plt.xlabel('Key Position')
    plt.ylabel('Query Position')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Attention weights visualization saved: {output_path}")


def main():
    print("=" * 60)
    print("Scaled Dot-Product Attention 구현")
    print("=" * 60)

    # 파라미터 설정
    batch_size = 1
    seq_len = 5
    d_model = 64
    d_k = 32

    # 예시 입력 (임베딩된 문장)
    # 실제로는 단어 임베딩이지만, 여기서는 랜덤 값 사용
    x = torch.randn(batch_size, seq_len, d_model)

    print(f"\n[입력 정보]")
    print(f"  배치 크기: {batch_size}")
    print(f"  시퀀스 길이: {seq_len}")
    print(f"  모델 차원 (d_model): {d_model}")
    print(f"  Key/Query 차원 (d_k): {d_k}")

    # Self-Attention 레이어 생성
    self_attn = SelfAttention(d_model, d_k=d_k, d_v=d_k)

    # Self-Attention 계산
    output, attn_weights = self_attn(x)

    print(f"\n[출력 정보]")
    print(f"  출력 shape: {output.shape}")
    print(f"  Attention weights shape: {attn_weights.shape}")

    # Attention weights 확인
    print(f"\n[Attention Weights (첫 번째 샘플)]")
    print(f"  각 행의 합 (softmax 확인): {attn_weights[0].sum(dim=-1).tolist()}")

    # 시각화용 토큰 (예시)
    tokens = ["나는", "학교에", "가서", "공부를", "한다"]
    visualize_attention(attn_weights, tokens)

    print("\n" + "=" * 60)
    print("스케일링의 효과 분석")
    print("=" * 60)

    # 스케일링 없이 계산
    Q = torch.randn(1, 10, 512)
    K = torch.randn(1, 10, 512)

    scores_unscaled = torch.matmul(Q, K.transpose(-2, -1))
    scores_scaled = scores_unscaled / math.sqrt(512)

    print(f"\n[스케일링 전]")
    print(f"  Score 평균: {scores_unscaled.mean().item():.4f}")
    print(f"  Score 표준편차: {scores_unscaled.std().item():.4f}")
    print(f"  Score 범위: [{scores_unscaled.min().item():.2f}, {scores_unscaled.max().item():.2f}]")

    print(f"\n[스케일링 후 (÷√512)]")
    print(f"  Score 평균: {scores_scaled.mean().item():.4f}")
    print(f"  Score 표준편차: {scores_scaled.std().item():.4f}")
    print(f"  Score 범위: [{scores_scaled.min().item():.2f}, {scores_scaled.max().item():.2f}]")

    # Softmax 비교
    softmax_unscaled = F.softmax(scores_unscaled, dim=-1)
    softmax_scaled = F.softmax(scores_scaled, dim=-1)

    print(f"\n[Softmax 출력 비교]")
    print(f"  스케일링 전 엔트로피: {-(softmax_unscaled * torch.log(softmax_unscaled + 1e-9)).sum(dim=-1).mean().item():.4f}")
    print(f"  스케일링 후 엔트로피: {-(softmax_scaled * torch.log(softmax_scaled + 1e-9)).sum(dim=-1).mean().item():.4f}")
    print(f"  (엔트로피가 높을수록 분포가 균일)")

    print("\n" + "=" * 60)
    print("Causal Mask (Look-ahead Mask) 예시")
    print("=" * 60)

    seq_len = 5
    # Causal mask: 하삼각 행렬 (미래 토큰 차단)
    causal_mask = torch.tril(torch.ones(seq_len, seq_len))

    print(f"\n[Causal Mask]")
    print(causal_mask)

    # 마스크 적용된 Attention
    x_small = torch.randn(1, seq_len, d_model)
    self_attn_small = SelfAttention(d_model, d_k=d_k, d_v=d_k)

    # 마스크 확장 (batch 차원 추가)
    mask = causal_mask.unsqueeze(0)
    output_masked, attn_weights_masked = self_attn_small(x_small, mask=mask)

    print(f"\n[Masked Attention Weights]")
    print(f"  첫 번째 행 (첫 토큰): {attn_weights_masked[0, 0].tolist()}")
    print(f"  두 번째 행 (둘째 토큰): {attn_weights_masked[0, 1].tolist()}")
    print(f"  마지막 행 (마지막 토큰): {attn_weights_masked[0, -1].tolist()}")
    print("  → 미래 위치(0.0)에는 attention이 적용되지 않음")


if __name__ == "__main__":
    main()
