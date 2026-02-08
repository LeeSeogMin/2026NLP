"""
3장 실습 2: Attention 메커니즘 구현

이 코드는 다음을 실습한다:
1. Scaled Dot-Product Attention 직접 구현
2. Attention Weight 계산 및 해석
3. Self-Attention 모듈 구현
4. Multi-Head Attention 구현
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

torch.manual_seed(42)
np.random.seed(42)


# ============================================================
# 1. Scaled Dot-Product Attention (NumPy)
# ============================================================
print("=" * 60)
print("1. Scaled Dot-Product Attention (NumPy 구현)")
print("=" * 60)

# 예시: 4개 단어, 임베딩 차원 8
seq_len = 4
d_k = 8  # Key 차원

# 임의의 Q, K, V 행렬 생성
np.random.seed(42)
Q = np.random.randn(seq_len, d_k)
K = np.random.randn(seq_len, d_k)
V = np.random.randn(seq_len, d_k)

print(f"Q shape: {Q.shape}")
print(f"K shape: {K.shape}")
print(f"V shape: {V.shape}")

# 단계 1: QKᵀ 계산
scores = Q @ K.T
print(f"\n[단계 1] QKᵀ (스케일링 전):")
print(np.round(scores, 3))

# 단계 2: √dₖ로 스케일링
scale = math.sqrt(d_k)
scaled_scores = scores / scale
print(f"\n[단계 2] QKᵀ / √{d_k} = QKᵀ / {scale:.2f} (스케일링 후):")
print(np.round(scaled_scores, 3))

# 단계 3: Softmax 적용
def softmax(x):
    """안정적인 softmax 구현"""
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

attention_weights = softmax(scaled_scores)
print(f"\n[단계 3] Attention Weights (softmax 후):")
print(np.round(attention_weights, 3))
print(f"각 행의 합: {np.round(attention_weights.sum(axis=1), 3)}")

# 단계 4: Attention Weights × V
output = attention_weights @ V
print(f"\n[단계 4] Attention Output (Weights × V):")
print(f"Output shape: {output.shape}")
print(np.round(output[:2], 3))

# √dₖ 스케일링의 효과 비교
print(f"\n[참고] √dₖ 스케일링의 효과:")
print(f"  스케일링 전 scores 분산: {scores.var():.3f}")
print(f"  스케일링 후 scores 분산: {scaled_scores.var():.3f}")
print(f"  기대 분산 (이론값): ~1.0")


# ============================================================
# 2. Scaled Dot-Product Attention (PyTorch)
# ============================================================
print("\n" + "=" * 60)
print("2. Scaled Dot-Product Attention (PyTorch 구현)")
print("=" * 60)


def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Scaled Dot-Product Attention

    Args:
        Q: Query (batch, seq_len, d_k)
        K: Key   (batch, seq_len, d_k)
        V: Value (batch, seq_len, d_v)
        mask: 마스크 텐서 (선택)

    Returns:
        output: Attention 출력
        weights: Attention 가중치
    """
    d_k = Q.size(-1)

    # QKᵀ / √dₖ
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    # 마스크 적용 (선택)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))

    # Softmax
    weights = F.softmax(scores, dim=-1)

    # Weights × V
    output = torch.matmul(weights, V)

    return output, weights


# 테스트: "나는 은행에서 돈을 찾았다"
words = ["나는", "은행에서", "돈을", "찾았다"]
seq_len = len(words)
d_model = 8

# 임의의 임베딩 (실제로는 학습된 임베딩 사용)
torch.manual_seed(42)
X = torch.randn(1, seq_len, d_model)  # (batch=1, seq=4, d=8)

# Self-Attention: Q = K = V = X
output, weights = scaled_dot_product_attention(X, X, X)

print(f"입력 문장: {' '.join(words)}")
print(f"\nAttention Weights:")
w = weights[0].detach().numpy()
for i, word in enumerate(words):
    scores_str = "  ".join(f"{w[i][j]:.3f}" for j in range(seq_len))
    print(f"  {word:<8} → [{scores_str}]")

print(f"\n해석: 각 행은 해당 단어가 다른 단어에 얼마나 주목하는지를 나타낸다.")


# ============================================================
# 3. Self-Attention 모듈 (PyTorch nn.Module)
# ============================================================
print("\n" + "=" * 60)
print("3. Self-Attention 모듈 구현")
print("=" * 60)


class SelfAttention(nn.Module):
    """Self-Attention 모듈"""

    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        # Q, K, V 변환 행렬
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, mask=None):
        """
        Args:
            x: 입력 텐서 (batch, seq_len, d_model)
            mask: 마스크 텐서 (선택)
        Returns:
            output: Attention 출력 (batch, seq_len, d_model)
            weights: Attention 가중치 (batch, seq_len, seq_len)
        """
        Q = self.W_q(x)  # (batch, seq, d_model)
        K = self.W_k(x)  # (batch, seq, d_model)
        V = self.W_v(x)  # (batch, seq, d_model)

        output, weights = scaled_dot_product_attention(Q, K, V, mask)
        return output, weights


# Self-Attention 테스트
d_model = 16
self_attn = SelfAttention(d_model)

# 입력: 5개 단어, 임베딩 차원 16
X = torch.randn(1, 5, d_model)
output, weights = self_attn(X)

print(f"입력 크기:   {X.shape}")
print(f"출력 크기:   {output.shape}")
print(f"가중치 크기: {weights.shape}")
print(f"학습 파라미터 수: {sum(p.numel() for p in self_attn.parameters()):,}")


# ============================================================
# 4. Multi-Head Attention 구현
# ============================================================
print("\n" + "=" * 60)
print("4. Multi-Head Attention 구현")
print("=" * 60)


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention 모듈"""

    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model은 num_heads로 나누어떨어져야 한다"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 각 Head의 차원

        # Q, K, V 변환 행렬
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)

        # 출력 변환 행렬
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, mask=None):
        """
        Args:
            x: 입력 텐서 (batch, seq_len, d_model)
        Returns:
            output: 출력 (batch, seq_len, d_model)
            weights: 각 Head의 Attention 가중치
        """
        batch_size, seq_len, _ = x.shape

        # Q, K, V 계산
        Q = self.W_q(x)  # (batch, seq, d_model)
        K = self.W_k(x)
        V = self.W_v(x)

        # Multi-Head로 분할: (batch, seq, d_model) → (batch, heads, seq, d_k)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # 각 Head에서 Attention 계산
        d_k = self.d_k
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        weights = F.softmax(scores, dim=-1)  # (batch, heads, seq, seq)
        attn_output = torch.matmul(weights, V)  # (batch, heads, seq, d_k)

        # Head 합치기: (batch, heads, seq, d_k) → (batch, seq, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.d_model)

        # 출력 변환
        output = self.W_o(attn_output)

        return output, weights


# Multi-Head Attention 테스트
d_model = 32
num_heads = 4
mha = MultiHeadAttention(d_model, num_heads)

X = torch.randn(1, 6, d_model)  # 6개 단어
output, weights = mha(X)

print(f"d_model = {d_model}, num_heads = {num_heads}, d_k = {d_model // num_heads}")
print(f"입력 크기:   {X.shape}")
print(f"출력 크기:   {output.shape}")
print(f"가중치 크기: {weights.shape}  (batch, heads, seq, seq)")
print(f"학습 파라미터 수: {sum(p.numel() for p in mha.parameters()):,}")

# 각 Head가 다른 패턴을 포착하는지 확인
print(f"\n각 Head의 Attention 패턴 (첫 번째 단어 기준):")
for h in range(num_heads):
    w = weights[0, h, 0].detach().numpy()
    pattern = "  ".join(f"{v:.3f}" for v in w)
    max_idx = w.argmax()
    print(f"  Head {h}: [{pattern}]  → 최대 주목: 단어 {max_idx}")


# ============================================================
# 5. Causal Mask (미래 토큰 마스킹)
# ============================================================
print("\n" + "=" * 60)
print("5. Causal Mask (Decoder용)")
print("=" * 60)

seq_len = 5

# 상삼각 마스크: 미래 위치를 0으로
causal_mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)
print("Causal Mask (1=참조 가능, 0=마스킹):")
print(causal_mask[0, 0].int().numpy())

# Causal Attention 테스트
X = torch.randn(1, seq_len, d_model)
output_causal, weights_causal = mha(X, mask=causal_mask)

print(f"\nCausal Attention Weights (Head 0):")
w = weights_causal[0, 0].detach().numpy()
for i in range(seq_len):
    row = "  ".join(f"{w[i][j]:.3f}" for j in range(seq_len))
    print(f"  단어 {i}: [{row}]")

print("\n해석: 각 단어는 자신과 이전 단어만 참조할 수 있다 (미래 단어는 0).")

print("\n실습 2 완료!")
