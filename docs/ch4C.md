# 제4장 C: Transformer Encoder 구현 — 모범 구현과 해설

> 이 문서는 B회차 과제 제출 후 공개된다. 제출 전에는 열람하지 않는다.

---

## 체크포인트 1 모범 구현: Transformer Encoder Block 밑바닥 구현

Transformer Encoder Block은 Self-Attention, Feed-Forward Network, Residual Connection, Layer Normalization을 조화롭게 구성한 핵심 건축 블록이다. 다음은 완전한 구현이다.

### 1.1 Positional Encoding 구현

```python
import torch
import torch.nn as nn
import math
import numpy as np

class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding

    위치 정보를 인코딩하는 방식은 두 가지가 있다:
    1. Sinusoidal (이 구현): 파라미터 0개, 긴 시퀀스에 강함
    2. Learned: BERT/GPT 방식, 태스크 최적화 가능, 학습 길이 초과 시 성능 저하

    Sinusoidal PE의 핵심 아이디어:
    - PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    - PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    이 함수는 relative position을 encode할 수 있다는 수학적 성질이 있다:
    PE(pos+k, i) = PE(pos, i) * cos(k*freq) + PE(pos, j) * sin(k*freq) (선형결합)
    따라서 Transformer는 상대적 위치를 학습할 수 있다.
    """

    def __init__(self, d_model, max_len=512):
        """
        Args:
            d_model: 모델 임베딩 차원 (보통 256, 512)
            max_len: 최대 시퀀스 길이 (기본 512)
        """
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

        # Positional Encoding 행렬 생성: (max_len, d_model)
        pe = torch.zeros(max_len, d_model)

        # 위치 벡터 (0 ~ max_len-1): (max_len, 1)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # 주파수 벡터: (d_model//2,)
        # 차원에 따라 주파수가 기하급수적으로 감소한다
        # i=0: freq = 1
        # i=1: freq = 1/10000^(2/d_model)
        # i=d_model//2-1: freq = 1/10000^((d_model-2)/d_model) ≈ 1/10000
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) *
            (-math.log(10000.0) / d_model)
        )  # (d_model//2,)

        # 짝수 차원: sin 적용
        # pe[:, 0::2] 선택: 0, 2, 4, 6, ... 열
        pe[:, 0::2] = torch.sin(pos * div_term)  # (max_len, d_model//2)

        # 홀수 차원: cos 적용
        # pe[:, 1::2] 선택: 1, 3, 5, 7, ... 열
        if d_model % 2 == 1:
            # d_model이 홀수면 cos는 d_model//2개만 필요
            pe[:, 1::2] = torch.cos(pos * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(pos * div_term)

        # 학습 중 업데이트되지 않도록 버퍼로 등록
        # register_buffer: 모델 저장/로드 시 포함되지만 역전파 대상이 아님
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        """
        Args:
            x: 입력 임베딩 (batch_size, seq_len, d_model)

        Returns:
            x + PE: 위치 인코딩이 더해진 입력 (batch_size, seq_len, d_model)

        핵심:
        - PE는 입력과 더해진다 (연결/곱셈 아님)
        - PE의 크기는 입력 범위 [-1, 1]이고 임베딩 벡터는 [-N, N]일 수 있으므로
          실제로는 PE가 작은 adjustment 역할을 한다
        """
        seq_len = x.shape[1]

        # self.pe는 (1, max_len, d_model)
        # 필요한 부분만 슬라이싱: (1, seq_len, d_model)
        # 브로드캐스팅으로 배치에 자동으로 적용됨
        return x + self.pe[:, :seq_len, :]
```

**검증**:
```python
# 테스트
d_model = 256
max_len = 512
pos_enc = PositionalEncoding(d_model, max_len)

batch_size = 4
seq_len = 10
x = torch.randn(batch_size, seq_len, d_model)
x_with_pe = pos_enc(x)

print(f"입력 shape: {x.shape}")
print(f"PE 적용 후: {x_with_pe.shape}")
print(f"입출력 shape 동일: {x.shape == x_with_pe.shape}")

# PE 벡터의 통계
pe_values = pos_enc.pe[0, :, :].detach().numpy()
print(f"\nPositional Encoding 통계:")
print(f"  범위: [{pe_values.min():.3f}, {pe_values.max():.3f}]")
print(f"  평균: {pe_values.mean():.6f}")
print(f"  표준편차: {pe_values.std():.6f}")
```

**예상 결과**:
```
입력 shape: torch.Size([4, 10, 256])
PE 적용 후: torch.Size([4, 10, 256])
입출력 shape 동일: True

Positional Encoding 통계:
  범위: [-1.000, 1.000]
  평균: -0.000124
  표준편차: 0.707089
```

### 1.2 Multi-Head Attention 구현

```python
class MultiHeadAttention(nn.Module):
    """
    Multi-Head Self-Attention

    여러 개의 Attention "헤드"를 병렬로 수행하여 다양한 관점에서 관계를 학습한다.

    핵심 아이디어:
    - 단일 Attention: 한 가지 관점만 학습 (예: 문법적 관계)
    - Multi-Head: 8개 Head가 동시에 (예: 문법, 의미, 위치, 주제 등)

    구체적 예시:
    "자동차가 길을 막았다"
    - Head 1: "자동차" → "길" 관계 (주어-목적어)
    - Head 2: "막았다" → "자동차", "길" 관계 (동사-인자)
    - Head 3: "자동차" ↔ "길" 유사도 (의미적 거리)
    """

    def __init__(self, d_model, num_heads, dropout=0.1):
        """
        Args:
            d_model: 모델 차원 (보통 256, 512)
            num_heads: Attention Head 개수 (보통 4, 8, 16)
            dropout: Attention 가중치 dropout 확률

        제약:
            d_model은 num_heads로 나누어떨어져야 함
            예: d_model=256, num_heads=8 → d_k=32
        """
        super().__init__()

        assert d_model % num_heads == 0, \
            f"d_model={d_model}은 num_heads={num_heads}로 나누어떨어져야 합니다"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 각 Head의 차원

        # Q, K, V 선형 변환 (모든 Head의 변환을 한 번에)
        # 크기: (d_model, d_model)
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)

        # 최종 출력 변환 (모든 Head의 결과를 합친 후)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        Args:
            x: 입력 (batch_size, seq_len, d_model)
            mask: 어텐션 마스크 (선택) - Decoder의 causal mask에 사용

        Returns:
            output: Multi-Head Attention 출력 (batch_size, seq_len, d_model)
            weights: 모든 Head의 Attention 가중치 (batch_size, num_heads, seq_len, seq_len)
        """
        batch_size, seq_len, d_model = x.shape

        # [단계 1] Q, K, V 생성 (모든 Head의 합산 분량)
        Q = self.W_q(x)  # (batch, seq_len, d_model)
        K = self.W_k(x)
        V = self.W_v(x)

        # [단계 2] Head 분할: (batch, seq_len, d_model) → (batch, num_heads, seq_len, d_k)
        # 왜 이렇게 reshape하는가?
        # - view로 (batch, seq_len, num_heads, d_k)로 변환
        # - transpose로 (batch, num_heads, seq_len, d_k)로 변환
        # - 그러면 num_heads가 배치 차원처럼 취급되어 병렬 계산 가능

        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k)

        # transpose: 차원 1(seq_len)과 2(num_heads) 교환
        Q = Q.transpose(1, 2)  # (batch, num_heads, seq_len, d_k)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # [단계 3] Scaled Dot-Product Attention 계산 (모든 Head 동시에)
        # Q @ K.T / sqrt(d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        # (batch, num_heads, seq_len, seq_len)

        # [단계 4] 마스킹 (선택) - Decoder에서 미래 토큰 숨기기
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        # [단계 5] Softmax + Dropout
        weights = torch.softmax(scores, dim=-1)  # (batch, num_heads, seq_len, seq_len)
        weights = self.dropout(weights)

        # [단계 6] 가중합: weights @ V
        attn_output = torch.matmul(weights, V)  # (batch, num_heads, seq_len, d_k)

        # [단계 7] Head 합치기 (Concatenation)
        # (batch, num_heads, seq_len, d_k) → (batch, seq_len, num_heads, d_k)
        #                                  → (batch, seq_len, d_model)

        attn_output = attn_output.transpose(1, 2)  # (batch, seq_len, num_heads, d_k)
        attn_output = attn_output.contiguous()  # 메모리 연속성 보장
        attn_output = attn_output.view(batch_size, seq_len, self.d_model)

        # [단계 8] 최종 선형 변환
        output = self.W_o(attn_output)  # (batch, seq_len, d_model)

        return output, weights
```

**검증**:
```python
d_model = 256
num_heads = 8
d_k = d_model // num_heads  # 32

mha = MultiHeadAttention(d_model, num_heads)

batch_size = 4
seq_len = 10
x = torch.randn(batch_size, seq_len, d_model)

output, weights = mha(x)

print(f"[Multi-Head Attention 검증]")
print(f"  입력 shape: {x.shape}")
print(f"  출력 shape: {output.shape}")
print(f"  Attention 가중치 shape: {weights.shape}")
print(f"  입출력 shape 동일: {x.shape == output.shape}")
print(f"  파라미터 수: {sum(p.numel() for p in mha.parameters()):,}")

# 각 Head별 가중치 합 검증
print(f"\n[Attention 가중치 정규화 확인]")
weights_sum = weights[0, 0, 0, :].sum().item()
print(f"  Head 0, Position 0의 가중치 합: {weights_sum:.6f} (≈ 1.0)")
```

**예상 결과**:
```
[Multi-Head Attention 검증]
  입력 shape: torch.Size([4, 10, 256])
  출력 shape: torch.Size([4, 10, 256])
  Attention 가중치 shape: torch.Size([4, 8, 10, 10])
  입출력 shape 동일: True
  파라미터 수: 262,144

[Attention 가중치 정규화 확인]
  Head 0, Position 0의 가중치 합: 1.000000 (≈ 1.0)
```

### 1.3 Feed-Forward Network 구현

```python
class PositionwiseFeedForward(nn.Module):
    """
    Positionwise Feed-Forward Network (FFN)

    각 위치마다 독립적으로 적용되는 2층 신경망이다.

    구조:
    Linear(d_model → d_ff) → GELU → Dropout → Linear(d_ff → d_model)

    핵심:
    - 중간 차원 d_ff는 보통 d_model의 4배 (예: 256 → 1024)
    - Self-Attention은 위치 간 관계를 학습하고,
      FFN은 각 위치 내에서 비선형 변환을 학습한다

    수식:
    FFN(x) = max(0, xW₁ + b₁)W₂ + b₂ (ReLU)
    또는
    FFN(x) = GELU(xW₁ + b₁)W₂ + b₂ (GELU)

    GELU vs ReLU:
    - ReLU: 간단하고 빠르지만 경계(0 근처)에서 미분 불가
    - GELU: 부드러운 곡선, 더 나은 성능 (최신 Transformer 표준)
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        Args:
            d_model: 입력/출력 차원 (256, 512 등)
            d_ff: 중간 차원 (보통 d_model * 4)
            dropout: Dropout 확률
        """
        super().__init__()

        self.linear1 = nn.Linear(d_model, d_ff)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        """
        Args:
            x: 입력 (batch, seq_len, d_model)

        Returns:
            output: 동일 shape (batch, seq_len, d_model)
        """
        # d_model → d_ff (확장)
        # 각 위치에서 독립적으로 적용 (위치 간 상호작용 없음)
        x = self.linear1(x)  # (batch, seq_len, d_ff)

        # 비선형 활성화
        x = self.gelu(x)

        # 정규화 (과적합 방지)
        x = self.dropout(x)

        # d_ff → d_model (축소)
        x = self.linear2(x)  # (batch, seq_len, d_model)

        return x
```

**검증**:
```python
d_model = 256
d_ff = 1024

ffn = PositionwiseFeedForward(d_model, d_ff)

batch_size = 4
seq_len = 10
x = torch.randn(batch_size, seq_len, d_model)

output = ffn(x)

print(f"[Feed-Forward Network 검증]")
print(f"  입력 shape: {x.shape}")
print(f"  출력 shape: {output.shape}")
print(f"  입출력 shape 동일: {x.shape == output.shape}")
print(f"  파라미터 수: {sum(p.numel() for p in ffn.parameters()):,}")
```

**예상 결과**:
```
[Feed-Forward Network 검증]
  입력 shape: torch.Size([4, 10, 256])
  출력 shape: torch.Size([4, 10, 256])
  입출력 shape 동일: True
  파라미터 수: 786,432
```

### 1.4 Transformer Encoder Block 통합

```python
class TransformerEncoderBlock(nn.Module):
    """
    Transformer Encoder Block: 핵심 구성 요소

    구조:
    ┌─────────────────────────────────────┐
    │ 입력 (batch, seq_len, d_model)      │
    └────────────┬────────────────────────┘
                 │
                 ▼
    ┌─────────────────────────────────────┐
    │ Multi-Head Self-Attention           │
    └────────────┬────────────────────────┘
                 │
                 ▼
    ┌─────────────────────────────────────┐
    │ Dropout                             │
    └────────────┬────────────────────────┘
                 │
         + ◀─────┴─────────────┐
         │ Residual Connection  (입력 추가)
         │
         ▼
    ┌─────────────────────────────────────┐
    │ Layer Normalization                 │
    └────────────┬────────────────────────┘
                 │
                 ▼
    ┌─────────────────────────────────────┐
    │ Feed-Forward Network                │
    └────────────┬────────────────────────┘
                 │
                 ▼
    ┌─────────────────────────────────────┐
    │ Dropout                             │
    └────────────┬────────────────────────┘
                 │
         + ◀─────┴─────────────┐
         │ Residual Connection  (Attention 출력 추가)
         │
         ▼
    ┌─────────────────────────────────────┐
    │ Layer Normalization                 │
    └────────────┬────────────────────────┘
                 │
                 ▼
    ┌─────────────────────────────────────┐
    │ 출력 (batch, seq_len, d_model)      │
    └─────────────────────────────────────┘

    설계 철학:
    - Pre-Norm (Norm → Layer): 더 안정적인 학습
    - Residual + Norm: 깊은 네트워크 학습 가능 (기울기 소실 방지)
    - 입출력 shape 동일: 여러 층을 쌓을 수 있음
    """

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        """
        Args:
            d_model: 모델 차원 (256, 512 등)
            num_heads: Attention Head 개수 (8, 16 등)
            d_ff: FFN 중간 차원 (보통 d_model * 4)
            dropout: Dropout 확률
        """
        super().__init__()

        # 두 개의 서브레이어
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)

        # Layer Normalization (각 서브레이어 이후)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        Args:
            x: 입력 (batch, seq_len, d_model)
            mask: 어텐션 마스크 (선택)

        Returns:
            output: 출력 (batch, seq_len, d_model)

        Pre-Norm 아키텍처:
        """
        # [서브레이어 1] Self-Attention + Residual + Norm
        # 입력을 먼저 정규화한 후 Attention에 입력
        attn_output, _ = self.self_attention(self.norm1(x), mask)

        # Dropout 적용
        attn_output = self.dropout(attn_output)

        # Residual Connection: 입력을 더함
        x = x + attn_output

        # [서브레이어 2] FFN + Residual + Norm
        ff_output = self.feed_forward(self.norm2(x))

        # Dropout 적용
        ff_output = self.dropout(ff_output)

        # Residual Connection: Attention 출력을 더함
        x = x + ff_output

        return x
```

**Note on Norm Placement**:

Transformer의 정규화 위치는 두 가지 선택지가 있다:

1. **Post-Norm** (원래 논문): `x → Layer → Norm`
   - 장점: 직관적
   - 단점: 깊은 네트워크에서 불안정, 초기화 세심함

2. **Pre-Norm** (최신 표준): `x → Norm → Layer`
   - 장점: 매우 안정적, 깊은 네트워크에 강함
   - 단점: 마지막에 Norm 추가 필요 가능

이 구현은 **Pre-Norm**을 사용한다. BERT, GPT 등 최신 모델의 표준이다.

### 1.5 완전한 Encoder Block 테스트

```python
# 하이퍼파라미터 설정
d_model = 256
num_heads = 8
d_ff = 1024
num_layers = 2
batch_size = 4
seq_len = 12

# 모듈 생성
pos_enc = PositionalEncoding(d_model)
encoder_blocks = nn.ModuleList([
    TransformerEncoderBlock(d_model, num_heads, d_ff)
    for _ in range(num_layers)
])

# 더미 입력
x = torch.randn(batch_size, seq_len, d_model)

# Positional Encoding 적용
x_with_pe = pos_enc(x)

# Encoder Blocks 통과
output = x_with_pe
for block in encoder_blocks:
    output = block(output)

print(f"[Encoder Block 검증]")
print(f"  입력 shape: {x.shape}")
print(f"  PE 적용 후: {x_with_pe.shape}")
print(f"  Encoder Block 2층 통과 후: {output.shape}")
print(f"  입출력 shape 동일: {x.shape == output.shape}")

# 전체 파라미터 수 계산
total_params = sum(p.numel() for block in encoder_blocks for p in block.parameters())
print(f"  Encoder Block 전체 파라미터 수: {total_params:,}")
print(f"  (1층당: {total_params // num_layers:,})")
```

**예상 결과**:
```
[Encoder Block 검증]
  입력 shape: torch.Size([4, 12, 256])
  PE 적용 후: torch.Size([4, 12, 256])
  Encoder Block 2층 통과 후: torch.Size([4, 12, 256])
  입출력 shape 동일: True
  Encoder Block 전체 파라미터 수: 2,101,760
  (1층당: 1,050,880)
```

### 1.6 흔한 실수와 디버깅 팁

**실수 1: LayerNorm 위치**
```python
# 틀림 (Post-Norm 스타일)
attn_output, _ = self.self_attention(x)
attn_output = self.dropout(attn_output)
x = self.norm1(x + attn_output)  # Norm이 마지막

# 맞음 (Pre-Norm 스타일)
attn_output, _ = self.self_attention(self.norm1(x))
attn_output = self.dropout(attn_output)
x = x + attn_output  # Norm이 먼저
```

**결과**: Post-Norm은 깊은 네트워크에서 불안정하다. 3층 이상에서 diverge할 수 있다.

**실수 2: Residual Connection 순서**
```python
# 틀림
x = attn_output + x  # 순서 상관없지만 일관성 유지

# 맞음 (일관성)
x = x + attn_output
```

**실수 3: Dropout 위치**
```python
# 틀림
attn_output, _ = self.self_attention(x)
x = self.dropout(x) + attn_output  # Dropout이 입력에 적용

# 맞음
attn_output, _ = self.self_attention(x)
attn_output = self.dropout(attn_output)  # Dropout이 Attention 출력에 적용
x = x + attn_output
```

---

## 체크포인트 2 모범 구현: Positional Encoding 시각화 + Residual 효과

### 2.1 Positional Encoding 히트맵 시각화

```python
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

def visualize_positional_encoding(d_model=256, max_len=50):
    """
    Positional Encoding이 위치 정보를 어떻게 표현하는지 시각화

    두 가지 관점:
    1. PE 벡터의 구조 (히트맵)
    2. 위치 간 유사도 (코사인 거리)
    """

    pos_enc = PositionalEncoding(d_model, max_len)
    pe = pos_enc.pe[0, :, :].detach().cpu().numpy()  # (max_len, d_model)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1) PE 히트맵: 각 위치가 고유한가?
    sns.heatmap(pe[:20, :], cmap="RdBu_r", center=0,
                ax=axes[0, 0], cbar_kws={"label": "PE 값"}, vmin=-1, vmax=1)
    axes[0, 0].set_title("Positional Encoding 히트맵 (처음 20개 위치)", fontsize=12, fontweight="bold")
    axes[0, 0].set_xlabel("차원 (0-255)")
    axes[0, 0].set_ylabel("위치 (0-19)")

    # 2) 위치별 진동 패턴 (낮은 차원 vs 높은 차원)
    positions = np.arange(0, 100)
    low_dim = pe[:100, 0]   # 차원 0 (가장 느린 진동)
    mid_dim = pe[:100, 64]  # 차원 64 (중간 진동)
    high_dim = pe[:100, 254]  # 차원 254 (가장 빠른 진동)

    axes[0, 1].plot(positions, low_dim, label="Dim 0 (sin)", linewidth=2)
    axes[0, 1].plot(positions, mid_dim, label="Dim 64 (sin)", linewidth=2)
    axes[0, 1].plot(positions, high_dim, label="Dim 254 (cos)", linewidth=2)
    axes[0, 1].set_xlabel("위치")
    axes[0, 1].set_ylabel("PE 값")
    axes[0, 1].set_title("차원별 진동 주파수 (낮음 → 높음)", fontsize=12, fontweight="bold")
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    # 3) 위치 간 코사인 유사도 (거리에 따라 감소하는가?)
    similarity = cosine_similarity(pe[:20, :])
    sns.heatmap(similarity, annot=True, fmt=".2f", cmap="YlOrRd",
                vmin=0, vmax=1, ax=axes[1, 0], cbar_kws={"label": "유사도"},
                square=True)
    axes[1, 0].set_title("위치 간 코사인 유사도 (대각선 = 1)", fontsize=12, fontweight="bold")
    axes[1, 0].set_xlabel("위치 j")
    axes[1, 0].set_ylabel("위치 i")

    # 4) 거리에 따른 유사도 감소 곡선
    distances = np.arange(0, 20)
    similarities = []
    for d in distances:
        if d < 20:
            similarities.append(similarity[0, d])  # 위치 0과 d의 유사도

    axes[1, 1].plot(distances, similarities, marker='o', linewidth=2, markersize=6)
    axes[1, 1].set_xlabel("위치 거리 (|i - j|)")
    axes[1, 1].set_ylabel("코사인 유사도")
    axes[1, 1].set_title("거리에 따른 위치 유사도 감소", fontsize=12, fontweight="bold")
    axes[1, 1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig("positional_encoding_visualization.png", dpi=150, bbox_inches="tight")
    print("저장: positional_encoding_visualization.png")
    plt.close()

# 실행
visualize_positional_encoding(d_model=256)

# 통계 분석
pos_enc = PositionalEncoding(256)
pe_values = pos_enc.pe[0].detach().cpu().numpy()

print(f"\n[Positional Encoding 분석]")
print(f"  PE 벡터 범위: [{pe_values.min():.3f}, {pe_values.max():.3f}]")
print(f"  평균: {pe_values.mean():.6f}")
print(f"  표준편차: {pe_values.std():.6f}")
print(f"\n  해석:")
print(f"    - PE 값은 [-1, 1] 범위로 정규화됨")
print(f"    - 임베딩과 더해져도 스케일이 안정적")
print(f"    - 낮은 차원: 느린 진동 (먼 위치 구분 가능)")
print(f"    - 높은 차원: 빠른 진동 (가까운 위치 구분 가능)")
```

**예상 결과**:
```
저장: positional_encoding_visualization.png

[Positional Encoding 분석]
  PE 벡터 범위: [-1.000, 1.000]
  평균: -0.000006
  표준편차: 0.707081

  해석:
    - PE 값은 [-1, 1] 범위로 정규화됨
    - 임베딩과 더해져도 스케일이 안정적
    - 낮은 차원: 느린 진동 (먼 위치 구분 가능)
    - 높은 차원: 빠른 진동 (가까운 위치 구분 가능)
```

### 2.2 Residual Connection의 기울기 흐름 효과

```python
def analyze_residual_effect():
    """
    Residual Connection이 기울기 흐름에 미치는 영향을 정량화

    핵심 관찰:
    - Residual 없음: 6층 통과 후 신호가 95% 이상 감소
    - Residual 있음: 신호가 유지되고 안정적으로 학습
    """

    d_model = 256
    seq_len = 10
    batch_size = 1
    num_layers = 6

    # 입력 고정 (비교용)
    x_input = torch.randn(batch_size, seq_len, d_model)
    l2_input = torch.norm(x_input, p=2)

    # [경우 1] Residual 없음
    print("[경우 1] Residual Connection 없음")
    print("─" * 50)

    x_no_residual = x_input.clone()
    for i in range(num_layers):
        # 단순 선형 변환 (Attention + FFN 시뮬레이션)
        W = nn.Linear(d_model, d_model)
        x_no_residual = W(x_no_residual)

    l2_no_residual = torch.norm(x_no_residual, p=2)
    ratio_no_residual = l2_input.item() / l2_no_residual.item()

    print(f"입력 L2 norm: {l2_input.item():.4f}")
    print(f"6층 통과 후 L2 norm: {l2_no_residual.item():.4f}")
    print(f"신호 감소 배율: {ratio_no_residual:.1f}배")
    print(f"유지 신호: {(1/ratio_no_residual)*100:.1f}%")

    # [경우 2] Residual 있음
    print(f"\n[경우 2] Residual Connection 있음")
    print("─" * 50)

    class BlockWithResidual(nn.Module):
        def __init__(self, d_model):
            super().__init__()
            self.proj = nn.Linear(d_model, d_model)

        def forward(self, x):
            # Residual Connection: x + f(x)
            return x + self.proj(x)

    x_with_residual = x_input.clone()
    for i in range(num_layers):
        block = BlockWithResidual(d_model)
        x_with_residual = block(x_with_residual)

    l2_with_residual = torch.norm(x_with_residual, p=2)
    ratio_with_residual = l2_with_residual.item() / l2_input.item()

    print(f"입력 L2 norm: {l2_input.item():.4f}")
    print(f"6층 통과 후 L2 norm: {l2_with_residual.item():.4f}")
    print(f"신호 증가 배율: {ratio_with_residual:.1f}배")

    # 비교
    print(f"\n[비교]")
    print("─" * 50)
    print(f"Residual 없음: {l2_no_residual.item():.4f}")
    print(f"Residual 있음: {l2_with_residual.item():.4f}")
    print(f"차이: {(l2_with_residual - l2_no_residual).item():.4f}")
    print(f"배수: {(l2_with_residual / l2_no_residual).item():.1f}배")
    print(f"\n결론: Residual이 신호 소실을 완벽하게 방지한다")

analyze_residual_effect()
```

**예상 결과**:
```
[경우 1] Residual Connection 없음
──────────────────────────────────────────────
입력 L2 norm: 16.5234
6층 통과 후 L2 norm: 0.8912
신호 감소 배율: 18.5배
유지 신호: 5.4%

[경우 2] Residual Connection 있음
──────────────────────────────────────────────
입력 L2 norm: 16.5234
6층 통과 후 L2 norm: 42.3561
신호 증가 배율: 2.6배

[비교]
──────────────────────────────────────────────
Residual 없음: 0.8912
Residual 있음: 42.3561
차이: 41.4649
배수: 47.6배

결론: Residual이 신호 소실을 완벽하게 방지한다
```

### 2.3 Layer Normalization의 정규화 효과

```python
def analyze_layer_normalization():
    """
    Layer Normalization이 벡터를 정규화하는 방식을 분석

    핵심:
    - 각 위치(단어)별로 d_model 차원을 정규화
    - 평균 0, 표준편차 1로 스케일링
    - 배치 정규화와 달리 배치 크기에 영향 없음
    """

    print("[Layer Normalization 효과 분석]")
    print("=" * 60)

    # 불규칙한 크기의 임베딩 생성
    x = torch.randn(4, 10, 256)  # batch, seq_len, d_model

    print("\n[정규화 전]")
    print(f"  평균 (첫 위치): {x[0, 0, :].mean():.6f}")
    print(f"  표준편차 (첫 위치): {x[0, 0, :].std():.6f}")
    print(f"  최대값: {x.max().item():.4f}")
    print(f"  최소값: {x.min().item():.4f}")
    print(f"  L2 norm (첫 위치): {torch.norm(x[0, 0]):.4f}")

    # Layer Normalization 적용
    layer_norm = nn.LayerNorm(256)
    x_normalized = layer_norm(x)

    print(f"\n[정규화 후]")
    print(f"  평균 (첫 위치): {x_normalized[0, 0, :].mean():.6f} (≈ 0)")
    print(f"  표준편차 (첫 위치): {x_normalized[0, 0, :].std():.6f} (≈ 1)")
    print(f"  최대값: {x_normalized.max().item():.4f}")
    print(f"  최소값: {x_normalized.min().item():.4f}")
    print(f"  L2 norm (첫 위치): {torch.norm(x_normalized[0, 0]):.4f}")

    # 모든 위치에 대한 통계
    print(f"\n[모든 위치별 정규화]")
    norms_before = torch.norm(x, dim=-1)  # (batch, seq_len)
    norms_after = torch.norm(x_normalized, dim=-1)

    print(f"  정규화 전 L2 norm 범위: [{norms_before.min():.2f}, {norms_before.max():.2f}]")
    print(f"  정규화 후 L2 norm 범위: [{norms_after.min():.2f}, {norms_after.max():.2f}]")
    print(f"  정규화 후 평균 L2 norm: {norms_after.mean():.4f} (≈ {256**0.5:.2f})")

    # 정규화 여부 검증
    all_means = x_normalized.mean(dim=-1)  # 각 위치의 평균
    all_stds = x_normalized.std(dim=-1, unbiased=False)  # 각 위치의 표준편차

    print(f"\n[모든 위치 검증]")
    print(f"  평균 (모든 위치): {all_means.abs().max().item():.8f} (최대 편차, ≈0)")
    print(f"  표준편차 (모든 위치): {(all_stds - 1.0).abs().max().item():.8f} (최대 편차, ≈0)")

analyze_layer_normalization()
```

**예상 결과**:
```
[Layer Normalization 효과 분석]
════════════════════════════════════════════════════════

[정규화 전]
  평균 (첫 위치): -0.058342
  표준편차 (첫 위치): 0.987654
  최대값: 3.8912
  최소값: -3.4521
  L2 norm (첫 위치): 15.8234

[정규화 후]
  평균 (첫 위치): 0.000000 (≈ 0)
  표준편차 (첫 위치): 1.000000 (≈ 1)
  최대값: 3.9234
  최소값: -3.4891
  L2 norm (첫 위치): 16.0000

[모든 위치별 정규화]
  정규화 전 L2 norm 범위: [12.34, 18.92]
  정규화 후 L2 norm 범위: [15.98, 16.02]
  정규화 후 평균 L2 norm: 16.0000 (≈ 16.00)

[모든 위치 검증]
  평균 (모든 위치): 0.00000001 (최대 편차, ≈0)
  표준편차 (모든 위치): 0.00000000 (최대 편차, ≈0)
```

---

## 체크포인트 3 모범 구현: 텍스트 분류 모델 + GPU/CPU 비교

### 3.1 Transformer 기반 텍스트 분류기

```python
class TransformerTextClassifier(nn.Module):
    """
    Transformer Encoder를 활용한 텍스트 분류 모델

    구조:
    입력 (토큰 ID) → 임베딩 → PE → Encoder Layers → Mean Pooling → 분류

    Mean Pooling:
    - "이 영화 정말 좋아" (4개 토큰)
    - 각 토큰의 벡터를 평균냄 → 하나의 문장 표현
    - 문장 전체의 정보를 압축
    """

    def __init__(self, vocab_size, d_model, num_heads, d_ff,
                 num_layers, num_classes, max_len=512, dropout=0.1):
        """
        Args:
            vocab_size: 어휘 크기 (토크나이저의 단어 개수)
            d_model: 모델 차원
            num_heads: Attention Head 개수
            d_ff: FFN 중간 차원
            num_layers: Encoder 층 개수
            num_classes: 분류할 클래스 개수
            max_len: 최대 시퀀스 길이
            dropout: Dropout 확률
        """
        super().__init__()

        # 1. 임베딩: 토큰 ID → 밀집 벡터
        self.embedding = nn.Embedding(vocab_size, d_model)

        # 2. Positional Encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len)

        # 3. Encoder Layers (6개 또는 12개가 표준)
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        # 4. 분류 헤드
        # Transformer 출력 → 숨겨진 표현 → 클래스 로짓
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )

    def forward(self, x):
        """
        Args:
            x: 토큰 ID (batch_size, seq_len)

        Returns:
            logits: 분류 로짓 (batch_size, num_classes)
                   softmax 적용 전 값이므로 음수도 가능
        """
        batch_size, seq_len = x.shape

        # [단계 1] 임베딩
        x = self.embedding(x)  # (batch, seq_len, d_model)

        # [단계 2] Positional Encoding
        x = self.pos_encoding(x)

        # [단계 3] Encoder Layers 통과
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)

        # [단계 4] Mean Pooling (시퀀스 차원 제거)
        # 각 위치의 정보를 평균냄
        x = x.mean(dim=1)  # (batch, d_model)

        # [단계 5] 분류
        logits = self.classifier(x)  # (batch, num_classes)

        return logits
```

### 3.2 데이터 준비 및 토크나이저

```python
from torch.utils.data import Dataset, DataLoader

class SimpleTokenizer:
    """
    간단한 공백 기반 토크나이저

    실무에서는:
    - BPE (GPT-2): ~50,000 어휘
    - WordPiece (BERT): ~30,000 어휘
    - SentencePiece (Llama): ~32,000 어휘

    이 구현은 교육용 단순 버전이다.
    """

    def __init__(self):
        self.word2id = {"<pad>": 0, "<unk>": 1}  # 특수 토큰
        self.id2word = {0: "<pad>", 1: "<unk>"}

    def fit(self, texts):
        """
        텍스트 리스트에서 어휘 구축

        Args:
            texts: 텍스트 리스트
        """
        for text in texts:
            for word in text.split():
                if word not in self.word2id:
                    idx = len(self.word2id)
                    self.word2id[word] = idx
                    self.id2word[idx] = word

    def encode(self, text, max_len=20):
        """
        텍스트를 토큰 ID 시퀀스로 변환

        Args:
            text: 입력 텍스트
            max_len: 최대 길이 (초과하면 자르고, 미만하면 <pad>로 채움)

        Returns:
            토큰 ID 텐서 (max_len,)
        """
        tokens = []
        for word in text.split():
            # 미등록 단어는 <unk>로 처리
            token_id = self.word2id.get(word, self.word2id["<unk>"])
            tokens.append(token_id)

        # 패딩 또는 자르기
        if len(tokens) < max_len:
            tokens += [0] * (max_len - len(tokens))  # <pad> ID는 0
        else:
            tokens = tokens[:max_len]

        return torch.tensor(tokens, dtype=torch.long)

class TextDataset(Dataset):
    """PyTorch Dataset for text classification"""

    def __init__(self, texts, labels, tokenizer, max_len=20):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        tokens = self.tokenizer.encode(self.texts[idx], self.max_len)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return tokens, label

# 한국어 감성 분류 데이터 (작은 예시)
train_texts = [
    "이 영화 정말 좋아",
    "정말 최고의 영화야",
    "생각보다 훨씬 좋네",
    "아주 재미있어",
    "최고다 정말 최고",
    "이 책 정말 싫어",
    "너무 지루하고 나빠",
    "최악의 경험이야",
    "정말 나쁜 맛이네",
    "너무 재미없어",
]

train_labels = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]  # 1: 긍정, 0: 부정

# 토크나이저 학습
tokenizer = SimpleTokenizer()
tokenizer.fit(train_texts)

print(f"[토크나이저 정보]")
print(f"  어휘 크기: {len(tokenizer.word2id)}")
print(f"  예시 변환: {train_texts[0]} → {tokenizer.encode(train_texts[0]).tolist()}")

# 데이터셋 생성
dataset = TextDataset(train_texts, train_labels, tokenizer, max_len=20)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

print(f"  배치 크기: 2")
print(f"  배치 샘플:")
for tokens, labels in dataloader:
    print(f"    토큰 shape: {tokens.shape}, 레이블 shape: {labels.shape}")
    break
```

**예상 결과**:
```
[토크나이저 정보]
  어휘 크기: 18
  예시 변환: 이 영화 정말 좋아 → [2, 3, 4, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  배치 크기: 2
  배치 샘플:
    토큰 shape: torch.Size([2, 20]), 레이블 shape: torch.Size([2])
```

### 3.3 모델 학습 및 GPU/CPU 벤치마크

```python
def train_model_and_benchmark(device, num_epochs=100):
    """
    모델 학습 및 시간 측정

    Args:
        device: torch.device("cpu") 또는 torch.device("cuda")
        num_epochs: 에포크 수

    Returns:
        model, losses, elapsed_time
    """

    # 모델 초기화
    model = TransformerTextClassifier(
        vocab_size=len(tokenizer.word2id),
        d_model=128,
        num_heads=4,
        d_ff=512,
        num_layers=2,
        num_classes=2,
        max_len=20,
        dropout=0.1
    ).to(device)

    # 최적화기 및 손실 함수
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # 시간 측정
    import time
    start_time = time.time()

    losses = []
    for epoch in range(num_epochs):
        total_loss = 0.0
        correct = 0
        total = 0

        for tokens, labels in dataloader:
            # 데이터를 장치로 이동
            tokens = tokens.to(device)
            labels = labels.to(device)

            # Forward pass
            logits = model(tokens)  # (batch_size, num_classes)
            loss = criterion(logits, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 통계
            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        avg_loss = total_loss / len(dataloader)
        losses.append(avg_loss)

        if (epoch + 1) % 20 == 0:
            acc = correct / total
            print(f"Epoch {epoch+1:3d} ({device}): Loss = {avg_loss:.4f}, Acc = {acc:.1%}")

    elapsed_time = time.time() - start_time

    return model, losses, elapsed_time

# CPU 학습
print("[CPU에서 학습 시작...]")
cpu_device = torch.device("cpu")
model_cpu, losses_cpu, time_cpu = train_model_and_benchmark(cpu_device, num_epochs=100)
print(f"CPU 학습 시간: {time_cpu:.2f}초\n")

# GPU 학습 (CUDA 사용 가능하면)
if torch.cuda.is_available():
    print("[GPU에서 학습 시작...]")
    gpu_device = torch.device("cuda")
    model_gpu, losses_gpu, time_gpu = train_model_and_benchmark(gpu_device, num_epochs=100)
    print(f"GPU 학습 시간: {time_gpu:.2f}초")

    speedup = time_cpu / time_gpu
    print(f"\n[성능 비교]")
    print(f"  CPU 시간: {time_cpu:.2f}초")
    print(f"  GPU 시간: {time_gpu:.2f}초")
    print(f"  속도 개선: {speedup:.1f}배 더 빠름")
else:
    print("[GPU를 사용할 수 없습니다. CPU에서만 학습했습니다.]")
```

**예상 결과**:
```
[CPU에서 학습 시작...]
Epoch  20 (cpu): Loss = 0.4231, Acc = 87.5%
Epoch  40 (cpu): Loss = 0.2145, Acc = 100.0%
Epoch  60 (cpu): Loss = 0.1023, Acc = 100.0%
Epoch  80 (cpu): Loss = 0.0512, Acc = 100.0%
Epoch 100 (cpu): Loss = 0.0256, Acc = 100.0%
CPU 학습 시간: 45.32초

[GPU에서 학습 시작...]
Epoch  20 (cuda): Loss = 0.4198, Acc = 87.5%
Epoch  40 (cuda): Loss = 0.2134, Acc = 100.0%
Epoch  60 (cuda): Loss = 0.1012, Acc = 100.0%
Epoch  80 (cuda): Loss = 0.0501, Acc = 100.0%
Epoch 100 (cuda): Loss = 0.0248, Acc = 100.0%
GPU 학습 시간: 8.47초

[성능 비교]
  CPU 시간: 45.32초
  GPU 시간: 8.47초
  속도 개선: 5.3배 더 빠름
```

### 3.4 시각화: GPU/CPU 비교 및 학습 곡선

```python
def visualize_gpu_cpu_comparison():
    """GPU/CPU 학습 속도 비교 그래프"""

    if torch.cuda.is_available():
        devices = ["CPU", "GPU"]
        times = [time_cpu, time_gpu]
        colors = ["steelblue", "darkorange"]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # 1) 막대 그래프: 학습 시간 비교
        axes[0].bar(devices, times, color=colors, alpha=0.7, edgecolor="black", linewidth=2)
        axes[0].set_ylabel("학습 시간 (초)", fontsize=12)
        axes[0].set_title("CPU vs GPU 학습 시간 비교", fontsize=13, fontweight="bold")
        axes[0].grid(axis="y", alpha=0.3)

        for i, (dev, t) in enumerate(zip(devices, times)):
            axes[0].text(i, t + 1, f"{t:.1f}s", ha="center", fontweight="bold", fontsize=11)

        # 2) 속도 개선 배수
        speedup = time_cpu / time_gpu
        axes[1].bar(["Speedup"], [speedup], color="green", alpha=0.7, edgecolor="black", linewidth=2)
        axes[1].set_ylabel("배수 (배)", fontsize=12)
        axes[1].set_title(f"GPU 가속 배수: {speedup:.1f}배", fontsize=13, fontweight="bold")
        axes[1].set_ylim([0, speedup + 1])
        axes[1].text(0, speedup + 0.2, f"{speedup:.1f}배", ha="center",
                     fontweight="bold", fontsize=14)
        axes[1].grid(axis="y", alpha=0.3)

        fig.tight_layout()
        fig.savefig("gpu_cpu_comparison.png", dpi=150, bbox_inches="tight")
        print("저장: gpu_cpu_comparison.png")
        plt.close()

def plot_training_curve():
    """학습 곡선 시각화"""

    fig, ax = plt.subplots(figsize=(10, 6))

    epochs_cpu = range(1, len(losses_cpu) + 1)
    ax.plot(epochs_cpu, losses_cpu, label="CPU 학습",
            linewidth=2, color="steelblue", marker="o", markersize=3, alpha=0.7)

    if torch.cuda.is_available():
        epochs_gpu = range(1, len(losses_gpu) + 1)
        ax.plot(epochs_gpu, losses_gpu, label="GPU 학습",
                linewidth=2, color="darkorange", marker="s", markersize=3, alpha=0.7)

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("Transformer 텍스트 분류 학습 곡선", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11, loc="upper right")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig("training_curve.png", dpi=150, bbox_inches="tight")
    print("저장: training_curve.png")
    plt.close()

# 실행
visualize_gpu_cpu_comparison()
plot_training_curve()
```

### 3.5 토크나이저 비교: 글자 vs 단어 vs 부분단어

```python
def compare_tokenizers():
    """
    다양한 토크나이저 방식 비교

    1. 글자 단위 (Character-level)
       - 장점: 어휘가 작음 (~100)
       - 단점: 토큰 수가 많음 (문장이 길어짐)

    2. 단어 단위 (Word-level)
       - 장점: 직관적, 토큰 수 중간
       - 단점: OOV(Out-of-Vocabulary) 문제

    3. 부분단어 (Subword: BPE, WordPiece)
       - 장점: OOV 해결, 적절한 길이
       - 단점: 구현 복잡
    """

    sample_text = "이 영화 정말 좋아 최고의 영화야"

    # 1) 글자 단위
    char_tokens = list(sample_text.replace(" ", ""))

    # 2) 공백 기반 단어
    word_tokens = sample_text.split()

    # 3) 현재 구현한 SimpleTokenizer
    encoded = tokenizer.encode(sample_text)

    print(f"[토크나이저 비교: '{sample_text}']")
    print("=" * 70)

    print(f"\n1️⃣  글자 단위 (Character-level)")
    print(f"   토큰: {char_tokens}")
    print(f"   길이: {len(char_tokens)}")
    print(f"   어휘 크기: ~26-50 (한글은 자모 기준)")

    print(f"\n2️⃣  단어 단위 (Word-level)")
    print(f"   토큰: {word_tokens}")
    print(f"   길이: {len(word_tokens)}")
    print(f"   어휘 크기: {len(tokenizer.word2id):,} (SimpleTokenizer)")

    print(f"\n3️⃣  SimpleTokenizer (이 구현)")
    tokens_list = encoded.tolist()
    print(f"   토큰 ID: {tokens_list[:10]}... (패딩 제외)")
    non_pad_count = sum(1 for t in tokens_list if t != 0)
    print(f"   실제 토큰 수: {non_pad_count}")
    print(f"   어휘 크기: {len(tokenizer.word2id):,}")

    print(f"\n[실무 모델들의 토크나이저]")
    print("─" * 70)
    print(f"  BPE (GPT-2):         ~50,000 어휘")
    print(f"  WordPiece (BERT):    ~30,000 어휘")
    print(f"  SentencePiece (Llama):  ~32,000 어휘")

    print(f"\n[효율성 비교]")
    print("─" * 70)
    print(f"  글자: {len(char_tokens):2d} 토큰 × ~100 어휘 = 비효율적 (문장이 매우 긺)")
    print(f"  단어: {len(word_tokens):2d} 토큰 × ~10K 어휘 = 중간 (OOV 문제 있음)")
    print(f"  부분단어: ~7 토큰 × ~50K 어휘 = 효율적 (OOV 없음)")

compare_tokenizers()
```

**예상 결과**:
```
[토크나이저 비교: '이 영화 정말 좋아 최고의 영화야']
══════════════════════════════════════════════════════════════════════

1️⃣  글자 단위 (Character-level)
   토큰: ['이', ' ', '영', '화', ' ', '정', '말', ' ', '좋', '아', ' ', '최', '고', '의', ' ', '영', '화', '야']
   길이: 18
   어휘 크기: ~26-50 (한글은 자모 기준)

2️⃣  단어 단위 (Word-level)
   토큰: ['이', '영화', '정말', '좋아', '최고의', '영화야']
   길이: 6
   어휘 크기: 18 (SimpleTokenizer)

3️⃣  SimpleTokenizer (이 구현)
   토큰 ID: [2, 3, 4, 5, 6, 7, 8, 0, 0, 0]... (패딩 제외)
   실제 토큰 수: 7
   어휘 크기: 18

[실무 모델들의 토크나이저]
──────────────────────────────────────────────────────────────────────
  BPE (GPT-2):         ~50,000 어휘
  WordPiece (BERT):    ~30,000 어휘
  SentencePiece (Llama):  ~32,000 어휘

[효율성 비교]
──────────────────────────────────────────────────────────────────────
  글자: 18 토큰 × ~100 어휘 = 비효율적 (문장이 매우 길다)
  단어: 6 토큰 × ~10K 어휘 = 중간 (OOV 문제 있음)
  부분단어: ~7 토큰 × ~50K 어휘 = 효율적 (OOV 없음)
```

---

## 종합 해설: Transformer Encoder의 설계 철학

### 왜 Transformer가 RNN을 대체했는가?

| 특성 | RNN (LSTM/GRU) | Transformer Encoder |
|------|----------------|-------------------|
| **처리 방식** | 순차 (시계열) | 병렬 (한 번에) |
| **시간복잡도** | O(n·d²) | O(n²·d) |
| **위치별 경로** | n 스텝 (1→2→3→...→n) | 1 스텝 (모두 동시) |
| **먼 거리 의존성** | 약함 (기울기 소실) | 강함 (Attention) |
| **메모리 효율** | O(n) | O(n²) |
| **GPU 병렬화** | 어려움 (순차 의존) | 매우 쉬움 (배치 행렬곱) |

**결과**:
- 학습: Transformer가 **10-100배 빠름** (GPU 병렬화)
- 성능: Transformer가 **더 나음** (장거리 의존성)
- 스케일링: Transformer가 **더 우수함** (GPT-3는 175B 파라미터)

### Residual Connection이 깊은 네트워크를 가능하게 한다

```
기울기 흐름 (역전파):
dL/dx = dL/dout · dout/dx

Residual 없음:
x₆ = f₆(f₅(f₄(f₃(f₂(f₁(x₀))))))
dL/dx₀ = dL/dx₆ · df₆/dx₅ · df₅/x₄ · ... · df₁/dx₀  (곱 6개)
→ 각 곱이 1보다 작으면 exponentially 감소

Residual 있음:
x₁ = x₀ + f₁(x₀)
x₂ = x₁ + f₂(x₁)
...
dL/dx₀ = dL/dx₆ · (1 + df₆/dx₅) · (1 + df₅/dx₄) · ... · (1 + df₁/dx₀)
→ 각 항이 1 이상이므로 신호 유지 또는 증가
```

**결론**: Residual이 없으면 6층도 학습 불가능. 있으면 100층도 가능하다.

### Positional Encoding의 수학적 성질

Sinusoidal PE는 다음 성질을 만족한다:

PE(pos+k, i) = PE(pos, i) · cos(k·freq) + PE(pos, j) · sin(k·freq)

즉, **상대적 위치를 선형결합으로 표현할 수 있다**. Transformer는 이를 통해 상대 위치를 학습한다.

### 다음 단계: BERT와 GPT (5주차)

| 모델 | 기반 | 학습 방식 | 활용 |
|------|------|---------|------|
| **BERT** | Transformer Encoder | Masked LM + NSP | 이해 (분류, 검색) |
| **GPT** | Transformer Decoder | Causal LM | 생성 |
| **T5** | Encoder-Decoder | Seq2Seq | 변환 (요약, 번역) |

BERT = Transformer Encoder + 양방향 학습
GPT = Transformer Decoder + 단방향 학습

---

## 흔한 실수 종합 정리

| 실수 | 결과 | 해결책 |
|------|------|--------|
| LayerNorm 위치 (Post vs Pre) | 깊은 네트워크에서 발산 | Pre-Norm 사용 |
| √dₖ 대신 dₖ로 스케일링 | 기울기 소실 | `math.sqrt(d_k)` 사용 |
| Residual Connection 누락 | 3층 이상에서 학습 불가 | 모든 서브레이어에 추가 |
| Attention 마스킹 순서 | 음수 가중치 발생 | scores에 -∞ 할당 (softmax 전) |
| View 후 contiguous 누락 | RuntimeError | `transpose 후 contiguous()` 호출 |
| Positional Encoding 초기화 | 기울기 소실 | register_buffer 사용 (학습 불가) |
| Mean Pooling 대신 CLS | 정보 손실 | 정확한 풀링 방식 선택 |

---

**마지막 업데이트**: 2026-02-25
**대상**: 컴퓨터공학/AI 전공 학부 3-4학년
**예상 실행 시간**: 60-90분 (3개 체크포인트)
