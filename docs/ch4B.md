## 4주차 B회차: Transformer Encoder 구현 실습

> **미션**: Transformer Encoder Block을 밑바닥부터 구현하고, Positional Encoding의 필요성을 확인하며, 실제 텍스트 분류 모델을 완성하여 GPU와 CPU의 학습 속도 차이를 체감할 수 있다

### 수업 타임라인

| 시간 | 내용 | Copilot 사용 |
|------|------|-------------|
| 00:00~00:05 | 조 편성 + 역할 배분 (조원 A/B) | 사용 안 함 |
| 00:05~00:10 | A회차 핵심 리캡 + 과제 스펙 재확인 | 사용 안 함 |
| 00:10~00:55 | 2인1조 Copilot 구현 (체크포인트 3회) | 적극 사용 |
| 00:55~01:00 | Google Classroom 제출 (조별 1부) | 사용 안 함 |
| 01:00~01:20 | 결과 토론 (구현 전략 비교·성능 차이 분석) | 사용 안 함 |
| 01:20~01:28 | 교수 피드백 + 핵심 정리 | 사용 안 함 |
| 01:28~01:30 | 다음 주 예고 | 사용 안 함 |

---

### A회차 핵심 리캡

**Transformer의 핵심 혁신**:
- RNN의 순차 처리를 제거하고, Self-Attention으로 모든 위치를 **동시에** 처리한다
- 결과: 1,000단어를 RNN은 1,000 스텝에 처리하지만, Transformer는 1 스텝에 처리한다
- O(1) 경로 길이: 먼 거리의 단어도 1 스텝에 정보를 교환할 수 있다

**Positional Encoding의 필요성**:
- Self-Attention은 위치에 무관하게 작동한다 ("개가 사람을 물었다" = "사람이 개를 물었다")
- Sinusoidal PE: 수학적 함수로 위치를 인코딩, 파라미터 0개, 긴 시퀀스에 강함
- Learned PE: BERT/GPT의 방식, 학습 가능, 태스크 최적화 가능, 학습 길이 초과 시 성능 저하

**Encoder Block의 구조**:
- Multi-Head Self-Attention → LayerNorm + Residual
- Feed-Forward Network → LayerNorm + Residual
- 각 블록은 입출력 shape이 동일하여 원하는 만큼 쌓을 수 있다

**Residual Connection의 역할**:
- 입력을 더하여 기울기의 "고속도로"를 만든다
- 깊은 네트워크(6층 이상)에서 기울기 소실을 방지한다
- Residual 없이는 신호가 95% 손실되어 학습이 불가능하다

**Layer Normalization의 역할**:
- 각 위치의 벡터를 평균 0, 표준편차 1로 정규화
- 학습 안정화, 수렴 속도 향상, 배치 크기에 덜 민감해짐

---

### 과제 스펙

**과제**: Transformer Encoder Block 밑바닥 구현 + Positional Encoding 시각화 + 텍스트 분류 모델 + GPU/CPU 속도 비교

**제출 형태**: 조별 1부, Google Classroom 업로드

**파일 구성**:
- 구현 코드 파일 (`*.py`)
- Positional Encoding 시각화 이미지 (히트맵 1개)
- GPU/CPU 속도 비교 그래프 (1개)
- 간단한 분석 리포트 (1-2페이지)

**검증 기준**:
- ✓ Transformer Encoder Block의 입출력 shape 확인
- ✓ Positional Encoding (Sinusoidal) 구현 및 시각화
- ✓ Multi-Head Self-Attention, FFN, LayerNorm, Residual Connection 모두 포함
- ✓ 텍스트 분류 모델 학습 및 결과 해석
- ✓ GPU 학습이 CPU보다 몇 배 빠른지 정량화
- ✓ 토크나이저 비교 (BPE vs WordPiece 시뮬레이션)

---

### 2인1조 실습

> **Copilot 활용**: Encoder Block의 기본 골격(MultiHeadAttention, LayerNorm 정의)을 직접 작성한 뒤, Copilot에게 "이 두 서브레이어를 하나의 EncoderBlock으로 만들어줄래? Residual Connection과 Forward Pass를 포함해야 해"라고 요청한다. Copilot의 제안을 검토하고, 왜 이 순서인지(Norm의 위치, Dropout 추가 등)를 이해하는 과정에서 Transformer 설계 철학을 깊이 있게 학습할 수 있다.

**역할 분담**:
- **조원 A (드라이버)**: 코드 작성 및 실행, 모델 학습 및 결과 수집
- **조원 B (네비게이터)**: 로직 검토, Copilot 프롬프트 설계, 수학적 원리 검증
- **체크포인트마다 역할 교대**: 각자 전체 구현을 이해하도록 진행

---

#### 체크포인트 1: Transformer Encoder Block 밑바닥 구현 (15분)

**목표**: Positional Encoding, Multi-Head Attention, Feed-Forward, Residual Connection, Layer Normalization을 모두 포함한 Encoder Block을 PyTorch로 구현하고 동작 확인

**핵심 단계**:

① **Positional Encoding 구현** — Sinusoidal 방식으로 위치 정보 추가

```python
import torch
import torch.nn as nn
import numpy as np
import math

class PositionalEncoding(nn.Module):
    """Sinusoidal Positional Encoding"""

    def __init__(self, d_model, max_len=512):
        super().__init__()
        self.d_model = d_model

        # Positional Encoding 행렬 생성
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # 차원별 주파수 계산
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) *
            (-math.log(10000.0) / d_model)
        )

        # sin/cos 적용 (짝수 차원: sin, 홀수 차원: cos)
        pe[:, 0::2] = torch.sin(pos * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(pos * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(pos * div_term)

        # 학습 가능하지 않도록 등록
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            x + positional_encoding
        """
        seq_len = x.shape[1]
        return x + self.pe[:, :seq_len, :]
```

예상 동작:

```
[Positional Encoding 확인]
  입력: (batch=2, seq_len=10, d_model=256)

  [위치별 PE 벡터의 코사인 유사도]
    위치 0 ↔ 1: 0.983 (매우 가깝다)
    위치 0 ↔ 5: 0.749 (중간)
    위치 0 ↔ 10: 0.614 (먼다)

  [PE 행렬 통계]
    평균: ≈ 0.000
    표준편차: ≈ 0.707
```

② **Multi-Head Attention 모듈** (3장 복습)

```python
class MultiHeadAttention(nn.Module):
    """Multi-Head Self-Attention"""

    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape

        # Q, K, V 계산
        Q = self.W_q(x)  # (batch, seq, d_model)
        K = self.W_k(x)
        V = self.W_v(x)

        # Multi-Head 분할
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # Attention 계산
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        weights = torch.softmax(scores, dim=-1)
        weights = self.dropout(weights)

        attn_output = torch.matmul(weights, V)

        # Head 합치기
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.d_model)

        # 출력 변환
        output = self.W_o(attn_output)

        return output, weights
```

③ **Feed-Forward Network**

```python
class PositionwiseFeedForward(nn.Module):
    """Positionwise Feed-Forward Network"""

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.gelu = nn.GELU()  # ReLU 대신 GELU
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        d_model → d_ff (확장) → d_model (축소)
        각 위치마다 독립적으로 적용
        """
        return self.linear2(
            self.dropout(self.gelu(self.linear1(x)))
        )
```

④ **Encoder Block 통합 — 핵심!**

```python
class TransformerEncoderBlock(nn.Module):
    """Transformer Encoder Block: Attention + FFN + Residual + Norm"""

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()

        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        [패턴]:
        1) Self-Attention → Dropout → Residual + LayerNorm
        2) FFN → Dropout → Residual + LayerNorm
        """
        # 첫 번째 서브레이어: Self-Attention + Residual + Norm
        attn_output, _ = self.self_attention(x, mask)
        attn_output = self.dropout(attn_output)
        x = self.norm1(x + attn_output)  # Residual!

        # 두 번째 서브레이어: FFN + Residual + Norm
        ff_output = self.feed_forward(x)
        ff_output = self.dropout(ff_output)
        x = self.norm2(x + ff_output)  # Residual!

        return x
```

⑤ **테스트 및 파라미터 확인**

```python
# 하이퍼파라미터 설정
d_model = 256
num_heads = 8
d_ff = 1024
batch_size = 4
seq_len = 12

# 모듈 생성
pos_enc = PositionalEncoding(d_model)
encoder_block = TransformerEncoderBlock(d_model, num_heads, d_ff)

# 더미 입력
x = torch.randn(batch_size, seq_len, d_model)

# Positional Encoding 적용
x_with_pos = pos_enc(x)

# Encoder Block 통과
output = encoder_block(x_with_pos)

print(f"[Encoder Block 검증]")
print(f"  입력 shape: {x.shape}")
print(f"  Positional Encoding 적용 후: {x_with_pos.shape}")
print(f"  Encoder Block 출력: {output.shape}")
print(f"  입출력 shape 동일: {x.shape == output.shape}")
print(f"  Encoder Block 파라미터 수: {sum(p.numel() for p in encoder_block.parameters()):,}")
```

예상 결과:

```
[Encoder Block 검증]
  입력 shape: torch.Size([4, 12, 256])
  Positional Encoding 적용 후: torch.Size([4, 12, 256])
  Encoder Block 출력: torch.Size([4, 12, 256])
  입출력 shape 동일: True
  Encoder Block 파라미터 수: 1,050,880

[파라미터 분석]
  Self-Attention: 262,144 (256×256×4)
  Feed-Forward: 786,432 (256×1024×2)
  LayerNorm: 512 (2×d_model)
```

**검증 체크리스트**:
- [ ] Positional Encoding이 올바르게 입력에 더해지는가?
- [ ] Multi-Head Attention의 출력이 (batch, seq, d_model) 형태인가?
- [ ] FFN의 중간 차원이 d_model보다 크게 확장되는가?
- [ ] Residual Connection으로 입력이 두 번 더해지는가?
- [ ] LayerNorm 이후의 출력이 평균 0, 표준편차 ≈ 1인가?
- [ ] 입출력 shape이 정확히 동일한가?

**Copilot 프롬프트 1**:
```
"Sinusoidal Positional Encoding을 PyTorch로 구현해줄래?
sin은 짝수 차원에, cos는 홀수 차원에 적용하고,
주파수는 pos / 10000^(2i/d_model) 형태로 계산해야 해."
```

**Copilot 프롬프트 2**:
```
"MultiHeadAttention과 PositionwiseFeedForward를 받아서
TransformerEncoderBlock을 만들어줄래?
Residual Connection과 LayerNorm을 각 서브레이어 뒤에 넣고,
forward 함수를 구현해야 해."
```

---

#### 체크포인트 2: Positional Encoding 시각화 + Residual 효과 확인 (15분)

**목표**: Positional Encoding이 실제로 위치 정보를 인코딩하는지 시각화하고, Residual Connection이 기울기 흐름에 미치는 영향을 정량화

**핵심 단계**:

① **Positional Encoding 히트맵 시각화**

```python
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_positional_encoding(d_model, max_len=50):
    """PE 벡터의 구조를 시각화"""

    pos_enc = PositionalEncoding(d_model, max_len)
    pe = pos_enc.pe[0].detach().numpy()  # (max_len, d_model)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 1) PE 벡터들의 히트맵 (각 위치가 고유한가?)
    sns.heatmap(pe[:20, :], cmap="RdBu_r", center=0,
                ax=axes[0], cbar_kws={"label": "PE 값"})
    axes[0].set_title("Positional Encoding 히트맵 (처음 20개 위치)")
    axes[0].set_xlabel("차원 (0-255)")
    axes[0].set_ylabel("위치 (0-19)")

    # 2) 위치 간 코사인 유사도 (가깝거나 먼 위치는?)
    from sklearn.metrics.pairwise import cosine_similarity
    similarity = cosine_similarity(pe[:20, :])

    sns.heatmap(similarity, annot=True, fmt=".3f", cmap="YlOrRd",
                vmin=0, vmax=1, ax=axes[1], cbar_kws={"label": "유사도"})
    axes[1].set_title("위치 간 코사인 유사도")
    axes[1].set_xlabel("위치")
    axes[1].set_ylabel("위치")

    fig.tight_layout()
    fig.savefig("positional_encoding.png", dpi=150, bbox_inches="tight")
    print("저장: positional_encoding.png")
    plt.close()

# 시각화 실행
visualize_positional_encoding(d_model=256)
```

예상 결과:

```
[Positional Encoding 분석]
  PE 벡터 분포:
    - 낮은 차원: 느린 진동 (먼 위치도 구분)
    - 높은 차원: 빠른 진동 (가까운 위치만 구분)

  코사인 유사도:
    - 대각선: 1.000 (자신과의 유사도)
    - 인접한 위치: 0.98+ (매우 유사)
    - 거리 5: ≈ 0.75
    - 거리 10+: ≈ 0.60 이하 (충분히 다르다)
```

② **Residual Connection의 기울기 흐름 효과 정량화**

```python
import torch.optim as optim

def analyze_residual_effect():
    """Residual Connection이 기울기 흐름에 미치는 영향"""

    d_model = 256
    seq_len = 10
    batch_size = 1
    num_layers = 6

    # 더미 입력 (고정)
    x_input = torch.randn(batch_size, seq_len, d_model)

    # 1) Residual 없는 순차 통과
    x_no_residual = x_input.clone()
    for i in range(num_layers):
        # 단순 선형 변환 (Residual 없음)
        proj = nn.Linear(d_model, d_model)
        x_no_residual = proj(x_no_residual)

    l2_norm_no_residual = torch.norm(x_no_residual, p=2)

    # 2) Residual 있는 통과
    class BlockWithResidual(nn.Module):
        def __init__(self, d_model):
            super().__init__()
            self.proj = nn.Linear(d_model, d_model)

        def forward(self, x):
            return x + self.proj(x)  # Residual!

    x_with_residual = x_input.clone()
    for i in range(num_layers):
        block = BlockWithResidual(d_model)
        x_with_residual = block(x_with_residual)

    l2_norm_with_residual = torch.norm(x_with_residual, p=2)

    # 결과
    l2_input = torch.norm(x_input, p=2)

    print(f"\n[Residual Connection의 효과 — 6층 통과 후]")
    print(f"  입력 L2 norm: {l2_input.item():.4f}")
    print(f"  Residual 없음: {l2_norm_no_residual.item():.4f}")
    print(f"    → 신호 감소: {(l2_input.item() / l2_norm_no_residual.item()):.1f}배")
    print(f"  Residual 있음: {l2_norm_with_residual.item():.4f}")
    print(f"    → 신호 유지됨 (증가할 수도!)")

analyze_residual_effect()
```

예상 결과:

```
[Residual Connection의 효과 — 6층 통과 후]
  입력 L2 norm: 16.5234
  Residual 없음: 0.8912
    → 신호 감소: 18.5배 (신호 거의 사라짐!)
  Residual 있음: 42.3561
    → 신호 유지됨 (증가할 수도!)

(깊은 네트워크에서 Residual이 없으면 신호가 극적으로 사라진다!)
```

③ **Layer Normalization의 정규화 효과**

```python
def analyze_layer_normalization():
    """Layer Normalization이 벡터를 어떻게 정규화하는가"""

    # 불규칙한 크기의 입력 생성
    x = torch.randn(4, 10, 256)  # 배치, 시퀀스, 차원

    print(f"\n[Layer Normalization 효과]")
    print(f"정규화 전:")
    print(f"  평균: {x.mean(dim=-1)[0, 0]:.4f}")
    print(f"  표준편차: {x.std(dim=-1)[0, 0]:.4f}")
    print(f"  최대값: {x.max():.4f}, 최소값: {x.min():.4f}")
    print(f"  L2 norm (첫 번째 위치): {torch.norm(x[0, 0]):.4f}")

    # LayerNorm 적용
    layer_norm = nn.LayerNorm(256)
    x_normalized = layer_norm(x)

    print(f"\n정규화 후:")
    print(f"  평균: {x_normalized.mean(dim=-1)[0, 0]:.4f} (≈ 0)")
    print(f"  표준편차: {x_normalized.std(dim=-1)[0, 0]:.4f} (≈ 1)")
    print(f"  최대값: {x_normalized.max():.4f}, 최소값: {x_normalized.min():.4f}")
    print(f"  L2 norm (첫 번째 위치): {torch.norm(x_normalized[0, 0]):.4f}")

analyze_layer_normalization()
```

예상 결과:

```
[Layer Normalization 효과]
정규화 전:
  평균: -0.1234
  표준편차: 0.8945
  최대값: 3.5612, 최소값: -3.2145
  L2 norm (첫 번째 위치): 15.234

정규화 후:
  평균: 0.0000 (수치 정밀도로 ≈0)
  표준편차: 1.0000
  최대값: 3.4567, 최소값: -3.1234
  L2 norm (첫 번째 위치): 16.000 (정규화된 차원)
```

**검증 체크리스트**:
- [ ] PE 히트맵에서 각 위치가 고유한 패턴을 보이는가?
- [ ] 위치 간 유사도가 거리에 따라 감소하는가?
- [ ] Residual 없이는 신호가 급격히 감소하는가?
- [ ] Residual 있으면 신호가 유지 또는 증가하는가?
- [ ] LayerNorm 후 평균이 0에 가까운가?
- [ ] LayerNorm 후 표준편차가 1에 가까운가?

**Copilot 프롬프트 3**:
```
"Positional Encoding 벡터들을 히트맵으로 시각화해줄래?
각 행이 위치, 각 열이 차원이고, 색상으로 PE 값을 표시해야 해.
그리고 위치 간 코사인 유사도도 계산하고 히트맵으로 그려줘."
```

**Copilot 프롬프트 4**:
```
"6층의 Linear 층을 순차로 통과시키고,
Residual Connection이 있을 때와 없을 때의 출력 벡터 L2 norm을 비교해줄래?
신호 감소를 정량화해야 해."
```

---

#### 체크포인트 3: 텍스트 분류 모델 + GPU/CPU 속도 비교 (25분)

**목표**: Transformer Encoder를 활용한 텍스트 분류 모델을 구현하고, GPU와 CPU에서 학습하여 실제 속도 차이를 체감하고 정량화

**핵심 단계**:

① **Transformer 기반 텍스트 분류기**

```python
class TransformerTextClassifier(nn.Module):
    """Transformer Encoder 기반 텍스트 분류 모델"""

    def __init__(self, vocab_size, d_model, num_heads, d_ff,
                 num_layers, num_classes, max_len=512, dropout=0.1):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)

        self.encoder_layers = nn.ModuleList([
            TransformerEncoderBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )

    def forward(self, x):
        """
        Args:
            x: 토큰 ID (batch, seq_len)
        Returns:
            logits: 분류 확률 (batch, num_classes)
        """
        # 임베딩
        x = self.embedding(x)  # (batch, seq, d_model)

        # Positional Encoding
        x = self.pos_encoding(x)

        # Encoder Layers
        for layer in self.encoder_layers:
            x = layer(x)

        # Mean Pooling (시퀀스 길이 제거)
        x = x.mean(dim=1)  # (batch, d_model)

        # 분류
        logits = self.classifier(x)

        return logits
```

② **데이터 준비**

```python
from torch.utils.data import Dataset, DataLoader

# 간단한 한국어 텍스트 분류 데이터
# 0: 부정 (싫다, 나쁘다), 1: 긍정 (좋다, 최고다)
train_texts = [
    "이 영화 정말 좋아",
    "정말 최고의 영화야",
    "생각보다 훨씬 좋네",
    "이 책 정말 싫어",
    "너무 지루하고 나빠",
    "최악의 경험이야",
    "이 음식 맛있어",
    "정말 나쁜 맛이네",
    "완벽한 영화다",
    "너무 재미없어",
]

train_labels = [1, 1, 1, 0, 0, 0, 1, 0, 1, 0]

class SimpleTokenizer:
    """간단한 공백 기반 토크나이저"""

    def __init__(self):
        self.word2id = {"<pad>": 0, "<unk>": 1}
        self.id2word = {0: "<pad>", 1: "<unk>"}

    def fit(self, texts):
        """단어 사전 구축"""
        for text in texts:
            for word in text.split():
                if word not in self.word2id:
                    idx = len(self.word2id)
                    self.word2id[word] = idx
                    self.id2word[idx] = word

    def encode(self, text, max_len=20):
        """텍스트를 토큰 ID로 변환"""
        tokens = []
        for word in text.split():
            token_id = self.word2id.get(word, self.word2id["<unk>"])
            tokens.append(token_id)

        # 패딩 또는 자르기
        if len(tokens) < max_len:
            tokens += [0] * (max_len - len(tokens))
        else:
            tokens = tokens[:max_len]

        return torch.tensor(tokens, dtype=torch.long)

# 토크나이저 학습
tokenizer = SimpleTokenizer()
tokenizer.fit(train_texts)

class TextDataset(Dataset):
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

dataset = TextDataset(train_texts, train_labels, tokenizer)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
```

③ **모델 학습 + GPU/CPU 속도 비교**

```python
def train_model_and_benchmark(device, num_epochs=100):
    """모델 학습 및 시간 측정"""

    # 모델 초기화
    model = TransformerTextClassifier(
        vocab_size=len(tokenizer.word2id),
        d_model=128,
        num_heads=4,
        d_ff=512,
        num_layers=2,
        num_classes=2,
        max_len=20
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # 학습 시간 측정
    import time
    start_time = time.time()

    losses = []
    for epoch in range(num_epochs):
        total_loss = 0
        correct = 0
        total = 0

        for tokens, labels in dataloader:
            tokens = tokens.to(device)
            labels = labels.to(device)

            logits = model(tokens)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        losses.append(total_loss / len(dataloader))

        if (epoch + 1) % 20 == 0:
            acc = correct / total
            print(f"Epoch {epoch+1:3d} ({device}): Loss = {losses[-1]:.4f}, Acc = {acc:.1%}")

    elapsed_time = time.time() - start_time

    return model, losses, elapsed_time

# CPU 학습
print("\n[CPU에서 학습 시작...]")
cpu_device = torch.device("cpu")
model_cpu, losses_cpu, time_cpu = train_model_and_benchmark(cpu_device, num_epochs=100)
print(f"CPU 학습 시간: {time_cpu:.2f}초")

# GPU 학습 (CUDA 사용 가능하면)
if torch.cuda.is_available():
    print("\n[GPU에서 학습 시작...]")
    gpu_device = torch.device("cuda")
    model_gpu, losses_gpu, time_gpu = train_model_and_benchmark(gpu_device, num_epochs=100)
    print(f"GPU 학습 시간: {time_gpu:.2f}초")

    speedup = time_cpu / time_gpu
    print(f"\n[속도 개선]")
    print(f"  GPU가 CPU보다 {speedup:.1f}배 빠름")
else:
    print("\nGPU를 사용할 수 없습니다. CPU에서만 학습했습니다.")
```

예상 결과:

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

[속도 개선]
  GPU가 CPU보다 5.3배 빠름
```

④ **GPU/CPU 속도 비교 시각화**

```python
def visualize_speed_comparison():
    """GPU/CPU 학습 속도 비교 그래프"""

    if torch.cuda.is_available():
        devices = ["CPU", "GPU"]
        times = [time_cpu, time_gpu]
        colors = ["steelblue", "darkorange"]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # 1) 막대 그래프
        axes[0].bar(devices, times, color=colors, alpha=0.7, edgecolor="black", linewidth=2)
        axes[0].set_ylabel("학습 시간 (초)")
        axes[0].set_title("CPU vs GPU 학습 시간")
        axes[0].grid(axis="y", alpha=0.3)
        for i, (dev, t) in enumerate(zip(devices, times)):
            axes[0].text(i, t + 1, f"{t:.1f}s", ha="center", fontweight="bold")

        # 2) 속도 개선 배수
        speedup = time_cpu / time_gpu
        axes[1].bar(["속도 개선"], [speedup], color="green", alpha=0.7, edgecolor="black", linewidth=2)
        axes[1].set_ylabel("배수 (배)")
        axes[1].set_title(f"GPU 가속 배수: {speedup:.1f}배 빠름")
        axes[1].set_ylim([0, speedup + 1])
        axes[1].text(0, speedup + 0.2, f"{speedup:.1f}배", ha="center", fontweight="bold", fontsize=14)
        axes[1].grid(axis="y", alpha=0.3)

        fig.tight_layout()
        fig.savefig("gpu_cpu_comparison.png", dpi=150, bbox_inches="tight")
        print("저장: gpu_cpu_comparison.png")
        plt.close()
    else:
        print("GPU를 사용할 수 없어 비교 그래프를 생성할 수 없습니다.")

visualize_speed_comparison()
```

⑤ **학습 곡선 시각화**

```python
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
    ax.set_title("Transformer 텍스트 분류 학습 곡선", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    fig.savefig("training_curve.png", dpi=150, bbox_inches="tight")
    print("저장: training_curve.png")
    plt.close()

plot_training_curve()
```

⑥ **토크나이저 비교 실험** (BPE 시뮬레이션)

```python
def compare_tokenizers():
    """BPE vs WordPiece 시뮬레이션 비교"""

    sample_text = "이 영화 정말 좋아 최고의 영화야"

    # 1) 글자 단위 (Character-level)
    char_tokens = list(sample_text.replace(" ", ""))

    # 2) 공백 기반 (Whitespace) — 간단한 Word-level
    word_tokens = sample_text.split()

    # 3) 현재 구현한 SimpleTokenizer
    encoded = tokenizer.encode(sample_text)

    print(f"\n[토크나이저 비교: '{sample_text}']")
    print(f"  글자 단위 (26개 기호): {len(char_tokens)} 토큰")
    print(f"    {char_tokens[:15]}...")
    print(f"  단어 단위: {len(word_tokens)} 토큰")
    print(f"    {word_tokens}")
    print(f"  SimpleTokenizer: {len([t for t in encoded if t.item() != 0])} 토큰 (패딩 제외)")
    print(f"    {encoded.tolist()}")

    print(f"\n[어휘 크기 비교]")
    print(f"  글자 단위: ~26-50 (언어에 따라)")
    print(f"  단어 단위: {len(tokenizer.word2id):,} (학습된 어휘)")
    print(f"  BPE (GPT-2): ~50,000")
    print(f"  WordPiece (BERT): ~30,000")
    print(f"  SentencePiece (Llama): ~32,000")

compare_tokenizers()
```

예상 결과:

```
[토크나이저 비교: '이 영화 정말 좋아 최고의 영화야']
  글자 단위 (26개 기호): 18 토큰
    ['이', ' ', '영', '화', ' ', '정', '말', ' ', '좋', '아', ' ', '최', '고', '의', ' ', '영', '화', '야']
  단어 단위: 5 토큰
    ['이', '영화', '정말', '좋아', '최고의', '영화야']
  SimpleTokenizer: 5 토큰 (패딩 제외)
    [2, 3, 4, 5, 6, 7, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

[어휘 크기 비교]
  글자 단위: ~26-50 (언어에 따라)
  단어 단위: 12 (학습된 어휘)
  BPE (GPT-2): ~50,000
  WordPiece (BERT): ~30,000
  SentencePiece (Llama): ~32,000

[토큰화 효율성]
  글자 단위는 어휘는 작지만 토큰이 18개 (길다!)
  단어 단위는 어휘가 12개, 토큰 5개 (가짓수: 18개)
```

**검증 체크리스트**:
- [ ] 모델이 학습되는가? (Loss가 감소하는가?)
- [ ] 정확도가 증가하는가? (100%에 수렴하는가?)
- [ ] GPU가 CPU보다 빠른가? (최소 2배 이상)
- [ ] 속도 비교 그래프가 생성되었는가?
- [ ] 학습 곡선이 생성되었는가?
- [ ] 토크나이저 비교가 올바르게 동작하는가?

**Copilot 프롬프트 5**:
```
"Transformer Encoder를 활용한 텍스트 분류 모델을 만들어줄래?
임베딩 → Positional Encoding → N개의 Encoder Layer → Mean Pooling → 분류 레이어 순서야.
forward 함수에서 토큰 ID (batch, seq_len)를 받아서 로짓을 반환해야 해."
```

**Copilot 프롬프트 6**:
```
"CPU와 GPU에서 동일한 모델을 학습시키고 시간을 측정해줄래?
시작 시간과 종료 시간의 차이로 elapsed_time을 계산하고,
GPU가 CPU보다 몇 배 빠른지 계산해야 해.
time.time()을 사용하면 돼."
```

**선택 프롬프트**:
```
"GPU/CPU 속도 비교를 막대 그래프로 그려줄래?
그리고 학습 곡선 (x축: Epoch, y축: Loss)도 matplotlib으로 그려줘."
```

---

### 제출 안내 (Google Classroom)

**제출 방법**:
- Google Classroom의 "4주차 B회차" 과제에 조별 1부 제출
- 파일명 형식: `group_{조번호}_ch4B.zip`

**포함할 파일**:
```
group_{조번호}_ch4B/
├── ch4B_transformer.py           # 전체 구현 코드
├── positional_encoding.png        # PE 히트맵 + 유사도
├── gpu_cpu_comparison.png         # 속도 비교 그래프
├── training_curve.png             # 학습 곡선
└── report.md                      # 분석 리포트 (1-2페이지)
```

**리포트 포함 항목** (report.md):
- Transformer Encoder Block 구현 과정 및 핵심 설계 (3-4문장)
  - 왜 Residual Connection이 필수인가?
  - LayerNorm의 위치와 역할?
- Positional Encoding 분석 (2-3문장)
  - PE 벡터가 위치를 어떻게 구분하는가?
  - Sinusoidal vs Learned의 트레이드오프?
- GPU/CPU 성능 비교 (2-3문장)
  - 속도 개선이 몇 배인가?
  - 왜 Transformer가 GPU 병렬화에 유리한가?
- 모델 성능 결과 (2문장)
  - 최종 정확도와 수렴 속도?
  - 토크나이저의 효율성?
- Copilot 사용 경험 (2문장)
  - 어떤 프롬프트가 효과적이었는가?
  - 코드 생성에서 어떤 부분을 수정했는가?

**제출 마감**: 수업 종료 후 24시간 이내

---

### 결과 토론 가이드

> **토론 가이드**: 조별 구현 결과를 공유하며, Positional Encoding의 실제 효과, Residual Connection의 필요성, 그리고 GPU 가속의 현실적 이점을 함께 분석한다

**토론 주제**:

① **Positional Encoding의 필요성**
- PE 없이 모델을 학습시키면 어떻게 될까?
- Sinusoidal PE의 코사인 유사도가 위치 거리를 반영하는가?
- 각 조가 측정한 최대 유사도 감소율은?

② **Residual Connection의 효과**
- 6층 통과 후 신호 감소가 몇 배인가? (조별 수치 비교)
- Residual 없이는 정말 학습이 불가능한가?
- 더 깊은 층(12층, 24층)에서는 어떻게 될까?

③ **Layer Normalization의 역할**
- 정규화 전후 벡터의 평균과 표준편차가 실제로 0과 1에 가까운가?
- LayerNorm의 위치(Pre-LN vs Post-LN)가 중요한가?
- 없으면 학습이 더 느려지는가?

④ **GPU 가속의 현실성**
- GPU가 CPU보다 정확히 몇 배 빠른가? (각 조의 수치)
- 왜 Transformer는 병렬화에 유리한가? (RNN과의 차이)
- 실제로 BERT/GPT를 학습하려면 GPU가 필수인가?

⑤ **토크나이저의 실무적 영향**
- 글자 단위 vs 단어 단위 vs 서브워드의 토큰 수 비교
- 토큰 수가 모델 성능에 어떻게 영향을 미치는가?
- 한국어와 영어의 토크나이저 효율성 차이는?

**발표 형식**:
- 각 조 3~5분 발표 (구현 전략 + 측정 결과)
- 다른 조의 질문에 답변 (2~3개 질문)
- 교수의 보충 설명 및 피드백

---

### 교수 피드백 포인트

**강화할 점**:
- Residual Connection과 Layer Normalization이 단순한 "기법"이 아니라, **깊은 신경망의 학습 가능성 자체를 결정**한다는 점을 강조
- Transformer의 병렬 처리가 단순한 "속도 개선"이 아니라, **현대 LLM의 등장을 가능하게 한 핵심 혁신**임을 강조
- Positional Encoding이 "필수"가 아니라는 오해 방지: PE 없이는 순서 정보를 완전히 잃어 모델이 작동하지 않음

**주의할 점**:
- Residual의 효과를 "신호 유지"로만 해석하지 않기: 실제로는 **기울기 역전파**에서의 역할이 핵심
- GPU 속도 비교 시 배치 크기, 시퀀스 길이 등 여러 요인을 고려하도록 유도
- "모델이 100% 정확도를 달성했다"는 것이 데이터가 작고 단순하기 때문임을 명시
- Transformer Encoder만으로는 "생성"을 할 수 없으며, 다음 주 Decoder 학습의 필요성 강조

**다음 학습으로의 연결**:
- Transformer Encoder의 구조를 완전히 이해했으므로, 5주차에서 **Decoder의 Causal Masking과 Cross-Attention**을 쉽게 이해할 수 있을 것
- BERT(Encoder-only)와 GPT(Decoder-only)의 구조 차이를 이 Encoder 이해를 바탕으로 설명
- 실제로 LLM의 규모(파라미터, 데이터)를 키우면 성능이 어떻게 향상되는지는 5주차 실습에서 확인

---

### 다음 주 예고

다음 주 5장 A회차에서는 **LLM의 두 가지 주요 아키텍처**를 깊이 있게 다룬다.

**예고 내용**:
- **BERT (Encoder-only)**: 양방향 문맥을 활용한 표현 학습, Masked Language Modeling (MLM) 사전학습 목표
- **GPT (Decoder-only)**: 인과적 생성, Causal Masking, 다음 토큰 예측 사전학습 목표
- **사전학습(Pre-training) vs 파인튜닝(Fine-tuning)**: BERT/GPT를 어떻게 재사용하는가?
- **실습**: 사전학습된 BERT/GPT 모델을 활용한 감성 분석, 개체명 인식, 텍스트 생성

**사전 준비**:
- 4장 내용 (특히 Encoder Block과 Positional Encoding)을 다시 읽어두기
- Causal Masking의 개념을 먼저 생각해보기 ("미래를 보지 못한다는 게 무슨 뜻?")

---

## 참고 자료

**실습 코드**:
- _전체 구현 코드는 practice/chapter4/code/4-2-encoder-block.py 참고_
- _텍스트 분류 모델은 practice/chapter4/code/4-3-classification.py 참고_
- _토크나이저 비교는 practice/chapter4/code/4-5-tokenizer-comparison.py 참고_

**권장 읽기**:
- Jay Alammar. The Illustrated Transformer. https://jalammar.github.io/illustrated-transformer/
- Lilian Weng. (2018). Attention? Attention!. https://lilianweng.github.io/posts/2018-06-24-attention/
- Vaswani, A. et al. (2017). Attention Is All You Need. *NeurIPS*. https://arxiv.org/abs/1706.03762
- Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2018). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. *arXiv*. https://arxiv.org/abs/1810.04805

