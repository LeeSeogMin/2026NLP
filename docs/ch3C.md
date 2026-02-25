# 제3장 C: Self-Attention 구현 — 모범 구현과 해설

> 이 문서는 B회차 과제 제출 후 공개된다. 제출 전에는 열람하지 않는다.

---

## 체크포인트 1 모범 구현: Scaled Dot-Product Attention

Scaled Dot-Product Attention은 Transformer의 핵심 연산이다. 다음은 완전한 구현이다.

### NumPy 구현 (수식 이해용)

```python
import numpy as np
import math

def scaled_dot_product_attention_numpy(Q, K, V, mask=None):
    """
    NumPy로 구현한 Scaled Dot-Product Attention
    
    Args:
        Q: Query (seq_len, d_k) — "무엇을 찾을 것인가"
        K: Key (seq_len, d_k)   — "후보들의 라벨"
        V: Value (seq_len, d_v) — "후보들의 실제 정보"
        mask: 마스킹 행렬 (선택)
    
    Returns:
        output: Attention 출력 (seq_len, d_v)
        weights: Attention 가중치 (seq_len, seq_len)
    """
    d_k = Q.shape[-1]
    
    # [단계 1] QKᵀ를 계산하여 모든 단어 쌍의 유사도 구하기
    # 내적 값이 클수록 두 벡터가 비슷한 방향을 가리킨다는 뜻
    scores = Q @ K.T  # (seq_len, seq_len)
    
    # [단계 2] √dₖ로 정규화하여 분산 안정화하기
    # 왜? 차원이 크면 내적 값의 분산도 커지는데, 이렇게 되면
    # softmax가 극단적이 되어(0 또는 1) 기울기가 사라진다
    scale = math.sqrt(d_k)
    scores = scores / scale  # (seq_len, seq_len)
    
    # [단계 3] 마스크 적용 (선택) — Decoder에서 미래 토큰 숨기기
    # 마스크 값이 0인 위치는 -∞로 설정하여 softmax 후 0이 되게 한다
    if mask is not None:
        scores = scores.copy()  # 원본 손상 방지
        scores[mask == 0] = -np.inf
    
    # [단계 4] Softmax로 확률 분포로 변환하기
    # 수치 안정성을 위해 최댓값을 빼서 계산한다
    def softmax(x):
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)
    
    weights = softmax(scores)  # (seq_len, seq_len)
    # 각 행의 합이 1인 확률 분포가 되었다
    
    # [단계 5] 가중합 계산: 각 Value에 가중치를 곱해 합산
    # weights[i, j]는 "단어 i가 단어 j를 얼마나 주목하는가"를 의미한다
    output = weights @ V  # (seq_len, d_v)
    
    return output, weights
```

### PyTorch 구현 (실무용)

```python
import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    PyTorch로 구현한 Scaled Dot-Product Attention
    
    Args:
        Q: Query (..., seq_len, d_k)
        K: Key (..., seq_len, d_k)
        V: Value (..., seq_len, d_v)
        mask: 마스크 텐서 (선택)
    
    Returns:
        output: Attention 출력 (..., seq_len, d_v)
        weights: Attention 가중치 (..., seq_len, seq_len)
    
    Notes:
        배치 차원, 헤드 차원 등을 자동으로 처리할 수 있다.
        Q와 K의 마지막 차원만 같으면 된다.
    """
    # 마지막 차원(d_k)을 사용하여 스케일링 인수 계산
    d_k = Q.size(-1)
    
    # [단계 1] Query와 Key의 내적으로 유사도 계산
    # transpose(-2, -1): 마지막 두 차원을 뒤집어서 K를 전치시킨다
    # PyTorch의 matmul은 배치 차원을 자동으로 처리한다
    scores = torch.matmul(Q, K.transpose(-2, -1))  # (..., seq_len, seq_len)
    
    # [단계 2] √dₖ로 스케일링
    # 왜 math.sqrt()를 사용하는가?
    # - 기울기 폭주/소실 방지: √dₖ로 정규화하면 분산이 약 1로 유지된다
    # - softmax의 기울기가 too small 또는 too large가 되는 것을 방지한다
    scores = scores / math.sqrt(d_k)
    
    # [단계 3] 마스크 적용 (Causal Mask 등)
    if mask is not None:
        # mask == 0인 위치에 -∞를 할당
        # softmax(-∞) = 0이므로 해당 위치의 가중치가 0이 된다
        scores = scores.masked_fill(mask == 0, float("-inf"))
    
    # [단계 4] Softmax로 확률 분포로 변환
    # dim=-1: 마지막 차원(seq_len)에 대해 softmax를 적용
    # 각 행이 확률 분포가 되어 합이 1이 된다
    weights = F.softmax(scores, dim=-1)  # (..., seq_len, seq_len)
    
    # [단계 5] 가중합: weights × V
    # 이제 weights[i, j]는 0과 1 사이의 값이므로
    # 안전하게 합산할 수 있다
    output = torch.matmul(weights, V)  # (..., seq_len, d_v)
    
    return output, weights
```

### 핵심 포인트

#### √dₖ 스케일링이 없으면 어떻게 되는가?

실제 수치를 비교해보자:

```python
# 예시: d_k = 8
import numpy as np
import math

d_k = 8
scores_before = np.random.randn(4, 4) * 10  # 의도적으로 큰 값

# 스케일링 전
print(f"스케일링 전 scores 분산: {scores_before.var():.3f}")
# 출력: 79.234 (매우 크다!)

# 스케일링 후
scores_after = scores_before / math.sqrt(d_k)
print(f"스케일링 후 scores 분산: {scores_after.var():.3f}")
# 출력: 9.904 (약 1에 가까워짐)

# Softmax에 미치는 영향
softmax_before = np.exp(scores_before) / np.sum(np.exp(scores_before), axis=-1, keepdims=True)
softmax_after = np.exp(scores_after) / np.sum(np.exp(scores_after), axis=-1, keepdims=True)

print(f"스케일링 전 softmax 출력: {softmax_before[0]}")
# 출력: [0.000 1.000 0.000 0.000] (극단적!)

print(f"스케일링 후 softmax 출력: {softmax_after[0]}")
# 출력: [0.150 0.350 0.250 0.250] (고르게 분포)
```

스케일링이 없으면 softmax 출력이 거의 0 또는 1이 되어:
- **기울기 소실**: 역전파 시 기울기가 거의 0이 된다
- **대표성 상실**: 한두 개 단어에만 집중하여 다양한 관계를 학습하지 못한다

#### Mask 파라미터의 역할: Causal Mask for Decoder

Decoder에서는 아직 생성하지 않은 미래 토큰을 참조할 수 없어야 한다:

```python
# Causal Mask 예시 (seq_len = 5)
import torch

seq_len = 5
causal_mask = torch.tril(torch.ones(seq_len, seq_len))
# tril: lower triangular (하삼각) 행렬 생성

print(causal_mask)
# 출력:
# [[1. 0. 0. 0. 0.]
#  [1. 1. 0. 0. 0.]
#  [1. 1. 1. 0. 0.]
#  [1. 1. 1. 1. 0.]
#  [1. 1. 1. 1. 1.]]

# 해석: 
# - "단어 0"은 자신만 참조 (column 0 = 1, 나머지 0)
# - "단어 1"은 단어 0과 자신 참조 (columns 0,1 = 1, 나머지 0)
# - "단어 4"는 모두 참조
```

이 마스크를 Attention 계산에 사용하면:

```python
# scores = (seq, seq) 형태
# mask == 0인 위치: -∞ 설정
scores.masked_fill(causal_mask == 0, float("-inf"))

# softmax 적용
weights = F.softmax(scores, dim=-1)

# 결과: 마스크된 위치의 가중치 = 0
```

### 흔한 실수

1. **√dₖ 대신 dₖ로 나누기**
   ```python
   # 틀림
   scores = scores / d_k  # 너무 많이 스케일링
   
   # 맞음
   scores = scores / math.sqrt(d_k)
   ```
   결과: 기울기가 더 작아져 학습이 느려진다.

2. **Mask를 적용한 후에도 가중치가 음수가 되는 경우**
   ```python
   # 틀림
   weights = weights.masked_fill(mask == 0, 0)  # 마스킹 후 처리
   
   # 맞음
   scores = scores.masked_fill(mask == 0, float("-inf"))  # 마스킹 먼저
   weights = F.softmax(scores, dim=-1)
   ```
   이유: -∞를 softmax에 넣으면 자동으로 0이 된다.

3. **배치와 헤드 차원 혼동**
   ```python
   # 틀림
   scores = torch.matmul(Q, K)  # Q: (batch, d_k), K: (batch, d_k)
   
   # 맞음
   scores = torch.matmul(Q, K.transpose(-2, -1))  # 마지막 두 차원만 전치
   ```
   결과: 차원 오류로 코드가 실행되지 않는다.

---

## 체크포인트 2 모범 구현: Self-Attention + Multi-Head Attention

### Self-Attention 모듈

```python
import torch
import torch.nn as nn
import math

class SelfAttention(nn.Module):
    """
    Self-Attention 모듈
    
    같은 입력에서 Q, K, V를 생성하여 Attention을 수행한다.
    이를 통해 문장 내 단어들이 서로를 참조하는 관계를 학습한다.
    """
    
    def __init__(self, d_model):
        """
        Args:
            d_model: 입력 및 출력 차원 (보통 512)
        """
        super().__init__()
        self.d_model = d_model
        
        # Q, K, V 변환 행렬
        # bias=False: Transformer 표준 설정
        # 각 행렬은 (d_model, d_model) 크기의 학습 가능한 가중치
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: 입력 (batch, seq_len, d_model)
                예: "나는 은행에서 돈을 찾았다"의 4개 단어
                   각 단어가 d_model=512 차원 벡터로 표현됨
            mask: 마스크 텐서 (선택)
                예: Causal Mask (Decoder용)
        
        Returns:
            output: Self-Attention 출력 (batch, seq_len, d_model)
                   같은 모양이므로 여러 층을 쌓을 수 있다
            weights: Attention 가중치 (batch, seq_len, seq_len)
                    히트맵 시각화에 사용됨
        """
        # [단계 1] Q, K, V 생성
        # 각각은 입력과 같은 차원이다
        Q = self.W_q(x)  # (batch, seq_len, d_model)
        K = self.W_k(x)  # (batch, seq_len, d_model)
        V = self.W_v(x)  # (batch, seq_len, d_model)
        
        # [단계 2] Scaled Dot-Product Attention 계산
        output, weights = scaled_dot_product_attention(Q, K, V, mask)
        # output: (batch, seq_len, d_model)
        # weights: (batch, seq_len, seq_len)
        
        return output, weights
```

### Multi-Head Attention 모듈

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention 모듈
    
    여러 개의 Attention Head를 병렬로 수행하여
    문법적 관계, 의미적 관계, 위치적 관계 등 다양한 패턴을 동시에 포착한다.
    """
    
    def __init__(self, d_model, num_heads):
        """
        Args:
            d_model: 모델 차원 (보통 512)
            num_heads: Attention Head 개수 (보통 8)
                d_model이 512면 각 Head는 512/8 = 64 차원에서 작동
        """
        super().__init__()
        
        # 검증: d_model이 num_heads로 나누어떨어져야 함
        assert d_model % num_heads == 0, \
            f"d_model({d_model})이 num_heads({num_heads})로 나누어떨어지지 않음"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 각 Head의 차원
        
        # Q, K, V 변환 행렬 (모든 Head의 변환을 한 번에 처리)
        # 크기: (d_model, d_model)
        # 내부적으로는 (d_model, num_heads * d_k)로 생각할 수 있다
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        
        # 출력 변환 행렬 (모든 Head의 출력을 합친 후)
        # 크기: (d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: 입력 (batch, seq_len, d_model)
            mask: 마스크 텐서 (선택)
        
        Returns:
            output: Multi-Head Attention 출력 (batch, seq_len, d_model)
            weights: 모든 Head의 Attention 가중치 (batch, num_heads, seq_len, seq_len)
        """
        batch_size, seq_len, d_model = x.shape
        
        # [단계 1] Q, K, V 생성 (모든 Head 분량)
        Q = self.W_q(x)  # (batch, seq_len, d_model)
        K = self.W_k(x)  # (batch, seq_len, d_model)
        V = self.W_v(x)  # (batch, seq_len, d_model)
        
        # [단계 2] 각 Head의 부분으로 분할
        # (batch, seq_len, d_model) → (batch, seq_len, num_heads, d_k)
        #                           → (batch, num_heads, seq_len, d_k)
        
        # view로 reshape
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k)
        
        # transpose로 차원 순서 바꾸기
        # (batch, seq_len, num_heads, d_k) → (batch, num_heads, seq_len, d_k)
        # 왜? Attention을 num_heads개 병렬로 계산하기 위해
        Q = Q.transpose(1, 2)  # (batch, num_heads, seq_len, d_k)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # [단계 3] 각 Head에서 Attention 계산
        # 모든 Head에 대해 동시에 계산 (배치 차원이 num_heads 포함)
        d_k = self.d_k
        
        # QKᵀ / √dₖ
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        # (batch, num_heads, seq_len, seq_len)
        
        # 마스크 적용
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        
        # Softmax
        weights = F.softmax(scores, dim=-1)
        # (batch, num_heads, seq_len, seq_len)
        
        # Weights × V
        attn_output = torch.matmul(weights, V)
        # (batch, num_heads, seq_len, d_k)
        
        # [단계 4] Head 합치기 (Concatenation)
        # (batch, num_heads, seq_len, d_k) → (batch, seq_len, num_heads, d_k)
        #                                   → (batch, seq_len, d_model)
        
        # transpose로 순서 복원
        attn_output = attn_output.transpose(1, 2)  # (batch, seq_len, num_heads, d_k)
        
        # contiguous: 메모리 연속성 보장 (view 사용하기 전에 필수)
        attn_output = attn_output.contiguous()
        
        # view로 모든 Head 합치기
        attn_output = attn_output.view(batch_size, seq_len, self.d_model)
        # (batch, seq_len, d_model)
        
        # [단계 5] 최종 선형 변환
        # num_heads개 Head의 결과를 통합하는 역할
        output = self.W_o(attn_output)  # (batch, seq_len, d_model)
        
        return output, weights
```

### 핵심 포인트

#### View와 Transpose로 텐서 차원을 변환하는 이유

```python
# 예시: batch=2, seq=4, d_model=8, num_heads=2, d_k=4

# 변환 전: (2, 4, 8)
# 변환 후: (2, 2, 4, 4)  # batch, heads, seq, d_k

# 왜?
# 1. 각 Head를 독립적으로 처리할 수 있다 (병렬화)
# 2. 배치 차원의 일부가 되므로 matmul이 자동으로 처리한다
# 3. torch.matmul은 배치 차원을 모두 처리하므로 매우 효율적이다

# 구체적인 연산:
# Q: (2, 2, 4, 4)
# K.T: (2, 2, 4, 4) → transpose(-2, -1) → (2, 2, 4, 4)
# scores = matmul(Q, K.T): (2, 2, 4, 4)

# GPU: 4개의 배치×Head 조합을 병렬로 처리 (총 4개 계산)
```

#### Head 분할 시 d_k = d_model // num_heads인 이유

```python
# √dₖ 스케일링을 일관되게 유지하기 위해

# 방법 1: 올바른 분할 (d_k = d_model // num_heads)
# d_model = 512, num_heads = 8
# d_k = 512 // 8 = 64
# √dₖ = √64 = 8
# 각 Head에서: scores / 8

# 방법 2: 분할 없이 전체 d_model 사용 (틀림)
# √dₖ = √512 ≈ 22.6
# 각 Head의 실제 차원은 64인데 스케일링이 과도해짐

# 결과: 방법 1이 각 Head에서 일관된 기울기를 유지한다
```

### 흔한 실수

1. **View 후 Contiguous 호출 누락**
   ```python
   # 틀림
   attn_output = attn_output.transpose(1, 2)
   attn_output = attn_output.view(batch_size, seq_len, d_model)  # RuntimeError!
   
   # 맞음
   attn_output = attn_output.transpose(1, 2).contiguous()
   attn_output = attn_output.view(batch_size, seq_len, d_model)
   ```
   이유: transpose 후 메모리가 연속이 아닐 수 있다.

2. **Mask를 num_heads 차원에 맞게 확장하지 않기**
   ```python
   # 틀림
   mask = torch.ones(seq_len, seq_len)
   scores.masked_fill(mask == 0, ...)  # 차원 불일치
   
   # 맞음
   mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
   # 또는 자동으로 브로드캐스트되는 경우 그대로 사용
   ```

3. **transpose(-2, -1)을 두 번 할 때 순서 실수**
   ```python
   # 틀림 - 원상태로 돌아감
   attn_output = attn_output.transpose(-2, -1).transpose(-2, -1)
   
   # 맞음 - 앞뒤 차원을 정확히
   # (batch, num_heads, seq_len, d_k) → (batch, seq_len, num_heads, d_k)
   attn_output = attn_output.transpose(1, 2)
   ```

---

## 체크포인트 3 모범 구현: Attention 시각화 + 감성 분류

### Attention Weight 히트맵 시각화

```python
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from pathlib import Path

def visualize_attention_heatmap(words, attention_weights, title, filename):
    """
    Attention Weight를 히트맵으로 시각화
    
    Args:
        words: 단어 리스트 (예: ["나는", "은행에서", "돈을", "찾았다"])
        attention_weights: (num_heads, seq_len, seq_len) 또는 (seq_len, seq_len)
        title: 그래프 제목
        filename: 저장할 파일명
    """
    # Single Head인 경우를 다루기 위해 차원 확인
    if attention_weights.ndim == 2:
        # (seq_len, seq_len) → (1, seq_len, seq_len)
        attention_weights = attention_weights.unsqueeze(0)
    
    num_heads = attention_weights.shape[0]
    seq_len = attention_weights.shape[1]
    
    # matplotlib 한글 폰트 설정
    import matplotlib
    matplotlib.rcParams['font.family'] = 'AppleGothic'
    matplotlib.rcParams['axes.unicode_minus'] = False
    
    # 서브플롯 생성
    fig, axes = plt.subplots(1, num_heads, figsize=(5*num_heads, 4))
    
    # 단일 Head인 경우 axes를 리스트로 변환
    if num_heads == 1:
        axes = [axes]
    
    # 각 Head별로 히트맵 그리기
    for h in range(num_heads):
        w = attention_weights[h].detach().cpu().numpy()  # (seq_len, seq_len)
        
        # seaborn의 heatmap으로 그리기
        sns.heatmap(
            w,  # 히트맵 데이터
            xticklabels=words,  # x축: 단어들 (Key/Value)
            yticklabels=words,  # y축: 단어들 (Query)
            annot=True,  # 각 셀에 값 표시
            fmt='.2f',  # 소수점 2자리
            cmap='YlOrRd',  # 색상: 노랑→빨강
            vmin=0, vmax=1,  # 값의 범위
            ax=axes[h],  # 서브플롯
            cbar=(h == num_heads-1),  # 마지막 그래프에만 컬러바 표시
        )
        
        axes[h].set_title(f'Head {h+1}', fontsize=12)
        if h == 0:
            axes[h].set_ylabel('Query (관찰자)', fontsize=10)
        axes[h].set_xlabel('Key/Value (관찰 대상)', fontsize=10)
    
    fig.suptitle(title, fontsize=14, y=1.02)
    fig.tight_layout()
    
    # 저장
    output_dir = Path(__file__).parent.parent / 'data' / 'output'
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / filename, dpi=150, bbox_inches='tight')
    print(f'저장: {output_dir / filename}')
    plt.close()


# 사용 예시
# "나는 은행에서 돈을 찾았다"에 대한 Attention 계산
d_model = 32
num_heads = 4

mha = MultiHeadAttention(d_model, num_heads)
words = ["나는", "은행에서", "돈을", "찾았다"]
X = torch.randn(1, len(words), d_model)

output, weights = mha(X)  # weights: (1, num_heads, seq_len, seq_len)
weights = weights[0]  # 배치 차원 제거: (num_heads, seq_len, seq_len)

visualize_attention_heatmap(
    words, weights,
    title="Self-Attention: '나는 은행에서 돈을 찾았다'",
    filename="attention_heatmap_ko.png"
)
```

### 감성 분류 모델 (AttentionClassifier)

```python
import torch
import torch.nn as nn

class AttentionClassifier(nn.Module):
    """
    Attention을 활용한 텍스트 분류 모델
    
    구조:
    입력 → 임베딩 → Multi-Head Attention → 평균 풀링 → 분류기 → 출력
    """
    
    def __init__(self, vocab_size, d_model, num_heads, num_classes):
        """
        Args:
            vocab_size: 어휘 크기 (예: 14)
            d_model: 임베딩 및 Attention 차원 (예: 16)
            num_heads: Attention Head 개수 (예: 2)
            num_classes: 분류 클래스 수 (예: 2 for 긍정/부정)
        """
        super().__init__()
        
        # [계층 1] 임베딩
        # vocab_size개의 단어를 d_model 차원의 벡터로 변환
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # [계층 2] Multi-Head Attention
        # 입력 시퀀스의 단어들이 서로를 참조하도록 함
        self.attention = MultiHeadAttention(d_model, num_heads)
        
        # [계층 3] 분류기 (완전 연결층)
        # d_model 차원의 벡터를 num_classes 차원으로 축소
        self.classifier = nn.Linear(d_model, num_classes)
    
    def forward(self, x):
        """
        Args:
            x: 입력 인덱스 (batch, seq_len) 
               예: [1, 2, 3, 4] → "이 영화 정말 좋다"
        
        Returns:
            logits: 분류 점수 (batch, num_classes)
            weights: Attention 가중치 (batch, num_heads, seq_len, seq_len)
        """
        # [단계 1] 임베딩
        # (batch, seq_len) → (batch, seq_len, d_model)
        embedded = self.embedding(x)
        
        # [단계 2] Attention 계산
        # Attention을 통해 각 단어가 다른 단어들과의 관계를 학습
        attn_out, weights = self.attention(embedded)
        # attn_out: (batch, seq_len, d_model)
        # weights: (batch, num_heads, seq_len, seq_len)
        
        # [단계 3] 평균 풀링
        # 모든 토큰의 Attention 출력을 평균내어 문장 표현으로 만들기
        # 이렇게 하면 문장 길이에 관계없이 고정 크기 벡터를 얻는다
        pooled = attn_out.mean(dim=1)  # (batch, d_model)
        
        # [단계 4] 분류
        logits = self.classifier(pooled)  # (batch, num_classes)
        
        return logits, weights
```

### 학습 및 평가 코드

```python
import torch
import torch.nn as nn
import torch.optim as optim

# 데이터 준비
# 감성 분류: 0=부정, 1=긍정
train_data = [
    ([1, 2, 3, 4], 1),      # 이 영화 정말 좋다
    ([1, 2, 5, 6], 0),      # 이 영화 매우 싫다
    ([7, 8, 3, 4], 1),      # 그 책은 정말 좋다
    ([7, 8, 5, 6], 0),      # 그 책은 매우 싫다
    ([9, 10, 3, 11], 1),    # 오늘 기분 정말 최고
    ([9, 10, 5, 12], 0),    # 오늘 기분 매우 최악
    ([1, 13, 3, 4], 1),     # 이 음식 정말 좋다
    ([1, 13, 5, 6], 0),     # 이 음식 매우 싫다
]

# 단어 사전
vocab_map = {
    0: "<pad>", 1: "이", 2: "영화", 3: "정말", 4: "좋다",
    5: "매우", 6: "싫다", 7: "그", 8: "책은", 9: "오늘",
    10: "기분", 11: "최고", 12: "최악", 13: "음식",
}

# 하이퍼파라미터
vocab_size = 14
d_model = 16
num_heads = 2
num_classes = 2
learning_rate = 0.01
num_epochs = 200

# 모델 초기화
model = AttentionClassifier(vocab_size, d_model, num_heads, num_classes)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# 데이터 준비
X_train = torch.tensor([d[0] for d in train_data])  # (8, 4)
y_train = torch.tensor([d[1] for d in train_data])  # (8,)

print(f"모델 구조:")
print(f"  임베딩: vocab_size={vocab_size} → d_model={d_model}")
print(f"  Attention: {num_heads} heads, d_k={d_model//num_heads}")
print(f"  분류기: {d_model} → {num_classes}")
print(f"  총 파라미터: {sum(p.numel() for p in model.parameters()):,}")

# 학습 루프
losses = []
for epoch in range(num_epochs):
    # Forward pass
    logits, _ = model(X_train)  # (batch, num_classes)
    
    # Loss 계산
    loss = criterion(logits, y_train)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    losses.append(loss.item())
    
    # 로깅
    if (epoch + 1) % 50 == 0:
        with torch.no_grad():
            preds = logits.argmax(dim=1)
            accuracy = (preds == y_train).float().mean()
        print(f"Epoch {epoch+1:3d}: Loss = {loss.item():.4f}, Accuracy = {accuracy:.1%}")

# 예측 및 평가
model.eval()
with torch.no_grad():
    logits, weights = model(X_train)
    preds = logits.argmax(dim=1)

print(f"\n예측 결과:")
labels = ["부정", "긍정"]
for i, (x, y) in enumerate(train_data):
    words = [vocab_map[idx] for idx in x]
    sentence = " ".join(words)
    pred_label = labels[preds[i]]
    true_label = labels[y]
    correct = "O" if preds[i] == y else "X"
    print(f"  '{sentence}'")
    print(f"    → {pred_label} (정답: {true_label}) [{correct}]")

# 특정 예제의 Attention 시각화
sample_idx = 0  # "이 영화 정말 좋다"
sample_words = [vocab_map[idx] for idx in train_data[sample_idx][0]]
sample_weights = weights[sample_idx]  # (num_heads, seq_len, seq_len)

visualize_attention_heatmap(
    sample_words, sample_weights,
    title=f"감성 분류: '{' '.join(sample_words)}' (긍정)",
    filename="sentiment_attention.png"
)
```

### 핵심 포인트

#### 각 Head별 Attention 패턴의 차이가 의미하는 것

```python
# 예: 4개 Head, 5개 단어

# Head 0 가중치 (첫 번째 단어 기준):
# [0.189  0.268  0.073  0.085  0.171  0.214]
# → 단어 1에 집중 (26.8%)

# Head 1 가중치:
# [0.114  0.152  0.127  0.185  0.224  0.198]
# → 단어 4에 집중 (22.4%)

# 해석:
# - Head 0: 바로 다음 단어 관계 학습 (인접 단어 의존성)
# - Head 1: 멀리 떨어진 단어 관계 학습 (장거리 의존성)
# - Head 2, 3: 중간 거리 또는 특정 패턴 학습

# 이렇게 여러 Head가 협력하여 다양한 문법적, 의미적 관계를 포착한다
```

#### 감성 분류에서 Attention이 "좋다/싫다" 같은 단어에 집중하는 패턴

```python
# 훈련 데이터 예시:
# 문장: [이, 영화, 정말, 좋다]  (레이블: 긍정)
# 인덱스: [1,  2,    3,   4]

# 학습 후 Attention Weight 예상 패턴:
#         [이, 영화, 정말, 좋다]
# 이  →   [0.2  0.1   0.1  0.6]   ← 감성 단어 "좋다"에 높은 가중치
# 영화→   [0.1  0.3   0.2  0.4]   ← 관찰 대상과 감성 단어에 집중
# 정말→   [0.1  0.2   0.3  0.4]   ← 강도 단어가 감성 단어 보강
# 좋다→   [0.1  0.2   0.3  0.4]   ← 감성 단어는 자기 자신에 집중

# 이렇게 Attention이 감성 단어를 강조함으로써:
# 1. 분류기는 감성 단어의 영향을 크게 받는다
# 2. 모델이 "좋다/싫다"를 감성 분류의 핵심 신호로 학습한다
```

### 흔한 실수

1. **평균 풀링 대신 마지막 토큰만 사용하기**
   ```python
   # 틀림
   pooled = attn_out[:, -1, :]  # 마지막 토큰만
   
   # 맞음
   pooled = attn_out.mean(dim=1)  # 모든 토큰의 평균
   ```
   이유: Attention이 이미 모든 단어를 혼합했으므로 평균이 더 효과적이다.

2. **단어 인덱스를 0부터 시작하지 않기**
   ```python
   # 틀림
   vocab_map = {
       1: "<pad>", 2: "이", 3: "영화",  # 인덱스가 1부터 시작
   }
   model = AttentionClassifier(vocab_size=14, ...)
   
   # 맞음
   vocab_map = {
       0: "<pad>", 1: "이", 2: "영화",  # 인덱스가 0부터 시작
   }
   ```
   이유: PyTorch의 Embedding은 0부터 vocab_size-1까지 인덱스를 기대한다.

3. **분류 점수(logits)에 소프트맥스를 두 번 적용**
   ```python
   # 틀림
   logits, _ = model(X_train)
   probs = F.softmax(logits, dim=-1)
   preds = probs.argmax(dim=-1)  # 이미 정규화됨
   
   # 맞음
   logits, _ = model(X_train)
   preds = logits.argmax(dim=-1)  # 직접 argmax (소프트맥스 불필요)
   ```

---

## 대안적 구현 방법

### PyTorch 내장 nn.MultiheadAttention 사용

```python
import torch
import torch.nn as nn

class SimpleAttentionClassifier(nn.Module):
    """PyTorch 내장 모듈을 사용한 간단한 구현"""
    
    def __init__(self, vocab_size, d_model, num_heads, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # 내장 Multi-Head Attention 사용
        # batch_first=True: (batch, seq, d_model) 순서로 입력
        self.mha = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            batch_first=True
        )
        
        self.classifier = nn.Linear(d_model, num_classes)
    
    def forward(self, x):
        embedded = self.embedding(x)
        
        # nn.MultiheadAttention은 (seq, batch, d_model) 또는
        # batch_first=True일 때 (batch, seq, d_model)을 기대한다
        attn_out, attn_weights = self.mha(
            query=embedded,
            key=embedded,
            value=embedded,
            need_weights=True
        )
        # attn_out: (batch, seq_len, d_model)
        # attn_weights: (batch, seq_len, seq_len)
        
        pooled = attn_out.mean(dim=1)
        logits = self.classifier(pooled)
        
        return logits, attn_weights
```

**장점**:
- 구현이 간단함
- PyTorch에서 최적화됨 (더 빠를 수 있음)
- 버그 가능성 낮음

**단점**:
- 내부 구현을 제어할 수 없음
- 교육용으로는 직접 구현이 더 좋음

### Masking 없이 Cross-Attention 구현

```python
class CrossAttention(nn.Module):
    """Cross-Attention: Encoder 출력에서 Decoder가 정보를 가져오기"""
    
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Query는 Decoder에서, Key/Value는 Encoder에서
        self.W_q = nn.Linear(d_model, d_model, bias=False)  # Decoder
        self.W_k = nn.Linear(d_model, d_model, bias=False)  # Encoder
        self.W_v = nn.Linear(d_model, d_model, bias=False)  # Encoder
        self.W_o = nn.Linear(d_model, d_model, bias=False)
    
    def forward(self, decoder_x, encoder_output):
        """
        Args:
            decoder_x: Decoder 입력 (batch, tgt_seq, d_model)
            encoder_output: Encoder 출력 (batch, src_seq, d_model)
        """
        batch_size = decoder_x.shape[0]
        tgt_seq = decoder_x.shape[1]
        src_seq = encoder_output.shape[1]
        
        # Q는 Decoder에서, K와 V는 Encoder에서
        Q = self.W_q(decoder_x).view(batch_size, tgt_seq, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(encoder_output).view(batch_size, src_seq, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(encoder_output).view(batch_size, src_seq, self.num_heads, self.d_k).transpose(1, 2)
        
        # Attention 계산
        # Q와 K의 시퀀스 길이가 다를 수 있다
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        # (batch, heads, tgt_seq, src_seq)
        
        weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(weights, V)  # (batch, heads, tgt_seq, d_k)
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, tgt_seq, self.d_model)
        output = self.W_o(attn_output)
        
        return output, weights
```

**사용 예시**:
```python
# 기계 번역에서
cross_attn = CrossAttention(d_model=512, num_heads=8)

encoder_output = ...  # (batch=1, src_seq=5, d_model=512)
decoder_x = ...       # (batch=1, tgt_seq=3, d_model=512)

output, weights = cross_attn(decoder_x, encoder_output)
# output: (batch=1, tgt_seq=3, d_model=512)
# weights: (batch=1, num_heads=8, tgt_seq=3, src_seq=5)
```

---

## 심화 학습 포인트

### 계산 복잡도 분석

| 계산 단계 | 시간 복잡도 | 메모리 복잡도 |
|----------|-----------|-------------|
| QKᵀ 계산 | O(n²d) | O(n²) |
| Softmax | O(n²) | O(n²) |
| Weights × V | O(n²d) | O(n²d) |
| **전체 Self-Attention** | **O(n²d)** | **O(n²)** |
| RNN (h=은닉차원) | O(n·h²) | O(h²) |

**분석**:
- n < d인 경우: Self-Attention이 RNN보다 빠름
- n > d인 경우: RNN이 더 효율적 (그러나 병렬화 불가)
- 현실: Transformer는 병렬화로 인한 이득이 크므로 RNN보다 훨씬 빠름

### 메모리 최적화: Flash Attention

표준 Attention의 메모리 병목:
```python
# 표준 구현
scores = torch.matmul(Q, K.transpose(-2, -1))  # (n, n) 저장
weights = F.softmax(scores, dim=-1)              # (n, n) 저장
output = torch.matmul(weights, V)
```

seq_len=1000인 경우:
- scores: 1000×1000×4bytes = 4MB (크지 않음)
- 하지만 매우 많은 intermediate tensor가 생성됨

**Flash Attention**의 아이디어:
```python
# 블록 단위로 계산하여 중간 결과 저장 최소화
for block in blocks:
    sub_scores = compute_scores(block)
    sub_weights = softmax(sub_scores)
    partial_output = update_output(sub_weights)
```

결과: 메모리 사용량을 ~10배 감소, 계산 속도는 ~2배 향상

### 위치 인코딩 (Positional Encoding)

Self-Attention은 순서 정보가 없다:
```python
# 같은 단어 집합이면 순서가 달라도 같은 출력
words = ["나", "은행", "에서"]
words_shuffled = ["은행", "에서", "나"]
# Attention 결과가 같다 (단어 쌍의 관계만 고려하므로)
```

해결: Positional Encoding 추가
```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len=5000):
        super().__init__()
        
        # PE(pos, 2i) = sin(pos / 10000^(2i/d))
        # PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
        
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * 
                             -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        # x: (batch, seq_len, d_model)
        return x + self.pe[:, :x.size(1), :]
```

**효과**: 같은 위치에 있는 단어는 항상 같은 인코딩을 더하므로 순서 정보가 보존된다.

---

## 참고 코드 파일

다음 파일에서 전체 구현을 확인할 수 있다:

- **practice/chapter3/code/3-1-임베딩.py** — Word2Vec 구현 및 시각화
- **practice/chapter3/code/3-3-어텐션.py** — Scaled Dot-Product Attention, Self-Attention, Multi-Head Attention 구현
- **practice/chapter3/code/3-5-실습.py** — Attention 시각화, 감성 분류 모델, 학습 코드

### 코드 실행 방법

```bash
# 가상환경 활성화 (1장에서 이미 생성)
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate    # Windows

# 각 실습 실행
python practice/chapter3/code/3-1-임베딩.py
python practice/chapter3/code/3-3-어텐션.py
python practice/chapter3/code/3-5-실습.py

# 생성된 시각화 확인
ls practice/chapter3/data/output/
```

---

## 최종 학습 정리

### 3주차 핵심 개념 요약

1. **임베딩**: 단어를 고정된 벡터로 표현 (Word2Vec, GloVe, FastText)
2. **RNN**: 순차 데이터를 처리하지만 장기 의존성 문제 존재
3. **LSTM**: 셀 상태와 게이트로 장기 기억 유지
4. **GRU**: LSTM의 간소화 버전
5. **Attention**: "어디에 집중할 것인가"를 학습하는 메커니즘
6. **Self-Attention**: 같은 시퀀스 내 단어들이 서로를 참조
7. **Multi-Head Attention**: 여러 관점에서 동시에 Attention 수행
8. **√dₖ 스케일링**: 기울기 안정화의 핵심

### 다음 장으로의 연결

다음 장(4장)에서는:
- Positional Encoding으로 순서 정보 추가
- Residual Connection으로 깊은 네트워크 구성
- Layer Normalization으로 학습 안정화
- 이들을 조합한 **Transformer 아키텍처** 완성

이 3장의 Attention이 Transformer의 핵심이므로, 여기서 충분히 이해하고 다음 장으로 진행하자.

---

**생성일**: 2026-02-25
**대상 독자**: 컴퓨터공학/AI 전공 학부생 (3~4학년)
**난이도**: 중급 (파이썬, 딥러닝 기초 선수)
