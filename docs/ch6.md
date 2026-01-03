# 6장 혁신의 중심: Transformer 아키텍처

## 학습 목표

이 장을 마치면 다음을 수행할 수 있다:
- RNN/LSTM의 한계와 Transformer의 등장 배경을 설명할 수 있다
- Attention 메커니즘의 Query, Key, Value 개념을 이해한다
- Self-Attention의 계산 과정을 단계별로 설명할 수 있다
- Multi-Head Attention의 필요성과 구조를 이해한다
- Positional Encoding의 역할과 구현 방법을 설명할 수 있다
- Transformer Encoder/Decoder 블록의 구조를 이해한다
- PyTorch로 Self-Attention과 Transformer Encoder를 구현할 수 있다

---

## 6.1 Transformer 등장 배경

### 6.1.1 RNN/LSTM의 한계

5장에서 학습한 RNN과 LSTM은 순차 데이터 처리에 효과적이지만, 몇 가지 근본적인 한계를 가진다.

첫째, 순차 처리로 인한 병렬화의 어려움이다. RNN은 시간 t의 출력을 계산하기 위해 시간 t-1의 Hidden State가 필요하다. 이 순차적 의존성으로 인해 GPU의 병렬 처리 능력을 충분히 활용하기 어렵다. 시퀀스 길이가 n이면 n번의 순차적 연산이 필요하다.

둘째, 장거리 의존성 문제가 여전히 남아있다. LSTM이 기울기 소실을 크게 개선했지만, 매우 긴 시퀀스에서는 초기 정보가 희석된다. 위치 1과 위치 100 사이의 정보 전달을 위해서는 99개의 Hidden State를 거쳐야 한다.

셋째, 학습 시간이 오래 걸린다. 순차 처리와 장거리 의존성 문제로 인해 대규모 데이터셋에서의 학습이 비효율적이다.

### 6.1.2 "Attention is All You Need"

2017년 Google 연구팀은 "Attention is All You Need"라는 제목의 논문을 발표했다. 이 논문은 순환 구조를 완전히 제거하고, Attention 메커니즘만으로 시퀀스를 처리하는 Transformer 아키텍처를 제안했다.

Transformer의 핵심 혁신은 다음과 같다:

첫째, 순환 구조의 완전한 제거이다. 모든 위치의 입력을 동시에 처리할 수 있으므로 병렬화가 가능하다.

둘째, Self-Attention을 통한 직접 연결이다. 시퀀스 내 모든 위치가 직접 연결되므로, 어떤 두 위치 사이의 경로 길이가 O(1)이다. RNN에서 O(n)이었던 장거리 의존성 문제가 근본적으로 해결된다.

셋째, 확장성이다. 병렬 처리가 가능하므로 더 큰 모델과 더 많은 데이터로 학습할 수 있다. 이는 후에 BERT, GPT 등 대규모 언어 모델의 기반이 되었다.

2025년 현재, 이 논문은 173,000회 이상 인용되며 AI 분야에서 가장 영향력 있는 논문 중 하나로 자리잡았다.

---

## 6.2 Attention 메커니즘

### 6.2.1 Attention의 직관적 이해

Attention은 인간이 정보를 처리하는 방식에서 영감을 받았다. 우리가 문장을 읽을 때, 모든 단어를 동등하게 처리하지 않는다. 현재 이해하려는 부분과 관련된 단어에 더 많은 "주의"를 기울인다.

예를 들어, "나는 어제 서울에서 친구를 만났다"라는 문장에서 "만났다"의 주어가 무엇인지 파악하려면 "나는"에 주의를 기울여야 한다. "서울에서"는 장소를 나타내므로 주어 파악에는 덜 중요하다.

Attention 메커니즘은 이러한 선택적 집중을 수학적으로 모델링한다. 현재 처리하는 위치에서 다른 모든 위치를 "참조"하고, 각 위치의 중요도에 따라 가중 평균을 계산한다.

### 6.2.2 Query, Key, Value

Attention은 Query, Key, Value라는 세 가지 개념을 사용한다. 이를 도서관 비유로 이해할 수 있다.

**Query (질문)**: 현재 찾고 있는 정보를 나타낸다. 도서관에서 "머신러닝 입문서"를 찾고 있다면, 이것이 Query이다.

**Key (색인)**: 각 정보의 "라벨" 또는 "색인"이다. 도서관의 각 책은 제목, 분류번호 등의 Key를 가진다. Query와 Key를 비교하여 관련성을 측정한다.

**Value (값)**: 실제 정보의 내용이다. 책의 실제 내용이 Value에 해당한다. Query와 Key의 유사도가 높으면 해당 Value를 더 많이 참조한다.

### 6.2.3 Scaled Dot-Product Attention

Transformer에서 사용하는 Attention은 Scaled Dot-Product Attention이다. 수식은 다음과 같다:

Attention(Q, K, V) = softmax(QK^T / √d_k) V

계산 과정을 단계별로 살펴보자:

**1단계: Attention Score 계산**
Query와 Key의 내적(dot product)을 계산한다. 이 값이 클수록 두 벡터가 유사하다는 의미이다.

**2단계: 스케일링**
Score를 √d_k로 나눈다. d_k는 Key 벡터의 차원이다. 스케일링이 필요한 이유는 d_k가 커질수록 내적 값이 커져 Softmax 출력이 극단적으로 치우칠 수 있기 때문이다.

실행 결과에서 확인할 수 있다:

```
[스케일링 전]
  Score 표준편차: 24.5367
  Score 범위: [-51.89, 56.63]

[스케일링 후 (÷√512)]
  Score 표준편차: 1.0844
  Score 범위: [-2.29, 2.50]
```

스케일링 후 값의 범위가 크게 줄어들어 Softmax가 더 균일한 분포를 출력한다.

**3단계: Softmax**
스케일링된 Score에 Softmax를 적용하여 확률 분포로 변환한다. 모든 값의 합이 1이 되며, 각 위치에 대한 Attention Weight가 된다.

**4단계: 가중합**
Attention Weight와 Value의 가중합을 계산한다. 중요도가 높은 위치의 Value가 더 많이 반영된다.

_전체 코드는 practice/chapter6/code/6-2-attention.py 참고_

---

## 6.3 Self-Attention

### 6.3.1 Self-Attention의 개념

Self-Attention은 같은 시퀀스 내에서 각 위치가 다른 모든 위치를 참조하는 Attention이다. "Self"가 붙은 이유는 Query, Key, Value가 모두 동일한 입력 시퀀스에서 생성되기 때문이다.

예를 들어, "The cat sat on the mat because it was tired"라는 문장에서 "it"이 무엇을 가리키는지 파악하려면, "it"의 Query가 "cat"의 Key와 높은 유사도를 가져야 한다.

### 6.3.2 Self-Attention 계산 과정

입력 시퀀스 X가 주어지면, Self-Attention은 다음과 같이 계산된다:

**1단계: Q, K, V 생성**
입력 X에 세 개의 학습 가능한 가중치 행렬을 곱하여 Q, K, V를 생성한다:
- Q = X × W^Q
- K = X × W^K
- V = X × W^V

**2단계: Attention Score 계산**
각 Query와 모든 Key의 내적을 계산한다. n개의 위치가 있으면 n×n 크기의 Attention Score 행렬이 생성된다.

**3단계: Scaled Softmax**
Score를 √d_k로 나누고 Softmax를 적용한다.

**4단계: 가중합**
Attention Weight와 V의 행렬 곱을 계산한다.

실행 결과:

```
[입력 정보]
  배치 크기: 1
  시퀀스 길이: 5
  모델 차원 (d_model): 64
  Key/Query 차원 (d_k): 32

[출력 정보]
  출력 shape: torch.Size([1, 5, 32])
  Attention weights shape: torch.Size([1, 5, 5])

[Attention Weights (첫 번째 샘플)]
  각 행의 합 (softmax 확인): [1.0, 1.0, 1.0, 1.0, 1.0]
```

Attention Weight 행렬의 각 행의 합이 1인 것을 확인할 수 있다. 이는 Softmax가 올바르게 적용되었음을 의미한다.

### 6.3.3 Self-Attention의 장점

Self-Attention은 RNN 대비 몇 가지 중요한 장점을 가진다:

**병렬 처리**: 모든 위치의 Attention을 동시에 계산할 수 있다. 시퀀스 길이에 관계없이 단일 행렬 연산으로 처리된다.

**직접 연결**: 모든 위치 쌍이 직접 연결된다. 위치 1과 위치 100 사이의 정보 전달에 단 한 번의 Attention만 필요하다.

**해석 가능성**: Attention Weight를 시각화하면 모델이 어떤 정보에 집중하는지 파악할 수 있다.

단점은 계산 복잡도가 O(n²)라는 점이다. 시퀀스 길이 n에 대해 n×n 크기의 Attention 행렬을 계산해야 한다. 이는 매우 긴 시퀀스에서 메모리 문제를 야기할 수 있다.

---

## 6.4 Multi-Head Attention

### 6.4.1 Multi-Head Attention의 필요성

단일 Attention Head는 하나의 관점에서만 관계를 학습한다. 그러나 언어에는 다양한 유형의 관계가 존재한다:

- 문법적 관계: 주어-동사, 목적어-동사
- 의미적 관계: 동의어, 반의어
- 지시 관계: 대명사와 선행사
- 수식 관계: 형용사-명사, 부사-동사

단일 Attention으로는 이러한 다양한 관계를 동시에 포착하기 어렵다. Multi-Head Attention은 여러 개의 독립적인 Attention Head를 병렬로 실행하여 다양한 관점에서 관계를 학습한다.

### 6.4.2 Multi-Head Attention 구조

Multi-Head Attention은 다음과 같이 계산된다:

MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O

head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)

각 Head는 독립적인 가중치 행렬 W^Q, W^K, W^V를 가진다. 모든 Head의 출력을 연결(Concatenate)한 후, 최종 선형 변환 W^O를 적용한다.

### 6.4.3 차원 분할

효율성을 위해 각 Head의 차원을 줄인다. d_model이 전체 모델 차원이고 h가 Head 수일 때:

d_k = d_v = d_model / h

예를 들어, BERT-base는 d_model=768, h=12이므로 각 Head의 차원은 64이다.

실행 결과:

```
[모델 설정]
  d_model: 256
  num_heads: 8
  d_k (= d_model / num_heads): 32
  d_ff: 1024
  num_layers: 4

[Multi-Head Attention]
  입력 shape: torch.Size([2, 10, 256])
  출력 shape: torch.Size([2, 10, 256])
  Attention weights shape: torch.Size([2, 8, 10, 10])
```

Attention Weight가 (batch, num_heads, seq_len, seq_len) 형태인 것을 확인할 수 있다. 8개의 Head가 각각 독립적인 10×10 Attention 행렬을 생성한다.

_전체 코드는 practice/chapter6/code/6-6-transformer-encoder.py 참고_

---

## 6.5 Positional Encoding

### 6.5.1 위치 정보의 필요성

Self-Attention은 순서 정보를 가지지 않는다. 입력의 순서를 바꿔도 Attention 결과는 동일하게 순서만 바뀐다. 그러나 언어에서 순서는 매우 중요하다:

- "개가 사람을 물었다" vs "사람이 개를 물었다"
- "I love you" vs "You love I"

동일한 단어로 구성되어도 순서에 따라 의미가 완전히 달라진다. 따라서 Transformer는 입력에 위치 정보를 추가해야 한다.

### 6.5.2 Sinusoidal Positional Encoding

원본 Transformer 논문에서는 Sinusoidal Positional Encoding을 제안했다:

PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

pos는 위치, i는 차원 인덱스이다. 짝수 차원은 sine, 홀수 차원은 cosine 함수를 사용한다.

이 설계의 장점:

**파라미터가 없다**: 수학적 함수로 정의되므로 학습이 필요 없다.

**일반화 가능**: 학습 시 보지 못한 긴 시퀀스에도 적용할 수 있다.

**상대적 위치 표현**: 임의의 고정 오프셋 k에 대해, PE(pos+k)를 PE(pos)의 선형 함수로 표현할 수 있다.

실행 결과:

```
[Sinusoidal Positional Encoding]
  학습 가능한 파라미터 수: 0 (파라미터 없음)

  [첫 3개 위치의 PE 값 (처음 8차원)]
    위치 0: ['0.000', '1.000', '0.000', '1.000', '0.000', '1.000', '0.000', '1.000']
    위치 1: ['0.841', '0.540', '0.762', '0.648', '0.682', '0.732', '0.605', '0.796']
    위치 2: ['0.909', '-0.416', '0.987', '-0.160', '0.997', '0.071', '0.963', '0.269']

  [위치 간 코사인 유사도]
    위치 0 ↔ 위치 1: 0.9702
    위치 0 ↔ 위치 5: 0.7373
    위치 0 ↔ 위치 10: 0.6691
    위치 0 ↔ 위치 50: 0.5462
```

가까운 위치일수록 Positional Encoding의 유사도가 높다. 이는 상대적 위치 정보가 인코딩되었음을 의미한다.

### 6.5.3 Learned Positional Encoding

GPT-2, BERT 등에서는 학습 가능한 Positional Encoding을 사용한다. 각 위치에 대해 d_model 차원의 임베딩 벡터를 학습한다.

장점:
- 태스크에 최적화된 위치 표현 학습 가능

단점:
- 최대 시퀀스 길이가 고정됨
- 학습 시 보지 못한 위치에 대해 일반화 불가

실행 결과:

```
[Learned Positional Encoding]
  학습 가능한 파라미터 수: 12,800
  = max_len × d_model = 100 × 128
```

| 특성 | Sinusoidal | Learned |
|------|------------|---------|
| 파라미터 | 없음 | max_len × d_model |
| 긴 시퀀스 일반화 | 가능 | 불가능 |
| 표현력 | 고정 | 유연 |
| 사용 예 | 원본 Transformer | GPT-2, BERT |

**표 6.1** Positional Encoding 방식 비교

_전체 코드는 practice/chapter6/code/6-5-positional-encoding.py 참고_

---

## 6.6 Transformer 구조

### 6.6.1 Encoder 블록

Transformer Encoder는 동일한 구조의 블록을 N번 쌓은 것이다. 각 블록은 두 개의 서브레이어로 구성된다:

**1. Multi-Head Self-Attention**: 시퀀스 내 모든 위치 간의 관계를 학습한다.

**2. Position-wise Feed-Forward Network**: 각 위치에 독립적으로 적용되는 2층 MLP이다.

FFN(x) = ReLU(xW_1 + b_1)W_2 + b_2

일반적으로 은닉층의 차원은 d_model의 4배이다 (예: d_model=512이면 d_ff=2048).

각 서브레이어에는 Residual Connection과 Layer Normalization이 적용된다:

output = LayerNorm(x + Sublayer(x))

Residual Connection은 기울기 흐름을 개선하고, Layer Normalization은 학습을 안정화한다.

### 6.6.2 Decoder 블록

Decoder는 Encoder와 유사하지만 세 개의 서브레이어를 가진다:

**1. Masked Multi-Head Self-Attention**: 미래 위치를 참조하지 못하도록 마스킹된 Self-Attention이다. 자기회귀 생성 시 정보 누출을 방지한다.

**2. Encoder-Decoder Attention (Cross-Attention)**: Query는 Decoder에서, Key와 Value는 Encoder 출력에서 가져온다. 입력 시퀀스의 관련 정보를 참조한다.

**3. Position-wise Feed-Forward Network**: Encoder와 동일하다.

### 6.6.3 Masking

Transformer에서는 두 가지 마스킹을 사용한다:

**Padding Mask**: 배치 처리 시 길이가 다른 시퀀스를 맞추기 위해 추가된 PAD 토큰을 무시한다.

**Look-ahead Mask (Causal Mask)**: Decoder에서 미래 토큰을 보지 못하도록 한다. 하삼각 행렬 형태이다.

실행 결과:

```
[Causal Mask]
tensor([[1., 0., 0., 0., 0.],
        [1., 1., 0., 0., 0.],
        [1., 1., 1., 0., 0.],
        [1., 1., 1., 1., 0.],
        [1., 1., 1., 1., 1.]])

[Masked Attention Weights]
  첫 번째 행 (첫 토큰): [1.0, 0.0, 0.0, 0.0, 0.0]
  두 번째 행 (둘째 토큰): [0.42, 0.58, 0.0, 0.0, 0.0]
  → 미래 위치(0.0)에는 attention이 적용되지 않음
```

첫 번째 토큰은 자기 자신만 참조할 수 있고, 두 번째 토큰은 첫 번째와 자신만 참조할 수 있다.

---

## 6.7 실습: Transformer 구현

이 절에서는 Transformer Encoder를 PyTorch로 구현하고 텍스트 분류에 적용한다.

### 6.7.1 Multi-Head Attention 구현

핵심 코드:

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        # 1. Linear projection
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)

        # 2. Reshape for multi-head
        Q = Q.view(batch, -1, self.num_heads, self.d_k).transpose(1, 2)

        # 3. Scaled Dot-Product Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attention_weights = F.softmax(scores, dim=-1)

        # 4. Weighted sum and concatenate
        context = torch.matmul(attention_weights, V)
        output = self.W_o(context.view(batch, -1, d_model))
        return output
```

### 6.7.2 Transformer Encoder Block 구현

```python
class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        # Self-Attention + Residual + Norm
        attn_output, _ = self.self_attention(x, x, x, mask)
        x = self.norm1(x + attn_output)

        # Feed-Forward + Residual + Norm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)
        return x
```

### 6.7.3 모델 크기 비교

실행 결과:

```
[Transformer Encoder Block]
  파라미터 수: 789,760

[Transformer Encoder (4 layers)]
  총 파라미터 수: 3,159,040

[Transformer Classifier]
  총 파라미터 수: 5,850,626
```

| 모델 | d_model | heads | layers | 파라미터 |
|------|---------|-------|--------|----------|
| 현재 모델 | 256 | 8 | 4 | ~5.9M |
| BERT-base | 768 | 12 | 12 | ~110M |
| BERT-large | 1024 | 16 | 24 | ~340M |
| GPT-2 small | 768 | 12 | 12 | ~117M |

**표 6.2** 모델 크기 비교

_전체 코드는 practice/chapter6/code/6-6-transformer-encoder.py 참고_

---

## 요약

이 장에서는 현대 NLP의 핵심인 Transformer 아키텍처를 학습했다.

**Attention 메커니즘**: Query, Key, Value를 사용하여 관련 정보에 선택적으로 집중한다. Scaled Dot-Product Attention은 √d_k로 스케일링하여 학습을 안정화한다.

**Self-Attention**: 같은 시퀀스 내 모든 위치 간의 관계를 학습한다. O(1) 경로 길이로 장거리 의존성을 효과적으로 모델링하며, 병렬 처리가 가능하다.

**Multi-Head Attention**: 여러 관점에서 관계를 학습한다. 문법적, 의미적 관계 등 다양한 패턴을 동시에 포착한다.

**Positional Encoding**: Attention의 순서 무관성을 보완한다. Sinusoidal 방식은 파라미터 없이 긴 시퀀스에 일반화되고, Learned 방식은 태스크에 최적화된다.

**Transformer 구조**: Encoder는 Self-Attention과 FFN의 블록을 쌓는다. Decoder는 Masked Self-Attention과 Cross-Attention을 추가로 사용한다. Residual Connection과 Layer Normalization으로 학습을 안정화한다.

Transformer는 BERT, GPT, T5 등 모든 현대 언어 모델의 기반이 되었다. 다음 장에서는 이러한 사전학습 언어 모델에 대해 더 자세히 학습한다.

---

## 연습 문제

1. Self-Attention의 계산 복잡도가 O(n²)인 이유를 설명하고, 긴 시퀀스에서 이를 해결하기 위한 방법을 제안하라.

2. Multi-Head Attention에서 Head 수를 늘리면 어떤 영향이 있는가? d_model을 고정했을 때의 트레이드오프를 설명하라.

3. Sinusoidal Positional Encoding에서 다양한 주파수를 사용하는 이유를 설명하라.

4. Transformer Encoder 블록에서 Residual Connection과 Layer Normalization의 순서가 중요한가? Pre-LN과 Post-LN의 차이를 조사하라.

5. PyTorch의 nn.TransformerEncoder와 직접 구현한 Encoder의 출력을 비교하고, 동일한 결과를 얻도록 파라미터를 초기화하라.

---

## 참고문헌

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention Is All You Need. *NeurIPS*.

Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2018). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. *NAACL*.

The Illustrated Transformer. Jay Alammar. https://jalammar.github.io/illustrated-transformer/

PyTorch Documentation. (2025). MultiheadAttention. https://docs.pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html
