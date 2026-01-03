# 6장 리서치: Transformer 아키텍처

## 1. Transformer 개요

### 1.1 역사적 배경
- **"Attention is All You Need"** (Vaswani et al., 2017): Transformer 아키텍처 제안
- 2025년 기준 173,000회 이상 인용된 AI 분야 가장 영향력 있는 논문
- RNN의 순차 처리 한계를 극복하고 병렬 처리 가능한 아키텍처

### 1.2 핵심 혁신
- 순환 구조 완전 제거 → 병렬화 용이
- Self-Attention으로 장거리 의존성 직접 모델링
- 모든 위치 간 O(1) path length (RNN은 O(n))

---

## 2. Attention 메커니즘

### 2.1 Query, Key, Value 개념
- **Query (Q)**: 현재 처리하는 위치에서 "무엇을 찾고 있는가"
- **Key (K)**: 각 위치가 가진 "라벨" 또는 "색인"
- **Value (V)**: 각 위치의 실제 정보

### 2.2 Scaled Dot-Product Attention
```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

- **√d_k로 스케일링하는 이유**: d_k가 커지면 dot product 값이 커져 softmax 출력이 극단적이 됨
- 스케일링으로 gradient 안정화

### 2.3 Self-Attention 계산 과정
1. 입력 X에서 Q, K, V 행렬 생성 (Linear projection)
2. Q와 K의 dot product 계산
3. √d_k로 스케일링
4. Softmax 적용 → Attention weights
5. V와 가중합

---

## 3. Multi-Head Attention

### 3.1 필요성
- 단일 Attention은 하나의 관점만 학습
- Multi-Head는 다양한 관계 패턴 동시 학습
  - 주어-동사 관계
  - 수식어-명사 관계
  - 문법적 구조
  - 의미적 유사성

### 3.2 구조
```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
```

### 3.3 일반적인 설정
| 모델 | d_model | num_heads | d_k (= d_model/h) |
|------|---------|-----------|-------------------|
| BERT-base | 768 | 12 | 64 |
| BERT-large | 1024 | 16 | 64 |
| GPT-2 | 768 | 12 | 64 |

---

## 4. Positional Encoding

### 4.1 필요성
- Transformer는 순서 정보 없음 (위치 무관)
- 동일한 단어도 위치에 따라 의미 다름
- "나는 너를 좋아해" vs "너는 나를 좋아해"

### 4.2 Sinusoidal Positional Encoding
```
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

**장점**:
- 파라미터 없음 (parameter-free)
- 학습 시 보지 못한 긴 시퀀스에도 일반화 가능
- 상대적 위치를 선형 함수로 표현 가능

### 4.3 Learned Positional Encoding
- 학습 가능한 임베딩 (GPT-2 등에서 사용)
- 더 유연하지만 최대 길이 제한

### 4.4 최신 기법
- **RoPE (Rotary Position Embedding)**: Llama, Mistral 등에서 사용
- **ALiBi**: Attention에 bias 추가
- **Relative Positional Encoding**: T5에서 사용

---

## 5. Transformer 전체 구조

### 5.1 Encoder 블록 (x6)
```
Input → Multi-Head Self-Attention → Add & Norm → FFN → Add & Norm → Output
```

### 5.2 Decoder 블록 (x6)
```
Input → Masked Self-Attention → Add & Norm →
      Cross-Attention (encoder output) → Add & Norm → FFN → Add & Norm → Output
```

### 5.3 핵심 구성요소

**Feed-Forward Network (FFN)**:
```
FFN(x) = ReLU(xW_1 + b_1)W_2 + b_2
```
- 차원: d_model → 4*d_model → d_model
- 위치별 독립 적용 (position-wise)

**Residual Connection + Layer Normalization**:
```
LayerNorm(x + Sublayer(x))
```
- 기울기 흐름 개선
- 학습 안정화

### 5.4 Masking 종류
- **Padding Mask**: PAD 토큰 무시
- **Look-ahead Mask (Causal Mask)**: Decoder에서 미래 토큰 차단

---

## 6. PyTorch 구현 참고

### 6.1 nn.MultiheadAttention
```python
import torch.nn as nn

mha = nn.MultiheadAttention(embed_dim=512, num_heads=8)
attn_output, attn_weights = mha(query, key, value)
```

### 6.2 직접 구현 시 핵심
- Attention score 계산: `torch.matmul(Q, K.transpose(-2, -1))`
- Scaling: `/ math.sqrt(d_k)`
- Softmax: `F.softmax(scores, dim=-1)`
- 가중합: `torch.matmul(attn_weights, V)`

---

## 7. 참고 자료

### 논문
- Vaswani, A., et al. (2017). Attention is All You Need. NeurIPS.
- Devlin, J., et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers.

### 웹 자료
- The Illustrated Transformer: https://jalammar.github.io/illustrated-transformer/
- PyTorch MultiheadAttention: https://docs.pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html
- UvA DL Notebooks Tutorial 6: https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html
- Dive into Deep Learning 11.5: https://d2l.ai/chapter_attention-mechanisms-and-transformers/multihead-attention.html
