## 3주차 B회차: Self-Attention 구현 + 시각화

> **미션**: Self-Attention 모듈을 구현하고 Multi-Head Attention의 패턴을 시각화하며 Attention 기반 텍스트 분류 모델을 완성할 수 있다

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

**Self-Attention의 원리**:
- Query (Q), Key (K), Value (V)는 같은 입력에서 생성되어 문장 안에서 단어들이 서로를 참조한다
- Scaled Dot-Product Attention의 핵심: Attention(Q, K, V) = softmax(QKᵀ / √dₖ) · V
- √dₖ 스케일링은 내적 값의 분산을 안정화하여 softmax 기울기 소실을 방지한다

**Multi-Head Attention의 가치**:
- d_model 차원을 num_heads개로 분할하여 각 Head가 독립적으로 서로 다른 관계를 학습한다
- 문법적, 의미적, 위치적 관계 등 다양한 패턴을 동시에 포착한다
- 각 Head의 출력을 합쳐 최종 표현을 생성한다

**실습 연계**:
- 지난 수업에서 배운 Attention의 이론을 PyTorch로 직접 구현한다
- Attention Weight를 시각화하여 모델의 의사결정 과정을 해석한다
- 이러한 이해가 4장 Transformer 구현의 기초가 된다

---

### 과제 스펙

**과제**: Self-Attention 및 Multi-Head Attention 모듈 구현 + Attention Weight 시각화 리포트

**제출 형태**: 조별 1부, Google Classroom 업로드

**파일 구성**:
- 구현 코드 파일 (`*.py`)
- 시각화 결과 이미지 (히트맵 2개 이상)
- 간단한 분석 리포트 (1-2페이지)

**검증 기준**:
- ✓ Scaled Dot-Product Attention 함수 동작 확인
- ✓ Self-Attention 모듈 구현 및 출력 차원 검증
- ✓ Multi-Head Attention 구현 및 각 Head별 패턴 확인
- ✓ Attention Weight 히트맵 시각화
- ✓ 감성 분류 모델 학습 및 결과 해석

---

### 2인1조 실습

> **Copilot 활용**: Attention Score 계산 코드를 직접 작성해본 뒤, Copilot에게 "이 Attention 구현을 Multi-Head로 확장해줘", "Attention Weight를 시각화하는 코드 작성해줄래?" 같이 단계적으로 요청한다. Copilot의 제안을 검토하고 수정하는 과정에서 Attention의 작동 원리를 더욱 깊이 이해할 수 있다.

**역할 분담**:
- **조원 A (드라이버)**: 코드 작성 및 실행, 결과 확인
- **조원 B (네비게이터)**: 로직 검토, Copilot 프롬프트 설계, 오류 해석
- **체크포인트마다 역할 교대**: 드라이버와 네비게이터를 번갈아가며 진행하여 두 명 모두 전체 구현을 이해한다

---

#### 체크포인트 1: Scaled Dot-Product Attention 구현 (15분)

**목표**: NumPy와 PyTorch로 Scaled Dot-Product Attention을 구현하고 √dₖ 스케일링의 효과를 확인한다

**핵심 단계**:

① **NumPy 구현** — 로우 레벨에서 Attention의 계산 과정을 이해한다

```python
import numpy as np
import math

def scaled_dot_product_attention_numpy(Q, K, V):
    """
    Args:
        Q, K, V: (seq_len, d_k) 행렬
    Returns:
        output: Attention 출력
        weights: Attention 가중치
    """
    d_k = Q.shape[-1]
    
    # 1단계: QKᵀ 계산
    scores = Q @ K.T
    
    # 2단계: √dₖ로 스케일링
    scaled_scores = scores / math.sqrt(d_k)
    
    # 3단계: Softmax 적용
    weights = scipy.special.softmax(scaled_scores, axis=1)
    
    # 4단계: Weights × V
    output = weights @ V
    
    return output, weights
```

예상 동작:
```
입력: 4개 단어, 임베딩 차원 8
Q, K, V shape: (4, 8)

[단계 1] QKᵀ (스케일링 전):
  점수 분산: 약 6.8

[단계 2] QKᵀ / √8 (스케일링 후):
  점수 분산: 약 0.85 (기대값: ~1.0)

[단계 3] Softmax 적용:
  각 행의 합: [1.0, 1.0, 1.0, 1.0]
```

② **PyTorch 구현** — 배치 처리와 GPU 호환성을 지원한다

```python
import torch
import torch.nn.functional as F

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
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
    
    # 마스크 적용 (Causal Mask 등)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    
    # Softmax
    weights = F.softmax(scores, dim=-1)
    
    # Weights × V
    output = torch.matmul(weights, V)
    
    return output, weights
```

③ **테스트** — "나는 은행에서 돈을 찾았다" 문장으로 동작 확인

```python
words = ["나는", "은행에서", "돈을", "찾았다"]
X = torch.randn(1, 4, 8)  # (batch=1, seq=4, d_model=8)

output, weights = scaled_dot_product_attention(X, X, X)

# Attention Weight 출력
for i, word in enumerate(words):
    print(f"{word}: {weights[0, i].numpy()}")
```

예상 결과:
```
나는:     [0.240  0.258  0.246  0.256]
은행에서: [0.245  0.252  0.250  0.253]
돈을:     [0.251  0.247  0.264  0.238]
찾았다:   [0.238  0.255  0.241  0.266]
```

**검증 체크리스트**:
- [ ] NumPy 구현에서 softmax 함수가 올바르게 작동하는가?
- [ ] 스케일링 전후 분산 차이를 확인했는가? (약 8배)
- [ ] PyTorch 구현에서 배치 크기가 1일 때와 여러 개일 때 모두 동작하는가?
- [ ] 마스크 파라미터가 주어졌을 때 해당 위치의 가중치가 0이 되는가?

**Copilot 프롬프트 1**:
```
"NumPy로 Scaled Dot-Product Attention을 구현해줄래? 입력은 Q, K, V이고 √dₖ로 스케일링해야 해. 
softmax도 수치적으로 안정적으로 구현해줘."
```

**Copilot 프롬프트 2**:
```
"위의 NumPy 구현을 PyTorch 버전으로 바꿔줄래? 배치 처리를 지원해야 하고 mask 파라미터도 추가해줘."
```

---

#### 체크포인트 2: Self-Attention + Multi-Head Attention 구현 (20분)

**목표**: Self-Attention 모듈을 먼저 구현한 뒤, Multi-Head로 확장하고 각 Head가 다른 패턴을 포착하는지 확인한다

**핵심 단계**:

① **Self-Attention 모듈** — nn.Module로 구현하여 학습 가능한 가중치 추가

```python
import torch.nn as nn

class SelfAttention(nn.Module):
    """Self-Attention 모듈"""
    
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        # 학습 가능한 가중치 행렬
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: 입력 (batch, seq_len, d_model)
        Returns:
            output: Attention 출력 (batch, seq_len, d_model)
            weights: Attention 가중치 (batch, seq_len, seq_len)
        """
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        
        output, weights = scaled_dot_product_attention(Q, K, V, mask)
        return output, weights
```

② **Multi-Head Attention 모듈** — d_model을 num_heads로 분할하여 병렬 처리

```python
class MultiHeadAttention(nn.Module):
    """Multi-Head Attention 모듈"""
    
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model은 num_heads로 나누어떨어져야 함"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 각 Head의 차원
        
        # 학습 가능한 가중치
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)  # 출력 변환
    
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        
        # Q, K, V 계산: (batch, seq, d_model)
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        
        # Multi-Head로 분할: (batch, seq, d_model) → (batch, heads, seq, d_k)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # 각 Head에서 Attention 계산
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(weights, V)
        
        # Head 합치기: (batch, heads, seq, d_k) → (batch, seq, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.d_model)
        
        # 출력 변환
        output = self.W_o(attn_output)
        
        return output, weights
```

③ **테스트 및 검증**

```python
# Self-Attention 테스트
d_model = 16
self_attn = SelfAttention(d_model)

X = torch.randn(1, 5, d_model)
output, weights = self_attn(X)

print(f"Self-Attention:")
print(f"  입력 크기: {X.shape}")
print(f"  출력 크기: {output.shape}")
print(f"  가중치 크기: {weights.shape}")
print(f"  파라미터 수: {sum(p.numel() for p in self_attn.parameters()):,}")

# Multi-Head Attention 테스트
d_model = 32
num_heads = 4
mha = MultiHeadAttention(d_model, num_heads)

X = torch.randn(1, 6, d_model)
output, weights = mha(X)

print(f"\nMulti-Head Attention (4 heads):")
print(f"  입력 크기: {X.shape}")
print(f"  출력 크기: {output.shape}")
print(f"  가중치 크기: {weights.shape} (batch, heads, seq, seq)")
print(f"  파라미터 수: {sum(p.numel() for p in mha.parameters()):,}")

# 각 Head의 패턴 확인
print(f"\n각 Head가 서로 다른 패턴을 포착하는가?")
for h in range(num_heads):
    w = weights[0, h, 0].detach().numpy()
    max_idx = w.argmax()
    print(f"  Head {h}: 최대 주목 = 단어 {max_idx}")
```

예상 결과:
```
Self-Attention:
  입력 크기: (1, 5, 16)
  출력 크기: (1, 5, 16)
  가중치 크기: (1, 5, 5)
  파라미터 수: 768

Multi-Head Attention (4 heads):
  입력 크기: (1, 6, 32)
  출력 크기: (1, 6, 32)
  가중치 크기: (1, 4, 6, 6)
  파라미터 수: 4,096

각 Head가 서로 다른 패턴을 포착하는가?
  Head 0: 최대 주목 = 단어 1
  Head 1: 최대 주목 = 단어 4
  Head 2: 최대 주목 = 단어 3
  Head 3: 최대 주목 = 단어 2
```

**검증 체크리스트**:
- [ ] Self-Attention의 입출력 차원이 동일한가?
- [ ] Multi-Head Attention의 출력 차원이 입력과 같은가?
- [ ] 가중치 행렬 W_q, W_k, W_v, W_o가 모두 생성되었는가?
- [ ] 각 Head가 실제로 다른 단어에 집중하는가? (위 결과에서 최대 주목 단어가 다름)

**Copilot 프롬프트 3**:
```
"Self-Attention을 nn.Module로 구현해줄래? nn.Linear로 W_q, W_k, W_v를 만들고 
Scaled Dot-Product Attention을 호출해야 해."
```

**Copilot 프롬프트 4**:
```
"위의 Self-Attention을 Multi-Head로 확장해줄래? num_heads를 파라미터로 받아서 
d_model을 num_heads로 나누고, 각 Head에서 Attention을 계산한 뒤 합쳐야 해."
```

---

#### 체크포인트 3: Attention Weight 시각화 + 감성 분류 (20분)

**목표**: Attention Weight를 히트맵으로 시각화하여 모델의 의사결정을 해석하고, Attention 기반 텍스트 분류 모델을 완성한다

**핵심 단계**:

① **Attention Weight 히트맵 시각화**

```python
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_attention_heatmap(words, mha, title, filename):
    """Attention Weight를 히트맵으로 시각화"""
    seq_len = len(words)
    d_model = mha.d_model
    num_heads = mha.num_heads
    
    # 입력 생성 (임의)
    X = torch.randn(1, seq_len, d_model)
    _, weights = mha(X)  # (1, heads, seq, seq)
    
    # 각 Head별로 히트맵 생성
    fig, axes = plt.subplots(1, num_heads, figsize=(5*num_heads, 4))
    if num_heads == 1:
        axes = [axes]
    
    for h in range(num_heads):
        w = weights[0, h].detach().numpy()
        sns.heatmap(w, xticklabels=words, yticklabels=words,
                   annot=True, fmt=".2f", cmap="YlOrRd",
                   vmin=0, vmax=1, ax=axes[h], cbar=(h == num_heads-1))
        axes[h].set_title(f"Head {h+1}")
    
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"저장: {filename}")
    plt.close()

# 한국어 문장 시각화
sentence_ko = ["나는", "은행에서", "돈을", "찾았다"]
mha_ko = MultiHeadAttention(d_model=32, num_heads=4)
visualize_attention_heatmap(sentence_ko, mha_ko,
                           "Self-Attention: '나는 은행에서 돈을 찾았다'",
                           "attention_heatmap_ko.png")

# 영어 문장 시각화
sentence_en = ["The", "cat", "sat", "on", "the", "mat"]
mha_en = MultiHeadAttention(d_model=32, num_heads=4)
visualize_attention_heatmap(sentence_en, mha_en,
                           "Self-Attention: 'The cat sat on the mat'",
                           "attention_heatmap_en.png")
```

예상 결과:
```
저장: attention_heatmap_ko.png
저장: attention_heatmap_en.png
```

② **Single-Head vs Multi-Head 비교** — Multi-Head의 가치를 시각적으로 보여준다

```python
# Single Head와 Multi Head의 차이 시각화
words = ["I", "love", "natural", "language", "processing"]
X = torch.randn(1, len(words), 32)

single_head = MultiHeadAttention(32, num_heads=1)
_, single_weights = single_head(X)

multi_head = MultiHeadAttention(32, num_heads=4)
_, multi_weights = multi_head(X)

fig, axes = plt.subplots(1, 5, figsize=(20, 4))

# Single Head
sns.heatmap(single_weights[0, 0].numpy(), xticklabels=words, yticklabels=words,
           annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1,
           ax=axes[0], cbar=False)
axes[0].set_title("Single Head")

# Multi Head (4개)
for h in range(4):
    sns.heatmap(multi_weights[0, h].numpy(), xticklabels=words, yticklabels=words,
               annot=True, fmt=".2f", cmap="Reds", vmin=0, vmax=1,
               ax=axes[h+1], cbar=False)
    axes[h+1].set_title(f"Head {h+1}")

fig.suptitle("Single-Head vs Multi-Head 비교", fontsize=13)
fig.tight_layout()
fig.savefig("single_vs_multi_head.png", dpi=150, bbox_inches="tight")
print("저장: single_vs_multi_head.png")
plt.close()

# Head별 엔트로피 분석 (관심 분포의 다양성)
print("\nHead별 평균 엔트로피 (높을수록 고르게 분포):")
for h in range(4):
    w = multi_weights[0, h].detach().numpy()
    entropies = []
    for row in w:
        row = row + 1e-10
        entropy = -np.sum(row * np.log2(row))
        entropies.append(entropy)
    avg_entropy = np.mean(entropies)
    print(f"  Head {h+1}: {avg_entropy:.3f}")
```

예상 결과:
```
Head별 평균 엔트로피:
  Head 1: 2.145
  Head 2: 1.823
  Head 3: 2.087
  Head 4: 1.956

(각 Head가 서로 다른 분포를 가짐을 확인)
```

③ **Attention 기반 감성 분류 모델**

```python
class AttentionClassifier(nn.Module):
    """Self-Attention을 활용한 텍스트 분류 모델"""
    
    def __init__(self, vocab_size, d_model, num_heads, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.classifier = nn.Linear(d_model, num_classes)
    
    def forward(self, x):
        # 임베딩: (batch, seq) → (batch, seq, d_model)
        embedded = self.embedding(x)
        
        # Self-Attention
        attn_out, weights = self.attention(embedded)
        
        # 평균 풀링: (batch, seq, d_model) → (batch, d_model)
        pooled = attn_out.mean(dim=1)
        
        # 분류: (batch, d_model) → (batch, num_classes)
        logits = self.classifier(pooled)
        
        return logits, weights

# 간단한 한국어 감성 분류 데이터
# 0: 부정, 1: 긍정
train_data = [
    ([1, 2, 3, 4], 1),    # 이 영화 정말 좋다
    ([1, 2, 5, 6], 0),    # 이 영화 매우 싫다
    ([7, 8, 3, 4], 1),    # 그 책은 정말 좋다
    ([7, 8, 5, 6], 0),    # 그 책은 매우 싫다
    ([9, 10, 3, 11], 1),  # 오늘 기분 정말 최고
    ([9, 10, 5, 12], 0),  # 오늘 기분 매우 최악
    ([1, 13, 3, 4], 1),   # 이 음식 정말 좋다
    ([1, 13, 5, 6], 0),   # 이 음식 매우 싫다
]

vocab_size = 14
d_model = 16
num_heads = 2
num_classes = 2

model = AttentionClassifier(vocab_size, d_model, num_heads, num_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

# 학습 데이터 준비
X_train = torch.tensor([d[0] for d in train_data])
y_train = torch.tensor([d[1] for d in train_data])

print(f"모델 구조:")
print(f"  Embedding: vocab_size={vocab_size} → d_model={d_model}")
print(f"  Multi-Head Attention: {num_heads} heads, d_k={d_model//num_heads}")
print(f"  Classifier: {d_model} → {num_classes}")
print(f"  총 파라미터: {sum(p.numel() for p in model.parameters()):,}")

# 200 Epoch 학습
losses = []
for epoch in range(200):
    logits, _ = model(X_train)
    loss = criterion(logits, y_train)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    losses.append(loss.item())
    if (epoch + 1) % 50 == 0:
        preds = logits.argmax(dim=1)
        acc = (preds == y_train).float().mean()
        print(f"Epoch {epoch+1:3d}: Loss = {loss.item():.4f}, Accuracy = {acc:.1%}")

# 테스트 및 해석
model.eval()
with torch.no_grad():
    logits, weights = model(X_train)
    preds = logits.argmax(dim=1)

vocab_map = {
    0: "<pad>", 1: "이", 2: "영화", 3: "정말", 4: "좋다",
    5: "매우", 6: "싫다", 7: "그", 8: "책은", 9: "오늘",
    10: "기분", 11: "최고", 12: "최악", 13: "음식",
}

print(f"\n예측 결과:")
labels = ["부정", "긍정"]
for i, (x, y) in enumerate(train_data):
    words = [vocab_map[idx] for idx in x]
    sentence = " ".join(words)
    pred_label = labels[preds[i]]
    true_label = labels[y]
    correct = "O" if preds[i] == y else "X"
    print(f"  '{sentence}' → {pred_label} (정답: {true_label}) [{correct}]")
```

예상 결과:
```
모델 구조:
  Embedding: vocab_size=14 → d_model=16
  Multi-Head Attention: 2 heads, d_k=8
  Classifier: 16 → 2
  총 파라미터: 1,282

Epoch  50: Loss = 0.0001, Accuracy = 100.0%
Epoch 100: Loss = 0.0000, Accuracy = 100.0%
Epoch 150: Loss = 0.0000, Accuracy = 100.0%
Epoch 200: Loss = 0.0000, Accuracy = 100.0%

예측 결과:
  '이 영화 정말 좋다' → 긍정 (정답: 긍정) [O]
  '이 영화 매우 싫다' → 부정 (정답: 부정) [O]
  '그 책은 정말 좋다' → 긍정 (정답: 긍정) [O]
  '그 책은 매우 싫다' → 부정 (정답: 부정) [O]
  '오늘 기분 정말 최고' → 긍정 (정답: 긍정) [O]
  '오늘 기분 매우 최악' → 부정 (정답: 부정) [O]
  '이 음식 정말 좋다' → 긍정 (정답: 긍정) [O]
  '이 음식 매우 싫다' → 부정 (정답: 부정) [O]
```

④ **감성 분류 모델의 Attention 시각화**

```python
# 첫 번째 문장("이 영화 정말 좋다")의 Attention 시각화
sample_idx = 0
sample_words = [vocab_map[idx] for idx in train_data[sample_idx][0]]

fig, axes = plt.subplots(1, num_heads, figsize=(6*num_heads, 4))
if num_heads == 1:
    axes = [axes]

for h in range(num_heads):
    w = weights[sample_idx, h].numpy()
    sns.heatmap(w, xticklabels=sample_words, yticklabels=sample_words,
               annot=True, fmt=".2f", cmap="YlOrRd", vmin=0, vmax=1,
               ax=axes[h], cbar=(h == num_heads-1))
    axes[h].set_title(f"Head {h+1}")

fig.suptitle(f"감성 분류: '{' '.join(sample_words)}' → 긍정", fontsize=12)
fig.tight_layout()
fig.savefig("sentiment_attention.png", dpi=150, bbox_inches="tight")
print("저장: sentiment_attention.png")
plt.close()

# 학습 곡선 시각화
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(losses, color="steelblue", linewidth=1.5)
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.set_title("Attention 분류 모델 학습 곡선")
ax.grid(True, alpha=0.3)
fig.savefig("training_curve.png", dpi=150, bbox_inches="tight")
print("저장: training_curve.png")
plt.close()
```

예상 결과:
```
저장: sentiment_attention.png
저장: training_curve.png
```

**검증 체크리스트**:
- [ ] 한국어 문장의 Attention 히트맵이 생성되었는가?
- [ ] 영어 문장의 Attention 히트맵이 생성되었는가?
- [ ] 각 Head가 다른 패턴을 보이는가? (엔트로피 값이 다름)
- [ ] 감성 분류 모델이 100% 정확도로 수렴했는가?
- [ ] 모든 테스트 문장이 올바르게 분류되었는가?

**Copilot 프롬프트 5**:
```
"Attention Weight를 히트맵으로 시각화하는 코드 작성해줄래? seaborn을 써서 
각 Head별로 서브플롯을 만들어야 해."
```

**Copilot 프롬프트 6**:
```
"Self-Attention을 활용한 간단한 텍스트 분류 모델을 만들어줄래? 
Embedding → MultiHeadAttention → 평균 풀링 → Classifier 순서야."
```

**선택 프롬프트**:
```
"학습 곡선을 matplotlib으로 그려줄래? x축은 Epoch, y축은 Loss이고 그리드를 추가해줘."
```

---

### 제출 안내 (Google Classroom)

**제출 방법**:
- Google Classroom의 "3주차 B회차" 과제에 조별 1부 제출
- 파일명 형식: `group_{조번호}_ch3B.zip`

**포함할 파일**:
```
group_{조번호}_ch3B/
├── ch3B_attention.py           # 전체 구현 코드
├── attention_heatmap_ko.png    # 한국어 문장 히트맵
├── attention_heatmap_en.png    # 영어 문장 히트맵
├── single_vs_multi_head.png    # Single vs Multi 비교
├── sentiment_attention.png      # 감성 분류 Attention
├── training_curve.png           # 학습 곡선
└── report.md                   # 분석 리포트 (1-2페이지)
```

**리포트 포함 항목** (report.md):
- 각 체크포인트의 구현 과정 및 어려웠던 점 (3-4문장)
- Attention Weight 히트맵 해석: "어떤 단어들이 서로 주목했는가?" (3-4문장)
- Single-Head vs Multi-Head의 차이점 (2-3문장)
- 감성 분류 모델 결과 해석: "어떤 단어에 높은 Attention이 할당되었는가?" (2-3문장)
- Copilot 사용 경험: 어떤 프롬프트가 효과적이었는가? (2문장)

**제출 마감**: 수업 종료 후 24시간 이내

---

### 결과 토론 가이드

> **토론 가이드**: 조별 구현 결과를 공유하며, 다른 조의 Attention 패턴과 비교하고 모델의 의사결정을 함께 해석한다

**토론 주제**:

① **Head별 역할 분담**
- 각 조의 4개 Head가 어떤 패턴을 포착했는가?
- 예: Head 0은 인접 단어 집중, Head 1은 원거리 의존성 포착 등
- "왜 이런 차이가 생겼을까?" → 초기화, 학습률, 데이터의 영향

② **√dₖ 스케일링의 효과**
- 스케일링 없이 구현했다면 Attention Weight 분포가 어떻게 달랐을까?
- 실제로 제거하고 실험해본 조는? 결과는?
- 스케일링이 기울기 소실을 방지하는 메커니즘 재확인

③ **Single-Head vs Multi-Head**
- 같은 입력으로 Single-Head와 Multi-Head를 비교했을 때?
- Multi-Head (4 heads)가 더 풍부한 표현을 하는가?
- 파라미터 수 증가 대비 성능 개선이 얼마나 되는가?

④ **감성 분류 모델의 의사결정**
- 모델이 "좋다/최고/싫다/최악" 같은 감성 단어에 높은 Attention을 부여하는가?
- 이중부정("싫다지 않다")이 있다면 제대로 처리하는가?
- 만약 신뢰도가 낮다면, 어떤 부분을 개선할 수 있을까?

⑤ **실무적 시사**
- Self-Attention이 RNN보다 어떤 점에서 우월한가?
- Transformer가 BERT/GPT 등 현대 LLM의 기초가 되는 이유는?
- 이번 학습이 4장 Transformer 학습에 어떻게 연결될까?

**발표 형식**:
- 각 조 3~5분 발표 (구현 전략 + 주요 결과)
- 다른 조의 질문에 답변 (2~3개 질문)
- 교수의 보충 설명 및 피드백

---

### 교수 피드백 포인트

**강화할 점**:
- Attention Weight를 히트맵으로 시각화하는 것이 모델 이해의 가장 중요한 도구임을 강조
- 각 Head가 단순히 "다른 패턴"을 보는 것이 아니라, **구체적인 언어 현상**(문법 관계, 의미 관계 등)을 포착하고 있음
- Multi-Head의 병렬 처리가 단순 속도 개선이 아니라 **표현력 향상**을 위한 것임을 명확히 함

**주의할 점**:
- √dₖ 스케일링을 "임의의 마법"이 아니라 **수학적 필연성**(분산 안정화)으로 이해하도록 유도
- "Attention이 정답을 맞힌다"는 오해 방지: 학습된 임베딩 공간과 데이터의 특성에 따라 달라질 수 있음
- 감성 분류 모델이 100% 정확도인 것은 데이터가 작고 단순하기 때문임을 명시

**다음 학습으로의 연결**:
- Self-Attention은 이 형태로 Transformer Encoder에 포함되며, Decoder에서는 Causal Mask와 함께 사용됨
- 4장에서는 Positional Encoding, Residual Connection, Layer Normalization 등 Transformer의 추가 구성 요소를 학습함
- Position Encoding 없이는 Self-Attention이 단어 순서를 무시하므로, 이 문제를 4장에서 해결함

---

### 다음 주 예고

다음 주 4장 A회차에서는 **Transformer 아키텍처**를 깊이 있게 다룬다.

**예고 내용**:
- Positional Encoding: Self-Attention이 순서 정보를 잃지 않도록 위치 정보를 추가하는 방법
- Transformer Encoder/Decoder 구조: Multi-Head Attention 외에 Feed-Forward 네트워크, Residual Connection, Layer Normalization이 어떻게 조합되는가
- PyTorch nn.TransformerEncoder 구현: 밑바닥부터 Transformer Encoder를 만들고, 실제 시퀀스 라벨링(NER) 태스크에 적용
- B회차에서는 실제로 Transformer Encoder를 학습시켜 텍스트 분류를 수행한다

**사전 준비**:
- 3장 내용 (특히 Self-Attention)을 다시 읽어두기
- Positional Encoding의 sin/cos 함수가 왜 사용되는지 미리 생각해보기

---

## 참고 자료

**실습 코드**:
- _전체 구현 코드는 practice/chapter3/code/3-3-어텐션.py 참고_
- _감성 분류 및 시각화는 practice/chapter3/code/3-5-실습.py 참고_

**권장 읽기**:
- Jay Alammar. The Illustrated Transformer. https://jalammar.github.io/illustrated-transformer/
- Lilian Weng. (2018). Attention? Attention!. https://lilianweng.github.io/posts/2018-06-24-attention/
- Vaswani, A. et al. (2017). Attention Is All You Need. *NeurIPS*. https://arxiv.org/abs/1706.03762

