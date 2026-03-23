# 제2장 C: MLP 텍스트 분류 모범 구현 — 한국어 영화 리뷰 감성 분류

> 이 문서는 B회차 과제 제출 후 공개된다. 제출 전에는 열람하지 않는다.

---

## 체크포인트 1 모범 구현: 데이터 준비 및 BoW 벡터화

한국어 영화 리뷰 40개(긍정 20, 부정 20)를 Bag-of-Words 벡터로 변환하는 완전한 구현이다.

### _build_vocab 구현

```python
def _build_vocab(self, texts, max_vocab_size):
    """단어 빈도 기반 어휘 사전 구축"""
    word_counts = Counter()
    for text in texts:
        word_counts.update(text.split())

    # 빈도 상위 max_vocab_size-2개 선택 (PAD, UNK 자리 확보)
    most_common = word_counts.most_common(max_vocab_size - 2)
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for word, _ in most_common:
        vocab[word] = len(vocab)
    return vocab
```

**핵심 설명**:
- `Counter`로 모든 텍스트의 단어 빈도를 한 번에 센다
- `max_vocab_size - 2`인 이유: `<PAD>`(0번)과 `<UNK>`(1번)을 위한 자리를 남겨둔다
- `<UNK>`은 사전에 없는 단어를 처리하는 안전장치이다

### _text_to_bow 구현

```python
def _text_to_bow(self, text):
    """텍스트 → Bag-of-Words 벡터"""
    bow = np.zeros(self.vocab_size, dtype=np.float32)
    for word in text.split():
        idx = self.vocab.get(word, self.vocab["<UNK>"])
        bow[idx] += 1
    # L2 정규화
    norm = np.linalg.norm(bow)
    if norm > 0:
        bow = bow / norm
    return bow
```

**핵심 설명**:
- `vocab.get(word, self.vocab["<UNK>"])`: 사전에 없는 단어는 `<UNK>` 인덱스(1)로 매핑
- L2 정규화: 벡터 크기를 1로 만들어 긴 리뷰와 짧은 리뷰를 공평하게 비교
- 결과는 `vocab_size` 차원의 희소 벡터 (대부분 0)

### DataLoader 확인

```python
texts, labels = create_review_data()
dataset = TextClassificationDataset(texts, labels, max_vocab_size=200)

print(f"전체 데이터: {len(texts)} 샘플")
print(f"긍정: {sum(labels)}, 부정: {len(labels) - sum(labels)}")
print(f"어휘 크기: {dataset.vocab_size}")

# 어휘 사전 샘플
sample_words = list(dataset.vocab.items())[:10]
print(f"어휘 샘플: {sample_words}")

# Train / Val 분할
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(
    dataset, [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
print(f"학습: {len(train_dataset)}, 검증: {len(val_dataset)}")

# 첫 배치 확인
batch_x, batch_y = next(iter(train_loader))
print(f"배치 입력 크기: {batch_x.shape}")
print(f"배치 레이블: {batch_y.tolist()}")
```

**예상 출력**:
```
전체 데이터: 40 샘플
긍정: 20, 부정: 20
어휘 크기: 82
어휘 샘플: [('<PAD>', 0), ('<UNK>', 1), ('영화', 2), ('정말', 3), ...]
학습: 32, 검증: 8
배치 입력 크기: torch.Size([8, 82])
배치 레이블: [1, 0, 1, 0, 0, 1, 1, 0]
```

### 핵심 포인트

#### Bag-of-Words의 한계

```python
# BoW는 단어 순서를 무시한다
text1 = "정말 좋은 영화"
text2 = "정말 나쁜 영화"

bow1 = dataset._text_to_bow(text1)
bow2 = dataset._text_to_bow(text2)

# "정말"과 "영화"가 공유되므로 유사한 벡터가 나온다
# 그러나 의미는 반대다!
# 3주차 Self-Attention은 단어 간 관계를 학습하여 이 문제를 해결한다
```

### 흔한 실수

1. **`<UNK>` 처리 누락**
   ```python
   # 틀림
   idx = self.vocab[word]  # KeyError 발생!

   # 맞음
   idx = self.vocab.get(word, self.vocab["<UNK>"])
   ```

2. **L2 정규화 시 0 벡터 처리 누락**
   ```python
   # 틀림
   bow = bow / np.linalg.norm(bow)  # 빈 텍스트에서 0으로 나누기 에러!

   # 맞음
   norm = np.linalg.norm(bow)
   if norm > 0:
       bow = bow / norm
   ```

3. **vocab_size 계산 실수**
   ```python
   # 틀림: max_vocab_size 그대로 사용
   bow = np.zeros(max_vocab_size)  # 실제 사전 크기와 불일치 가능

   # 맞음: 실제 구축된 사전의 크기 사용
   bow = np.zeros(self.vocab_size)  # self.vocab_size = len(self.vocab)
   ```

---

## 체크포인트 2 모범 구현: MLP 모델 정의 및 학습

### TextClassifier 구현

```python
class TextClassifier(nn.Module):
    """MLP 기반 텍스트 분류 모델"""

    def __init__(self, vocab_size, hidden_size=64, num_classes=2, dropout=0.3):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(vocab_size, hidden_size),       # 82 → 64
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2), # 64 → 32
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes),  # 32 → 2
        )

    def forward(self, x):
        return self.classifier(x)
```

**핵심 설명**:
- `nn.Sequential`: 여러 층을 순서대로 연결. `forward()`가 자동으로 순전파를 수행한다
- **구조**: 입력(82) → 은닉1(64) → 은닉2(32) → 출력(2)
- 각 은닉층 후에 ReLU(비선형성) + Dropout(과적합 방지)
- 출력층에는 활성화 함수가 없다 → `CrossEntropyLoss`가 내부적으로 softmax를 처리

```python
# 모델 생성 및 정보 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TextClassifier(
    vocab_size=dataset.vocab_size,
    hidden_size=64,
    num_classes=2,
    dropout=0.3,
).to(device)

print(f"모델 구조:\n{model}")
total_params = sum(p.numel() for p in model.parameters())
print(f"총 파라미터 수: {total_params}")
```

**예상 출력**:
```
모델 구조:
TextClassifier(
  (classifier): Sequential(
    (0): Linear(in_features=82, out_features=64, bias=True)
    (1): ReLU()
    (2): Dropout(p=0.3, inplace=False)
    (3): Linear(in_features=64, out_features=32, bias=True)
    (4): ReLU()
    (5): Dropout(p=0.3, inplace=False)
    (6): Linear(in_features=32, out_features=2, bias=True)
  )
)
총 파라미터 수: 7458
```

**파라미터 수 계산**:
```
Layer 0 (Linear): (82 × 64) + 64 = 5,312
Layer 3 (Linear): (64 × 32) + 32 = 2,080
Layer 6 (Linear): (32 × 2) + 2 = 66
합계: 5,312 + 2,080 + 66 = 7,458
```

### 학습 루프 구현

```python
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

num_epochs = 30
history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

for epoch in range(num_epochs):
    # --- Training ---
    model.train()
    train_loss, train_correct, train_total = 0, 0, 0

    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)

        optimizer.zero_grad()          # 1. 이전 그래디언트 초기화
        outputs = model(batch_X)       # 2. 순전파
        loss = criterion(outputs, batch_y)  # 3. 손실 계산
        loss.backward()                # 4. 역전파 (그래디언트 계산)
        optimizer.step()               # 5. 파라미터 업데이트

        train_loss += loss.item() * batch_X.size(0)
        _, predicted = outputs.max(1)
        train_total += batch_y.size(0)
        train_correct += predicted.eq(batch_y).sum().item()

    train_loss /= train_total
    train_acc = 100.0 * train_correct / train_total

    # --- Validation ---
    model.eval()
    val_loss, val_correct, val_total = 0, 0, 0

    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            val_loss += loss.item() * batch_X.size(0)
            _, predicted = outputs.max(1)
            val_total += batch_y.size(0)
            val_correct += predicted.eq(batch_y).sum().item()

    val_loss /= val_total
    val_acc = 100.0 * val_correct / val_total

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["train_acc"].append(train_acc)
    history["val_acc"].append(val_acc)

    if (epoch + 1) % 10 == 0:
        print(
            f"Epoch [{epoch+1:2d}/{num_epochs}] "
            f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.1f}% | "
            f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.1f}%"
        )

print(f"\n최종 검증 정확도: {history['val_acc'][-1]:.1f}%")
```

**예상 출력**:
```
Epoch [10/30] Train Loss: 0.6655, Acc: 84.4% | Val Loss: 0.6940, Acc: 37.5%
Epoch [20/30] Train Loss: 0.5294, Acc: 100.0% | Val Loss: 0.6667, Acc: 50.0%
Epoch [30/30] Train Loss: 0.2439, Acc: 100.0% | Val Loss: 0.5154, Acc: 87.5%

최종 검증 정확도: 87.5%
```

### 핵심 포인트

#### Dropout의 작동 원리

```python
# 훈련 시 (model.train())
# Dropout(0.3): 30%의 뉴런을 무작위로 0으로 만들고, 나머지를 1/(1-0.3)배 스케일링
# → 앙상블 효과: 매번 다른 서브네트워크로 학습

# 평가 시 (model.eval())
# Dropout 비활성화: 모든 뉴런을 사용
# → 일관된 예측
```

#### 학습 루프의 5단계

```python
optimizer.zero_grad()    # 1. 이전 그래디언트 초기화 (누적 방지)
outputs = model(x)       # 2. 순전파
loss = criterion(outputs, y)  # 3. 손실 계산
loss.backward()          # 4. 역전파: 각 파라미터의 ∂loss/∂w 계산
optimizer.step()         # 5. 파라미터 업데이트: w = w - lr × ∂loss/∂w
```

### 흔한 실수

1. **model.train() / model.eval() 호출 누락**
   ```python
   # 틀림: Dropout이 항상 활성화되어 평가 결과가 불안정
   for batch_X, batch_y in val_loader:
       outputs = model(batch_X)

   # 맞음
   model.eval()
   with torch.no_grad():
       for batch_X, batch_y in val_loader:
           outputs = model(batch_X)
   ```

2. **optimizer.zero_grad() 호출 누락**
   ```python
   # 틀림: 그래디언트가 배치마다 누적됨!
   for batch_X, batch_y in train_loader:
       outputs = model(batch_X)
       loss = criterion(outputs, batch_y)
       loss.backward()
       optimizer.step()

   # 맞음
   for batch_X, batch_y in train_loader:
       optimizer.zero_grad()  # 매 배치 시작 시 초기화
       outputs = model(batch_X)
       loss = criterion(outputs, batch_y)
       loss.backward()
       optimizer.step()
   ```

3. **CrossEntropyLoss에 softmax 이중 적용**
   ```python
   # 틀림
   probs = torch.softmax(outputs, dim=1)
   loss = criterion(probs, batch_y)  # CrossEntropyLoss가 내부적으로 softmax를 이미 포함!

   # 맞음
   loss = criterion(outputs, batch_y)  # logits를 직접 전달
   ```

---

## 체크포인트 3 모범 구현: 성능 평가 및 시각화

### compute_metrics 구현

```python
def compute_metrics(y_true, y_pred, num_classes=2):
    """평가 지표 계산 (sklearn 없이 직접 구현)"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    accuracy = np.mean(y_true == y_pred)

    # Confusion Matrix 구축
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1

    # 이진 분류 (positive = 1)
    tp = cm[1][1]  # 실제 긍정을 긍정으로 예측
    fp = cm[0][1]  # 실제 부정을 긍정으로 예측
    fn = cm[1][0]  # 실제 긍정을 부정으로 예측

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm,
    }
```

**핵심 설명**:
- Confusion Matrix: `cm[실제][예측]` 형태로 구축. 이중 for 루프로 모든 (실제, 예측) 쌍을 카운트
- `tp / (tp + fp)`: "긍정이라 예측한 것 중 실제 긍정 비율" (Precision)
- `tp / (tp + fn)`: "실제 긍정 중 모델이 맞힌 비율" (Recall)
- 분모가 0인 경우 0.0 반환: 해당 클래스의 예측이 없거나 실제 샘플이 없을 때

### 평가 실행

```python
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for batch_X, batch_y in val_loader:
        batch_X = batch_X.to(device)
        outputs = model(batch_X)
        _, predicted = outputs.max(1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(batch_y.numpy())

metrics = compute_metrics(all_labels, all_preds)
print(f"Accuracy:  {metrics['accuracy']:.4f}")
print(f"Precision: {metrics['precision']:.4f}")
print(f"Recall:    {metrics['recall']:.4f}")
print(f"F1-Score:  {metrics['f1']:.4f}")

cm = metrics["confusion_matrix"]
print(f"\nConfusion Matrix:")
print(f"           Predicted")
print(f"           Neg   Pos")
print(f"Actual Neg  {cm[0][0]:3d}   {cm[0][1]:3d}")
print(f"       Pos  {cm[1][0]:3d}   {cm[1][1]:3d}")
```

**예상 출력**:
```
Accuracy:  0.8750
Precision: 1.0000
Recall:    0.8000
F1-Score:  0.8889

Confusion Matrix:
           Predicted
           Neg   Pos
Actual Neg    3     0
       Pos    1     4
```

**지표 해석**:
- Accuracy 0.875: 8개 검증 샘플 중 7개 정답
- Precision 1.00: 긍정이라 예측한 4개가 모두 실제 긍정
- Recall 0.80: 실제 긍정 5개 중 4개를 맞힘
- F1 0.889: Precision과 Recall의 조화평균

> 참고: 랜덤 시드와 환경에 따라 정확한 수치는 다를 수 있다. 전체적인 경향(손실 감소, 정확도 증가)이 중요하다.

### 시각화 구현

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

output_dir = Path("practice/chapter2/data/output")
output_dir.mkdir(parents=True, exist_ok=True)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# 1. 손실 곡선
axes[0].plot(history["train_loss"], label="Train")
axes[0].plot(history["val_loss"], label="Validation")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].set_title("Loss Curve")
axes[0].legend()
axes[0].grid(True)

# 2. 정확도 곡선
axes[1].plot(history["train_acc"], label="Train")
axes[1].plot(history["val_acc"], label="Validation")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy (%)")
axes[1].set_title("Accuracy Curve")
axes[1].legend()
axes[1].grid(True)

# 3. Confusion Matrix (matplotlib imshow 사용)
im = axes[2].imshow(cm, cmap="Blues")
axes[2].set_xticks([0, 1])
axes[2].set_yticks([0, 1])
axes[2].set_xticklabels(["Negative", "Positive"])
axes[2].set_yticklabels(["Negative", "Positive"])
axes[2].set_xlabel("Predicted")
axes[2].set_ylabel("Actual")
axes[2].set_title("Confusion Matrix")
for i in range(2):
    for j in range(2):
        axes[2].text(j, i, cm[i, j], ha="center", va="center", fontsize=14)
plt.colorbar(im, ax=axes[2])

plt.tight_layout()
save_path = output_dir / "ch2_text_classification.png"
plt.savefig(save_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"그래프 저장: {save_path}")
```

**출력**: 3개 그래프(손실 곡선, 정확도 곡선, Confusion Matrix)가 하나의 이미지로 저장

### 새 텍스트 예측

```python
test_texts = [
    "정말 재미있는 영화다",
    "지루하고 별로다",
    "최고의 명작 강력 추천",
    "시간 낭비 최악의 영화",
]

model.eval()
for text in test_texts:
    bow = dataset._text_to_bow(text)
    input_tensor = torch.FloatTensor(bow).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        prob = torch.softmax(output, dim=1)
        pred = output.argmax(1).item()

    sentiment = "긍정" if pred == 1 else "부정"
    confidence = prob[0][pred].item() * 100
    print(f"  '{text}' → {sentiment} (신뢰도: {confidence:.1f}%)")
```

**예상 출력**:
```
  '정말 재미있는 영화다' → 긍정 (신뢰도: 57.2%)
  '지루하고 별로다' → 부정 (신뢰도: 87.7%)
  '최고의 명작 강력 추천' → 긍정 (신뢰도: 66.2%)
  '시간 낭비 최악의 영화' → 부정 (신뢰도: 63.3%)
```

**핵심 설명**:
- `unsqueeze(0)`: 1차원 벡터를 (1, vocab_size) 배치 형태로 변환
- `torch.softmax(output, dim=1)`: logits를 확률로 변환 (합 = 1)
- 여기서는 손실 계산이 아닌 해석 목적이므로 softmax를 명시적으로 적용한다

### 흔한 실수

1. **Confusion Matrix 인덱싱 실수**
   ```python
   # 틀림: 행과 열을 반대로
   cm[p][t] += 1  # 예측을 행으로 넣으면 해석이 반대!

   # 맞음: cm[실제][예측]
   cm[t][p] += 1
   ```

2. **Precision/Recall 분모 0 체크 누락**
   ```python
   # 틀림: 모델이 한 클래스만 예측하면 ZeroDivisionError
   precision = tp / (tp + fp)

   # 맞음
   precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
   ```

3. **평가 시 GPU → CPU 변환 누락**
   ```python
   # 틀림: GPU 텐서를 numpy로 직접 변환 불가
   all_preds.extend(predicted.numpy())  # RuntimeError!

   # 맞음
   all_preds.extend(predicted.cpu().numpy())
   ```

---

## 종합 해설

### 2주차 전체 프로세스

#### 1단계: 텍스트 → 숫자 (전처리)
- 한국어 영화 리뷰 40개 (긍정 20, 부정 20)
- Bag-of-Words: 단어의 출현 횟수를 벡터로 표현 + L2 정규화
- 결과: (40, 82) 크기의 희소 행렬

#### 2단계: 분류 모델 구축 (MLP)
- 입력층 (82) → 은닉층1 (64) → 은닉층2 (32) → 출력층 (2)
- 은닉층마다 ReLU 활성화와 Dropout(0.3)으로 과적합 방지
- `nn.Sequential`로 간결하게 정의

#### 3단계: 학습 및 평가
- 30 에폭 훈련: 손실이 점차 감소
- 검증 정확도: ~87.5%
- `compute_metrics()`로 Precision, Recall, F1 직접 계산
- Confusion Matrix로 오분류 패턴 분석

### 핵심 개념 정리

**Bag-of-Words의 한계**:
- 단어 순서를 무시 → "좋은 영화"와 "영화 좋은"이 같은 벡터
- 희소 벡터 → 계산 비효율
- **해결책**: 3주차 임베딩과 Self-Attention

**MLP의 작동**:
- 은닉층이 입력을 점진적으로 추상화
- 각 층이 다른 수준의 특징 학습 (저수준→고수준)
- Dropout이 앙상블 효과로 과적합 방지

**성능 지표의 의미**:
- Accuracy: 전체 성능 (클래스 균형일 때 유효)
- Precision: "긍정이라 한 것 중 진짜 긍정 비율"
- Recall: "진짜 긍정 중 맞힌 비율"
- F1: 두 지표의 조화평균 (불균형 데이터에서 유용)

### 소규모 데이터의 특성

이번 실습은 40개 샘플로 진행한다. 실무 데이터(수만~수십만 개)와의 차이:

- **과적합 위험이 크다**: 학습 정확도 100%에 도달하기 쉬우나, 이는 일반화 성능을 의미하지 않는다
- **Dropout과 weight_decay가 더 중요하다**: 적은 데이터에서 정규화가 필수
- **검증 정확도 변동이 크다**: 8개 샘플에서 1개만 틀려도 정확도가 12.5% 변동
- **하이퍼파라미터 효과를 직접 확인하기 좋다**: 학습률, Dropout, 에폭 수 변경 결과를 빠르게 관찰 가능

### 다음 장으로의 연결

3주차에서는:
- **임베딩**: 단어를 고정 차원의 밀도 벡터로 표현 (Word2Vec)
- **Self-Attention**: "어디에 집중할지" 동적으로 결정
- **Multi-Head Attention**: 여러 관점에서 동시에 주목

BoW + MLP가 "단어를 세는" 방식이라면, 임베딩 + Attention은 "단어의 의미와 관계를 학습하는" 방식이다.

---

## 참고 코드 파일

완전한 구현은 다음 파일에서 확인할 수 있다:

- **practice/chapter2/code/2-4-텍스트분류.py** — 전체 파이프라인 (데이터 → 모델 → 학습 → 평가 → 시각화)
- **practice/chapter2/code/2-1-신경망기초.py** — 퍼셉트론, MLP, 활성화 함수, 역전파 데모
- **practice/chapter2/code/2-2-모델개발.py** — PyTorch 모델 개발, 학습 파이프라인, 옵티마이저 비교

### 코드 실행 방법

```bash
# 가상환경 활성화
source venv/bin/activate  # macOS/Linux

# 텍스트 분류 파이프라인 실행
python practice/chapter2/code/2-4-텍스트분류.py

# 생성된 이미지 확인
ls practice/chapter2/data/output/
```

---

## 최종 학습 정리

### 2주차 핵심 개념 요약

1. **Bag-of-Words**: 단어 빈도만 고려 (순서 무시)
2. **MLP**: 은닉층이 비선형 분류를 가능하게 함
3. **활성화 함수**: ReLU가 비선형성을 추가
4. **역전파**: 연쇄 법칙으로 그래디언트 계산
5. **Dropout**: 과적합 방지 (훈련 시에만 적용)
6. **성능 지표**: Accuracy, Precision, Recall, F1-Score
7. **Confusion Matrix**: 오분류 패턴 분석

### 3주차 미리보기

- **임베딩의 필요성**: BoW 대신 밀도 벡터 (더 효율적, 문맥 학습)
- **Word2Vec**: "비슷한 문맥의 단어는 비슷한 벡터" 학습
- **Self-Attention**: 문장 내 단어 간 관계 학습

이 MLP 기반 분류는 3주차 Self-Attention, 4주차 Transformer, 5주차 BERT/GPT로 연결되는 NLP 파이프라인의 첫 걸음이다.

---

**생성일**: 2026-02-25
**대상 독자**: 컴퓨터공학/AI 전공 학부생 (3~4학년)
**난이도**: 초급~중급 (Python, PyTorch 기초 선수)
