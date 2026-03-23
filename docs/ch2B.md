## 2주차 B회차: MLP 텍스트 분류 실습

> **미션**: PyTorch `nn.Module`로 한국어 영화 리뷰 감성 분류 MLP 모델을 구현하고, 평가 지표를 계산하며, 학습 곡선과 Confusion Matrix를 분석할 수 있다

### 수업 타임라인

| 시간 | 내용 | Copilot 사용 |
|------|------|-------------|
| 00:00~00:05 | 조 편성 + 역할 배분 (조원 A/B) | 사용 안 함 |
| 00:05~00:10 | A회차 핵심 리캡 + 과제 스펙 재확인 | 사용 안 함 |
| 00:10~00:55 | 2인1조 Copilot 구현 (체크포인트 3회) | 적극 사용 |
| 00:55~01:00 | Google Classroom 제출 (조별 1부) | 사용 안 함 |
| 01:00~01:20 | 결과 토론 (구현 전략 비교·성능 차이 분석) | 사용 안 함 |
| 01:20~01:28 | 핵심 정리 | 사용 안 함 |
| 01:28~01:30 | 다음 주 예고 | 사용 안 함 |

---

### A회차 핵심 리캡

- **신경망 구조**: 퍼셉트론은 직선으로만 분류 가능. 은닉층을 추가한 MLP가 XOR 같은 비선형 문제를 해결한다.
- **활성화 함수**: ReLU(max(0, x))가 비선형성을 추가한다. 없으면 층을 아무리 쌓아도 직선이다.
- **학습 원리**: 손실 함수로 오차를 측정하고, 경사 하강법 + 역전파로 파라미터를 업데이트한다.
- **PyTorch 패턴**: `nn.Module`로 모델 정의 → `zero_grad()` → `backward()` → `step()` 루프.

**이번 실습에서 사용할 평가 지표**:
- **Accuracy**: 전체 중 맞은 비율
- **Precision**: "긍정"이라 예측한 것 중 실제 긍정 비율
- **Recall**: 실제 긍정 중 모델이 맞힌 비율
- **F1-Score**: Precision과 Recall의 조화평균

---

### 과제 스펙

**과제**: 한국어 영화 리뷰 감성 분류 MLP 모델 구현 + 성능 분석

**제출 형태**: 조별 1부, Google Classroom 업로드

**검증 기준**:
- ✓ 한국어 리뷰 데이터 전처리 및 BoW 벡터화 구현
- ✓ MLP 모델 정의 (입력(vocab_size) → 은닉(64) → 은닉(32) → 출력(2))
- ✓ 모델 학습 및 손실 감소 확인
- ✓ Accuracy, Precision, Recall, F1-Score 계산
- ✓ 학습 곡선 및 Confusion Matrix 시각화

---

### 2인1조 실습

> **Copilot 활용**: Copilot에게 "한국어 영화 리뷰 감성 분류를 위한 Bag-of-Words 기반 MLP 모델을 PyTorch로 구현해줘"로 시작하여, 생성된 코드의 각 부분이 하는 역할을 분석한다.

**역할 분담**:
- **조원 A (드라이버)**: 코드 작성 및 실행, 결과 확인, Copilot과 대화
- **조원 B (네비게이터)**: 로직 검토, 오류 해석, 다음 단계 설계
- **체크포인트마다 역할 교대**

---

#### 체크포인트 1: 데이터 준비 및 BoW 벡터화 (15분)

**목표**: 한국어 영화 리뷰 데이터를 로드하고, Bag-of-Words 벡터화를 구현하여 DataLoader를 구성한다

**데이터 준비** — 다음 함수를 그대로 사용한다:

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from collections import Counter

def create_review_data():
    """한국어 영화 리뷰 샘플 데이터"""
    positive = [
        "이 영화 정말 재미있다 추천한다",
        "감동적인 스토리 배우 연기 최고",
        "완벽한 영화 다시 보고 싶다",
        "기대 이상이다 정말 좋았다",
        "훌륭한 작품 명작이다",
        "재미있고 감동적인 영화",
        "최고의 영화 강력 추천",
        "멋진 영화 배우들 연기 훌륭",
        "좋은 영화 재미있게 봤다",
        "감동 받았다 눈물이 났다",
        "연기 좋고 스토리 좋고 최고",
        "정말 좋은 영화 추천한다",
        "기대했던 대로 정말 좋았다",
        "완벽한 스토리 감동적이다",
        "배우들 연기가 훌륭하다",
        "최고의 감동 영화이다",
        "다시 보고 싶은 영화",
        "정말 재미있는 영화였다",
        "감동적인 명작 영화이다",
        "좋은 영화 감동 받았다",
    ]
    negative = [
        "지루하고 재미없다 최악",
        "시간 낭비였다 별로다",
        "기대 이하 실망했다",
        "스토리가 너무 지루하다",
        "연기도 별로 내용도 별로",
        "최악의 영화 돈 아깝다",
        "재미없다 추천 안한다",
        "보다가 잠들었다 지루했다",
        "실망스러운 영화였다",
        "기대했는데 별로였다",
        "스토리가 엉망이다",
        "연기가 너무 어색하다",
        "지루해서 끝까지 못봤다",
        "최악이다 보지마라",
        "시간이 아깝다 별로다",
        "실망이다 기대 이하",
        "재미없는 영화다",
        "별로다 추천 안한다",
        "스토리가 별로다",
        "지루하다 재미없다",
    ]
    texts = positive + negative
    labels = [1] * len(positive) + [0] * len(negative)

    np.random.seed(42)
    indices = np.random.permutation(len(texts))
    texts = [texts[i] for i in indices]
    labels = [labels[i] for i in indices]
    return texts, labels
```

**구현할 클래스** — `TextClassificationDataset`:

```python
class TextClassificationDataset(Dataset):
    """Bag-of-Words 기반 텍스트 분류 Dataset"""

    def __init__(self, texts, labels, vocab=None, max_vocab_size=200):
        self.texts = texts
        self.labels = labels
        if vocab is None:
            self.vocab = self._build_vocab(texts, max_vocab_size)
        else:
            self.vocab = vocab
        self.vocab_size = len(self.vocab)

    def _build_vocab(self, texts, max_vocab_size):
        """단어 빈도 기반 어휘 사전 구축"""
        # TODO: Counter로 모든 텍스트의 단어 빈도를 센다
        # TODO: 빈도 상위 max_vocab_size-2개 단어를 선택한다
        # TODO: {"<PAD>": 0, "<UNK>": 1, 단어1: 2, ...} 형태의 사전을 반환한다
        pass

    def _text_to_bow(self, text):
        """텍스트 → Bag-of-Words 벡터"""
        # TODO: vocab_size 크기의 영벡터를 만든다
        # TODO: 텍스트를 공백으로 분리하여 각 단어의 인덱스를 찾는다
        # TODO: 사전에 없는 단어는 <UNK> 인덱스를 사용한다
        # TODO: 해당 인덱스의 값을 +1 한다
        # 힌트: L2 정규화(벡터 크기를 1로)를 적용하면 긴 리뷰와 짧은 리뷰를 공평하게 비교 가능
        pass

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        bow = self._text_to_bow(self.texts[idx])
        return torch.FloatTensor(bow), torch.tensor(self.labels[idx], dtype=torch.long)
```

**DataLoader 구성** — 다음 코드로 데이터를 분할한다:

```python
texts, labels = create_review_data()
dataset = TextClassificationDataset(texts, labels, max_vocab_size=200)

# 80:20 Train/Val 분할
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(
    dataset, [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
```

**검증 체크리스트**:
- [ ] 어휘 사전에 `<PAD>`, `<UNK>` 토큰이 포함되었는가?
- [ ] BoW 벡터의 차원이 `vocab_size`와 일치하는가?
- [ ] DataLoader에서 첫 배치를 정상적으로 가져올 수 있는가?

**Copilot 프롬프트**:
```
"Counter를 사용해 텍스트 리스트에서 단어 빈도를 세고,
상위 N개 단어로 어휘 사전을 구축하는 _build_vocab 메서드를 구현해줘"
```

---

#### 체크포인트 2: MLP 모델 정의 및 학습 (20분)

**목표**: PyTorch nn.Module로 MLP 모델을 구현하고, 30 에폭 학습을 수행한다

**구현할 클래스** — `TextClassifier`:

```python
class TextClassifier(nn.Module):
    """MLP 기반 텍스트 분류 모델"""

    def __init__(self, vocab_size, hidden_size=64, num_classes=2, dropout=0.3):
        super().__init__()
        # TODO: nn.Sequential로 다음 구조를 정의한다
        #   Linear(vocab_size → hidden_size) → ReLU → Dropout
        #   → Linear(hidden_size → hidden_size//2) → ReLU → Dropout
        #   → Linear(hidden_size//2 → num_classes)
        pass

    def forward(self, x):
        # TODO: self.classifier(x)를 반환한다
        pass
```

**학습 설정**:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TextClassifier(vocab_size=dataset.vocab_size, hidden_size=64, num_classes=2).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
```

**구현할 학습 루프**:

```python
num_epochs = 30
history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

for epoch in range(num_epochs):
    # --- Training ---
    model.train()
    train_loss, train_correct, train_total = 0, 0, 0

    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        # TODO: 순전파 → 손실 계산 → 역전파 → 파라미터 업데이트
        # TODO: train_loss, train_correct, train_total 누적
        pass

    # --- Validation ---
    model.eval()
    val_loss, val_correct, val_total = 0, 0, 0

    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            # TODO: 순전파 → 손실 계산 → 정확도 누적
            pass

    # history에 기록
    # TODO: train_loss, val_loss, train_acc, val_acc를 history에 추가

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {train_loss/train_total:.4f}, Acc: {100*train_correct/train_total:.1f}% | "
              f"Val Loss: {val_loss/val_total:.4f}, Acc: {100*val_correct/val_total:.1f}%")
```

**검증 체크리스트**:
- [ ] 모델의 총 파라미터 수를 확인했는가?
- [ ] 손실이 에폭을 거치며 감소하는가?
- [ ] 학습 정확도가 점차 증가하는가?

**Copilot 프롬프트**:
```
"nn.Sequential로 3층 MLP를 정의하고,
AdamW 옵티마이저로 30 에폭 학습하는 코드를 완성해줘.
각 에폭마다 train/val 손실과 정확도를 기록해줘."
```

---

#### 체크포인트 3: 성능 평가 및 시각화 (15분)

**목표**: Accuracy, Precision, Recall, F1-Score를 계산하고, 학습 곡선과 Confusion Matrix를 시각화한다

**구현할 함수** — `compute_metrics`:

```python
def compute_metrics(y_true, y_pred, num_classes=2):
    """평가 지표 계산 (sklearn 없이 직접 구현)"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    accuracy = np.mean(y_true == y_pred)

    # TODO: Confusion Matrix 계산 (num_classes × num_classes 행렬)
    # TODO: TP, FP, FN 추출 (positive = 1 기준)
    # TODO: Precision = TP / (TP + FP)
    # TODO: Recall = TP / (TP + FN)
    # TODO: F1 = 2 × Precision × Recall / (Precision + Recall)
    # 힌트: 분모가 0인 경우 0.0을 반환하도록 처리

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm,
    }
```

**평가 실행**:

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
```

**시각화** — 학습 곡선과 Confusion Matrix:

```python
import matplotlib.pyplot as plt

# TODO: history의 train_loss, val_loss를 그래프로 그린다
# TODO: history의 train_acc, val_acc를 그래프로 그린다
# TODO: Confusion Matrix를 plt.imshow() 또는 seaborn.heatmap()으로 시각화한다
# 힌트: fig, axes = plt.subplots(1, 3, figsize=(15, 4))로 3개 그래프를 나란히 배치
```

**새 텍스트 예측** (선택):

```python
test_texts = ["정말 재미있는 영화다", "지루하고 별로다"]
# TODO: 각 텍스트를 BoW 벡터로 변환 → 모델 예측 → softmax 확률 출력
```

**검증 체크리스트**:
- [ ] 4개 평가 지표(Accuracy, Precision, Recall, F1)가 모두 출력되는가?
- [ ] 학습 곡선이 감소 추세를 보이는가?
- [ ] Confusion Matrix가 정상 시각화되었는가?

**Copilot 프롬프트**:
```
"sklearn 없이 Confusion Matrix에서 TP, FP, FN을 추출하여
Precision, Recall, F1-Score를 계산하는 함수를 구현해줘"
```

**확장 트랙** (가산점):
- Dropout 비율(0.1 vs 0.3 vs 0.5)을 변경하여 성능 비교
- 은닉층 크기(32 vs 64 vs 128)에 따른 정확도 변화 분석

---

### 제출 안내 (Google Classroom)

**제출 방법**: "2주차 B회차" 과제에 조별 1부 제출. 파일명: `group_{조번호}_ch2B.zip`

**포함할 파일**:
- 구현 코드 파일 (`.py`)
- 학습 곡선 + Confusion Matrix 이미지 (`.png`)
- 분석 리포트 (`report.md`, 1페이지): 구현 과정, 성능 해석, Copilot 활용 경험

**제출 마감**: 수업 종료 후 24시간 이내

---

### 결과 토론 가이드

> **토론 가이드**: 조별 구현 결과를 공유하며, 다른 조의 모델 성능과 하이퍼파라미터 선택을 비교하고 성능 차이의 원인을 함께 분석한다

**토론 주제**:

① **하이퍼파라미터 영향**
- 학습률(0.0001 vs 0.001 vs 0.01)과 Dropout 비율이 수렴 속도와 최종 성능에 어떤 영향을 주었는가?

② **모델 성능 분석**
- Confusion Matrix에서 어느 방향의 오분류(FP vs FN)가 더 많은가? 그 이유는?
- Precision과 Recall 중 어느 것이 더 높은가?

③ **BoW의 한계와 3주차 연결**
- "정말 좋은 영화"와 "정말 나쁜 영화"를 BoW로 구분할 수 있는가?
- 단어 순서를 무시하는 BoW의 한계를 3주차 Self-Attention은 어떻게 해결할까?

---

### 다음 주 예고

3주차 A회차에서는 **단어 임베딩과 Attention의 원리**를 다룬다.

- **Word2Vec**: "비슷한 문맥의 단어는 비슷한 의미" — 단어를 밀도 벡터로 표현
- **Self-Attention**: 문장 내 단어 간 관계를 학습하는 메커니즘
- **사전 준비**: 2주차 MLP 개념(활성화 함수, 역전파) 복습

---

## 참고 자료

**실습 코드**:
- _전체 구현 코드는 practice/chapter2/code/2-4-텍스트분류.py 참고_
- _신경망 기초 시연은 practice/chapter2/code/2-1-신경망기초.py 참고_
- _모델 개발 패턴은 practice/chapter2/code/2-2-모델개발.py 참고_

---

**마지막 업데이트**: 2026-02-25
**교과목**: 딥러닝 자연어처리 — LLM 시대의 NLP 엔지니어링
**수업 형태**: 주2회 90분 B회차 (실습 + 토론)
