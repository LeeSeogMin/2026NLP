## 2주차 B회차: MLP 텍스트 분류 실습

> **미션**: PyTorch `nn.Module`로 IMDb 영화 리뷰 감성 분류 MLP 모델을 구현하고, 하이퍼파라미터 튜닝을 통해 모델 성능을 최적화하며, Confusion Matrix와 학습 곡선을 분석할 수 있다

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

**신경망의 기본 구조**:
- 퍼셉트론은 직선으로만 분리 가능한 문제(AND, OR)를 풀 수 있지만, XOR 같은 비선형 문제는 풀 수 없다
- 은닉층을 추가한 다층 퍼셉트론(MLP)은 비선형 경계를 학습하여 XOR을 완벽하게 해결한다
- 각 층이 입력을 중간 표현으로 변환하고, 최종 층이 판단하는 **표현 학습**이 핵심이다

**활성화 함수의 역할**:
- ReLU(max(0, x))는 계산이 빠르고 기울기 소실 문제를 해결하여 가장 널리 사용된다
- GELU는 ReLU의 부드러운 버전으로, 최신 Transformer 모델의 표준이다
- Softmax는 출력을 확률 분포로 변환하여 다중 분류에 사용된다

**역전파와 경사 하강법**:
- 손실 함수(Loss)는 예측과 정답의 거리를 하나의 숫자로 표현한다
- 경사 하강법은 손실을 줄이는 방향으로 파라미터를 w_new = w_old - lr × ∂Loss/∂w로 업데이트한다
- 역전파(Backpropagation)는 연쇄 법칙을 이용하여 모든 파라미터의 그래디언트를 효율적으로 계산한다

**PyTorch nn.Module 패턴**:
- 모든 신경망은 nn.Module을 상속하여 `__init__`에서 구조, `forward()`에서 데이터 흐름을 정의한다
- nn.Linear(in_features, out_features)는 완전 연결층으로, w × x + b 연산을 수행한다
- `optimizer.zero_grad()`, `loss.backward()`, `optimizer.step()`이 학습 루프의 핵심이다

**실습 연계**:
- 이론에서 배운 MLP를 실제 데이터(IMDb 영화 리뷰)로 구현한다
- 텍스트 전처리와 Bag-of-Words 벡터화를 통해 단어를 숫자로 변환한다
- 하이퍼파라미터 튜닝(학습률, 배치 크기, 은닉층 크기)이 모델 성능에 미치는 영향을 체험한다

---

### 과제 스펙

**과제**: IMDb 영화 리뷰 감성 분류 MLP 모델 구현 + 성능 분석 보고서

**제출 형태**: 조별 1부, Google Classroom 업로드

**파일 구성**:
- 구현 코드 파일 (`*.py`)
- 평가 메트릭 시각화 이미지 (학습 곡선, Confusion Matrix)
- 분석 리포트 (1-2페이지)

**검증 기준**:
- ✓ 텍스트 전처리 (토큰화, 어휘 사전, 패딩) 완료
- ✓ Bag-of-Words 또는 TF-IDF 벡터화 구현
- ✓ MLP 모델 정의 (input → hidden(128) → hidden(64) → output)
- ✓ 모델 학습 및 손실 감소 확인
- ✓ Accuracy, Precision, Recall, F1-Score 계산
- ✓ Confusion Matrix 시각화
- ✓ 배치 크기 / 학습률 변화에 따른 성능 비교

---

### 2인1조 실습

> **Copilot 활용**: Copilot에게 "PyTorch nn.Module로 3층 MLP 텍스트 분류 모델을 작성해줘"로 시작하여, 생성된 forward() 메서드의 각 층이 하는 역할을 분석한다. "이 모델로 IMDb 데이터를 학습하는 코드를 작성해줄래?"라고 확장하여 전체 파이프라인을 구성하고, 각 단계에서 Copilot의 제안을 검토·수정하는 과정을 통해 MLP의 작동 원리를 깊이 이해한다.

**역할 분담**:
- **조원 A (드라이버)**: 코드 작성 및 실행, 결과 확인, Copilot과 대화
- **조원 B (네비게이터)**: 로직 검토, 오류 해석, 다음 단계 설계, 프롬프트 최적화
- **체크포인트마다 역할 교대**: 드라이버와 네비게이터를 번갈아가며 진행하여 두 명 모두 전체 구현을 이해한다

---

#### 체크포인트 1: 텍스트 전처리 및 벡터화 (15분)

**목표**: IMDb 데이터셋을 로드하여 텍스트 전처리와 Bag-of-Words 벡터화를 수행하고, 입력 차원을 확인한다

**핵심 단계**:

① **IMDb 데이터셋 로드**

```python
import torch
import torch.nn as nn
from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
from collections import Counter
import numpy as np

# 데이터셋 로드
train_dataset, test_dataset = IMDB(split=('train', 'test'))

# 첫 3개 샘플 확인
for label, text in list(train_dataset)[:3]:
    print(f"Label: {label}, Text: {text[:100]}")
```

예상 출력:
```
Label: 1, Text: This film set in the Vienna of 1912 has an aristocratic woman looking for her uncle...
Label: 1, Text: I would like to have seen this film to understand more about the sexual abuse...
Label: 0, Text: I want to like the film, but it is poorly put together...
```

② **토큰화 및 어휘 사전 구축**

```python
# 토크나이저 설정
tokenizer = get_tokenizer("basic_english")

# 어휘 사전 구축
vocab_counter = Counter()
for label, text in train_dataset:
    tokens = tokenizer(text)
    vocab_counter.update(tokens)

# 빈도 상위 10,000개 단어 선택
vocab_size = 10000
vocab = {word: idx for idx, (word, _) in enumerate(
    vocab_counter.most_common(vocab_size), start=1)}

print(f"어휘 사전 크기: {len(vocab)}")
print(f"샘플 어휘:")
for word, idx in list(vocab.items())[:5]:
    print(f"  '{word}': {idx}")
```

예상 출력:
```
어휘 사전 크기: 10000
샘플 어휘:
  'the': 1
  'and': 2
  'a': 3
  'of': 4
  'to': 5
```

③ **Bag-of-Words 벡터화 및 패딩**

```python
def preprocess_text(text, tokenizer, vocab, max_len=500):
    """
    텍스트를 BoW 벡터로 변환
    Args:
        text: 입력 문장
        tokenizer: 토크나이저
        vocab: 어휘 사전
        max_len: 최대 시퀀스 길이
    Returns:
        bow_vector: Bag-of-Words 벡터
    """
    tokens = tokenizer(text)
    # 단어를 인덱스로 변환
    indices = [vocab.get(token, 0) for token in tokens]  # OOV는 0
    # 길이 조정
    indices = indices[:max_len]
    indices = indices + [0] * (max_len - len(indices))  # 패딩

    # BoW 벡터 생성 (vocab_size 차원)
    bow_vector = np.zeros(vocab_size + 1)
    for idx in indices[:max_len]:
        bow_vector[idx] += 1

    return bow_vector

# 샘플 텍스트 전처리
sample_text = "This is a great movie"
bow_vec = preprocess_text(sample_text, tokenizer, vocab)
print(f"BoW 벡터 차원: {bow_vec.shape}")
print(f"0이 아닌 요소 개수: {np.count_nonzero(bow_vec)}")
print(f"BoW 벡터 (처음 10개): {bow_vec[:10]}")
```

예상 출력:
```
BoW 벡터 차원: (10001,)
0이 아닌 요소 개수: 4
BoW 벡터 (처음 10개): [485.  1.  1.  1.  1.  0.  0.  0.  0.  0.]
```

④ **전체 데이터셋 처리**

```python
# 학습 데이터 변환
X_train = []
y_train = []
for label, text in train_dataset:
    bow_vec = preprocess_text(text, tokenizer, vocab)
    X_train.append(bow_vec)
    # IMDb 레이블: 1 = 긍정, 2 = 부정 → 0/1로 변환
    y_train.append(0 if label == 1 else 1)

X_train = np.array(X_train)
y_train = np.array(y_train)

# 테스트 데이터 변환
X_test = []
y_test = []
for label, text in test_dataset:
    bow_vec = preprocess_text(text, tokenizer, vocab)
    X_test.append(bow_vec)
    y_test.append(0 if label == 1 else 1)

X_test = np.array(X_test)
y_test = np.array(y_test)

print(f"학습 데이터 형태: X_train={X_train.shape}, y_train={y_train.shape}")
print(f"테스트 데이터 형태: X_test={X_test.shape}, y_test={y_test.shape}")
print(f"클래스 분포 (학습): 부정={np.sum(y_train == 1)}, 긍정={np.sum(y_train == 0)}")
print(f"클래스 분포 (테스트): 부정={np.sum(y_test == 1)}, 긍정={np.sum(y_test == 0)}")
```

예상 출력:
```
학습 데이터 형태: X_train=(25000, 10001), y_train=(25000,)
테스트 데이터 형태: X_test=(25000, 10001), y_test=(25000,)
클래스 분포 (학습): 부정=12500, 긍정=12500
클래스 분포 (테스트): 부정=12500, 긍정=12500
```

⑤ **PyTorch DataLoader 구성**

```python
from torch.utils.data import TensorDataset, DataLoader

# NumPy → PyTorch Tensor 변환
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.LongTensor(y_train)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.LongTensor(y_test)

# 데이터셋 생성
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

# DataLoader 생성
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 첫 배치 확인
batch_x, batch_y = next(iter(train_loader))
print(f"배치 크기: X={batch_x.shape}, y={batch_y.shape}")
print(f"배치 레이블 예시: {batch_y[:5].tolist()}")
```

예상 출력:
```
배치 크기: X=torch.Size([32, 10001]), y=torch.Size([32])
배치 레이블 예시: [0, 1, 1, 0, 1]
```

**검증 체크리스트**:
- [ ] IMDb 데이터셋이 정상적으로 로드되었는가? (25,000개 학습, 25,000개 테스트)
- [ ] 어휘 사전 크기가 약 10,000인가?
- [ ] Bag-of-Words 벡터 차원이 10,001인가?
- [ ] 클래스가 균형 있게 분포하는가? (부정/긍정 50:50)
- [ ] DataLoader가 정상 작동하는가?

**Copilot 프롬프트 1**:
```
"IMDb 데이터셋을 로드하고 기본 영어 토크나이저로 토큰화한 뒤, 상위 10,000개 단어로 어휘 사전을 만들어줄래?"
```

**Copilot 프롬프트 2**:
```
"텍스트를 Bag-of-Words 벡터로 변환하는 함수를 작성해줄래?
각 단어의 빈도를 세고, 최대 길이 500으로 패딩하는 코드를 포함해줘."
```

---

#### 체크포인트 2: MLP 모델 정의 및 학습 (20분)

**목표**: PyTorch nn.Module로 3층 MLP 모델을 구현하고, Adam 옵티마이저로 5 에폭 학습을 수행한다

**핵심 단계**:

① **MLP 모델 클래스 정의**

```python
class MLPTextClassifier(nn.Module):
    """
    MLP 기반 텍스트 분류 모델
    구조: Input(10001) → Hidden(128) → ReLU → Hidden(64) → ReLU → Output(2)
    """

    def __init__(self, input_size, hidden1_size=128, hidden2_size=64, output_size=2, dropout_rate=0.5):
        super().__init__()

        # 층 정의
        self.fc1 = nn.Linear(input_size, hidden1_size)      # 입력층 → 은닉층1
        self.relu1 = nn.ReLU()                              # 활성화
        self.dropout1 = nn.Dropout(dropout_rate)            # 드롭아웃

        self.fc2 = nn.Linear(hidden1_size, hidden2_size)    # 은닉층1 → 은닉층2
        self.relu2 = nn.ReLU()                              # 활성화
        self.dropout2 = nn.Dropout(dropout_rate)            # 드롭아웃

        self.fc3 = nn.Linear(hidden2_size, output_size)     # 은닉층2 → 출력층

    def forward(self, x):
        """
        순전파
        Args:
            x: 입력 (batch_size, 10001)
        Returns:
            logits: 출력 로짓 (batch_size, 2)
        """
        # 첫 번째 블록: fc1 → relu1 → dropout1
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        # 두 번째 블록: fc2 → relu2 → dropout2
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)

        # 출력층 (활성화 함수 없음 — CrossEntropyLoss가 softmax 포함)
        logits = self.fc3(x)

        return logits

# 모델 생성
input_size = vocab_size + 1  # 10001
model = MLPTextClassifier(input_size, hidden1_size=128, hidden2_size=64, output_size=2)

# 모델 정보 출력
total_params = sum(p.numel() for p in model.parameters())
print(f"모델 구조:")
print(model)
print(f"\n총 파라미터 수: {total_params:,}")

# 각 층의 파라미터 수 확인
print(f"\n레이어별 파라미터:")
for name, param in model.named_parameters():
    print(f"  {name}: {param.shape} = {param.numel():,}")
```

예상 출력:
```
모델 구조:
MLPTextClassifier(
  (fc1): Linear(in_features=10001, out_features=128, bias=True)
  (relu1): ReLU()
  (dropout1): Dropout(p=0.5, inplace=False)
  (fc2): Linear(in_features=128, out_features=64, bias=True)
  (relu2): ReLU()
  (dropout2): Dropout(p=0.5, inplace=False)
  (fc3): Linear(in_features=64, out_features=2, bias=True)
)

총 파라미터 수: 1,285,440

레이어별 파라미터:
  fc1.weight: torch.Size([128, 10001]) = 1,280,128
  fc1.bias: torch.Size([128]) = 128
  fc2.weight: torch.Size([64, 128]) = 8,192
  fc2.bias: torch.Size([64]) = 64
  fc3.weight: torch.Size([2, 64]) = 128
  fc3.bias: torch.Size([2]) = 128
```

② **손실 함수 및 옵티마이저 설정**

```python
# GPU 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"사용 디바이스: {device}")

# 모델을 GPU/CPU로 이동
model = model.to(device)

# 손실 함수: 다중 분류
criterion = nn.CrossEntropyLoss()

# 옵티마이저: Adam (적응형 학습률)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 선택: 학습률 스케줄러 (학습 진행에 따라 lr 감소)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=1, verbose=True
)
```

③ **학습 루프**

```python
def train_epoch(model, train_loader, criterion, optimizer, device):
    """
    한 에폭 학습
    Returns:
        평균 손실값
    """
    model.train()  # 학습 모드
    total_loss = 0.0

    for batch_x, batch_y in train_loader:
        # 데이터를 GPU/CPU로 이동
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        # 순전파
        logits = model(batch_x)
        loss = criterion(logits, batch_y)

        # 역전파
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)

def evaluate(model, test_loader, criterion, device):
    """
    모델 평가
    Returns:
        평균 손실값, 정확도
    """
    model.eval()  # 평가 모드
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            total_loss += loss.item()

            # 정확도 계산
            preds = logits.argmax(dim=1)
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)

    return total_loss / len(test_loader), correct / total

# 학습 루프 (5 에폭)
num_epochs = 5
train_losses = []
val_losses = []
val_accuracies = []

print(f"학습 시작 (에폭: {num_epochs})")
for epoch in range(num_epochs):
    # 학습
    train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
    train_losses.append(train_loss)

    # 평가
    val_loss, val_acc = evaluate(model, test_loader, criterion, device)
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)

    # 학습률 스케줄러 업데이트
    scheduler.step(val_loss)

    print(f"Epoch {epoch+1}/{num_epochs} | "
          f"Train Loss: {train_loss:.4f} | "
          f"Val Loss: {val_loss:.4f} | "
          f"Val Accuracy: {val_acc:.4f}")

print(f"\n학습 완료!")
```

예상 출력:
```
사용 디바이스: cuda
학습 시작 (에폭: 5)
Epoch 1/5 | Train Loss: 0.6821 | Val Loss: 0.5432 | Val Accuracy: 0.7234
Epoch 2/5 | Train Loss: 0.4102 | Val Loss: 0.3891 | Val Accuracy: 0.8145
Epoch 3/5 | Train Loss: 0.2853 | Val Loss: 0.3245 | Val Accuracy: 0.8567
Epoch 4/5 | Train Loss: 0.2145 | Val Loss: 0.3012 | Val Accuracy: 0.8723
Epoch 5/5 | Train Loss: 0.1876 | Val Loss: 0.2987 | Val Accuracy: 0.8756

학습 완료!
```

④ **모델 저장**

```python
# 최상 모델 저장
torch.save(model.state_dict(), "best_mlp_classifier.pth")
print("모델 저장 완료: best_mlp_classifier.pth")
```

**검증 체크리스트**:
- [ ] 모델 파라미터 수가 약 1,285,440인가?
- [ ] 모든 층이 정상 작동하는가? (fc1, fc2, fc3)
- [ ] 손실이 에폭을 거치며 감소하는가?
- [ ] 정확도가 70% 이상인가?
- [ ] GPU 사용이 활성화되었는가?

**Copilot 프롬프트 3**:
```
"PyTorch nn.Module로 3층 MLP 텍스트 분류 모델을 작성해줄래?
입력 10001, 은닉층 128, 64, 출력 2 크기를 갖고
ReLU와 Dropout을 포함해줘."
```

**Copilot 프롬프트 4**:
```
"위의 MLP 모델을 Adam 옵티마이저로 5 에폭 학습하는 코드를 작성해줄래?
각 에폭마다 손실과 정확도를 출력하고, 학습 루프를 명확히 해줘."
```

---

#### 체크포인트 3: 성능 평가 및 분석 (15분)

**목표**: Confusion Matrix, 정밀도, 재현율, F1-Score를 계산하고 시각화하여 모델 성능을 깊이 있게 분석한다

**핵심 단계**:

① **전체 테스트셋 예측**

```python
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# 모델 평가 모드
model.eval()

all_preds = []
all_targets = []

with torch.no_grad():
    for batch_x, batch_y in test_loader:
        batch_x = batch_x.to(device)

        logits = model(batch_x)
        preds = logits.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(batch_y.numpy())

all_preds = np.array(all_preds)
all_targets = np.array(all_targets)

# 성능 지표 계산
accuracy = accuracy_score(all_targets, all_preds)
precision = precision_score(all_targets, all_preds)
recall = recall_score(all_targets, all_preds)
f1 = f1_score(all_targets, all_preds)

print(f"테스트셋 성능 지표:")
print(f"  Accuracy:  {accuracy:.4f} (전체 맞은 비율)")
print(f"  Precision: {precision:.4f} (부정 판정 중 실제 부정 비율)")
print(f"  Recall:    {recall:.4f} (실제 부정 중 부정으로 맞힌 비율)")
print(f"  F1-Score:  {f1:.4f} (Precision과 Recall의 조화평균)")
```

예상 출력:
```
테스트셋 성능 지표:
  Accuracy:  0.8756 (전체 맞은 비율)
  Precision: 0.8834 (부정 판정 중 실제 부정 비율)
  Recall:    0.8643 (실제 부정 중 부정으로 맞힌 비율)
  F1-Score:  0.8738 (Precision과 Recall의 조화평균)
```

② **Confusion Matrix 계산 및 시각화**

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Confusion Matrix 계산
cm = confusion_matrix(all_targets, all_preds)

# 정규화 버전 (백분율)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# 시각화
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# 절대값
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['긍정', '부정'], yticklabels=['긍정', '부정'])
axes[0].set_title('Confusion Matrix (절대값)', fontsize=12, fontweight='bold')
axes[0].set_ylabel('실제 레이블')
axes[0].set_xlabel('예측 레이블')

# 백분율
sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='YlOrRd', ax=axes[1],
            xticklabels=['긍정', '부정'], yticklabels=['긍정', '부정'])
axes[1].set_title('Confusion Matrix (정규화)', fontsize=12, fontweight='bold')
axes[1].set_ylabel('실제 레이블')
axes[1].set_xlabel('예측 레이블')

plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
print("저장: confusion_matrix.png")
plt.show()

# 의미 해석
print(f"\nConfusion Matrix 해석:")
print(f"  True Negative (TN): {cm[0, 0]} (긍정을 긍정으로 맞힘)")
print(f"  False Positive (FP): {cm[0, 1]} (긍정을 부정으로 오분류)")
print(f"  False Negative (FN): {cm[1, 0]} (부정을 긍정으로 오분류)")
print(f"  True Positive (TP): {cm[1, 1]} (부정을 부정으로 맞힘)")
```

예상 출력:
```
저장: confusion_matrix.png

Confusion Matrix 해석:
  True Negative (TN): 10928 (긍정을 긍정으로 맞힘)
  False Positive (FP): 1572 (긍정을 부정으로 오분류)
  False Negative (FN): 1339 (부정을 긍정으로 오분류)
  True Positive (TP): 11161 (부정을 부정으로 맞힘)
```

③ **학습 곡선 시각화**

```python
# 학습 곡선 그리기
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# 손실 곡선
axes[0].plot(range(1, num_epochs+1), train_losses, marker='o', label='Train Loss', linewidth=2)
axes[0].plot(range(1, num_epochs+1), val_losses, marker='s', label='Validation Loss', linewidth=2)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('학습 곡선 (손실)', fontsize=12, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 정확도 곡선
axes[1].plot(range(1, num_epochs+1), val_accuracies, marker='o', color='green', linewidth=2)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].set_title('검증 정확도', fontsize=12, fontweight='bold')
axes[1].set_ylim([0.7, 0.9])
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
print("저장: training_curves.png")
plt.show()
```

예상 출력:
```
저장: training_curves.png
```

④ **배치 크기 및 학습률 비교 분석**

```python
# 다양한 하이퍼파라미터로 빠른 테스트 (1 에폭)
results = []

for batch_size in [16, 32, 64]:
    # 다른 batch_size로 DataLoader 재구성
    temp_train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    temp_test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    for learning_rate in [0.0001, 0.001, 0.01]:
        # 모델 재초기화
        temp_model = MLPTextClassifier(input_size, hidden1_size=128, hidden2_size=64, output_size=2)
        temp_model = temp_model.to(device)

        temp_optimizer = torch.optim.Adam(temp_model.parameters(), lr=learning_rate)

        # 1 에폭 학습
        train_loss = train_epoch(temp_model, temp_train_loader, criterion, temp_optimizer, device)
        val_loss, val_acc = evaluate(temp_model, temp_test_loader, criterion, device)

        results.append({
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_acc': val_acc
        })

        print(f"Batch={batch_size}, LR={learning_rate}: Train Loss={train_loss:.4f}, Val Acc={val_acc:.4f}")

# 결과 시각화
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Batch Size 효과
for bs in [16, 32, 64]:
    bs_results = [r['val_acc'] for r in results if r['batch_size'] == bs]
    axes[0].plot([0.0001, 0.001, 0.01], bs_results, marker='o', label=f'Batch={bs}')
axes[0].set_xscale('log')
axes[0].set_xlabel('Learning Rate')
axes[0].set_ylabel('Validation Accuracy')
axes[0].set_title('학습률에 따른 정확도 (배치 크기 비교)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Learning Rate 효과
for lr in [0.0001, 0.001, 0.01]:
    lr_results = [r['val_acc'] for r in results if r['learning_rate'] == lr]
    axes[1].plot([16, 32, 64], lr_results, marker='o', label=f'LR={lr}')
axes[1].set_xlabel('Batch Size')
axes[1].set_ylabel('Validation Accuracy')
axes[1].set_title('배치 크기에 따른 정확도 (학습률 비교)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# 히트맵: Batch Size × Learning Rate
heatmap_data = np.zeros((3, 3))
for i, bs in enumerate([16, 32, 64]):
    for j, lr in enumerate([0.0001, 0.001, 0.01]):
        for r in results:
            if r['batch_size'] == bs and r['learning_rate'] == lr:
                heatmap_data[i, j] = r['val_acc']

sns.heatmap(heatmap_data, annot=True, fmt='.4f', cmap='YlGn', ax=axes[2],
            xticklabels=['0.0001', '0.001', '0.01'],
            yticklabels=['16', '32', '64'],
            cbar_kws={'label': 'Validation Accuracy'})
axes[2].set_xlabel('Learning Rate')
axes[2].set_ylabel('Batch Size')
axes[2].set_title('하이퍼파라미터 그리드 (정확도)')

plt.tight_layout()
plt.savefig('hyperparameter_analysis.png', dpi=150, bbox_inches='tight')
print("저장: hyperparameter_analysis.png")
plt.show()
```

예상 출력:
```
Batch=16, LR=0.0001: Train Loss=0.5234, Val Acc=0.7234
Batch=16, LR=0.001: Train Loss=0.4103, Val Acc=0.8012
Batch=16, LR=0.01: Train Loss=0.3891, Val Acc=0.7856
Batch=32, LR=0.0001: Train Loss=0.5121, Val Acc=0.7145
Batch=32, LR=0.001: Train Loss=0.3987, Val Acc=0.8145
Batch=32, LR=0.01: Train Loss=0.4215, Val Acc=0.7934
Batch=64, LR=0.0001: Train Loss=0.5456, Val Acc=0.6987
Batch=64, LR=0.001: Train Loss=0.4234, Val Acc=0.8067
Batch=64, LR=0.01: Train Loss=0.4891, Val Acc=0.7745
저장: hyperparameter_analysis.png
```

**검증 체크리스트**:
- [ ] Accuracy가 80% 이상인가?
- [ ] Precision, Recall이 0.80 이상인가?
- [ ] Confusion Matrix가 정상 시각화되었는가?
- [ ] 학습 곡선이 감소 추세를 보이는가?
- [ ] 배치 크기/학습률 분석이 완료되었는가?

**Copilot 프롬프트 5**:
```
"sklearn을 사용해서 테스트셋에 대한
Accuracy, Precision, Recall, F1-Score를 계산하는 코드를 작성해줄래?"
```

**Copilot 프롬프트 6**:
```
"Confusion Matrix를 seaborn으로 히트맵으로 시각화하고,
절대값과 정규화 버전을 나란히 표시해줄래?"
```

**선택 프롬프트**:
```
"배치 크기와 학습률을 변화시키면서 정확도를 비교하는
그래프를 matplotlib으로 그려줄래?"
```

---

### 제출 안내 (Google Classroom)

**제출 방법**:
- Google Classroom의 "2주차 B회차" 과제에 조별 1부 제출
- 파일명 형식: `group_{조번호}_ch2B.zip`

**포함할 파일**:
```
group_{조번호}_ch2B/
├── ch2B_mlp_classifier.py      # 전체 구현 코드
├── confusion_matrix.png         # Confusion Matrix 히트맵
├── training_curves.png          # 학습 곡선 + 검증 정확도
├── hyperparameter_analysis.png  # 배치 크기/학습률 비교
└── report.md                    # 분석 리포트 (1-2페이지)
```

**리포트 포함 항목** (report.md):
- 각 체크포인트의 구현 과정 및 어려웠던 점 (3-4문장)
- 최종 모델 성능: Accuracy, Precision, Recall, F1-Score 및 해석 (3-4문장)
- Confusion Matrix 분석: 어떤 경우에 오분류가 발생했는가? (2-3문장)
- 배치 크기/학습률의 영향 분석: 어떤 조합이 최적이었는가, 그 이유는? (3-4문장)
- Copilot 활용 경험: 어떤 프롬프트가 효과적이었는가, 생성된 코드를 어떻게 수정했는가? (2-3문장)

**제출 마감**: 수업 종료 후 24시간 이내

---

### 결과 토론 가이드

> **토론 가이드**: 조별 구현 결과를 공유하며, 다른 조의 모델 성능과 하이퍼파라미터 선택을 비교하고 성능 차이의 원인을 함께 분석한다

**토론 주제**:

① **모델 아키텍처 선택**
- 모든 조가 동일한 3층 MLP (128-64)를 사용했는가?
- 은닉층 크기를 다르게 설정한 조는? 그 결과는?
- Dropout 비율을 조정한 경우 과적합 방지 효과는?

② **텍스트 전처리 전략**
- 모두 Bag-of-Words를 사용했는가, 아니면 TF-IDF를 시도한 조도 있는가?
- 최대 시퀀스 길이(500)를 조정한 경우 성능 변화는?
- 어휘 사전 크기(10,000)가 최적인가? 더 크거나 작으면?

③ **하이퍼파라미터 영향**
- 배치 크기 (16 vs 32 vs 64): 어떤 크기가 가장 좋았는가?
- 학습률 (0.0001 vs 0.001 vs 0.01): 수렴 속도와 최종 성능은?
- "배치 크기가 크면 정확도가 낮다"는 것이 맞는가? 왜 그럴까?

④ **모델 성능 분석**
- 모든 조의 정확도가 85% 이상인가? 차이가 나는 이유는?
- Precision과 Recall 중 어느 것이 더 높은가? 불균형이 있는 이유는?
- Confusion Matrix에서 어느 방향의 오분류(FP vs FN)가 더 많은가?

⑤ **실무적 시사**
- 이 모델을 실제 영화 리뷰 사이트에 배포하면 어떤 문제가 생길까?
- 정확도 85%가 높은가, 낮은가? 어떤 상황에서는 충분하고 어떤 상황에서는 부족한가?
- 텍스트 분류를 더 잘하려면 어떤 접근을 해야 할까? (3주차 Self-Attention과의 연결)

**발표 형식**:
- 각 조 3~5분 발표 (구현 전략 + 주요 결과)
- 다른 조의 질문에 답변 (2~3개 질문)
- 교수의 보충 설명 및 피드백

---

### 다음 주 예고

다음 주 3장 A회차에서는 **단어 임베딩과 Attention의 원리**를 깊이 있게 다룬다.

**예고 내용**:
- Word2Vec: "비슷한 문맥의 단어는 비슷한 의미" — 수학적 구현
- 사전학습 임베딩(GloVe, FastText) 활용: 원칙부터 코드까지
- RNN/LSTM/GRU의 기본: 왜 순차 처리가 필요했는가
- Attention 메커니즘: Query, Key, Value의 직관적 이해
- Self-Attention 구현: PyTorch로 밑바닥부터
- B회차에서는 Self-Attention을 직접 구현하고 Attention Weight를 시각화한다

**사전 준비**:
- 2주차 MLP 개념(활성화 함수, 역전파) 복습
- "단어"와 "숫자"의 변환 과정(임베딩) 미리 생각해보기
- 시험 공부할 때 "무엇에 집중할지 선택"하는 과정과 Attention을 비유로 연결해보기

---

## 참고 자료

**실습 코드**:
- _전체 구현 코드는 practice/chapter2/code/2-2-mlp분류.py 참고_
- _데이터 전처리 및 벡터화는 practice/chapter2/code/2-2-전처리.py 참고_
- _배치 처리 패턴은 practice/chapter2/code/2-2-dataloader.py 참고_

**권장 읽기**:
- PyTorch 공식 튜토리얼: Text Classification with TorchText. https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press. Ch. 6-7.
- Graves, A. (2012). Supervised Sequence Labelling with Recurrent Neural Networks. https://arxiv.org/abs/1308.0850

**데이터셋**:
- IMDb Large Movie Review Dataset: http://ai.stanford.edu/~amaas/data/sentiment/ (25,000 train, 25,000 test)
- 또는 torchtext.datasets.IMDB로 자동 로드 가능

---

**마지막 업데이트**: 2026-02-25
**교과목**: 딥러닝 자연어처리 — LLM 시대의 NLP 엔지니어링
**수업 형태**: 주2회 90분 B회차 (실습 + 토론)
