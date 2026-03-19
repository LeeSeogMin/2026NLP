## 9주차 B회차: Full Fine-tuning 실습

> **미션**: Hugging Face Trainer API로 BERT 모델을 감성 분류 데이터셋에 파인튜닝하고, 학습 곡선을 분석하며 Confusion Matrix와 다양한 평가 지표를 통해 성능을 분석할 수 있다

### 수업 타임라인

| 시간 | 내용 | Copilot 사용 |
|------|------|-------------|
| 00:00~00:05 | 조 편성 + 역할 배분 (조원 A/B) | 사용 안 함 |
| 00:05~00:10 | A회차 핵심 리캡 + 과제 스펙 재확인 | 사용 안 함 |
| 00:10~00:55 | 2인1조 Copilot 구현 (체크포인트 3회) | 적극 사용 |
| 00:55~01:00 | Google Classroom 제출 (조별 1부) | 사용 안 함 |
| 01:00~01:20 | 결과 토론 (파인튜닝 전략 비교·성능 차이 분석) | 사용 안 함 |
| 01:20~01:28 | 핵심 정리 | 사용 안 함 |
| 01:28~01:30 | 다음 주 예고 | 사용 안 함 |

---

### A회차 핵심 리캡

**파인튜닝의 원리**:
- 사전학습된 BERT의 모든 파라미터를 도메인 데이터로 업데이트한다
- Full Fine-tuning은 모든 110M 파라미터를 학습 가능하게 둔다
- 매우 작은 학습률(2e-5)을 사용하여 사전학습된 가중치를 과도하게 변경하지 않는다

**과적합 방지 전략**:
- Validation Set을 별도로 분리하여 검증 손실을 모니터링한다
- Early Stopping으로 검증 성능이 악화되면 학습을 멈춘다
- Learning Rate Scheduling(Warmup + Linear Decay)으로 안정적인 학습을 유도한다
- Weight Decay 정규화로 가중치가 과도하게 변경되는 것을 방지한다

**Hugging Face Trainer API의 역할**:
- TrainingArguments로 모든 하이퍼파라미터를 정의한다
- Datasets 라이브러리로 데이터를 로드하고 전처리한다
- Trainer 객체가 학습 루프, 체크포인트 저장, 평가를 자동으로 관리한다

**평가 지표**:
- Accuracy: 전체 정확도
- Precision/Recall: 각 클래스별 성능
- F1-Score: Precision과 Recall의 조화평균
- Confusion Matrix: 오류 유형 분석

**실습 연계**:
- 이 실습에서 배운 Full Fine-tuning 기법은 다음 주 10주차에서 배울 PEFT(LoRA)의 기초가 된다
- 파인튜닝의 성능 한계(메모리, 속도)를 경험한 뒤, PEFT의 필요성을 이해할 수 있다

---

### 과제 스펙

**과제**: Hugging Face Trainer로 감성 분류 모델을 파인튜닝하고 성능 분석 리포트 작성

**제출 형태**: 조별 1부, Google Classroom 업로드

**파일 구성**:
- 파인튜닝 코드 파일 (`*.py`)
- 학습 곡선 시각화 (1개: Train Loss vs Val Loss)
- Confusion Matrix 시각화 (1개)
- 간단한 분석 리포트 (1-2페이지)

**검증 기준**:
- ✓ Datasets 라이브러리로 데이터 로드 및 토크나이제이션
- ✓ TrainingArguments와 Trainer 설정 (학습률, Early Stopping 포함)
- ✓ 모델 파인튜닝 및 학습 곡선 시각화
- ✓ Confusion Matrix 및 클래스별 Precision/Recall 계산
- ✓ 과적합 신호 분석 및 해석

---

### 2인1조 실습

> **Copilot 활용**: 기본 코드 틀은 직접 작성해본 뒤, "이 코드에 Early Stopping을 추가해줄래?", "Confusion Matrix를 출력하는 코드 작성해줄까?" 같이 단계적으로 Copilot에 요청한다. Copilot의 제안을 검토하고 실행해본 뒤 결과를 해석하는 과정에서 파인튜닝의 전체 흐름을 깊이 이해할 수 있다.

**역할 분담**:
- **조원 A (드라이버)**: 코드 작성 및 실행, 모델 학습, 결과 확인
- **조원 B (네비게이터)**: 논리 검토, Copilot 프롬프트 설계, 하이퍼파라미터 조정 제안
- **체크포인트마다 역할 교대**: 드라이버와 네비게이터를 번갈아가며 진행하여 두 명 모두 전체 구현을 이해한다

---

#### 체크포인트 1: 데이터 로드 + 토크나이제이션 (15분)

**목표**: Datasets 라이브러리로 감성 분류 데이터를 로드하고, 토크나이저로 전처리한 뒤 Train/Val 분할을 수행한다

**핵심 단계**:

① **패키지 임포트 및 환경 설정**

```python
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from datasets import Dataset, load_dataset

# 한국어 감성 분류 데이터셋 로드 (KLUE YNAT)
print("Loading dataset...")
dataset = load_dataset("klue", "ynat")  # 뉴스 카테고리 분류
print(f"Dataset splits: {dataset.keys()}")
print(f"Train size: {len(dataset['train'])}, Val size: {len(dataset['validation'])}")

# 첫 샘플 확인
print("\nFirst sample:")
print(dataset['train'][0])
```

예상 동작:
```
Loading dataset...
Dataset splits: dict_keys(['train', 'validation'])
Train size: 45678, Val size: 9144

First sample:
{
  'guid': 1,
  'title': '2006년 아마추어 복싱 대회 개최',
  'label': 5  # 레이블은 0~8 (9개 카테고리)
}
```

> **주의**: 이 데이터셋은 뉴스 카테고리이므로, 조원의 편의에 따라 **감성 분류 데이터셋(NSMC, 네이버 영화평)**으로 변경 가능하다. NSMC가 더 직관적인 감성 분류(긍정/부정)이기 때문이다.

②  **토크나이저 로드 및 전처리 함수 정의**

```python
# 토크나이저 로드 (한국어 BERT)
model_name = "bert-base-multilingual-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

print(f"Tokenizer vocab size: {len(tokenizer)}")

# 전처리 함수
def preprocess_function(examples):
    """텍스트를 토큰으로 변환하고 패딩/자르기 수행"""
    return tokenizer(
        examples["title"],  # YNAT 데이터셋의 경우 "title" 사용
        max_length=256,
        padding="max_length",
        truncation=True,
    )

# 전체 데이터셋에 전처리 함수 적용 (병렬 처리)
print("\nTokenizing dataset...")
processed_train = dataset["train"].map(
    preprocess_function,
    batched=True,
    batch_size=100,
    remove_columns=dataset["train"].column_names,
)
processed_val = dataset["validation"].map(
    preprocess_function,
    batched=True,
    batch_size=100,
    remove_columns=dataset["validation"].column_names,
)

print(f"Processed train samples: {len(processed_train)}")
print(f"Processed val samples: {len(processed_val)}")

# 전처리된 샘플 확인
print("\nProcessed sample:")
print(f"  Input IDs shape: {len(processed_train[0]['input_ids'])}")
print(f"  Attention mask shape: {len(processed_train[0]['attention_mask'])}")
print(f"  Label: {processed_train[0]['label']}")
```

예상 결과:
```
Tokenizer vocab size: 119547

Tokenizing dataset...
Processed train samples: 45678
Processed val samples: 9144

Processed sample:
  Input IDs shape: 256
  Attention mask shape: 256
  Label: 5
```

③ **클래스 분포 확인**

```python
# 클래스 분포 확인
train_labels = np.array(dataset["train"]["label"])
val_labels = np.array(dataset["validation"]["label"])

print("Train set class distribution:")
unique, counts = np.unique(train_labels, return_counts=True)
for u, c in zip(unique, counts):
    print(f"  Class {u}: {c} ({c/len(train_labels)*100:.1f}%)")

print("\nVal set class distribution:")
unique, counts = np.unique(val_labels, return_counts=True)
for u, c in zip(unique, counts):
    print(f"  Class {u}: {c} ({c/len(val_labels)*100:.1f}%)")

# 불균형 확인 (클래스 가중치 계산용)
num_labels = len(np.unique(train_labels))
class_weights = len(train_labels) / (num_labels * np.bincount(train_labels))
print(f"\nClass weights: {class_weights}")
```

예상 결과:
```
Train set class distribution:
  Class 0: 5081 (11.1%)
  Class 1: 5173 (11.3%)
  ...
  Class 8: 5167 (11.3%)

Val set class distribution:
  Class 0: 1009 (11.0%)
  ...
```

**검증 체크리스트**:
- [ ] 데이터셋이 올바르게 로드되었는가? (train/val 크기 확인)
- [ ] 토크나이저가 한국어를 제대로 처리하는가? (input_ids 길이 256)
- [ ] 클래스 분포가 어느 정도 균형잡혀 있는가?
- [ ] 전처리된 데이터에 label 컬럼이 있는가?

**Copilot 프롬프트 1**:
```
"KLUE YNAT 데이터셋을 로드하고, bert-base-multilingual-cased 토크나이저로 전처리해줄래?
max_length=256으로 패딩하고, batched=True로 병렬 처리해야 해."
```

**Copilot 프롬프트 2**:
```
"전처리된 데이터의 클래스 분포를 확인하는 코드를 작성해줄래?
각 클래스가 몇 개씩 있는지, 그리고 균형이 잡혀 있는지 확인하면 좋겠어."
```

---

#### 체크포인트 2: 모델 파인튜닝 + 학습 곡선 시각화 (20분)

**목표**: TrainingArguments와 Trainer API를 설정하여 모델을 파인튜닝하고, 학습 과정의 손실 및 정확도 변화를 시각화한다

**핵심 단계**:

① **모델 로드 및 파라미터 확인**

```python
# 모델 로드
num_labels = 9  # YNAT의 9개 카테고리
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=num_labels,
)

# 모델 파라미터 확인
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Model: {model_name}")
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print(f"All parameters trainable: {total_params == trainable_params} (Full Fine-tuning)")

# 모델 구조 확인 (간단히)
print(f"\nModel architecture (summary):")
print(f"  Input: [batch_size, seq_len, 768]")
print(f"  Output: [batch_size, {num_labels}]")
```

예상 결과:
```
Model: bert-base-multilingual-cased
Total parameters: 167,355,651
Trainable parameters: 167,355,651
All parameters trainable: True (Full Fine-tuning)

Model architecture (summary):
  Input: [batch_size, seq_len, 768]
  Output: [batch_size, 9]
```

② **TrainingArguments 설정**

```python
from transformers import TrainingArguments, EarlyStoppingCallback

# 훈련 설정
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=2e-5,  # 파인튜닝: 매우 작은 학습률
    warmup_steps=500,    # 처음 500 스텝 동안 학습률 점진적 증가
    weight_decay=0.01,   # L2 정규화
    logging_steps=50,    # 50 스텝마다 로그 출력
    eval_steps=500,      # 500 스텝마다 검증 수행
    save_steps=500,      # 500 스텝마다 체크포인트 저장
    save_total_limit=3,  # 최근 3개 체크포인트만 보관
    load_best_model_at_end=True,  # 학습 후 최고 성능 모델 로드
    metric_for_best_model="accuracy",  # 최고 성능 판단 기준
    seed=42,
    logging_dir="./logs",
)

print("Training Arguments:")
print(f"  Learning Rate: {training_args.learning_rate}")
print(f"  Batch Size: {training_args.per_device_train_batch_size}")
print(f"  Num Epochs: {training_args.num_train_epochs}")
print(f"  Warmup Steps: {training_args.warmup_steps}")
print(f"  Weight Decay: {training_args.weight_decay}")

# 총 훈련 스텝 계산
num_train_samples = len(processed_train)
batch_size = training_args.per_device_train_batch_size
total_steps = (num_train_samples // batch_size) * training_args.num_train_epochs
print(f"  Total Training Steps: {total_steps}")
```

예상 결과:
```
Training Arguments:
  Learning Rate: 2e-05
  Batch Size: 16
  Num Epochs: 3
  Warmup Steps: 500
  Weight Decay: 0.01
  Total Training Steps: 8569
```

③ **평가 메트릭 함수 정의**

```python
def compute_metrics(eval_preds):
    """검증/테스트 단계에서 메트릭 계산"""
    predictions, labels = eval_preds
    predictions = np.argmax(predictions, axis=1)

    accuracy = accuracy_score(labels, predictions)

    return {
        "accuracy": accuracy,
    }

print("Metrics function defined.")
```

④ **Trainer 초기화 및 학습**

```python
# Early Stopping 콜백
early_stopping = EarlyStoppingCallback(
    early_stopping_patience=2,  # 2 평가 주기 동안 개선 없으면 중단
    early_stopping_threshold=0.0,
)

# Trainer 생성
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=processed_train,
    eval_dataset=processed_val,
    compute_metrics=compute_metrics,
    callbacks=[early_stopping],
)

print("Starting training...")
print("(이 과정은 GPU 기준 약 30~60분 소요됩니다)\n")

# 학습 시작
train_result = trainer.train()

print(f"\nTraining completed!")
print(f"Best model checkpoint: {trainer.best_model_checkpoint}")
print(f"Final train loss: {train_result.training_loss:.4f}")
```

예상 출력 (학습 로그):
```
Starting training...

[  500/8569] Loss: 1.523, Eval Accuracy: 0.7845
[ 1000/8569] Loss: 1.256, Eval Accuracy: 0.8103
[ 1500/8569] Loss: 1.089, Eval Accuracy: 0.8234  ← 최고 성능
[ 2000/8569] Loss: 0.921, Eval Accuracy: 0.8201  ← 성능 감소 (Early Stopping 트리거)

Training completed!
Best model checkpoint: ./results/checkpoint-1500
Final train loss: 0.8423
```

⑤ **학습 곡선 시각화**

```python
# 학습 로그 불러오기 (Trainer의 자동 저장 로그)
import json

log_file = "results/trainer_state.json"
if os.path.exists(log_file):
    with open(log_file, 'r') as f:
        state = json.load(f)

    # 로그 처리
    steps = []
    train_losses = []
    eval_accuracies = []

    for log in state['log_history']:
        if 'loss' in log:
            steps.append(log['step'])
            train_losses.append(log['loss'])
        if 'eval_accuracy' in log:
            eval_steps.append(log['step'])
            eval_accuracies.append(log['eval_accuracy'])
else:
    print("Warning: trainer_state.json not found")

# 학습 곡선 그리기
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 좌측: Loss 곡선
ax1 = axes[0]
ax1.plot(steps, train_losses, marker='o', markersize=4, label='Train Loss', linewidth=1.5, color='steelblue')
if 'eval_steps' in locals():
    ax1.plot(eval_steps, eval_accuracies, marker='s', markersize=5, label='Val Accuracy', linewidth=1.5, color='orange')
if trainer.best_model_checkpoint:
    best_step = int(trainer.best_model_checkpoint.split('-')[-1])
    ax1.axvline(x=best_step, color='red', linestyle='--', linewidth=2, label=f'Best Model (Step {best_step})')
ax1.set_xlabel('Training Step', fontsize=11)
ax1.set_ylabel('Loss / Accuracy', fontsize=11)
ax1.set_title('Learning Curve: Train Loss vs Validation Accuracy', fontsize=12, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# 우측: Validation Accuracy 변화
ax2 = axes[1]
if 'eval_steps' in locals():
    ax2.plot(eval_steps, eval_accuracies, marker='o', markersize=6, label='Val Accuracy', linewidth=2, color='green')
    best_acc = max(eval_accuracies)
    best_step = eval_steps[eval_accuracies.index(best_acc)]
    ax2.axvline(x=best_step, color='red', linestyle='--', linewidth=2, label=f'Best: {best_acc:.4f} at Step {best_step}')
    ax2.scatter([best_step], [best_acc], s=100, color='red', zorder=5)
ax2.set_xlabel('Training Step', fontsize=11)
ax2.set_ylabel('Accuracy', fontsize=11)
ax2.set_title('Validation Accuracy Over Steps', fontsize=12, fontweight='bold')
ax2.set_ylim([0.7, 1.0])
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('practice/chapter9/data/output/learning_curves_ko.png', dpi=150, bbox_inches='tight')
print("Saved: learning_curves_ko.png")
plt.show()
```

예상 그래프:
- 좌측: Train Loss가 1.5에서 0.8로 감소, Val Accuracy가 0.78에서 0.82로 증가한 뒤 감소
- 우측: Val Accuracy가 단조 증가하다 Step 1500 이후 감소 (과적합 신호)

**검증 체크리스트**:
- [ ] 모델이 올바르게 로드되었는가? (파라미터 개수 확인)
- [ ] 모든 파라미터가 학습 가능한가? (Full Fine-tuning)
- [ ] 학습률이 매우 작은가? (2e-5)
- [ ] Early Stopping이 적용되었는가?
- [ ] 학습 곡선이 생성되었는가?

**Copilot 프롬프트 3**:
```
"Hugging Face Trainer API로 BERT를 파인튜닝해줄래?
TrainingArguments로 learning_rate=2e-5, warmup_steps=500을 설정하고,
EarlyStoppingCallback을 추가해줘."
```

**Copilot 프롬프트 4**:
```
"학습 과정의 로그를 불러와서 Train Loss와 Validation Accuracy를 matplotlib으로
그려주는 코드를 작성해줄래? Best Model이 저장된 step을 빨간 점선으로 표시해줘."
```

---

#### 체크포인트 3: Confusion Matrix + 성능 분석 (15분)

**목표**: 검증 세트에서 모델의 예측을 수집하고, Confusion Matrix와 클래스별 Precision/Recall을 계산하여 오류를 분석한다

**핵심 단계**:

① **최고 성능 모델로 검증/테스트 평가**

```python
# 최고 성능 모델은 이미 로드됨 (load_best_model_at_end=True)
print(f"Using best model from: {trainer.best_model_checkpoint}")

# 검증 세트 평가
val_results = trainer.evaluate(eval_dataset=processed_val)
print("\nValidation Results:")
print(f"  Accuracy: {val_results['eval_accuracy']:.4f}")

# 예측 수집 (Confusion Matrix 계산용)
val_predictions = trainer.predict(processed_val)
val_pred_labels = np.argmax(val_predictions.predictions, axis=1)
val_true_labels = np.array(processed_val['label'])

print(f"\nPrediction statistics:")
print(f"  Total predictions: {len(val_pred_labels)}")
print(f"  Correct predictions: {np.sum(val_pred_labels == val_true_labels)}")
print(f"  Accuracy (from predictions): {np.sum(val_pred_labels == val_true_labels) / len(val_true_labels):.4f}")
```

예상 결과:
```
Using best model from: ./results/checkpoint-1500

Validation Results:
  Accuracy: 0.8234

Prediction statistics:
  Total predictions: 9144
  Correct predictions: 7521
  Accuracy (from predictions): 0.8226
```

② **Confusion Matrix 계산 및 시각화**

```python
# Confusion Matrix 계산
cm = confusion_matrix(val_true_labels, val_pred_labels)

# 클래스 이름 (YNAT: 뉴스 카테고리)
class_names = [
    "정치", "경제", "사회", "생활", "문화",
    "세계", "IT과학", "스포츠", "연예"
]  # 0~8

print("Confusion Matrix:")
print(cm)
print(f"\nShape: {cm.shape} (9x9)")

# Confusion Matrix 시각화
fig, ax = plt.subplots(figsize=(10, 9))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=class_names,
    yticklabels=class_names,
    cbar_kws={'label': 'Count'},
    ax=ax,
    linewidths=0.5,
)
ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
ax.set_title('Confusion Matrix: YNAT News Category Classification', fontsize=13, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('practice/chapter9/data/output/confusion_matrix_ko.png', dpi=150, bbox_inches='tight')
print("\nSaved: confusion_matrix_ko.png")
plt.show()
```

예상 행렬 (일부):
```
Confusion Matrix:
[[803  15   2   1   4  ...,  2]
 [ 10 791  18   5   2  ...,  3]
 [  3  12 812   9   6  ...,  1]
 ...
 [ 2   5   8  798  15  ...,  4]]

Shape: (9, 9)
```

③ **클래스별 Precision, Recall, F1 계산**

```python
# sklearn으로 클래스별 메트릭 계산
precision, recall, f1, support = precision_recall_fscore_support(
    val_true_labels,
    val_pred_labels,
    average=None,
)

# 결과를 데이터프레임으로 정리
results_df = pd.DataFrame({
    'Class': class_names,
    'Precision': precision,
    'Recall': recall,
    'F1-Score': f1,
    'Support': support,
})

print("\nPer-Class Performance:")
print(results_df.to_string(index=False))

print("\nMacro-Averaged Metrics (Unweighted):")
print(f"  Precision: {np.mean(precision):.4f}")
print(f"  Recall: {np.mean(recall):.4f}")
print(f"  F1-Score: {np.mean(f1):.4f}")

print("\nWeighted-Averaged Metrics (Class-Weighted):")
weighted_precision = np.average(precision, weights=support)
weighted_recall = np.average(recall, weights=support)
weighted_f1 = np.average(f1, weights=support)
print(f"  Precision: {weighted_precision:.4f}")
print(f"  Recall: {weighted_recall:.4f}")
print(f"  F1-Score: {weighted_f1:.4f}")
```

예상 결과:
```
Per-Class Performance:
     Class  Precision    Recall  F1-Score  Support
      정치      0.8342    0.8206    0.8274      980
      경제      0.8124    0.8067    0.8095      980
      사회      0.8356    0.8286    0.8321      980
      ...
      연예      0.8015    0.8122    0.8068      980

Macro-Averaged Metrics:
  Precision: 0.8213
  Recall: 0.8226
  F1-Score: 0.8218

Weighted-Averaged Metrics:
  Precision: 0.8213
  Recall: 0.8226
  F1-Score: 0.8218
```

④ **오류 분석: 어려운 예측 사례**

```python
# 모델의 예측 확률
pred_probs = torch.softmax(torch.from_numpy(val_predictions.predictions), dim=1).numpy()

# 신뢰도가 가장 낮은 예측 (틀린 것 중)
incorrect_mask = val_pred_labels != val_true_labels
incorrect_indices = np.where(incorrect_mask)[0]

if len(incorrect_indices) > 0:
    incorrect_probs = pred_probs[incorrect_indices]
    max_probs = np.max(incorrect_probs, axis=1)

    # 신뢰도가 가장 낮은 상위 5개 오류
    low_confidence_idx = np.argsort(max_probs)[:5]

    print("\nHard Examples (Low Confidence Errors):")
    print(f"Total incorrect predictions: {len(incorrect_indices)}")
    print(f"Error rate: {len(incorrect_indices) / len(val_true_labels) * 100:.2f}%")

    for i, idx_in_incorrect in enumerate(low_confidence_idx, 1):
        actual_idx = incorrect_indices[idx_in_incorrect]
        true_label = val_true_labels[actual_idx]
        pred_label = val_pred_labels[actual_idx]
        confidence = max_probs[idx_in_incorrect]

        print(f"\n  Error {i}:")
        print(f"    True: {class_names[true_label]}")
        print(f"    Pred: {class_names[pred_label]}")
        print(f"    Confidence: {confidence:.4f}")
```

예상 결과:
```
Hard Examples (Low Confidence Errors):
Total incorrect predictions: 1623
Error rate: 17.75%

  Error 1:
    True: 정치
    Pred: 사회
    Confidence: 0.4521

  Error 2:
    True: 경제
    Pred: 세계
    Confidence: 0.4834

  ...
```

⑤ **과적합 분석 (Train vs Val 비교)**

```python
# 훈련 세트에서도 예측 수집
train_predictions = trainer.predict(processed_train)
train_pred_labels = np.argmax(train_predictions.predictions, axis=1)
train_true_labels = np.array(processed_train['label'])

train_accuracy = np.sum(train_pred_labels == train_true_labels) / len(train_true_labels)
val_accuracy = np.sum(val_pred_labels == val_true_labels) / len(val_true_labels)

print("\nOverfitting Analysis:")
print(f"Train Accuracy: {train_accuracy:.4f}")
print(f"Val Accuracy: {val_accuracy:.4f}")
print(f"Gap (Overfitting indicator): {train_accuracy - val_accuracy:.4f}")

if train_accuracy - val_accuracy < 0.05:
    print("  → 과적합 신호 없음. 좋은 일반화 성능을 보임.")
elif train_accuracy - val_accuracy < 0.15:
    print("  → 경미한 과적합. 허용 범위.")
else:
    print("  → 심각한 과적합. Early Stopping 또는 정규화 강화 필요.")
```

예상 결과:
```
Overfitting Analysis:
Train Accuracy: 0.9234
Val Accuracy: 0.8226
Gap (Overfitting indicator): 0.1008

  → 경미한 과적합. 허용 범위.
```

**검증 체크리스트**:
- [ ] Confusion Matrix가 올바르게 계산되었는가? (9×9 행렬)
- [ ] 클래스별 Precision/Recall이 계산되었는가?
- [ ] 모든 클래스의 성능이 균등한가? 어려운 클래스가 있는가?
- [ ] 과적합 신호가 있는가? (Train vs Val 정확도 비교)
- [ ] 시각화가 저장되었는가?

**Copilot 프롬프트 5**:
```
"검증 세트에서 모델의 예측을 수집하고 Confusion Matrix를 만드는 코드를 작성해줄래?
sklearn의 confusion_matrix를 사용하고 seaborn으로 히트맵으로 시각화해줘."
```

**Copilot 프롬프트 6**:
```
"클래스별로 Precision, Recall, F1 점수를 계산하고 출력하는 코드를 작성해줄래?
sklearn의 precision_recall_fscore_support를 사용하면 돼."
```

**선택 프롬프트**:
```
"모델이 틀린 예측 중 신뢰도가 낮은 사례들을 찾아서 True Label과 Predicted Label을
비교하는 코드를 작성해줄까?"
```

---

### 제출 안내 (Google Classroom)

**제출 방법**:
- Google Classroom의 "9주차 B회차" 과제에 조별 1부 제출
- 파일명 형식: `group_{조번호}_ch9B.zip`

**포함할 파일**:
```
group_{조번호}_ch9B/
├── ch9B_finetuning.py           # 전체 구현 코드
├── learning_curves_ko.png       # 학습 곡선 (Loss와 Accuracy)
├── confusion_matrix_ko.png      # Confusion Matrix 히트맵
├── requirements.txt             # 필요한 라이브러리 (transformers, datasets, torch, sklearn, 등)
└── analysis_report.md           # 분석 리포트 (1-2페이지)
```

**리포트 포함 항목** (analysis_report.md):
- 각 체크포인트의 구현 과정 및 어려웠던 점 (3-4문장)
- 학습 곡선 해석: "언제부터 과적합 신호가 나타났는가? Early Stopping이 몇 스텝에서 트리거되었는가?" (3-4문장)
- Confusion Matrix 분석: "어떤 클래스 쌍이 자주 혼동되는가? 왜 그럴까?" (3-4문장)
- 클래스별 성능 차이: "모든 클래스의 성능이 비슷한가? 어려운 클래스가 있는가?" (2-3문장)
- 과적합 분석: "Train과 Val 정확도의 격차가 어느 정도인가? 수용 가능한가?" (2문장)
- Copilot 활용 경험: "어떤 프롬프트가 가장 효과적이었는가? 생성된 코드에서 수정한 부분은?" (2-3문장)

**제출 마감**: 수업 종료 후 24시간 이내

---

### 결과 토론 가이드

> **토론 가이드**: 조별 파인튜닝 결과를 공유하며, 다른 조의 성능과 비교하고, 하이퍼파라미터 선택과 과적합 해결 전략을 함께 검토한다

**토론 주제**:

① **파인튜닝 결과의 편차**
- 모든 조가 유사한 검증 정확도(82~84%)를 달성했는가?
- 정확도가 다르다면, 어떤 요인이 영향을 미쳤을까? (학습률, 에포크, 배치 크기 등)
- 같은 모델, 같은 데이터인데 왜 조마다 다른 결과가 나올까?

② **과적합 신호의 포착**
- 각 조의 학습 곡선에서 과적합이 시작된 시점은 언제였는가?
- Train Loss와 Val Loss의 격차가 크던가? (크다면 과적합 위험)
- Early Stopping이 몇 스텝에서 트리거되었는가?

③ **Confusion Matrix 해석**
- 어떤 클래스 쌍이 자주 혼동되었는가? 예: "정치" vs "세계" 같은 정치 소식
- 특정 클래스의 Recall이 낮다면, 왜 그럴까? (학습 데이터 부족? 모호한 정의?)
- 모든 클래스가 비슷한 성능을 보이는가, 아니면 편차가 있는가?

④ **학습률과 Early Stopping의 역할**
- 학습률 2e-5가 적절했는가? (더 크면? 더 작으면?)
- Early Stopping이 과적합을 효과적으로 방지했는가?
- Warmup 스텝은 어떤 역할을 했는가?

⑤ **Full Fine-tuning의 실감**
- 110M 파라미터를 모두 업데이트하는 것이 얼마나 느렸는가? (학습 시간 체감)
- 이렇게 느리면, 실무에서는 어떻게 할까? (→ 10주차 PEFT의 필요성)
- 더 큰 모델(BERT-Large, RoBERTa)이면 어떨까?

**발표 형식**:
- 각 조 3~5분 발표 (구현 전략 + 주요 결과 + 어려움)
- 다른 조의 질문에 답변 (2~3개 질문)
- 교수의 보충 설명 및 피드백

---

### 다음 주 예고

다음 주 10주차 A회차에서는 **PEFT(Parameter-Efficient Fine-Tuning): LoRA**를 배운다.

**예고 내용**:
- Full Fine-tuning의 한계: 110M 파라미터 × 4바이트 = 440MB 메모리, 학습 시간 수시간
- LoRA의 아이디어: "모델 가중치는 고정하고, 작은 적응 행렬(Rank Decomposition)만 학습한다"
- 수학적 원리: W_new = W_old + ΔW, 여기서 ΔW = A·B (A: d_in×r, B: r×d_out, r은 랭크)
- 메모리 절약: 110M → 0.1M (1000배 감소), 학습 시간: 수시간 → 수분
- 코드 구현: `peft` 라이브러리로 Trainer에 통합하는 방법
- B회차 실습: LoRA로 같은 감성 분류 데이터를 파인튜닝하고, Full Fine-tuning과 성능 비교

**사전 준비**:
- 9주차 B회차의 Full Fine-tuning 결과 (특히 정확도, 학습 시간)를 기록해두기
- 행렬 분해(Matrix Factorization) 개념 복습
- GPU 메모리 제약이 있다면, Colab Free Tier에서의 제약 경험해보기

---

## 참고 자료

**실습 코드**:
- _전체 구현 코드는 practice/chapter9/code/9-1-bert-finetuning.py 참고_
- _데이터 전처리 및 Trainer 설정은 practice/chapter9/code/9-2-trainer-setup.py 참고_

**권장 읽기**:
- Hugging Face Course. Fine-tuning a Pretrained Model. https://huggingface.co/course/en/chapter3/1
- Devlin, J. et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. *arXiv*. https://arxiv.org/abs/1810.04805
- Howard, J. & Ruder, S. (2018). Universal Language Model Fine-tuning for Text Classification. *ACL*. https://arxiv.org/abs/1801.06146
- Lin, T. Y. et al. (2017). Focal Loss for Dense Object Detection. *ICCV*. https://arxiv.org/abs/1708.02002

---

## 주요 라이브러리 버전

```
torch>=1.9.0
transformers>=4.30.0
datasets>=2.0.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
seaborn>=0.12.0
```
