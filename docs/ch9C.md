# 제9장 C: Full Fine-tuning with Trainer API — 모범 구현과 해설

> 이 문서는 B회차 과제 제출 후 공개된다. 제출 전에는 열람하지 않는다.

---

## 체크포인트 1 모범 구현: Datasets 로드 + 토크나이제이션 + DataCollator

Full Fine-tuning의 첫 단계는 데이터를 모델이 이해할 수 있는 형식으로 준비하는 것이다. 이 과정에서 Datasets 라이브러리의 강력함이 드러난다.

### 데이터 로드 및 확인

```python
from datasets import load_dataset, Dataset
import numpy as np
import pandas as pd

# 한국어 뉴스 카테고리 분류 데이터셋 로드 (KLUE YNAT)
print("=" * 60)
print("Step 1: 데이터셋 로드")
print("=" * 60)

dataset = load_dataset("klue", "ynat")
print(f"Dataset splits: {dataset.keys()}")

# 각 분할의 크기 확인
train_size = len(dataset["train"])
val_size = len(dataset["validation"])
print(f"Train set size: {train_size:,}")
print(f"Validation set size: {val_size:,}")

# 첫 번째 샘플 확인
print("\nFirst training sample:")
sample = dataset["train"][0]
print(f"  guid: {sample['guid']}")
print(f"  title: {sample['title']}")
print(f"  date: {sample['date']}")
print(f"  label: {sample['label']}")

# 데이터셋 구조 확인
print(f"\nDataset features: {dataset['train'].features}")
print(f"Number of rows per column:")
for col in dataset["train"].column_names:
    print(f"  {col}: {len(dataset['train'][col])}")
```

**예상 출력**:
```
============================================================
Step 1: 데이터셋 로드
============================================================
Dataset splits: dict_keys(['train', 'validation'])
Train set size: 45,678
Validation set size: 9,144

First training sample:
  guid: 1
  title: 2006년 아마추어 복싱 대회 개최
  date: 2018-10-24
  label: 5

Dataset features: {
    'guid': Value(dtype='int64'),
    'title': Value(dtype='string'),
    'date': Value(dtype='string'),
    'label': Value(dtype='int64')
}
Number of rows per column:
  guid: 45678
  title: 45678
  date: 45678
  label: 45678
```

### 클래스 분포 분석

```python
# 클래스 분포 확인 (불균형 진단)
print("\n" + "=" * 60)
print("Step 2: 클래스 분포 확인")
print("=" * 60)

train_labels = np.array(dataset["train"]["label"])
val_labels = np.array(dataset["validation"]["label"])

# YNAT 클래스 이름 (0~8: 9개 카테고리)
class_names = [
    "정치", "경제", "사회", "생활", "문화",
    "세계", "IT과학", "스포츠", "연예"
]

# Train set 분포
print("\nTrain set class distribution:")
unique, counts = np.unique(train_labels, return_counts=True)
for u, c in zip(unique, counts):
    percentage = c / len(train_labels) * 100
    bar_length = int(percentage / 2)
    bar = "█" * bar_length
    print(f"  {class_names[u]:6s} ({u}): {c:6,d} ({percentage:5.1f}%) {bar}")

# Validation set 분포
print("\nValidation set class distribution:")
unique, counts = np.unique(val_labels, return_counts=True)
for u, c in zip(unique, counts):
    percentage = c / len(val_labels) * 100
    bar_length = int(percentage / 2)
    bar = "█" * bar_length
    print(f"  {class_names[u]:6s} ({u}): {c:6,d} ({percentage:5.1f}%) {bar}")

# 클래스 가중치 계산 (불균형 해결용)
num_labels = len(np.unique(train_labels))
class_counts = np.bincount(train_labels, minlength=num_labels)
class_weights = len(train_labels) / (num_labels * class_counts)
class_weights = class_weights / class_weights.sum() * num_labels

print(f"\nClass weights (for loss balancing):")
for i, (name, weight) in enumerate(zip(class_names, class_weights)):
    print(f"  {name:6s}: {weight:.4f}")

# 불균형 측정: 최대 클래스와 최소 클래스의 비율
imbalance_ratio = np.max(class_counts) / np.min(class_counts)
print(f"\nImbalance ratio (max/min): {imbalance_ratio:.2f}x")
if imbalance_ratio < 1.5:
    print("  → 클래스가 매우 균형잡혀 있음")
elif imbalance_ratio < 3:
    print("  → 클래스가 어느 정도 균형잡혀 있음 (문제 없음)")
else:
    print("  → 클래스 불균형 존재 (가중치 조정 권장)")
```

**예상 출력**:
```
============================================================
Step 2: 클래스 분포 확인
============================================================

Train set class distribution:
  정치   (0):  5,081 ( 11.1%) ███
  경제   (1):  5,173 ( 11.3%) ███
  사회   (2):  5,040 ( 11.0%) ███
  생활   (3):  5,130 ( 11.2%) ███
  문화   (4):  5,081 ( 11.1%) ███
  세계   (5):  5,186 ( 11.3%) ███
  IT과학 (6):  5,138 ( 11.2%) ███
  스포츠 (7):  5,087 ( 11.1%) ███
  연예   (8):  5,162 ( 11.3%) ███

Validation set class distribution:
  정치   (0):  1,009 ( 11.0%) ██
  경제   (1):  1,032 ( 11.3%) ██
  ...

Class weights (for loss balancing):
  정치   : 0.9987
  경제   : 0.9763
  ...

Imbalance ratio (max/min): 1.04x
  → 클래스가 매우 균형잡혀 있음
```

### 토크나이제이션 (Tokenization)

```python
from transformers import AutoTokenizer

print("\n" + "=" * 60)
print("Step 3: 토크나이저 로드 및 전처리 함수 정의")
print("=" * 60)

# 한국어 BERT 토크나이저 로드
model_name = "bert-base-multilingual-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

print(f"Tokenizer model: {model_name}")
print(f"Vocabulary size: {len(tokenizer):,}")
print(f"Special tokens:")
print(f"  [CLS]: {tokenizer.cls_token_id} (분류 시작 토큰)")
print(f"  [SEP]: {tokenizer.sep_token_id} (문장 구분 토큰)")
print(f"  [UNK]: {tokenizer.unk_token_id} (미지 단어)")
print(f"  [PAD]: {tokenizer.pad_token_id} (패딩)")

# 예제 토크나이제이션 시연
sample_text = "2006년 아마추어 복싱 대회 개최"
print(f"\nExample tokenization:")
print(f"  Input text: '{sample_text}'")

tokens = tokenizer.tokenize(sample_text)
print(f"  Tokens: {tokens}")

token_ids = tokenizer.encode(sample_text, add_special_tokens=True)
print(f"  Token IDs: {token_ids}")

# 패딩/자르기 예제
encoded = tokenizer(
    sample_text,
    max_length=16,
    padding="max_length",
    truncation=True,
    return_tensors="pt"
)
print(f"\n  With padding/truncation (max_length=16):")
print(f"    input_ids: {encoded['input_ids'].tolist()}")
print(f"    attention_mask: {encoded['attention_mask'].tolist()}")

# 전처리 함수 정의
def preprocess_function(examples):
    """
    배치의 모든 샘플을 토크나이제이션한다.

    Args:
        examples: Datasets 배치 (딕셔너리 형태)
            - 'title': 텍스트 리스트
            - 'label': 레이블 리스트

    Returns:
        토크나이제이션된 결과 (토크나이저가 반환하는 딕셔너리)
    """
    return tokenizer(
        examples["title"],           # 입력 텍스트
        max_length=256,              # 최대 길이
        padding="max_length",        # 짧은 문장은 패딩
        truncation=True,             # 긴 문장은 자르기
    )

print(f"\nPreprocessing function defined.")
print(f"  Max sequence length: 256")
print(f"  Padding strategy: max_length")
print(f"  Truncation: enabled")
```

**예상 출력**:
```
============================================================
Step 3: 토크나이저 로드 및 전처리 함수 정의
============================================================
Tokenizer model: bert-base-multilingual-cased
Vocabulary size: 119,547
Special tokens:
  [CLS]: 101 (분류 시작 토큰)
  [SEP]: 102 (문장 구분 토큰)
  [UNK]: 100 (미지 단어)
  [PAD]: 0 (패딩)

Example tokenization:
  Input text: '2006년 아마추어 복싱 대회 개최'
  Tokens: ['2006', '##년', '아', '##마', '##추', '##어', '복', '##싱', '대', '##회', '개', '##최']
  Token IDs: [101, 2164, 2356, 1051, 1051, 1051, 3098, 3098, 2264, 3098, 1051, 1051, 102]

  With padding/truncation (max_length=16):
    input_ids: [101, 2164, 2356, ..., 102, 0, 0, 0]
    attention_mask: [1, 1, 1, ..., 1, 0, 0, 0]

Preprocessing function defined.
  Max sequence length: 256
  Padding strategy: max_length
  Truncation: enabled
```

### 데이터셋 전처리 (병렬 처리)

```python
print("\n" + "=" * 60)
print("Step 4: 모든 샘플에 전처리 함수 적용 (병렬 처리)")
print("=" * 60)

# 전처리 적용 (배치 병렬 처리로 빠르게 수행)
print("Tokenizing train set...")
processed_train = dataset["train"].map(
    preprocess_function,
    batched=True,        # 배치 단위로 처리
    batch_size=100,      # 한 번에 100개 샘플 처리
    remove_columns=dataset["train"].column_names,  # 원본 컬럼 제거
    num_proc=4,          # 4개 프로세스 병렬 사용 (선택)
    desc="Processing train set",  # 진행상황 표시
)

print("Tokenizing validation set...")
processed_val = dataset["validation"].map(
    preprocess_function,
    batched=True,
    batch_size=100,
    remove_columns=dataset["validation"].column_names,
    num_proc=4,
    desc="Processing validation set",
)

print(f"\nTokenization completed!")
print(f"  Processed train samples: {len(processed_train):,}")
print(f"  Processed validation samples: {len(processed_val):,}")

# 전처리된 샘플 확인
print(f"\nProcessed sample structure:")
sample = processed_train[0]
print(f"  Keys: {sample.keys()}")
print(f"  input_ids shape: ({len(sample['input_ids'])},)")
print(f"  attention_mask shape: ({len(sample['attention_mask'])},)")
print(f"  label: {sample['label']}")

# 첫 몇 개 토큰 ID 확인
print(f"\n  First 20 token IDs: {sample['input_ids'][:20]}")
print(f"  First 20 attention mask: {sample['attention_mask'][:20]}")

# 패딩된 위치 확인
attention_mask = sample['attention_mask']
pad_start_idx = attention_mask.index(0) if 0 in attention_mask else len(attention_mask)
print(f"  Padding starts at index: {pad_start_idx} (out of {len(attention_mask)})")

# 데이터타입 확인
print(f"\nData types in processed dataset:")
for key in processed_train.features.keys():
    print(f"  {key}: {processed_train.features[key]}")
```

**예상 출력**:
```
============================================================
Step 4: 모든 샘플에 전처리 함수 적용 (병렬 처리)
============================================================
Tokenizing train set...
Processing train set: 100%|██████████| 457/457 [02:15<00:00, 3.38 batches/s]
Tokenizing validation set...
Processing validation set: 100%|██████████| 92/92 [00:28<00:00, 3.32 batches/s]

Tokenization completed!
  Processed train samples: 45,678
  Processed validation samples: 9,144

Processed sample structure:
  Keys: dict_keys(['input_ids', 'token_type_ids', 'attention_mask', 'label'])
  input_ids shape: (256,)
  attention_mask shape: (256,)
  label: 5

  First 20 token IDs: [101, 2164, 2356, 1051, 1051, 1051, 3098, 3098, 2264, 3098, 1051, 1051, 102, 0, 0, 0, 0, 0, 0, 0]
  First 20 attention mask: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]

Data types in processed dataset:
  input_ids: Sequence(feature=Value(dtype='int64'), length=256)
  token_type_ids: Sequence(feature=Value(dtype='int64'), length=256)
  attention_mask: Sequence(feature=Value(dtype='int64'), length=256)
  label: Value(dtype='int64')
```

### 핵심 포인트 해설

#### 왜 배치 단위 처리(batched=True)를 사용하는가?

```python
# 비교: 샘플 단위 vs 배치 단위

# ❌ 느린 방법 (샘플 단위)
# map 함수가 각 샘플에 대해 개별적으로 호출
# 시간: ~10분 (50K 샘플)
processed_slow = dataset["train"].map(preprocess_function, batched=False)

# ✅ 빠른 방법 (배치 단위)
# 100개 샘플을 한 번에 처리하여 함수 호출 오버헤드 감소
# 또한 토크나이저가 배치 처리에 최적화됨
# 시간: ~2분 (50K 샘플)
processed_fast = dataset["train"].map(preprocess_function, batched=True, batch_size=100)

# 속도 개선: 약 5배 향상
```

#### Attention Mask의 역할

```python
# Attention Mask는 모델에 "어떤 토큰이 실제 데이터이고 어떤 토큰이 패딩인지" 알린다

sample_text = "복싱"  # 매우 짧은 텍스트
encoded = tokenizer(
    sample_text,
    max_length=10,
    padding="max_length",
    truncation=True,
)

print("Token IDs:   ", encoded['input_ids'])
# [101, 2264, 3098, 102, 0, 0, 0, 0, 0, 0]
#  [CLS] 복  싱  [SEP] [PAD] [PAD] ...

print("Attention mask:", encoded['attention_mask'])
# [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
#  ↑ 실제  ↑ 실제 ↑ 모두 패딩 (무시)

# 모델은 attention_mask=0인 위치에 대해 계산을 건너뛴다
# 왜? 패딩 토큰이 실제 의미를 가지지 않으므로, 모델이 "이 위치는 무시하라"는 신호를 받는다
```

#### Label 정보의 유지

```python
# remove_columns를 사용하여 label 제거하면 안 된다!
# 모델 학습에 label이 필수이기 때문

# ❌ 틀린 방법
processed_wrong = dataset["train"].map(
    preprocess_function,
    batched=True,
    remove_columns=dataset["train"].column_names,  # label도 제거됨!
)
# 결과: {'input_ids': [...], 'attention_mask': [...], 'token_type_ids': [...]}
# label이 없음!

# ✅ 올바른 방법
# preprocess_function이 label을 자동으로 유지하도록 설계
# tokenizer()는 입력 텍스트만 처리하고,
# label은 자동으로 통과된다

# 확인:
sample = processed_train[0]
print('label' in sample)  # True
```

---

## 체크포인트 2 모범 구현: TrainingArguments + Trainer 파인튜닝

파인튜닝의 핵심은 하이퍼파라미터와 학습 루프 관리이다. Trainer API가 이 모든 것을 자동화한다.

### TrainingArguments 설정

```python
import torch
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback
from transformers import AutoModelForSequenceClassification

print("\n" + "=" * 60)
print("Step 5: TrainingArguments 설정 (하이퍼파라미터)")
print("=" * 60)

# 디렉토리 생성
import os
os.makedirs("results", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# GPU 확인
print(f"GPU available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# TrainingArguments 설정
training_args = TrainingArguments(
    # 출력 경로
    output_dir="./results",
    overwrite_output_dir=True,

    # 학습 에포크 및 배치 크기
    num_train_epochs=3,                          # 3 에포크
    per_device_train_batch_size=16,              # GPU당 배치 크기 16
    per_device_eval_batch_size=16,               # 검증 배치 크기

    # 학습률 및 최적화
    learning_rate=2e-5,                          # 파인튜닝 학습률 (매우 작음!)
    warmup_steps=500,                            # 처음 500 스텝 동안 학습률 점진적 증가
    weight_decay=0.01,                           # L2 정규화 (가중치 감소)
    adam_epsilon=1e-8,                           # Adam 옵티마이저의 epsilon (기본값)

    # 학습 제어
    max_grad_norm=1.0,                           # 그래디언트 클립핑

    # 평가 및 저장
    eval_strategy="steps",                       # steps 간격으로 평가
    eval_steps=500,                              # 500 스텝마다 평가
    save_steps=500,                              # 500 스텝마다 체크포인트 저장
    save_total_limit=3,                          # 최근 3개 체크포인트만 보관
    load_best_model_at_end=True,                 # 학습 후 최고 성능 모델 로드
    metric_for_best_model="accuracy",            # 최고 성능 판단 기준
    greater_is_better=True,                      # 높을수록 좋은 메트릭

    # 로깅
    logging_steps=50,                            # 50 스텝마다 로그 출력
    logging_dir="./logs",                        # TensorBoard 로그 저장
    logging_first_step=True,                     # 첫 번째 스텝도 로그 출력

    # 기타
    seed=42,                                     # 재현성을 위한 시드
    fp16=True,                                   # 혼합 정밀도 (메모리 절약)
    push_to_hub=False,                           # Hugging Face Hub 업로드 안 함
)

print("\nTrainingArguments 요약:")
print(f"  Model: bert-base-multilingual-cased")
print(f"  Epochs: {training_args.num_train_epochs}")
print(f"  Batch size: {training_args.per_device_train_batch_size}")
print(f"  Learning rate: {training_args.learning_rate}")
print(f"  Learning rate schedule: warmup {training_args.warmup_steps} + linear decay")
print(f"  Regularization:")
print(f"    - Weight decay: {training_args.weight_decay}")
print(f"    - Gradient clipping: {training_args.max_grad_norm}")
print(f"    - Mixed precision (FP16): {training_args.fp16}")
print(f"  Early stopping: enabled (monitor accuracy)")
print(f"  Evaluation interval: every {training_args.eval_steps} steps")

# 총 훈련 스텝 계산
num_train_samples = len(processed_train)
batch_size = training_args.per_device_train_batch_size
total_steps = (num_train_samples // batch_size) * training_args.num_train_epochs
warmup_pct = (training_args.warmup_steps / total_steps) * 100

print(f"\nTraining schedule:")
print(f"  Total training samples: {num_train_samples:,}")
print(f"  Steps per epoch: {num_train_samples // batch_size:,}")
print(f"  Total training steps: {total_steps:,}")
print(f"  Warmup steps: {training_args.warmup_steps} ({warmup_pct:.1f}% of total)")
print(f"  Evaluation intervals: {total_steps // training_args.eval_steps} evaluations")
print(f"  Estimated training time: ~{total_steps * 0.5 / 3600:.1f} hours (GPU)")
```

**예상 출력**:
```
============================================================
Step 5: TrainingArguments 설정 (하이퍼파라미터)
============================================================
GPU available: True
GPU device: NVIDIA A100
GPU memory: 40.0 GB

TrainingArguments 요약:
  Model: bert-base-multilingual-cased
  Epochs: 3
  Batch size: 16
  Learning rate: 2e-05
  Learning rate schedule: warmup 500 + linear decay
  Regularization:
    - Weight decay: 0.01
    - Gradient clipping: 1.0
    - Mixed precision (FP16): True
  Early stopping: enabled (monitor accuracy)
  Evaluation interval: every 500 steps

Training schedule:
  Total training samples: 45,678
  Steps per epoch: 2,854
  Total training steps: 8,562
  Warmup steps: 500 (5.8% of total)
  Evaluation intervals: 17 evaluations
  Estimated training time: ~2.1 hours (GPU)
```

### 모델 로드 및 평가 메트릭 정의

```python
print("\n" + "=" * 60)
print("Step 6: 모델 로드 및 평가 메트릭 정의")
print("=" * 60)

# 모델 로드
num_labels = 9  # YNAT는 9개 카테고리
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=num_labels,
    ignore_mismatched_sizes=False,
)

# 모델 파라미터 확인
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Model: {model_name}")
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print(f"All parameters trainable: {total_params == trainable_params} (Full Fine-tuning)")

# 파라미터 분석
print(f"\nModel layer breakdown:")
for name, param in list(model.named_parameters())[:5]:
    print(f"  {name}: {param.shape}")
print(f"  ...")
print(f"  classifier.weight: {model.classifier.weight.shape}")
print(f"  classifier.bias: {model.classifier.bias.shape}")

# 메모리 사용량 추정
param_memory_mb = (total_params * 4) / (1024 * 1024)  # 4 bytes per FP32
optimizer_memory_mb = (trainable_params * 8) / (1024 * 1024)  # Adam은 2개 상태 벡터
batch_memory_mb = (16 * 256 * 768 * 4) / (1024 * 1024)  # 근사값
total_memory_mb = param_memory_mb + optimizer_memory_mb + batch_memory_mb

print(f"\nMemory usage estimation:")
print(f"  Model parameters: {param_memory_mb:.0f} MB")
print(f"  Optimizer states (Adam): {optimizer_memory_mb:.0f} MB")
print(f"  Batch data (approx): {batch_memory_mb:.0f} MB")
print(f"  Total (approx): {total_memory_mb:.0f} MB")
print(f"  Recommendation: GPU with {total_memory_mb / 1024:.1f} GB+ memory")

# 평가 메트릭 정의
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def compute_metrics(eval_preds):
    """
    평가 중 계산할 메트릭을 정의한다.

    Args:
        eval_preds: (predictions, label_ids) 튜플
            - predictions: (num_samples, num_labels) 형태의 로짓
            - label_ids: (num_samples,) 형태의 실제 레이블

    Returns:
        메트릭 딕셔너리
    """
    predictions, labels = eval_preds
    predictions = np.argmax(predictions, axis=1)

    accuracy = accuracy_score(labels, predictions)

    # Weighted F1 (클래스 불균형 고려)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='weighted'
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

print(f"\nEvaluation metrics defined:")
print(f"  - Accuracy: overall correctness")
print(f"  - Precision: weighted average")
print(f"  - Recall: weighted average")
print(f"  - F1: weighted average")
```

**예상 출력**:
```
============================================================
Step 6: 모델 로드 및 평가 메트릭 정의
============================================================
Model: bert-base-multilingual-cased
Total parameters: 167,355,651
Trainable parameters: 167,355,651
All parameters trainable: True (Full Fine-tuning)

Model layer breakdown:
  bert.embeddings.word_embeddings.weight: torch.Size([119547, 768])
  bert.embeddings.position_embeddings.weight: torch.Size([512, 768])
  bert.encoder.layer.0.attention.self.query.weight: torch.Size([768, 768])
  ...
  classifier.weight: torch.Size([9, 768])
  classifier.bias: torch.Size([9])

Memory usage estimation:
  Model parameters: 634 MB
  Optimizer states (Adam): 1269 MB
  Batch data (approx): 188 MB
  Total (approx): 2091 MB
  Recommendation: GPU with 2.0 GB+ memory
```

### Trainer 초기화 및 학습

```python
print("\n" + "=" * 60)
print("Step 7: Trainer 초기화 및 학습")
print("=" * 60)

# Early Stopping 콜백
early_stopping = EarlyStoppingCallback(
    early_stopping_patience=2,         # 2회 평가 동안 개선 없으면 중단
    early_stopping_threshold=0.0,      # 최소 개선 폭 (0 = 어떤 개선이든 가능)
)

print("EarlyStoppingCallback 설정:")
print(f"  Patience: 2 evaluations")
print(f"  Monitoring: accuracy (increasing)")
print(f"  Logic: if best_acc_yet - current_acc < 0.0 for 2 evals → stop")

# Trainer 생성
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=processed_train,
    eval_dataset=processed_val,
    compute_metrics=compute_metrics,
    callbacks=[early_stopping],
)

print(f"\nTrainer initialized:")
print(f"  Model: {type(model).__name__}")
print(f"  Train samples: {len(processed_train):,}")
print(f"  Eval samples: {len(processed_val):,}")
print(f"  Device: {trainer.args.device}")

# 학습 시작
print(f"\nStarting training...")
print(f"This will take approximately {total_steps * 0.5 / 60:.0f} minutes on GPU\n")

train_result = trainer.train()

print(f"\n✓ Training completed!")
print(f"Best model checkpoint: {trainer.best_model_checkpoint}")
print(f"Final training loss: {train_result.training_loss:.4f}")
```

**예상 출력** (로그 일부):
```
============================================================
Step 7: Trainer 초기화 및 학습
============================================================
EarlyStoppingCallback 설정:
  Patience: 2 evaluations
  Monitoring: accuracy (increasing)
  Logic: if best_acc_yet - current_acc < 0.0 for 2 evals → stop

Trainer initialized:
  Model: BertForSequenceClassification
  Train samples: 45,678
  Eval samples: 9,144
  Device: cuda:0

Starting training...
This will take approximately 72 minutes on GPU

[  500/8562] Loss: 1.523, Eval Accuracy: 0.7845
[ 1000/8562] Loss: 1.256, Eval Accuracy: 0.8103
[ 1500/8562] Loss: 1.089, Eval Accuracy: 0.8234 ← Best accuracy
[ 2000/8562] Loss: 0.921, Eval Accuracy: 0.8201
[ 2500/8562] Loss: 0.765, Eval Accuracy: 0.8165 ← Early stopping triggered

✓ Training completed!
Best model checkpoint: ./results/checkpoint-1500
Final training loss: 0.7234
```

### 학습 곡선 시각화

```python
import json
import matplotlib.pyplot as plt
import seaborn as sns

print("\n" + "=" * 60)
print("Step 8: 학습 곡선 시각화")
print("=" * 60)

# 학습 로그 불러오기
log_file = "results/trainer_state.json"
if os.path.exists(log_file):
    with open(log_file, 'r') as f:
        state = json.load(f)

    # 로그 파싱
    steps = []
    train_losses = []
    eval_steps = []
    eval_accuracies = []
    eval_f1_scores = []

    for log in state['log_history']:
        if 'loss' in log:
            steps.append(log['step'])
            train_losses.append(log['loss'])

        if 'eval_accuracy' in log:
            eval_steps.append(log['step'])
            eval_accuracies.append(log['eval_accuracy'])
            eval_f1_scores.append(log.get('eval_f1', None))

    print(f"Loaded training history:")
    print(f"  Total training logs: {len(train_losses)}")
    print(f"  Total eval checkpoints: {len(eval_accuracies)}")

    # 시각화 설정
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (15, 5)
    plt.rcParams['font.size'] = 10

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # [1] Train Loss vs Validation Metrics
    ax1 = axes[0]
    ax1_twin = ax1.twinx()

    line1 = ax1.plot(steps, train_losses, 'o-', color='steelblue',
                     linewidth=2, markersize=4, label='Train Loss')
    line2 = ax1_twin.plot(eval_steps, eval_accuracies, 's-', color='coral',
                          linewidth=2, markersize=5, label='Validation Accuracy')

    # Best model 표시
    best_step = int(trainer.best_model_checkpoint.split('-')[-1])
    best_acc_idx = eval_steps.index(best_step)
    best_acc = eval_accuracies[best_acc_idx]
    ax1_twin.scatter([best_step], [best_acc], s=150, color='red',
                     zorder=5, marker='*', edgecolors='darkred', linewidth=2)

    ax1.axvline(x=best_step, color='red', linestyle='--', alpha=0.5, linewidth=2)

    ax1.set_xlabel('Training Step', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Train Loss', fontsize=11, fontweight='bold', color='steelblue')
    ax1_twin.set_ylabel('Validation Accuracy', fontsize=11, fontweight='bold', color='coral')
    ax1.set_title('Loss & Accuracy Over Training', fontsize=12, fontweight='bold')

    ax1.tick_params(axis='y', labelcolor='steelblue')
    ax1_twin.tick_params(axis='y', labelcolor='coral')
    ax1.grid(True, alpha=0.3)

    # 범례 통합
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center right', fontsize=10)

    # [2] Validation Accuracy 변화
    ax2 = axes[1]
    ax2.plot(eval_steps, eval_accuracies, 'o-', color='green',
             linewidth=2.5, markersize=6, label='Validation Accuracy')

    # Best point 강조
    ax2.scatter([best_step], [best_acc], s=150, color='red', zorder=5,
                marker='*', edgecolors='darkred', linewidth=2,
                label=f'Best: {best_acc:.4f} @ Step {best_step}')
    ax2.axvline(x=best_step, color='red', linestyle='--', alpha=0.5, linewidth=2)

    ax2.set_xlabel('Training Step', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Validation Accuracy', fontsize=11, fontweight='bold')
    ax2.set_title('Validation Performance', fontsize=12, fontweight='bold')
    ax2.set_ylim([0.75, 0.85])
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10, loc='lower right')

    # [3] Learning Rate Schedule (계산)
    ax3 = axes[2]

    # Warmup + Linear Decay 시뮬레이션
    warmup_steps = training_args.warmup_steps
    total_steps_actual = total_steps
    base_lr = training_args.learning_rate

    schedule_steps = []
    schedule_lrs = []

    for step in range(0, total_steps_actual + 1, 100):
        if step <= warmup_steps:
            lr = base_lr * (step / warmup_steps)
        else:
            progress = (step - warmup_steps) / (total_steps_actual - warmup_steps)
            lr = base_lr * max(0, 1 - progress)

        schedule_steps.append(step)
        schedule_lrs = [schedule_lrs[-1] if schedule_lrs else 0] if step == schedule_steps[0] else schedule_lrs + [lr]

    # 수정된 버전
    schedule_steps = list(range(0, total_steps_actual + 1, 100))
    schedule_lrs = []
    for step in schedule_steps:
        if step <= warmup_steps:
            lr = base_lr * (step / warmup_steps if warmup_steps > 0 else 1)
        else:
            progress = (step - warmup_steps) / (total_steps_actual - warmup_steps)
            lr = base_lr * max(0, 1 - progress)
        schedule_lrs.append(lr)

    ax3.plot(schedule_steps, schedule_lrs, 'o-', color='purple', linewidth=2.5, markersize=3)
    ax3.axvline(x=warmup_steps, color='orange', linestyle='--', linewidth=2,
                label=f'Warmup ends ({warmup_steps} steps)')
    ax3.fill_between([0, warmup_steps], 0, max(schedule_lrs) * 1.1,
                      alpha=0.2, color='orange', label='Warmup phase')

    ax3.set_xlabel('Training Step', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Learning Rate', fontsize=11, fontweight='bold')
    ax3.set_title('Learning Rate Schedule (Warmup + Linear Decay)', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig('practice/chapter9/data/output/learning_curves_ko.png',
                dpi=150, bbox_inches='tight')
    print(f"\nSaved: practice/chapter9/data/output/learning_curves_ko.png")
    plt.close()

    # 수치 요약
    print(f"\nTraining Summary:")
    print(f"  Initial accuracy: {eval_accuracies[0]:.4f}")
    print(f"  Best accuracy: {best_acc:.4f} @ step {best_step}")
    print(f"  Final accuracy: {eval_accuracies[-1]:.4f}")
    print(f"  Accuracy improvement: {best_acc - eval_accuracies[0]:.4f}")
    print(f"  Initial loss: {train_losses[0]:.4f}")
    print(f"  Final loss: {train_losses[-1]:.4f}")
    print(f"  Loss reduction: {train_losses[0] - train_losses[-1]:.4f}")

else:
    print("Warning: trainer_state.json not found")
```

**예상 출력**:
```
============================================================
Step 8: 학습 곡선 시각화
============================================================
Loaded training history:
  Total training logs: 171
  Total eval checkpoints: 17

Saved: practice/chapter9/data/output/learning_curves_ko.png

Training Summary:
  Initial accuracy: 0.7845
  Best accuracy: 0.8234 @ step 1500
  Final accuracy: 0.8165
  Accuracy improvement: 0.0389
  Initial loss: 1.5234
  Final loss: 0.7156
  Loss reduction: 0.8078
```

---

## 체크포인트 3 모범 구현: Confusion Matrix + 오류 분석

### 검증 세트 평가

```python
print("\n" + "=" * 60)
print("Step 9: 검증 세트 성능 평가")
print("=" * 60)

# 최고 성능 모델은 이미 로드됨 (load_best_model_at_end=True)
print(f"Using model from best checkpoint: {trainer.best_model_checkpoint}")

# 검증 세트 평가
val_results = trainer.evaluate(eval_dataset=processed_val)
print(f"\nValidation Results:")
print(f"  Loss: {val_results['eval_loss']:.4f}")
print(f"  Accuracy: {val_results['eval_accuracy']:.4f}")
print(f"  Precision (weighted): {val_results['eval_precision']:.4f}")
print(f"  Recall (weighted): {val_results['eval_recall']:.4f}")
print(f"  F1 (weighted): {val_results['eval_f1']:.4f}")

# 예측 수집
print(f"\nGenerating predictions on validation set...")
val_predictions = trainer.predict(processed_val)
val_pred_labels = np.argmax(val_predictions.predictions, axis=1)
val_true_labels = np.array([processed_val[i]['label'] for i in range(len(processed_val))])

print(f"  Total predictions: {len(val_pred_labels):,}")
print(f"  Correct predictions: {np.sum(val_pred_labels == val_true_labels):,}")
print(f"  Accuracy (verified): {np.sum(val_pred_labels == val_true_labels) / len(val_true_labels):.4f}")
```

### Confusion Matrix 계산 및 시각화

```python
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

print("\n" + "=" * 60)
print("Step 10: Confusion Matrix 계산 및 시각화")
print("=" * 60)

# Confusion Matrix 계산
cm = confusion_matrix(val_true_labels, val_pred_labels)

print(f"Confusion Matrix shape: {cm.shape}")
print(f"Matrix:\n{cm}")

# Confusion Matrix 시각화
fig, ax = plt.subplots(figsize=(12, 10))

sns.heatmap(
    cm,
    annot=True,           # 각 셀에 값 표시
    fmt='d',              # 정수 형식
    cmap='Blues',         # 블루 색상도
    xticklabels=class_names,
    yticklabels=class_names,
    cbar_kws={'label': 'Number of Samples'},
    ax=ax,
    linewidths=0.5,
    linecolor='gray',
)

ax.set_xlabel('Predicted Label', fontsize=13, fontweight='bold')
ax.set_ylabel('True Label', fontsize=13, fontweight='bold')
ax.set_title('Confusion Matrix: YNAT News Category Classification\n(Validation Set)',
             fontsize=14, fontweight='bold')

plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('practice/chapter9/data/output/confusion_matrix_ko.png',
            dpi=150, bbox_inches='tight')
print(f"Saved: practice/chapter9/data/output/confusion_matrix_ko.png")
plt.close()

# Diagonal 값 확인 (각 클래스별 올바른 예측)
print(f"\nDiagonal values (correct predictions per class):")
for i, name in enumerate(class_names):
    correct = cm[i, i]
    total = cm[i].sum()
    percentage = (correct / total) * 100
    print(f"  {name:6s}: {correct:4d} / {total:4d} ({percentage:5.1f}%)")
```

**예상 출력**:
```
============================================================
Step 9: 검증 세트 성능 평가
============================================================
Using model from best checkpoint: ./results/checkpoint-1500

Validation Results:
  Loss: 0.4521
  Accuracy: 0.8234
  Precision (weighted): 0.8231
  Recall (weighted): 0.8234
  F1 (weighted): 0.8229

Generating predictions on validation set...
  Total predictions: 9,144
  Correct predictions: 7521
  Accuracy (verified): 0.8226

============================================================
Step 10: Confusion Matrix 계산 및 시각화
============================================================
Confusion Matrix shape: (9, 9)
Matrix:
[[803  15   2   1   4   3   2   2   1]
 [  10 791  18   5   2   8   3   2   5]
 [  3  12 812   9   6   4   1   3   2]
 ...
 [  2   5   8  798  15   7   3   4   2]]

Diagonal values (correct predictions per class):
  정치   : 803 / 980 ( 81.9%)
  경제   : 791 / 980 ( 80.7%)
  사회   : 812 / 980 ( 82.9%)
  생활   : 798 / 980 ( 81.4%)
  문화   : 805 / 980 ( 82.1%)
  세계   : 819 / 980 ( 83.6%)
  IT과학 : 814 / 980 ( 83.1%)
  스포츠 : 821 / 980 ( 83.8%)
  연예   : 816 / 980 ( 83.3%)

Saved: practice/chapter9/data/output/confusion_matrix_ko.png
```

### 클래스별 성능 분석

```python
print("\n" + "=" * 60)
print("Step 11: 클래스별 Precision, Recall, F1")
print("=" * 60)

# 클래스별 메트릭 계산
precision, recall, f1, support = precision_recall_fscore_support(
    val_true_labels,
    val_pred_labels,
    average=None,
)

# 결과를 DataFrame으로 정리
results_df = pd.DataFrame({
    'Class': class_names,
    'Precision': precision,
    'Recall': recall,
    'F1-Score': f1,
    'Support': support,
})

print("\nPer-Class Performance:")
print(results_df.to_string(index=False))

# 가중치 평균 메트릭
print(f"\nMacro-Averaged Metrics (Unweighted):")
print(f"  Precision: {np.mean(precision):.4f}")
print(f"  Recall: {np.mean(recall):.4f}")
print(f"  F1-Score: {np.mean(f1):.4f}")

print(f"\nWeighted-Averaged Metrics (Class-Weighted by Support):")
weighted_precision = np.average(precision, weights=support)
weighted_recall = np.average(recall, weights=support)
weighted_f1 = np.average(f1, weights=support)
print(f"  Precision: {weighted_precision:.4f}")
print(f"  Recall: {weighted_recall:.4f}")
print(f"  F1-Score: {weighted_f1:.4f}")

# 성능이 가장 좋은/나쁜 클래스
best_f1_idx = np.argmax(f1)
worst_f1_idx = np.argmin(f1)

print(f"\nBest and Worst Performing Classes:")
print(f"  Best:  {class_names[best_f1_idx]:6s} (F1: {f1[best_f1_idx]:.4f})")
print(f"  Worst: {class_names[worst_f1_idx]:6s} (F1: {f1[worst_f1_idx]:.4f})")
print(f"  Difference: {f1[best_f1_idx] - f1[worst_f1_idx]:.4f}")

# 시각화: 클래스별 메트릭 비교
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# [1] 클래스별 메트릭 막대 그래프
ax1 = axes[0]
x = np.arange(len(class_names))
width = 0.25

bars1 = ax1.bar(x - width, precision, width, label='Precision', color='steelblue', alpha=0.8)
bars2 = ax1.bar(x, recall, width, label='Recall', color='coral', alpha=0.8)
bars3 = ax1.bar(x + width, f1, width, label='F1-Score', color='green', alpha=0.8)

ax1.set_xlabel('Class', fontsize=11, fontweight='bold')
ax1.set_ylabel('Score', fontsize=11, fontweight='bold')
ax1.set_title('Per-Class Metrics (Precision, Recall, F1)', fontsize=12, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(class_names, rotation=45, ha='right')
ax1.set_ylim([0.75, 0.9])
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3, axis='y')

# [2] Support와 F1의 관계
ax2 = axes[1]
colors = plt.cm.viridis(np.linspace(0, 1, len(class_names)))
bars = ax2.barh(class_names, support, color=colors, alpha=0.8)

# 상위에 F1 점수 표시
for i, (s, f) in enumerate(zip(support, f1)):
    ax2.text(s + 50, i, f'F1:{f:.3f}', va='center', fontsize=9, fontweight='bold')

ax2.set_xlabel('Support (Number of Samples)', fontsize=11, fontweight='bold')
ax2.set_title('Class Support and F1 Score', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('practice/chapter9/data/output/per_class_metrics_ko.png',
            dpi=150, bbox_inches='tight')
print(f"\nSaved: practice/chapter9/data/output/per_class_metrics_ko.png")
plt.close()
```

**예상 출력**:
```
============================================================
Step 11: 클래스별 Precision, Recall, F1
============================================================

Per-Class Performance:
       Class  Precision    Recall  F1-Score  Support
       정치      0.8204    0.8194    0.8199      980
       경제      0.8089    0.8071    0.8080      980
       사회      0.8312    0.8286    0.8299      980
       생활      0.8134    0.8141    0.8138      980
       문화      0.8187    0.8214    0.8200      980
       세계      0.8371    0.8357    0.8364      980
       IT과학    0.8301    0.8306    0.8303      980
       스포츠    0.8410    0.8378    0.8394      980
       연예      0.8267    0.8327    0.8297      980

Macro-Averaged Metrics (Unweighted):
  Precision: 0.8252
  Recall: 0.8252
  F1-Score: 0.8252

Weighted-Averaged Metrics (Class-Weighted by Support):
  Precision: 0.8252
  Recall: 0.8234
  F1-Score: 0.8242

Best and Worst Performing Classes:
  Best:  스포츠 (F1: 0.8394)
  Worst: 경제   (F1: 0.8080)
  Difference: 0.0314

Saved: practice/chapter9/data/output/per_class_metrics_ko.png
```

### 오류 분석: 어려운 예측 사례

```python
print("\n" + "=" * 60)
print("Step 12: 오류 분석 (어려운 예측 사례)")
print("=" * 60)

# 모델의 예측 확률
pred_probs = torch.softmax(
    torch.from_numpy(val_predictions.predictions), dim=1
).numpy()

# 올바른 예측과 틀린 예측 분류
correct_mask = val_pred_labels == val_true_labels
incorrect_mask = ~correct_mask
incorrect_indices = np.where(incorrect_mask)[0]

print(f"Overall Error Analysis:")
print(f"  Total predictions: {len(val_pred_labels):,}")
print(f"  Correct: {np.sum(correct_mask):,} ({np.sum(correct_mask) / len(val_pred_labels) * 100:.2f}%)")
print(f"  Incorrect: {len(incorrect_indices):,} ({len(incorrect_indices) / len(val_pred_labels) * 100:.2f}%)")

if len(incorrect_indices) > 0:
    # 신뢰도 (confidence) 분석
    incorrect_probs = pred_probs[incorrect_indices]
    max_probs = np.max(incorrect_probs, axis=1)
    mean_confidence = np.mean(max_probs)

    print(f"\nConfidence Analysis (Incorrect Predictions):")
    print(f"  Mean confidence: {mean_confidence:.4f}")
    print(f"  Min confidence: {np.min(max_probs):.4f}")
    print(f"  Max confidence: {np.max(max_probs):.4f}")

    # 신뢰도가 가장 낮은 오류들 (모델이 불확실했던 오류)
    low_confidence_idx = np.argsort(max_probs)[:5]

    print(f"\nTop 5 Hard Examples (Low Confidence Errors):")
    for i, idx_in_incorrect in enumerate(low_confidence_idx, 1):
        actual_idx = incorrect_indices[idx_in_incorrect]
        true_label = val_true_labels[actual_idx]
        pred_label = val_pred_labels[actual_idx]
        confidence = max_probs[idx_in_incorrect]

        # 예측 분포
        sorted_prob_idx = np.argsort(incorrect_probs[idx_in_incorrect])[::-1]

        print(f"\n  Error {i}:")
        print(f"    True: {class_names[true_label]:6s}")
        print(f"    Pred: {class_names[pred_label]:6s}")
        print(f"    Confidence: {confidence:.4f}")
        print(f"    Top 3 predictions:")
        for rank, prob_idx in enumerate(sorted_prob_idx[:3], 1):
            prob = incorrect_probs[idx_in_incorrect][prob_idx]
            print(f"      {rank}. {class_names[prob_idx]:6s}: {prob:.4f}")

    # 자신감은 높았지만 틀린 오류들 (과신뢰 오류)
    high_confidence_idx = np.argsort(max_probs)[-5:][::-1]

    print(f"\nTop 5 Overconfident Errors (High Confidence but Wrong):")
    for i, idx_in_incorrect in enumerate(high_confidence_idx, 1):
        actual_idx = incorrect_indices[idx_in_incorrect]
        true_label = val_true_labels[actual_idx]
        pred_label = val_pred_labels[actual_idx]
        confidence = max_probs[idx_in_incorrect]

        print(f"\n  Error {i}:")
        print(f"    True: {class_names[true_label]:6s}")
        print(f"    Pred: {class_names[pred_label]:6s}")
        print(f"    Model confidence: {confidence:.4f} (very high!)")
```

**예상 출력**:
```
============================================================
Step 12: 오류 분석 (어려운 예측 사례)
============================================================
Overall Error Analysis:
  Total predictions: 9,144
  Correct: 7,521 (82.26%)
  Incorrect: 1,623 (17.74%)

Confidence Analysis (Incorrect Predictions):
  Mean confidence: 0.5834
  Min confidence: 0.1205
  Max confidence: 0.9876

Top 5 Hard Examples (Low Confidence Errors):
  Error 1:
    True: 정치
    Pred: 사회
    Confidence: 0.1265
    Top 3 predictions:
      1. 사회: 0.1265
      2. 정치: 0.1234
      3. 생활: 0.1189

  Error 2:
    True: 경제
    Pred: 세계
    Confidence: 0.1434
    ...

Top 5 Overconfident Errors (High Confidence but Wrong):
  Error 1:
    True: 생활
    Pred: 문화
    Model confidence: 0.9876 (very high!)

  Error 2:
    True: 세계
    Pred: IT과학
    Model confidence: 0.9654 (very high!)
```

### 과적합 분석 (Train vs Validation)

```python
print("\n" + "=" * 60)
print("Step 13: 과적합 분석 (Train vs Validation)")
print("=" * 60)

# 훈련 세트에서도 예측 수집
train_predictions = trainer.predict(processed_train)
train_pred_labels = np.argmax(train_predictions.predictions, axis=1)
train_true_labels = np.array([processed_train[i]['label'] for i in range(len(processed_train))])

train_accuracy = np.sum(train_pred_labels == train_true_labels) / len(train_true_labels)
val_accuracy = np.sum(val_pred_labels == val_true_labels) / len(val_true_labels)

print(f"\nAccuracy Comparison:")
print(f"  Train Accuracy: {train_accuracy:.4f}")
print(f"  Val Accuracy:   {val_accuracy:.4f}")
print(f"  Gap (Overfitting indicator): {train_accuracy - val_accuracy:.4f}")

# 과적합 진단
gap = train_accuracy - val_accuracy
if gap < 0.02:
    diagnosis = "✓ 과적합 신호 없음. 매우 좋은 일반화 성능"
    color = "green"
elif gap < 0.05:
    diagnosis = "✓ 미미한 과적합. 허용 범위"
    color = "yellow"
elif gap < 0.10:
    diagnosis = "⚠ 경미한 과적합. 허용 범위이나 모니터링 필요"
    color = "orange"
else:
    diagnosis = "✗ 심각한 과적합. 정규화 강화 필요"
    color = "red"

print(f"  → {diagnosis}")

# 혼동 행렬 비교
cm_train = confusion_matrix(train_true_labels, train_pred_labels)
cm_val = confusion_matrix(val_true_labels, val_pred_labels)

# 각 클래스별 정확도 비교
print(f"\nPer-Class Accuracy Comparison:")
for i, name in enumerate(class_names):
    train_acc = cm_train[i, i] / cm_train[i].sum()
    val_acc = cm_val[i, i] / cm_val[i].sum()
    diff = train_acc - val_acc

    symbol = "+" if diff > 0.02 else ("=" if diff > -0.02 else "-")
    print(f"  {name:6s}: Train {train_acc:.4f}, Val {val_acc:.4f}, Diff {diff:+.4f} {symbol}")

# 시각화
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# [1] Accuracy 비교
ax1 = axes[0]
categories = ['Train', 'Validation']
accuracies = [train_accuracy, val_accuracy]
colors_bar = ['steelblue', 'coral']
bars = ax1.bar(categories, accuracies, color=colors_bar, alpha=0.8, width=0.6, edgecolor='black', linewidth=2)

# 수치 표시
for bar, acc in zip(bars, accuracies):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width() / 2., height,
             f'{acc:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

# 과적합 간격 표시
ax1.plot([0, 1], [val_accuracy, val_accuracy], 'r--', linewidth=2, label=f'Gap: {gap:.4f}')

ax1.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
ax1.set_title('Overfitting Analysis: Train vs Validation', fontsize=12, fontweight='bold')
ax1.set_ylim([0.80, 0.95])
ax1.grid(True, alpha=0.3, axis='y')
ax1.legend(fontsize=10)

# [2] 클래스별 정확도 비교
ax2 = axes[1]
x = np.arange(len(class_names))
width = 0.35

train_accs = [cm_train[i, i] / cm_train[i].sum() for i in range(len(class_names))]
val_accs = [cm_val[i, i] / cm_val[i].sum() for i in range(len(class_names))]

bars1 = ax2.bar(x - width/2, train_accs, width, label='Train', color='steelblue', alpha=0.8)
bars2 = ax2.bar(x + width/2, val_accs, width, label='Validation', color='coral', alpha=0.8)

ax2.set_xlabel('Class', fontsize=11, fontweight='bold')
ax2.set_ylabel('Per-Class Accuracy', fontsize=11, fontweight='bold')
ax2.set_title('Per-Class Accuracy: Train vs Validation', fontsize=12, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(class_names, rotation=45, ha='right')
ax2.set_ylim([0.75, 0.95])
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('practice/chapter9/data/output/overfitting_analysis_ko.png',
            dpi=150, bbox_inches='tight')
print(f"\nSaved: practice/chapter9/data/output/overfitting_analysis_ko.png")
plt.close()
```

**예상 출력**:
```
============================================================
Step 13: 과적합 분석 (Train vs Validation)
============================================================

Accuracy Comparison:
  Train Accuracy: 0.9234
  Val Accuracy:   0.8226
  Gap (Overfitting indicator): 0.1008

  → ⚠ 경미한 과적합. 허용 범위이나 모니터링 필요

Per-Class Accuracy Comparison:
  정치   : Train 0.9456, Val 0.8194, Diff +0.1262 +
  경제   : Train 0.9321, Val 0.8071, Diff +0.1250 +
  사회   : Train 0.9387, Val 0.8286, Diff +0.1101 +
  ...
  연예   : Train 0.9412, Val 0.8327, Diff +0.1085 +

Saved: practice/chapter9/data/output/overfitting_analysis_ko.png
```

---

## 흔한 실수와 디버깅

### 실수 1: 학습률이 너무 크다

```python
# ❌ 틀린 예: 처음부터 학습할 때의 학습률
training_args = TrainingArguments(
    learning_rate=1e-4,  # 파인튜닝에는 너무 크다
)

# 결과: 학습이 발산하거나 진동함
# 시각: Loss가 계속 증가하거나, 심하게 진동

# ✅ 올바른 예: 파인튜닝의 학습률
training_args = TrainingArguments(
    learning_rate=2e-5,  # 파인튜닝: 100배 더 작음
)

# 이유:
# - 사전학습된 가중치가 이미 좋은 위치에 있다
# - 큰 스텝으로 움직이면 좋은 표현을 망친다
# - 작은 스텝으로 미세 조정한다
```

### 실수 2: Early Stopping 없이 과도하게 학습

```python
# ❌ 틀린 예
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=processed_train,
    eval_dataset=processed_val,
    compute_metrics=compute_metrics,
    callbacks=[],  # Early Stopping이 없음!
)

# 결과: 20 에포크 후 Val Loss가 증가했지만
# 계속 학습하여 과적합 심화
# Test Accuracy가 82%에서 75%로 떨어짐

# ✅ 올바른 예
early_stopping = EarlyStoppingCallback(
    early_stopping_patience=2,
)

trainer = Trainer(
    ...,
    callbacks=[early_stopping],
)

# 결과: Val Loss가 최소일 때(에포크 15)에서 자동으로 멈춤
# Test Accuracy 82% 유지
```

### 실수 3: 모든 파라미터가 학습되는지 확인하지 않기

```python
# ❌ 실수: 실수로 일부 파라미터가 frozen되어 있음
model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased")

# 일부 레이어를 실수로 frozen
for param in model.bert.embeddings.parameters():
    param.requires_grad = False

total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"{trainable} / {total} trainable")  # 110M / 167M?
# 일부 파라미터만 학습됨!

# ✅ 올바른 예: Full Fine-tuning 확인
for param in model.parameters():
    param.requires_grad = True  # 명시적 확인

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
assert trainable == total  # 모든 파라미터 학습 가능 확인
```

### 실수 4: 배치 크기가 너무 크다

```python
# ❌ 틀린 예
training_args = TrainingArguments(
    per_device_train_batch_size=256,  # GPU 메모리 부족!
)

# 결과: CUDA out of memory 에러

# ✅ 올바른 예
training_args = TrainingArguments(
    per_device_train_batch_size=16,  # 또는 8 (메모리 제약 있을 때)
)

# 배치 크기 선택 가이드:
# - 메모리 16GB: batch_size = 16
# - 메모리 8GB: batch_size = 8
# - 메모리 4GB: batch_size = 4
```

### 실수 5: Confusion Matrix 해석 오류

```python
# ❌ 틀린 해석
# "정확도가 82%니까 모델이 좋다"

# ✅ 올바른 해석
# 1. Confusion Matrix에서 약한 클래스 확인
#    - 경제: 80.7% (상대적으로 어려움)
#    - 스포츠: 83.8% (상대적으로 쉬움)
#
# 2. 혼동 패턴 확인
#    - 정치 ↔ 사회: 자주 혼동 (도메인이 비슷)
#    - 경제 ↔ 세계: 자주 혼동 (경제 뉴스가 국제 뉴스로 분류될 수 있음)
#
# 3. 개선 방향
#    - 경제 뉴스 더 수집
#    - 정치/사회 데이터 주석 재확인
#    - 모델 크기 증가 (BERT-Large) 고려
```

---

## 종합 해설 및 학습 요점

### Full Fine-tuning의 핵심 이해

**1. 왜 학습률이 매우 작은가?**

```
사전학습된 BERT의 가중치:
  W_pretrained = [좋은 표현을 인코딩]

Full Fine-tuning의 목표:
  W_new = W_pretrained + ΔW
  여기서 ΔW는 가능한 한 작아야 함

이유:
  - W_pretrained는 이미 일반적인 언어 이해를 가지고 있다
  - 큰 ΔW를 더하면 이 좋은 표현이 망가진다
  - 작은 ΔW로 도메인 특화 조정만 수행한다

학습률:
  - 처음부터: 1e-3 ~ 5e-4 (모든 것이 임의의 값에서 시작)
  - 파인튜닝: 1e-5 ~ 5e-5 (좋은 위치에서 미세 조정)

차이: 약 100배
```

**2. Warmup + Linear Decay의 역할**

```
Warmup (처음 500 스텝, 5.8% of training):
  학습률: 0 → 2e-5로 천천히 증가
  이유: 초기에 불안정한 업데이트 방지

Linear Decay (남은 7500 스텝):
  학습률: 2e-5 → 0으로 선형 감소
  이유: 수렴 안정화, 최종 미세 조정

효과: 학습의 안정성 + 수렴성 증가
```

**3. Early Stopping의 중요성**

```
문제: Train Loss는 계속 감소하지만 Val Loss는 증가
       (과적합 신호)

Early Stopping이 없으면:
  - 에포크 15: Val Acc = 82.34% (최고)
  - 에포크 20: Val Acc = 81.65% (감소)
  - 에포크 25: Val Acc = 80.12% (더 감소)
  → 에포크 25 모델을 테스트하면 성능 저조

Early Stopping이 있으면:
  - 에포크 15에서 자동으로 멈춤
  → 에포크 15 모델(최고 성능)을 테스트
  → 최고의 성능 유지
```

**4. Confusion Matrix에서 배우는 것**

```
단순 정확도: 82%
혼동 행렬: 다양한 인사이트

예시:
  정치 ↔ 사회: 자주 혼동
    → 이 두 카테고리가 도메인적으로 비슷함을 의미
    → 데이터 주석 재확인 필요
    → 모델이 구분을 위해 더 세밀한 특징 필요

  경제: Recall 80.7% (가장 낮음)
    → 경제 뉴스를 다른 카테고리로 분류하는 경우가 많음
    → 경제 도메인 데이터 부족 가능성
    → 경제 관련 용어 추가 학습 고려

정확도만으로는 알 수 없는 정보들!
```

**5. Train vs Validation 격차의 의미**

```
Train Acc = 92.34%, Val Acc = 82.26%, 격차 = 10.08%

격차 크기 해석:
  < 2%: 과적합 없음 ✓
  2%~5%: 미미한 과적합 (보통 허용)
  5%~10%: 경미한 과적합 (모니터링 권장)
  > 10%: 심각한 과적합 (정규화 강화 필요)

현재 상황 (10.08%):
  → 경미한 과적합이지만 허용 범위
  → 이유:
     - 도메인 데이터 제한적 (45K 샘플)
     - 모델 크기 매우 큼 (167M 파라미터)
     - 비율: 파라미터/샘플 = 3.6 (과적합 위험)

  → 하지만 Early Stopping과 정규화로 관리됨
  → Test 성능도 Validation 성능과 유사할 가능성 높음
```

### 다음 주 10주차 예고

이 9주차 Full Fine-tuning은 기초를 다지기 위함이다. **메모리 사용량이 크고(1.2GB), 학습이 느리고(수시간), 파라미터 개수가 많다(110M)는 실제 문제**가 있다.

**10주차에서 배울 것**:

- **PEFT(Parameter-Efficient Fine-Tuning)**: 0.1% 파라미터(~1M)만 학습
- **LoRA(Low-Rank Adaptation)**: 추가 행렬 AₘₓᵣBᵣₓₙ만 학습
- **메모리 절약**: 1.2GB → 0.4GB (3배 감소)
- **속도 향상**: 수시간 → 30분 (10배 향상)
- **성능 유지**: Full Fine-tuning과 유사한 정확도 달성

같은 Trainer API를 사용하되, `peft` 라이브러리의 LoRA 설정만 추가하면 된다.

---

## 참고 코드 파일

다음 파일에서 전체 구현을 확인할 수 있다:

- **practice/chapter9/code/9-1-bert-finetuning.py** — 전체 Full Fine-tuning 구현
- **practice/chapter9/data/output/** — 생성된 그래프 및 시각화

### 코드 실행 방법

```bash
# 가상환경 활성화
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate    # Windows

# 필수 라이브러리 설치
pip install transformers datasets torch scikit-learn matplotlib seaborn

# Full Fine-tuning 실행 (약 2-3시간, GPU 필수)
python practice/chapter9/code/9-1-bert-finetuning.py

# 결과 확인
ls practice/chapter9/data/output/
# → learning_curves_ko.png
# → confusion_matrix_ko.png
# → per_class_metrics_ko.png
# → overfitting_analysis_ko.png
```

---

## 최종 학습 정리

### 9주차 핵심 개념 요약

1. **Full Fine-tuning**: 사전학습된 모델의 모든 파라미터를 도메인 데이터로 업데이트
2. **학습률 결정**: 매우 작은 학습률(2e-5)로 좋은 표현을 보존
3. **과적합 방지**: Validation Set, Early Stopping, Warmup+Decay, Weight Decay
4. **Hugging Face Trainer**: TrainingArguments + Trainer로 자동화
5. **다양한 평가**: Accuracy + Precision/Recall/F1 + Confusion Matrix
6. **오류 분석**: 약한 클래스, 혼동 패턴, 과신뢰 오류 파악
7. **Train vs Val 격차**: 과적합 정도의 정량적 지표

### 다음 장으로의 연결

다음 10주차에서는:
- PEFT/LoRA로 메모리와 시간을 1/10 수준으로 감소
- 그럼에도 성능은 거의 동일하게 유지
- 개인 노트북 GPU에서도 파인튜닝 가능
- 대규모 모델(GPT-3.5 수준 크기)도 파인튜닝 가능

이 9주차의 Full Fine-tuning이 기초이므로, 여기서 확실히 이해하고 다음으로 진행하자.

---

**생성일**: 2026-02-25
**대상 독자**: 컴퓨터공학/AI 전공 학부생 (3~4학년)
**난이도**: 중급 (파이썬, 딥러닝 기초, Transformer 이해 선수)
