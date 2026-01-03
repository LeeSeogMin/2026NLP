# 11장 리서치 결과: LLM 파인튜닝 (1) - 전이 학습과 Full Fine-tuning

**조사일**: 2026-01-03
**조사 주제**: Transfer Learning, Fine-tuning, Hugging Face Trainer API, 하이퍼파라미터 튜닝

---

## 1. 전이 학습 (Transfer Learning)

### 1.1 개념
- 사전학습된 모델의 지식을 새로운 태스크에 재활용
- Pre-training → Fine-tuning 패러다임
- NLP에서는 BERT, GPT 등 대규모 언어 모델을 다운스트림 태스크에 적용

### 1.2 장점
- 적은 데이터로 높은 성능 달성
- 학습 시간 및 계산 비용 절감
- 일반화 성능 향상
- 도메인 간 지식 전이 가능

### 1.3 워크플로우
1. 사전학습된 모델의 레이어 가져오기
2. 레이어 동결 (Freeze)
3. 새로운 학습 가능한 레이어 추가
4. 새 레이어를 데이터셋으로 학습
5. (선택) 전체 모델 Fine-tuning

---

## 2. 파인튜닝 전략

### 2.1 Feature Extraction (특징 추출)
- 사전학습 가중치 완전 고정
- 분류 헤드(Classifier Head)만 학습
- 빠른 학습, 적은 데이터에 적합
- 도메인이 유사할 때 효과적

### 2.2 Full Fine-tuning
- 전체 모델 가중치 업데이트
- 도메인이 다를 때 필요
- 더 많은 데이터와 계산 자원 필요
- 최고 성능 달성 가능

### 2.3 Partial Fine-tuning
- 일부 레이어만 학습
- Layer-wise Learning Rate 적용
- 상위 레이어: 높은 학습률
- 하위 레이어: 낮은 학습률

### 2.4 전략 선택 기준

| 상황 | 권장 전략 |
|------|-----------|
| 적은 데이터, 유사 도메인 | Feature Extraction |
| 적은 데이터, 다른 도메인 | Partial Fine-tuning |
| 많은 데이터, 다른 도메인 | Full Fine-tuning |
| 대규모 모델, 제한된 자원 | PEFT (LoRA 등) |

---

## 3. Hugging Face Trainer API

### 3.1 핵심 클래스
```python
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
```

### 3.2 TrainingArguments 주요 파라미터
```python
TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_steps=500,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    logging_dir="./logs",
    fp16=True  # Mixed precision
)
```

### 3.3 Trainer 워크플로우
1. 모델 로드 (`AutoModelForSequenceClassification`)
2. 토크나이저 로드 (`AutoTokenizer`)
3. 데이터셋 토큰화 (`dataset.map()`)
4. TrainingArguments 설정
5. compute_metrics 함수 정의
6. Trainer 생성 및 학습
7. 평가 및 저장

### 3.4 compute_metrics 예시
```python
from sklearn.metrics import accuracy_score, f1_score

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.argmax(axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average="weighted")
    }
```

---

## 4. 하이퍼파라미터 튜닝

### 4.1 Learning Rate
- **BERT 권장**: 2e-5 ~ 5e-5
- 너무 높으면: 사전학습 지식 손상
- 너무 낮으면: 학습 느림, 수렴 실패
- Fine-tuning은 매우 낮은 학습률 필요

### 4.2 Warmup Steps
- 초기 학습 안정화
- 원래 BERT: 10,000 steps warmup
- Fine-tuning: 전체 스텝의 6-10%
- 학습률을 점진적으로 증가시킴

### 4.3 Weight Decay
- L2 정규화 효과
- BERT 기본값: 0.01
- AdamW 옵티마이저와 함께 사용
- LayerNorm, Bias는 decay 제외

### 4.4 Batch Size
- 메모리 vs 성능 트레이드오프
- 작은 배치: 더 많은 업데이트, 노이즈
- 큰 배치: 안정적이지만 메모리 필요
- Gradient Accumulation으로 가상 배치 크기 증가

### 4.5 Epochs
- BERT Fine-tuning: 3-5 epochs 권장
- 너무 많으면 과적합
- Early Stopping과 함께 사용

### 4.6 권장 하이퍼파라미터 범위

| 파라미터 | 범위 | 기본값 |
|----------|------|--------|
| learning_rate | 1e-5 ~ 5e-5 | 2e-5 |
| batch_size | 8 ~ 32 | 16 |
| epochs | 2 ~ 5 | 3 |
| warmup_ratio | 0.06 ~ 0.1 | 0.06 |
| weight_decay | 0.01 ~ 0.1 | 0.01 |

---

## 5. 과적합 방지

### 5.1 Early Stopping
- Validation loss 모니터링
- patience 설정 (예: 3 epochs)
- 가장 좋은 모델 저장

### 5.2 Dropout
- BERT 기본: 0.1
- 과적합 시 0.2-0.3으로 증가
- 추론 시 비활성화

### 5.3 Label Smoothing
- Hard label → Soft label
- 과신(overconfidence) 방지
- 일반화 성능 향상

### 5.4 Data Augmentation (NLP)
- 동의어 치환
- 역번역 (Back-translation)
- 랜덤 삽입/삭제/교환
- 한국어: KoEDA 라이브러리

---

## 6. 학습 모니터링

### 6.1 Loss Curve 분석
- Train Loss 감소: 학습 진행 중
- Validation Loss 증가: 과적합 시작
- 두 Loss 모두 높음: 언더피팅

### 6.2 모니터링 도구
- TensorBoard: 기본 제공
- Weights & Biases: 고급 시각화
- MLflow: 실험 관리

### 6.3 Hugging Face 통합
```python
from transformers.integrations import TensorBoardCallback

training_args = TrainingArguments(
    logging_dir="./logs",
    logging_steps=10,
    report_to=["tensorboard"]
)
```

---

## 7. 파인튜닝 태스크

### 7.1 Sequence Classification
- 입력: 문장/문서
- 출력: 클래스 레이블
- 예: 감성 분석, 주제 분류
- 모델: `AutoModelForSequenceClassification`

### 7.2 Token Classification
- 입력: 토큰 시퀀스
- 출력: 각 토큰의 레이블
- 예: NER, POS Tagging
- 모델: `AutoModelForTokenClassification`

### 7.3 Question Answering
- 입력: (질문, 문맥)
- 출력: 시작/끝 위치
- 예: SQuAD 데이터셋
- 모델: `AutoModelForQuestionAnswering`

---

## 8. 데이터셋 준비

### 8.1 Hugging Face Datasets
```python
from datasets import load_dataset

# 공개 데이터셋 로드
dataset = load_dataset("imdb")

# 커스텀 데이터 로드
dataset = load_dataset("csv", data_files="data.csv")
dataset = load_dataset("json", data_files="data.json")
```

### 8.2 데이터 분할
```python
dataset = dataset.train_test_split(test_size=0.2)
train_dataset = dataset["train"]
test_dataset = dataset["test"]
```

### 8.3 토큰화
```python
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=512
    )

tokenized_datasets = dataset.map(tokenize_function, batched=True)
```

---

## 9. 성능 비교 참고

### BERT Fine-tuning 결과 (참고)
| 데이터셋 | Base 모델 | Fine-tuned | 향상 |
|----------|-----------|------------|------|
| SST-2 | 92.8% | 93.5% | +0.7% |
| MRPC | 84.8% | 89.3% | +4.5% |
| CoLA | 52.1% | 60.5% | +8.4% |

### 주의사항
- BERT-Large는 작은 데이터셋에서 불안정할 수 있음
- 여러 번 실행하여 평균 성능 측정 권장
- 시드(seed) 고정으로 재현성 확보

---

## 10. 참고문헌

- Hugging Face Trainer Documentation: https://huggingface.co/docs/transformers/training
- Hugging Face LLM Course: https://huggingface.co/learn/llm-course/
- Devlin, J., et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers. NAACL 2019
- Keras Transfer Learning Guide: https://keras.io/guides/transfer_learning/
- "On the Stability of Fine-Tuning BERT" - OpenReview
