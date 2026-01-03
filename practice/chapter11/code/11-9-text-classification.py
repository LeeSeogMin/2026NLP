"""
11장 실습 코드 11-9: 텍스트 분류 파인튜닝 (IMDb 감성 분석)
- IMDb 데이터셋 로드 및 전처리
- BERT 모델 파인튜닝
- 학습 결과 분석
"""

import torch
import numpy as np
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import warnings
warnings.filterwarnings("ignore")


def load_imdb_dataset(sample_size=1000):
    """IMDb 데이터셋 로드 (샘플링)"""
    print("1. IMDb 데이터셋 로드")
    print("-" * 40)

    # 전체 데이터셋 로드
    dataset = load_dataset("imdb")

    # 메모리 절약을 위해 샘플링
    train_dataset = dataset["train"].shuffle(seed=42).select(range(sample_size))
    test_dataset = dataset["test"].shuffle(seed=42).select(range(sample_size // 4))

    print(f"  원본 데이터셋:")
    print(f"    훈련: {len(dataset['train'])}개")
    print(f"    테스트: {len(dataset['test'])}개")
    print(f"\n  샘플링 후:")
    print(f"    훈련: {len(train_dataset)}개")
    print(f"    테스트: {len(test_dataset)}개")

    # 샘플 확인
    print(f"\n  샘플 데이터:")
    sample = train_dataset[0]
    print(f"    텍스트 길이: {len(sample['text'])} 문자")
    print(f"    레이블: {sample['label']} ({'긍정' if sample['label'] == 1 else '부정'})")
    print(f"    텍스트 미리보기: {sample['text'][:100]}...")

    return train_dataset, test_dataset


def tokenize_data(train_dataset, test_dataset, tokenizer, max_length=256):
    """데이터 토큰화"""
    print("\n2. 데이터 토큰화")
    print("-" * 40)

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length
        )

    print(f"  최대 시퀀스 길이: {max_length}")
    print(f"  토큰화 진행 중...")

    train_tokenized = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"]
    )
    test_tokenized = test_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"]
    )

    print(f"  토큰화 완료!")

    return train_tokenized, test_tokenized


def setup_model_and_trainer(model_name, train_dataset, test_dataset, tokenizer):
    """모델 및 Trainer 설정"""
    print("\n3. 모델 및 Trainer 설정")
    print("-" * 40)

    # 모델 로드
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"  모델: {model_name}")
    print(f"  총 파라미터: {total_params:,}")
    print(f"  학습 가능 파라미터: {trainable_params:,}")

    # compute_metrics 함수
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=-1)

        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='binary', zero_division=0
        )

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

    # TrainingArguments
    training_args = TrainingArguments(
        output_dir="./imdb_results",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=50,
        report_to="none",
        seed=42
    )

    print(f"\n  학습 설정:")
    print(f"    에폭: {training_args.num_train_epochs}")
    print(f"    배치 크기: {training_args.per_device_train_batch_size}")
    print(f"    학습률: {training_args.learning_rate}")
    print(f"    가중치 감쇠: {training_args.weight_decay}")
    print(f"    워밍업 비율: {training_args.warmup_ratio}")

    # Data Collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Trainer 생성
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    return trainer, model


def train_model(trainer):
    """모델 학습"""
    print("\n4. 모델 학습")
    print("-" * 40)
    print("  학습을 시작합니다...")
    print()

    train_result = trainer.train()

    print(f"\n  학습 완료!")
    print(f"    총 스텝: {train_result.global_step}")
    print(f"    학습 손실: {train_result.training_loss:.4f}")

    return train_result


def evaluate_model(trainer):
    """모델 평가"""
    print("\n5. 모델 평가")
    print("-" * 40)

    eval_result = trainer.evaluate()

    print(f"  평가 결과:")
    print(f"    정확도 (Accuracy): {eval_result['eval_accuracy']:.4f}")
    print(f"    정밀도 (Precision): {eval_result['eval_precision']:.4f}")
    print(f"    재현율 (Recall): {eval_result['eval_recall']:.4f}")
    print(f"    F1 스코어: {eval_result['eval_f1']:.4f}")
    print(f"    손실: {eval_result['eval_loss']:.4f}")

    return eval_result


def test_predictions(trainer, tokenizer):
    """예측 테스트"""
    print("\n6. 예측 테스트")
    print("-" * 40)

    test_texts = [
        "This movie was absolutely amazing! Great story and wonderful acting.",
        "Terrible movie. Complete waste of time. Very boring and predictable.",
        "An average film with some good moments but nothing special.",
    ]

    for text in test_texts:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256
        )

        with torch.no_grad():
            outputs = trainer.model(**inputs)
            logits = outputs.logits
            prediction = torch.argmax(logits, dim=-1).item()
            probs = torch.softmax(logits, dim=-1)[0]

        sentiment = "긍정" if prediction == 1 else "부정"
        confidence = probs[prediction].item()

        print(f"\n  텍스트: \"{text[:50]}...\"")
        print(f"  예측: {sentiment} (확신도: {confidence:.2%})")


def analyze_hyperparameters():
    """하이퍼파라미터 분석 가이드"""
    print("\n7. 하이퍼파라미터 튜닝 가이드")
    print("-" * 40)

    print("""
  권장 하이퍼파라미터 범위:
  ┌─────────────────┬─────────────────┬─────────────────┐
  │    파라미터     │    권장 범위    │     기본값      │
  ├─────────────────┼─────────────────┼─────────────────┤
  │ learning_rate   │ 1e-5 ~ 5e-5     │ 2e-5           │
  │ batch_size      │ 8 ~ 32          │ 16             │
  │ epochs          │ 2 ~ 5           │ 3              │
  │ warmup_ratio    │ 0.06 ~ 0.1      │ 0.1            │
  │ weight_decay    │ 0.01 ~ 0.1      │ 0.01           │
  │ max_length      │ 128 ~ 512       │ 256            │
  └─────────────────┴─────────────────┴─────────────────┘

  과적합 방지 전략:
    - Early Stopping (patience=2-3)
    - Dropout 증가 (0.1 → 0.2)
    - Weight Decay 증가
    - Data Augmentation (역번역, 동의어 치환)

  학습률 스케줄:
    - 선형 감소 (linear decay)
    - 코사인 감소 (cosine decay)
    - 워밍업 + 감소 조합
""")


def main():
    print("=" * 60)
    print("11장: 텍스트 분류 파인튜닝 (IMDb 감성 분석)")
    print("=" * 60)
    print()

    # 설정
    model_name = "distilbert-base-uncased"
    sample_size = 500  # 데모용 작은 샘플

    print(f"모델: {model_name}")
    print(f"샘플 크기: {sample_size}")
    print()

    # 1. 데이터 로드
    train_dataset, test_dataset = load_imdb_dataset(sample_size)

    # 2. 토크나이저 로드 및 토큰화
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_tokenized, test_tokenized = tokenize_data(
        train_dataset, test_dataset, tokenizer
    )

    # 3. 모델 및 Trainer 설정
    trainer, model = setup_model_and_trainer(
        model_name, train_tokenized, test_tokenized, tokenizer
    )

    # 4. 학습
    train_result = train_model(trainer)

    # 5. 평가
    eval_result = evaluate_model(trainer)

    # 6. 예측 테스트
    test_predictions(trainer, tokenizer)

    # 7. 하이퍼파라미터 가이드
    analyze_hyperparameters()

    print("\n" + "=" * 60)
    print("실습 완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()
