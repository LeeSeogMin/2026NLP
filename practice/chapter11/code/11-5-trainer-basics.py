"""
11장 실습 코드 11-5: Hugging Face Trainer API 기초
- TrainingArguments 설정
- Trainer 클래스 활용
- compute_metrics 함수 정의
"""

import torch
import numpy as np
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def create_sample_dataset():
    """샘플 데이터셋 생성"""
    print("1. 샘플 데이터셋 생성")
    print("-" * 40)

    # 간단한 감성 분석 데이터
    texts = [
        "This movie was absolutely fantastic!",
        "I really enjoyed watching this film.",
        "Great acting and wonderful story.",
        "Best movie I have seen this year.",
        "Highly recommended for everyone.",
        "This was a terrible waste of time.",
        "I hated every minute of it.",
        "Worst movie ever made.",
        "Complete disaster and boring.",
        "Do not watch this awful film.",
        "The plot was interesting and engaging.",
        "Loved the characters and the ending.",
        "Not worth the ticket price at all.",
        "Disappointing and poorly directed.",
        "An amazing cinematic experience!",
        "Dull and unimaginative storytelling.",
    ]

    labels = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0]  # 1: positive, 0: negative

    # Dataset 객체 생성
    dataset = Dataset.from_dict({
        "text": texts,
        "label": labels
    })

    # Train/Test 분할
    dataset = dataset.train_test_split(test_size=0.25, seed=42)

    print(f"  훈련 데이터: {len(dataset['train'])}개")
    print(f"  테스트 데이터: {len(dataset['test'])}개")
    print(f"  레이블: 0 (부정), 1 (긍정)")

    return dataset


def tokenize_dataset(dataset, tokenizer):
    """데이터셋 토큰화"""
    print("\n2. 데이터셋 토큰화")
    print("-" * 40)

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=128
        )

    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    # 불필요한 컬럼 제거
    tokenized_dataset = tokenized_dataset.remove_columns(["text"])

    print(f"  토큰화 완료")
    print(f"  최대 시퀀스 길이: 128")

    # 샘플 확인
    sample = tokenized_dataset["train"][0]
    print(f"\n  샘플 입력:")
    print(f"    input_ids 길이: {len(sample['input_ids'])}")
    print(f"    레이블: {sample['label']}")

    return tokenized_dataset


def define_compute_metrics():
    """평가 메트릭 함수 정의"""
    print("\n3. 평가 메트릭 함수 정의")
    print("-" * 40)

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=-1)

        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='weighted', zero_division=0
        )

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

    print("  compute_metrics 함수:")
    print("    - accuracy: 정확도")
    print("    - precision: 정밀도")
    print("    - recall: 재현율")
    print("    - f1: F1 스코어")

    return compute_metrics


def setup_training_arguments():
    """TrainingArguments 설정"""
    print("\n4. TrainingArguments 설정")
    print("-" * 40)

    training_args = TrainingArguments(
        output_dir="./results",           # 출력 디렉토리
        num_train_epochs=3,               # 에폭 수
        per_device_train_batch_size=4,    # 학습 배치 크기
        per_device_eval_batch_size=4,     # 평가 배치 크기
        learning_rate=2e-5,               # 학습률
        weight_decay=0.01,                # 가중치 감쇠
        warmup_steps=10,                  # 워밍업 스텝
        evaluation_strategy="epoch",      # 평가 전략
        save_strategy="epoch",            # 저장 전략
        load_best_model_at_end=True,      # 최종 모델 로드
        logging_steps=5,                  # 로깅 간격
        report_to="none",                 # 리포팅 비활성화
        seed=42                           # 랜덤 시드
    )

    print("  주요 설정:")
    print(f"    output_dir: {training_args.output_dir}")
    print(f"    num_train_epochs: {training_args.num_train_epochs}")
    print(f"    learning_rate: {training_args.learning_rate}")
    print(f"    batch_size: {training_args.per_device_train_batch_size}")
    print(f"    weight_decay: {training_args.weight_decay}")
    print(f"    warmup_steps: {training_args.warmup_steps}")
    print(f"    evaluation_strategy: {training_args.evaluation_strategy}")

    return training_args


def create_trainer(model, training_args, train_dataset, eval_dataset,
                   tokenizer, compute_metrics):
    """Trainer 객체 생성"""
    print("\n5. Trainer 객체 생성")
    print("-" * 40)

    # Data Collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    print("  Trainer 구성 요소:")
    print("    - model: AutoModelForSequenceClassification")
    print("    - args: TrainingArguments")
    print("    - train_dataset: 토큰화된 훈련 데이터")
    print("    - eval_dataset: 토큰화된 평가 데이터")
    print("    - data_collator: DataCollatorWithPadding")
    print("    - compute_metrics: 평가 메트릭 함수")

    return trainer


def train_and_evaluate(trainer):
    """모델 학습 및 평가"""
    print("\n6. 모델 학습 및 평가")
    print("-" * 40)

    print("\n학습 시작...")
    train_result = trainer.train()

    print(f"\n학습 완료!")
    print(f"  총 학습 스텝: {train_result.global_step}")
    print(f"  최종 학습 손실: {train_result.training_loss:.4f}")

    # 평가
    print("\n평가 결과:")
    eval_result = trainer.evaluate()
    for key, value in eval_result.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")

    return eval_result


def main():
    print("=" * 60)
    print("11장: Hugging Face Trainer API 기초")
    print("=" * 60)
    print()

    # 모델 및 토크나이저 로드
    model_name = "distilbert-base-uncased"
    print(f"모델: {model_name}")
    print("-" * 40)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  총 파라미터: {total_params:,}")
    print()

    # 1. 데이터셋 생성
    dataset = create_sample_dataset()

    # 2. 토큰화
    tokenized_dataset = tokenize_dataset(dataset, tokenizer)

    # 3. 메트릭 함수
    compute_metrics = define_compute_metrics()

    # 4. TrainingArguments
    training_args = setup_training_arguments()

    # 5. Trainer 생성
    trainer = create_trainer(
        model=model,
        training_args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    # 6. 학습 및 평가
    eval_result = train_and_evaluate(trainer)

    print("\n" + "=" * 60)
    print("실습 완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()
