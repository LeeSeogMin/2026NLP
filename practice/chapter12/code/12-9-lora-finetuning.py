"""
12-9-lora-finetuning.py
LoRA를 활용한 텍스트 분류 파인튜닝 실습

이 코드는 IMDb 감성분석 태스크에 LoRA를 적용하여
Full Fine-tuning과 성능을 비교한다.
"""

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import time

print("=" * 60)
print("LoRA 파인튜닝 실습: IMDb 감성분석")
print("=" * 60)

# GPU 확인
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"사용 디바이스: {device}")

# ============================================================
# 1. 데이터 준비
# ============================================================
print("\n[1] 데이터 준비")
print("-" * 50)

# IMDb 데이터셋 로드 (샘플링)
dataset = load_dataset("imdb")

# 학습 효율을 위해 샘플링
train_size = 500
eval_size = 200

train_dataset = dataset["train"].shuffle(seed=42).select(range(train_size))
eval_dataset = dataset["test"].shuffle(seed=42).select(range(eval_size))

print(f"학습 데이터: {len(train_dataset)} 샘플")
print(f"평가 데이터: {len(eval_dataset)} 샘플")

# 토크나이저 로드
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)


def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=256,
        padding=False
    )


# 토큰화
train_tokenized = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
eval_tokenized = eval_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


# 평가 함수
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="binary")
    return {"accuracy": acc, "f1": f1}


# ============================================================
# 2. Full Fine-tuning (비교 기준)
# ============================================================
print("\n[2] Full Fine-tuning (비교 기준)")
print("-" * 50)

# 모델 로드
full_model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2
)
full_model.to(device)

# 파라미터 수 확인
full_total_params = sum(p.numel() for p in full_model.parameters())
full_trainable = sum(p.numel() for p in full_model.parameters() if p.requires_grad)
print(f"전체 파라미터: {full_total_params:,}")
print(f"학습 가능 파라미터: {full_trainable:,} (100%)")

# 학습 설정
full_training_args = TrainingArguments(
    output_dir="./results_full",
    num_train_epochs=2,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    learning_rate=2e-5,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="no",
    logging_steps=50,
    report_to="none",
    fp16=False,
)

# Trainer 생성
full_trainer = Trainer(
    model=full_model,
    args=full_training_args,
    train_dataset=train_tokenized,
    eval_dataset=eval_tokenized,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# 학습
print("\nFull Fine-tuning 학습 시작...")
start_time = time.time()
full_trainer.train()
full_time = time.time() - start_time

# 평가
full_results = full_trainer.evaluate()
print(f"\nFull Fine-tuning 결과:")
print(f"  - 정확도: {full_results['eval_accuracy']*100:.2f}%")
print(f"  - F1 Score: {full_results['eval_f1']:.4f}")
print(f"  - 학습 시간: {full_time:.1f}초")

# 메모리 정리
del full_model, full_trainer
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# ============================================================
# 3. LoRA Fine-tuning
# ============================================================
print("\n[3] LoRA Fine-tuning")
print("-" * 50)

# 새 모델 로드
lora_base_model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2
)

# LoRA 설정
lora_config = LoraConfig(
    r=8,                          # 저랭크 차원
    lora_alpha=16,                # 스케일링 팩터
    target_modules=["q_lin", "v_lin"],  # Query, Value
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.SEQ_CLS
)

# LoRA 적용
lora_model = get_peft_model(lora_base_model, lora_config)
lora_model.to(device)

# 파라미터 수 확인
lora_total_params = sum(p.numel() for p in lora_model.parameters())
lora_trainable = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)

print("LoRA 설정:")
print(f"  - Rank: {lora_config.r}")
print(f"  - Alpha: {lora_config.lora_alpha}")
print(f"  - Target Modules: {lora_config.target_modules}")
print(f"\n파라미터:")
print(f"  - 전체: {lora_total_params:,}")
print(f"  - 학습 가능: {lora_trainable:,}")
print(f"  - 학습 비율: {lora_trainable/lora_total_params*100:.4f}%")
print(f"  - Full 대비 감소: {(1 - lora_trainable/full_trainable)*100:.2f}%")

# 학습 설정
lora_training_args = TrainingArguments(
    output_dir="./results_lora",
    num_train_epochs=2,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    learning_rate=3e-4,  # LoRA는 더 높은 학습률 사용
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="no",
    logging_steps=50,
    report_to="none",
    fp16=False,
)

# Trainer 생성
lora_trainer = Trainer(
    model=lora_model,
    args=lora_training_args,
    train_dataset=train_tokenized,
    eval_dataset=eval_tokenized,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# 학습
print("\nLoRA Fine-tuning 학습 시작...")
start_time = time.time()
lora_trainer.train()
lora_time = time.time() - start_time

# 평가
lora_results = lora_trainer.evaluate()
print(f"\nLoRA Fine-tuning 결과:")
print(f"  - 정확도: {lora_results['eval_accuracy']*100:.2f}%")
print(f"  - F1 Score: {lora_results['eval_f1']:.4f}")
print(f"  - 학습 시간: {lora_time:.1f}초")

# ============================================================
# 4. 다양한 Rank 실험
# ============================================================
print("\n[4] Rank 값에 따른 성능 비교")
print("-" * 50)

rank_results = []

for r in [4, 8, 16]:
    print(f"\nRank = {r} 실험...")

    # 새 모델
    temp_model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2
    )

    # LoRA 설정
    temp_config = LoraConfig(
        r=r,
        lora_alpha=r * 2,
        target_modules=["q_lin", "v_lin"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.SEQ_CLS
    )

    temp_peft = get_peft_model(temp_model, temp_config)
    temp_peft.to(device)

    trainable_p = sum(p.numel() for p in temp_peft.parameters() if p.requires_grad)

    # 학습
    temp_args = TrainingArguments(
        output_dir=f"./results_rank_{r}",
        num_train_epochs=2,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        learning_rate=3e-4,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=100,
        report_to="none",
        fp16=False,
    )

    temp_trainer = Trainer(
        model=temp_peft,
        args=temp_args,
        train_dataset=train_tokenized,
        eval_dataset=eval_tokenized,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    temp_trainer.train()
    temp_results = temp_trainer.evaluate()

    rank_results.append({
        "rank": r,
        "alpha": r * 2,
        "trainable_params": trainable_p,
        "accuracy": temp_results["eval_accuracy"],
        "f1": temp_results["eval_f1"]
    })

    # 정리
    del temp_model, temp_peft, temp_trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# 결과 출력
print("\n" + "=" * 70)
print("Rank별 성능 비교")
print("=" * 70)
print(f"{'Rank':>6} | {'Alpha':>6} | {'학습 파라미터':>15} | {'정확도':>10} | {'F1':>8}")
print("-" * 70)

for res in rank_results:
    print(f"{res['rank']:>6} | {res['alpha']:>6} | {res['trainable_params']:>15,} | {res['accuracy']*100:>9.2f}% | {res['f1']:>8.4f}")

# ============================================================
# 5. Full Fine-tuning vs LoRA 최종 비교
# ============================================================
print("\n[5] Full Fine-tuning vs LoRA 최종 비교")
print("-" * 50)

print(f"\n{'방법':^20} | {'학습 파라미터':^15} | {'정확도':^10} | {'학습 시간':^10}")
print("-" * 60)
print(f"{'Full Fine-tuning':^20} | {full_trainable:>15,} | {full_results['eval_accuracy']*100:>9.2f}% | {full_time:>9.1f}s")
print(f"{'LoRA (r=8)':^20} | {lora_trainable:>15,} | {lora_results['eval_accuracy']*100:>9.2f}% | {lora_time:>9.1f}s")

# 효율성 분석
param_reduction = (1 - lora_trainable / full_trainable) * 100
accuracy_diff = lora_results['eval_accuracy'] - full_results['eval_accuracy']
time_ratio = full_time / lora_time if lora_time > 0 else 0

print(f"\n효율성 분석:")
print(f"  - 파라미터 감소: {param_reduction:.2f}%")
print(f"  - 정확도 차이: {accuracy_diff*100:+.2f}%p")
print(f"  - 학습 속도 비율: {time_ratio:.2f}x")

# ============================================================
# 6. 핵심 요약
# ============================================================
print("\n" + "=" * 60)
print("핵심 요약")
print("=" * 60)
print(f"""
1. LoRA는 Full Fine-tuning 대비 {param_reduction:.1f}% 적은 파라미터로 학습

2. 성능:
   - Full Fine-tuning: {full_results['eval_accuracy']*100:.1f}% 정확도
   - LoRA (r=8): {lora_results['eval_accuracy']*100:.1f}% 정확도

3. LoRA 장점:
   - 메모리 사용량 대폭 감소
   - 태스크별 어댑터만 저장 (수 MB)
   - 베이스 모델 공유 가능
   - 카타스트로픽 포겟팅 방지

4. 권장 설정:
   - Rank: 8-16 (태스크 복잡도에 따라 조정)
   - Alpha: 2 * Rank
   - Target: Query, Value (최소) → 모든 Attention (최적)
""")

print("=" * 60)
print("LoRA 파인튜닝 실습 완료")
print("=" * 60)
