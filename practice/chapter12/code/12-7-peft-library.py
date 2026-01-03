"""
12-7-peft-library.py
Hugging Face PEFT 라이브러리 기본 사용법

이 코드는 PEFT 라이브러리를 활용하여 LoRA를 적용하는
기본적인 방법을 보여준다.
"""

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

print("=" * 60)
print("Hugging Face PEFT 라이브러리 기본 사용법")
print("=" * 60)

# ============================================================
# 1. 기본 모델 로드
# ============================================================
print("\n[1] 기본 모델 로드")
print("-" * 50)

model_name = "distilbert-base-uncased"
print(f"모델: {model_name}")

# 모델과 토크나이저 로드
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 원본 모델 파라미터 확인
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"전체 파라미터: {total_params:,}")
print(f"학습 가능 파라미터: {trainable_params:,}")

# ============================================================
# 2. LoraConfig 설정
# ============================================================
print("\n[2] LoraConfig 설정")
print("-" * 50)

# LoRA 설정
lora_config = LoraConfig(
    r=8,                          # 저랭크 차원
    lora_alpha=16,                # 스케일링 팩터
    target_modules=["q_lin", "v_lin"],  # DistilBERT의 Query, Value 프로젝션
    lora_dropout=0.1,             # 드롭아웃
    bias="none",                  # 바이어스 학습 안함
    task_type=TaskType.SEQ_CLS    # 시퀀스 분류
)

print("LoraConfig 파라미터:")
print(f"  - r (rank): {lora_config.r}")
print(f"  - lora_alpha: {lora_config.lora_alpha}")
print(f"  - target_modules: {lora_config.target_modules}")
print(f"  - lora_dropout: {lora_config.lora_dropout}")
print(f"  - bias: {lora_config.bias}")
print(f"  - task_type: {lora_config.task_type}")

# ============================================================
# 3. PEFT 모델 생성
# ============================================================
print("\n[3] PEFT 모델 생성 (get_peft_model)")
print("-" * 50)

# LoRA 적용
peft_model = get_peft_model(model, lora_config)

# 학습 가능 파라미터 출력
peft_model.print_trainable_parameters()

# 상세 파라미터 분석
total_after = sum(p.numel() for p in peft_model.parameters())
trainable_after = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
print(f"\n상세 분석:")
print(f"  - 전체 파라미터: {total_after:,}")
print(f"  - 학습 가능: {trainable_after:,}")
print(f"  - 동결됨: {total_after - trainable_after:,}")
print(f"  - 학습 비율: {trainable_after/total_after*100:.4f}%")

# ============================================================
# 4. 학습 가능 파라미터 확인
# ============================================================
print("\n[4] 학습 가능 파라미터 목록")
print("-" * 50)

print("학습 가능한 LoRA 파라미터:")
for name, param in peft_model.named_parameters():
    if param.requires_grad:
        print(f"  - {name}: {param.shape}")

# ============================================================
# 5. Forward Pass 테스트
# ============================================================
print("\n[5] Forward Pass 테스트")
print("-" * 50)

# 테스트 입력
test_text = "This is a great movie!"
inputs = tokenizer(test_text, return_tensors="pt", padding=True, truncation=True)

# 추론
peft_model.eval()
with torch.no_grad():
    outputs = peft_model(**inputs)
    logits = outputs.logits
    probs = torch.softmax(logits, dim=-1)

print(f"입력: '{test_text}'")
print(f"로짓: {logits[0].tolist()}")
print(f"확률: Negative={probs[0][0]:.4f}, Positive={probs[0][1]:.4f}")

# ============================================================
# 6. 모델 저장 및 로드
# ============================================================
print("\n[6] 모델 저장 및 로드")
print("-" * 50)

import os
import tempfile

# 임시 디렉토리에 저장
save_path = tempfile.mkdtemp()
adapter_path = os.path.join(save_path, "lora_adapter")

# LoRA 어댑터만 저장 (수 MB)
peft_model.save_pretrained(adapter_path)

# 저장된 파일 크기 확인
saved_files = os.listdir(adapter_path)
total_size = sum(os.path.getsize(os.path.join(adapter_path, f)) for f in saved_files)
print(f"저장 경로: {adapter_path}")
print(f"저장된 파일: {saved_files}")
print(f"총 크기: {total_size / 1024:.2f} KB")

# 새 베이스 모델에 어댑터 로드
print("\n어댑터 로드 테스트:")
base_model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=2
)
loaded_model = PeftModel.from_pretrained(base_model, adapter_path)
print("  - 어댑터 로드 성공!")

# ============================================================
# 7. 어댑터 병합 (Merge)
# ============================================================
print("\n[7] 어댑터 병합 (merge_and_unload)")
print("-" * 50)

print("병합 전:")
print(f"  - 모델 타입: {type(loaded_model).__name__}")

# 병합
merged_model = loaded_model.merge_and_unload()

print("병합 후:")
print(f"  - 모델 타입: {type(merged_model).__name__}")
print("  - LoRA 가중치가 베이스 모델에 병합됨")
print("  - 추론 시 추가 오버헤드 없음")

# 병합 후 동일 출력 확인
merged_model.eval()
with torch.no_grad():
    merged_outputs = merged_model(**inputs)
    merged_probs = torch.softmax(merged_outputs.logits, dim=-1)

print(f"\n병합 후 추론 결과: Negative={merged_probs[0][0]:.4f}, Positive={merged_probs[0][1]:.4f}")

# ============================================================
# 8. 다양한 Rank 설정 비교
# ============================================================
print("\n[8] Rank 설정별 파라미터 비교")
print("-" * 50)

print(f"{'Rank':>6} | {'Alpha':>6} | {'학습 파라미터':>15} | {'비율':>10}")
print("-" * 50)

for r in [4, 8, 16, 32]:
    alpha = r * 2  # 권장: alpha = 2 * r
    config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=["q_lin", "v_lin"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.SEQ_CLS
    )

    # 새 모델 생성
    temp_model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2
    )
    temp_peft = get_peft_model(temp_model, config)

    total_p = sum(p.numel() for p in temp_peft.parameters())
    trainable_p = sum(p.numel() for p in temp_peft.parameters() if p.requires_grad)
    ratio = trainable_p / total_p * 100

    print(f"{r:>6} | {alpha:>6} | {trainable_p:>15,} | {ratio:>9.4f}%")

    # 메모리 정리
    del temp_model, temp_peft

# 정리
import shutil
shutil.rmtree(save_path)

print("\n" + "=" * 60)
print("PEFT 라이브러리 기본 사용법 완료")
print("=" * 60)
