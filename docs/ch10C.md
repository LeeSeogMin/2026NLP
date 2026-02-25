# 제10장 C: PEFT와 LoRA/QLoRA — 모범 구현과 해설

> 이 문서는 B회차 과제 제출 후 공개된다. 제출 전에는 열람하지 않는다.

---

## 체크포인트 1 모범 구현: 4-bit 모델 로딩 및 메모리 확인

LoRA와 QLoRA의 핵심은 메모리 효율성이다. 먼저 Full Fine-tuning의 이론적 메모리 필요량을 계산하고, 실제 4-bit 로딩으로 얼마나 감소하는지 확인하는 것이 중요하다.

### Full Fine-tuning 메모리 계산 (이론)

```python
import torch
import math

def calculate_fft_memory_comprehensive(
    num_params: int,
    dtype_bits: int = 32,
    optimizer: str = "adam",
    gradient_checkpointing: bool = False
) -> dict:
    """
    Full Fine-tuning의 이론적 메모리 필요량 계산

    Args:
        num_params: 모델의 총 파라미터 수
        dtype_bits: 데이터 타입 비트 수 (32=float32, 16=float16)
        optimizer: 옵티마이저 타입 ("adam", "sgd", "adamw")
        gradient_checkpointing: Gradient Checkpointing 사용 여부

    Returns:
        메모리 분석 딕셔너리
    """

    # 바이트 단위 계산
    bytes_per_param = dtype_bits // 8

    # 1. 모델 가중치 메모리
    param_memory = num_params * bytes_per_param

    # 2. 옵티마이저 상태 메모리
    # Adam: momentum (m) + variance (v) + 선택사항: weight decay 복사
    if optimizer == "adam":
        optimizer_states = 2  # m, v
    elif optimizer == "adamw":
        optimizer_states = 2  # Adam과 동일
    elif optimizer == "sgd":
        optimizer_states = 0  # 상태 없음
    else:
        optimizer_states = 2

    optimizer_memory = num_params * bytes_per_param * optimizer_states

    # 3. 그래디언트 메모리
    gradient_memory = num_params * bytes_per_param

    # 4. 활성화(Activation) 메모리
    # 역전파를 위해 순전파 중 계산된 중간값들을 저장해야 함
    # Gradient Checkpointing을 사용하면 이를 재계산으로 대체 (계산 오버헤드 증가)
    activation_memory = num_params * bytes_per_param * 0.5
    if gradient_checkpointing:
        activation_memory *= 0.3  # 약 70% 감소, 계산 20% 증가

    # 5. 총 메모리
    total_memory = (
        param_memory +
        optimizer_memory +
        gradient_memory +
        activation_memory
    )

    return {
        'num_params': num_params,
        'param_memory_gb': param_memory / 1e9,
        'optimizer_memory_gb': optimizer_memory / 1e9,
        'gradient_memory_gb': gradient_memory / 1e9,
        'activation_memory_gb': activation_memory / 1e9,
        'total_memory_gb': total_memory / 1e9,
        'optimizer_type': optimizer,
        'gradient_checkpointing': gradient_checkpointing,
    }

# Llama 모델 파라미터 수
models = {
    'Llama-2-7B': 7 * 10**9,
    'Llama-2-13B': 13 * 10**9,
    'Llama-2-70B': 70 * 10**9,
}

print("=" * 90)
print("Full Fine-tuning 메모리 추정 (float32, Adam 옵티마이저)")
print("=" * 90)
print()

for model_name, num_params in models.items():
    calc = calculate_fft_memory_comprehensive(
        num_params, dtype_bits=32, optimizer="adam", gradient_checkpointing=False
    )

    print(f"모델: {model_name}")
    print(f"  파라미터 수: {calc['num_params']/1e9:.1f}B")
    print(f"  모델 가중치: {calc['param_memory_gb']:.1f} GB (float32)")
    print(f"  옵티마이저 상태: {calc['optimizer_memory_gb']:.1f} GB (Adam의 m, v)")
    print(f"  그래디언트: {calc['gradient_memory_gb']:.1f} GB")
    print(f"  활성화 값: {calc['activation_memory_gb']:.1f} GB")
    print(f"  총 메모리: {calc['total_memory_gb']:.1f} GB")

    # 실무 평가
    if calc['total_memory_gb'] > 1000:
        feasibility = "❌ 불가능 (대규모 클러스터 필수)"
    elif calc['total_memory_gb'] > 80:
        feasibility = "⚠️ 극도로 어려움 (다중 GPU 고사양)"
    elif calc['total_memory_gb'] > 48:
        feasibility = "⚠️ 어려움 (고사양 워크스테이션)"
    else:
        feasibility = "✓ 가능 (일반 워크스테이션)"

    print(f"  → {feasibility}")
    print()

print("=" * 90)
print()
```

**예상 출력**:
```
==========================================================================================
Full Fine-tuning 메모리 추정 (float32, Adam 옵티마이저)
==========================================================================================

모델: Llama-2-7B
  파라미터 수: 7.0B
  모델 가중치: 28.0 GB (float32)
  옵티마이저 상태: 56.0 GB (Adam의 m, v)
  그래디언트: 28.0 GB
  활성화 값: 14.0 GB
  총 메모리: 126.0 GB
  → ⚠️ 극도로 어려움 (다중 GPU 고사양)

모델: Llama-2-13B
  파라미터 수: 13.0B
  모델 가중치: 52.0 GB (float32)
  옵티마이저 상태: 104.0 GB (Adam의 m, v)
  그래디언트: 52.0 GB
  활성화 값: 26.0 GB
  총 메모리: 234.0 GB
  → ❌ 불가능 (대규모 클러스터 필수)

모델: Llama-2-70B
  파라미터 수: 70.0B
  모델 가중치: 280.0 GB (float32)
  옵티마이저 상태: 560.0 GB (Adam의 m, v)
  그래디언트: 280.0 GB
  활성화 값: 140.0 GB
  총 메모리: 1260.0 GB
  → ❌ 불가능 (대규모 클러스터 필수)
```

### 4-bit 양자화 로딩 및 실제 메모리 측정

```python
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import prepare_model_for_kbit_training

# 0. GPU 사전 확인
print("사전 GPU 메모리 확인:")
torch.cuda.empty_cache()  # 불필요한 메모리 해제
allocated_before = torch.cuda.memory_allocated() / 1e9
reserved_before = torch.cuda.memory_reserved() / 1e9
print(f"  - 할당됨: {allocated_before:.2f} GB")
print(f"  - 예약됨: {reserved_before:.2f} GB")
print()

# 1. BitsAndBytes 4-bit 설정
print("Step 1: BitsAndBytes 4-bit 양자화 설정")
print("-" * 60)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",                # NF4 양자화 (정규분포 기반)
    bnb_4bit_compute_dtype=torch.bfloat16,    # 연산 시 정밀도
    bnb_4bit_use_double_quant=True,           # Double Quantization 활성화
)

print(f"양자화 설정:")
print(f"  - 4-bit 로드: {bnb_config.load_in_4bit}")
print(f"  - 양자화 타입: {bnb_config.bnb_4bit_quant_type}")
print(f"  - 연산 정밀도: {bnb_config.bnb_4bit_compute_dtype}")
print(f"  - Double Quant: {bnb_config.bnb_4bit_use_double_quant}")
print()

# 2. 모델 로딩 (Llama 7B 사용, 실습 친화적)
print("Step 2: Llama 7B 모델 로딩 (4-bit)")
print("-" * 60)

model_name = "meta-llama/Llama-2-7b-hf"

print(f"모델 로딩 시작: {model_name}")
print("(첫 실행 시 약 3-5분 소요, Hugging Face 인증 필요)")
print()

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

print("✓ 모델 로딩 완료")
print()

# 3. 메모리 측정
print("Step 3: GPU 메모리 사용량 측정")
print("-" * 60)

def print_memory_stats(label: str):
    """GPU 메모리 통계 출력"""
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9

    print(f"{label}:")
    print(f"  - 할당됨 (allocated): {allocated:.2f} GB")
    print(f"  - 예약됨 (reserved): {reserved:.2f} GB")
    print(f"  - 총 용량: {total:.2f} GB")
    print(f"  - 남은 여유: {total - reserved:.2f} GB")

    return allocated, reserved

allocated_4bit, reserved_4bit = print_memory_stats("4-bit 로딩 후")
print()

# 4. 모델 정보 확인
print("Step 4: 모델 파라미터 정보")
print("-" * 60)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"전체 파라미터: {total_params:,} ({total_params/1e9:.2f}B)")
print(f"학습 가능 파라미터: {trainable_params:,}")
print(f"  → Full Fine-tuning에 필요한 그래디언트: {trainable_params * 4 / 1e9:.1f} GB (float32)")
print()

# 5. 메모리 절감 분석
print("Step 5: 메모리 절감 효과 분석")
print("-" * 60)

# Llama 7B float32 메모리
fft_calc = calculate_fft_memory_comprehensive(total_params, dtype_bits=32)
fft_total = fft_calc['total_memory_gb']

# 4-bit 메모리 (실제 측정)
quantized_memory = reserved_4bit

# 절감율
reduction_ratio = (1 - quantized_memory / fft_total) * 100
reduction_factor = fft_total / quantized_memory

print(f"Full Fine-tuning (float32): {fft_total:.1f} GB")
print(f"4-bit 양자화 (실제): {quantized_memory:.1f} GB")
print(f"메모리 절감: {reduction_ratio:.1f}%")
print(f"감소 배수: {reduction_factor:.1f}배")
print()

# 6. 토크나이저 로딩
print("Step 6: 토크나이저 로딩")
print("-" * 60)

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"✓ 토크나이저 로딩 완료")
print(f"  - 어휘 크기: {len(tokenizer)}")
print(f"  - Pad 토큰: {tokenizer.pad_token}")
print()

# 7. 요약 테이블
print("=" * 60)
print("메모리 비교 요약")
print("=" * 60)

summary_data = {
    'Full FT (float32)': fft_total,
    'float16': fft_total / 2,
    'int8': fft_total / 4,
    '4-bit (측정)': quantized_memory,
}

print(f"{'방법':<20} {'메모리':<15} {'상대값':<15}")
print("-" * 50)

for method, memory in summary_data.items():
    relative = memory / fft_total
    print(f"{method:<20} {memory:>6.1f} GB      {relative:>6.2f}×")

print()
```

**예상 출력**:
```
사전 GPU 메모리 확인:
  - 할당됨: 0.12 GB
  - 예약됨: 0.25 GB

Step 1: BitsAndBytes 4-bit 양자화 설정
------------------------------------------------------------
양자화 설정:
  - 4-bit 로드: True
  - 양자화 타입: nf4
  - 연산 정밀도: torch.bfloat16
  - Double Quant: True

Step 2: Llama 7B 모델 로딩 (4-bit)
------------------------------------------------------------
모델 로딩 시작: meta-llama/Llama-2-7b-hf
(첫 실행 시 약 3-5분 소요, Hugging Face 인증 필요)

✓ 모델 로딩 완료

Step 3: GPU 메모리 사용량 측정
------------------------------------------------------------
4-bit 로딩 후:
  - 할당됨 (allocated): 5.23 GB
  - 예약됨 (reserved): 5.67 GB
  - 총 용량: 48.00 GB
  - 남은 여유: 42.33 GB

Step 4: 모델 파라미터 정보
------------------------------------------------------------
전체 파라미터: 6,738,415,616 (6.74B)
학습 가능 파라미터: 6,738,415,616
  → Full Fine-tuning에 필요한 그래디언트: 27.0 GB (float32)

Step 5: 메모리 절감 효과 분석
------------------------------------------------------------
Full Fine-tuning (float32): 126.0 GB
4-bit 양자화 (실제): 5.67 GB
메모리 절감: 95.5%
감소 배수: 22.2배

메모리 비교 요약
============================================================
방법                  메모리            상대값
----------------------------------------------------
Full FT (float32)     126.0 GB          1.00×
float16                63.0 GB          0.50×
int8                   31.5 GB          0.25×
4-bit (측정)            5.67 GB         0.045×
```

### 핵심 이해 포인트

**4-bit 양자화의 작동 원리**:

1. **범위 매핑**: 원본 가중치(보통 -2~2 범위)를 4-bit 정수(0~15)로 변환
2. **NF4의 장점**: 정규분포 기반으로 자주 나타나는 값에 더 높은 해상도 할당
3. **Double Quantization**: 스케일값까지 양자화하여 추가 메모리 절감

**메모리 절감 효과**:
- float32 → float16: 2배 감소
- float32 → int8: 4배 감소
- float32 → int4: 8배 감소 (+ Double Quant로 추가 절감)

---

## 체크포인트 2 모범 구현: LoRA 설정 및 파인튜닝

이 단계에서는 4-bit 로딩된 모델에 LoRA를 적용하고, 실제 파인튜닝을 수행하며, 메모리 변화를 모니터링한다.

### LoRA 설정 (다양한 target_modules 비교)

```python
from peft import LoraConfig, get_peft_model
import pandas as pd

print("=" * 80)
print("Step 1: LoRA 설정 비교 (target_modules별)")
print("=" * 80)
print()

# 모델을 4-bit 학습을 위해 준비
model = prepare_model_for_kbit_training(model)

# 세 가지 LoRA 전략 정의
lora_strategies = {
    '전략 1: Conservative (Q, V만)': {
        'target_modules': ['q_proj', 'v_proj'],
        'description': 'Query와 Value만 적응 (가장 효율적)',
    },
    '전략 2: Balanced (Attention 전부)': {
        'target_modules': ['q_proj', 'k_proj', 'v_proj', 'out_proj'],
        'description': 'Attention 모든 프로젝션 (균형잡힌 선택)',
    },
    '전략 3: Comprehensive (Attention+FFN)': {
        'target_modules': ['q_proj', 'k_proj', 'v_proj', 'out_proj', 'up_proj', 'down_proj'],
        'description': 'Attention + FFN 가중 모두 (최대 표현력)',
    }
}

# Llama 7B의 아키텍처 파라미터
hidden_dim = 4096        # Self-Attention 차원
num_heads = 32
head_dim = hidden_dim // num_heads  # 128
ffn_dim = 11008          # Feed-Forward Network 중간 차원
num_layers = 32

rank = 16  # LoRA rank

strategies_info = {}

print(f"Llama-2-7B 아키텍처 정보:")
print(f"  - 은닉 차원 (hidden_dim): {hidden_dim}")
print(f"  - FFN 중간 차원: {ffn_dim}")
print(f"  - Transformer 층 수: {num_layers}")
print(f"  - LoRA rank: {rank}")
print()

for strategy_name, strategy_dict in lora_strategies.items():
    print(f"{strategy_name}")
    print(f"  설명: {strategy_dict['description']}")
    print(f"  Target Modules: {strategy_dict['target_modules']}")

    # LoRA 파라미터 수 계산
    # 각 모듈: input_dim × rank + rank × output_dim
    lora_params = 0
    module_details = []

    target_modules = strategy_dict['target_modules']

    for module in target_modules:
        if module in ['q_proj', 'k_proj', 'v_proj', 'out_proj']:
            # Self-Attention 프로젝션: hidden_dim × hidden_dim
            module_params = 2 * rank * hidden_dim  # AB의 합
            module_details.append(f"{module}: {module_params:,}")
        elif module in ['up_proj', 'down_proj']:
            # FFN 프로젝션
            module_params = 2 * rank * max(hidden_dim, ffn_dim)
            module_details.append(f"{module}: {module_params:,}")

    # 전체 LoRA 파라미터 (모든 층에 적용)
    total_lora_params = 0
    for module in target_modules:
        if module in ['q_proj', 'k_proj', 'v_proj', 'out_proj']:
            total_lora_params += 2 * rank * hidden_dim * num_layers
        elif module in ['up_proj', 'down_proj']:
            total_lora_params += 2 * rank * max(hidden_dim, ffn_dim) * num_layers

    strategies_info[strategy_name] = {
        'config': None,  # 나중에 할당
        'total_params': total_lora_params,
        'modules': target_modules,
    }

    print(f"  LoRA 파라미터: {total_lora_params:,} ({total_lora_params/1e6:.2f}M)")
    print(f"  메모리 추정: {total_lora_params * 4 / 1e9:.3f} GB (float32)")
    print()

# 실제 LoRA 설정 생성 및 적용 (Balanced 전략 선택)
print("=" * 80)
print("Step 2: 선택된 전략으로 LoRA 모델 변환")
print("=" * 80)
print()

selected_strategy = '전략 2: Balanced (Attention 전부)'
selected_modules = strategies_info[selected_strategy]['modules']

lora_config = LoraConfig(
    r=rank,
    lora_alpha=32,                          # rank의 2배 (관례)
    target_modules=selected_modules,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

print(f"선택된 전략: {selected_strategy}")
print(f"LoRA 설정:")
print(f"  - Rank (r): {lora_config.r}")
print(f"  - Alpha (스케일 인자): {lora_config.lora_alpha}")
print(f"  - Alpha / r: {lora_config.lora_alpha / lora_config.r}")
print(f"  - Target Modules: {lora_config.target_modules}")
print(f"  - Dropout: {lora_config.lora_dropout}")
print()

# LoRA 모델로 변환
print("모델에 LoRA 적용 중...")
model = get_peft_model(model, lora_config)
print()

print("✓ LoRA 적용 완료")
print()

# 학습 가능 파라미터 출력
print("Step 3: 학습 파라미터 통계")
print("-" * 80)

model.print_trainable_parameters()
print()

# 메모리 상태 확인
allocated_lora, reserved_lora = print_memory_stats("LoRA 적용 후")
print()

# LoRA 추가로 인한 메모리 증가
additional_memory = reserved_lora - reserved_4bit
print(f"4-bit 모델 대비 추가 메모리: {additional_memory:.3f} GB")
print()
```

**예상 출력**:
```
================================================================================
Step 1: LoRA 설정 비교 (target_modules별)
================================================================================

Llama-2-7B 아키텍처 정보:
  - 은닉 차원 (hidden_dim): 4096
  - FFN 중간 차원: 11008
  - Transformer 층 수: 32
  - LoRA rank: 16

전략 1: Conservative (Q, V만)
  설명: Query와 Value만 적응 (가장 효율적)
  Target Modules: ['q_proj', 'v_proj']
  LoRA 파라미터: 4,194,304 (4.19M)
  메모리 추정: 0.017 GB (float32)

전략 2: Balanced (Attention 전부)
  설명: Attention 모든 프로젝션 (균형잡힌 선택)
  Target Modules: ['q_proj', 'k_proj', 'v_proj', 'out_proj']
  LoRA 파라미터: 8,388,608 (8.39M)
  메모리 추정: 0.034 GB (float32)

전략 3: Comprehensive (Attention+FFN)
  설명: Attention + FFN 가중 모두 (최대 표현력)
  Target Modules: ['q_proj', 'k_proj', 'v_proj', 'out_proj', 'up_proj', 'down_proj']
  LoRA 파라미터: 50,331,648 (50.33M)
  메모리 추정: 0.201 GB (float32)

================================================================================
Step 2: 선택된 전략으로 LoRA 모델 변환
================================================================================

선택된 전략: 전략 2: Balanced (Attention 전부)
LoRA 설정:
  - Rank (r): 16
  - Alpha (스케일 인자): 32
  - Alpha / r: 2.0
  - Target Modules: ['q_proj', 'k_proj', 'v_proj', 'out_proj']
  - Dropout: 0.05

모델에 LoRA 적용 중...

✓ LoRA 적용 완료

Step 3: 학습 파라미터 통계
--------------------------------------------------------------------------------
trainable params: 8,388,608 || all params: 6,746,804,224 || trainable%: 0.1243

LoRA 적용 후:
  - 할당됨 (allocated): 5.36 GB
  - 예약됨 (reserved): 5.78 GB
  - 총 용량: 48.00 GB
  - 남은 여유: 42.22 GB

4-bit 모델 대비 추가 메모리: 0.11 GB
```

### 파인튜닝 실행

```python
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset
import os

print("=" * 80)
print("Step 4: 데이터셋 준비")
print("=" * 80)
print()

# Wikitext-2 데이터셋 로드 (실습용 소규모)
print("Wikitext-2 데이터셋 로딩 (1% 샘플)...")
dataset = load_dataset("wikitext", "wikitext-2", split="train[:1%]")
print(f"✓ 로딩 완료: {len(dataset)} 샘플")
print()

# 토크나이제이션 함수
def tokenize_function(examples):
    """텍스트를 토큰으로 변환"""
    outputs = tokenizer(
        examples["text"],
        truncation=True,
        max_length=512,
        padding="max_length",
    )
    outputs["labels"] = outputs["input_ids"].copy()  # 언어 모델링에서 labels = input_ids
    return outputs

print("토크나이제이션 진행 중...")
tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"],
    batch_size=100,
)
print(f"✓ 토크나이제이션 완료: {len(tokenized_dataset)} 샘플")
print(f"  샘플 구조: {list(tokenized_dataset[0].keys())}")
print()

# 학습 설정
print("=" * 80)
print("Step 5: 학습 설정")
print("=" * 80)
print()

# Gradient Checkpointing 활성화
model.gradient_checkpointing_enable()

training_args = TrainingArguments(
    output_dir="./llama-lora-finetuned",
    num_train_epochs=1,
    per_device_train_batch_size=4,           # 배치 크기
    gradient_accumulation_steps=2,           # 효과적 배치 = 8
    gradient_checkpointing=True,             # 메모리 절약
    learning_rate=5e-4,
    weight_decay=0.01,
    logging_steps=5,
    save_steps=25,
    save_total_limit=2,
    lr_scheduler_type="cosine",
    warmup_steps=10,
    bf16=True,                               # bfloat16 혼합 정밀도
    max_grad_norm=0.3,
    seed=42,
)

print(f"학습 설정:")
print(f"  - 배치 크기 (GPU당): {training_args.per_device_train_batch_size}")
print(f"  - 그래디언트 누적 단계: {training_args.gradient_accumulation_steps}")
print(f"  - 효과적 배치 크기: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
print(f"  - 학습률: {training_args.learning_rate}")
print(f"  - 학습 에포크: {training_args.num_train_epochs}")
print(f"  - 혼합 정밀도 (bfloat16): {training_args.bf16}")
print(f"  - Gradient Checkpointing: {training_args.gradient_checkpointing}")
print()

# 학습 시작 전 메모리 확인
print("학습 시작 전 메모리 상태:")
allocated_before_train, reserved_before_train = print_memory_stats("학습 준비 후")
print()

# 학습
print("=" * 80)
print("Step 6: 파인튜닝 실행")
print("=" * 80)
print()

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # Causal Language Modeling (다음 토큰 예측)
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

# 학습 실행
train_result = trainer.train()

print()
print("=" * 80)
print("학습 완료")
print("=" * 80)
print()

print(f"최종 손실: {train_result.training_loss:.4f}")
print()

# 학습 후 메모리 확인
print("학습 완료 후 메모리 상태:")
allocated_after_train, reserved_after_train = print_memory_stats("학습 후")
print()

# 메모리 피크 추적
print("메모리 사용 흐름:")
print(f"  초기 상태: {reserved_4bit:.2f} GB (4-bit 모델만)")
print(f"  LoRA 적용: {reserved_lora:.2f} GB (+{reserved_lora - reserved_4bit:.2f} GB)")
print(f"  학습 진행: {reserved_before_train:.2f} GB (최대)")
print(f"  학습 후: {reserved_after_train:.2f} GB")
print()
```

**예상 출력**:
```
================================================================================
Step 4: 데이터셋 준비
================================================================================

Wikitext-2 데이터셋 로딩 (1% 샘플)...
✓ 로딩 완료: 183 샘플

토크나이제이션 진행 중...
✓ 토크나이제이션 완료: 183 샘플
  샘플 구조: ['input_ids', 'attention_mask', 'labels']

================================================================================
Step 5: 학습 설정
================================================================================

학습 설정:
  - 배치 크기 (GPU당): 4
  - 그래디언트 누적 단계: 2
  - 효과적 배치 크기: 8
  - 학습률: 5e-04
  - 학습 에포크: 1
  - 혼합 정밀도 (bfloat16): True
  - Gradient Checkpointing: True

학습 시작 전 메모리 상태:
학습 준비 후:
  - 할당됨 (allocated): 6.45 GB
  - 예약됨 (reserved): 6.89 GB
  - 총 용량: 48.00 GB
  - 남은 여유: 41.11 GB

================================================================================
Step 6: 파인튜닝 실행
================================================================================

[1/1 00:23<00:00, 23.21it/s, loss=4.2341]

================================================================================
학습 완료
================================================================================

최종 손실: 4.2341

학습 완료 후 메모리 상태:
학습 후:
  - 할당됨 (allocated): 5.58 GB
  - 예약됨 (reserved): 5.82 GB
  - 총 용량: 48.00 GB
  - 남은 여유: 42.18 GB

메모리 사용 흐름:
  초기 상태: 5.67 GB (4-bit 모델만)
  LoRA 적용: 5.78 GB (+0.11 GB)
  학습 진행: 6.89 GB (최대)
  학습 후: 5.82 GB
```

### 핵심 이해 포인트

**LoRA의 목표와 메커니즘**:

1. **목표**: 모든 파라미터를 업데이트하지 않고, 작은 어댑터 AB로 충분한 표현력 확보
2. **초기값**: B를 0으로 초기화하여 초기 성능이 원본 모델과 같게 유지
3. **Rank 선택**: r=16은 대부분의 태스크에서 Full Fine-tuning 성능의 90% 이상 달성

**메모리 효율성 비교**:
- Full FT: 126GB (파라미터, 옵티마이저, 그래디언트 모두 필요)
- LoRA: 5.8GB (4-bit + LoRA 어댑터 + 옵티마이저 상태의 일부만)
- **메모리 절감: 95% 이상**

---

## 체크포인트 3 모범 구현: 메모리·성능 비교 및 추론

### 종합 메모리 비교 분석

```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

print("=" * 90)
print("Step 1: 메모리 비교 종합 분석")
print("=" * 90)
print()

# 메모리 및 성능 비교 데이터 (Llama 7B 기준)
comparison_data = {
    '방법': [
        'Full Fine-tuning (이론)',
        'Full FT + Gradient Ckpt',
        'LoRA (r=16)',
        'LoRA (r=16) + 4-bit',
        'QLoRA (r=16)',
    ],
    '모델 메모리 (GB)': [28, 28, 28, 5.7, 5.7],
    '옵티마이저 상태 (GB)': [56, 56, 0.4, 0.4, 0.4],
    '그래디언트 (GB)': [28, 28, 0.4, 0.4, 0.4],
    '활성화 값 (GB)': [14, 4, 0.4, 0.4, 0.4],
    '학습 가능 파라미터': ['100%', '100%', '0.12%', '0.12%', '0.12%'],
    '추론 레이턴시': ['+0%', '+0%', '+0%', '+2%', '+5%'],
}

df = pd.DataFrame(comparison_data)

# 총 메모리 계산
df['총 메모리 (GB)'] = (
    df['모델 메모리 (GB)'] +
    df['옵티마이저 상태 (GB)'] +
    df['그래디언트 (GB)'] +
    df['활성화 값 (GB)']
)

print("메모리 사용 비교표")
print("=" * 90)
print()

display_df = df[['방법', '모델 메모리 (GB)', '옵티마이저 상태 (GB)',
                  '그래디언트 (GB)', '활성화 값 (GB)', '총 메모리 (GB)']].copy()
print(display_df.to_string(index=False))
print()

print("성능 및 파라미터")
print("=" * 90)
print()

perf_df = df[['방법', '학습 가능 파라미터', '추론 레이턴시']].copy()
print(perf_df.to_string(index=False))
print()

# 메모리 절감율 계산
fft_memory = df.loc[0, '총 메모리 (GB)']
df['절감율 (%)'] = ((fft_memory - df['총 메모리 (GB)']) / fft_memory * 100).round(1)
df['감소 배수'] = (fft_memory / df['총 메모리 (GB)']).round(2)

print("메모리 절감 효과")
print("=" * 90)
print()

reduction_df = df[['방법', '총 메모리 (GB)', '절감율 (%)', '감소 배수']].copy()
print(reduction_df.to_string(index=False))
print()

# 실무 평가
print("실무 적용 가능성")
print("=" * 90)
print()

feasibility = {
    '8GB GPU': [],
    '24GB GPU': [],
    '48GB GPU': [],
}

for idx, row in df.iterrows():
    method = row['방법']
    memory = row['총 메모리 (GB)']

    for gpu_memory in [8, 24, 48]:
        if memory <= gpu_memory * 0.8:  # 80% 사용을 안전 한계로 설정
            feasibility[f'{gpu_memory}GB GPU'].append('✓')
        else:
            feasibility[f'{gpu_memory}GB GPU'].append('✗')

feasibility_df = pd.DataFrame({
    '방법': df['방법'],
    '8GB GPU': feasibility['8GB GPU'],
    '24GB GPU': feasibility['24GB GPU'],
    '48GB GPU': feasibility['48GB GPU'],
})

print(feasibility_df.to_string(index=False))
print()

# 시각화
print("=" * 90)
print("Step 2: 메모리 비교 시각화")
print("=" * 90)
print()

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. 메모리 구성 (누적 막대 그래프)
ax = axes[0, 0]
methods = df['방법'].tolist()
model_mem = df['모델 메모리 (GB)'].tolist()
opt_mem = df['옵티마이저 상태 (GB)'].tolist()
grad_mem = df['그래디언트 (GB)'].tolist()
act_mem = df['활성화 값 (GB)'].tolist()

x = np.arange(len(methods))
width = 0.6

bars1 = ax.bar(x, model_mem, width, label='Model Weights', color='#1f77b4')
bars2 = ax.bar(x, opt_mem, width, bottom=model_mem, label='Optimizer State', color='#ff7f0e')
bottom2 = np.array(model_mem) + np.array(opt_mem)
bars3 = ax.bar(x, grad_mem, width, bottom=bottom2, label='Gradients', color='#2ca02c')
bottom3 = bottom2 + np.array(grad_mem)
bars4 = ax.bar(x, act_mem, width, bottom=bottom3, label='Activations', color='#d62728')

ax.set_ylabel('메모리 (GB)', fontsize=11, fontweight='bold')
ax.set_title('메모리 구성 비교 (누적)', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(methods, rotation=30, ha='right', fontsize=9)
ax.legend(fontsize=10, loc='upper left')
ax.grid(axis='y', alpha=0.3)

# 2. 총 메모리 비교 (로그 스케일)
ax = axes[0, 1]
total_memories = df['총 메모리 (GB)'].tolist()
colors = ['#d62728' if m > 80 else '#ff7f0e' if m > 30 else '#2ca02c'
          for m in total_memories]

bars = ax.bar(methods, total_memories, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
ax.set_ylabel('총 메모리 (GB, 로그 스케일)', fontsize=11, fontweight='bold')
ax.set_title('총 메모리 비교', fontsize=12, fontweight='bold')
ax.set_yscale('log')
ax.set_xticklabels(methods, rotation=30, ha='right', fontsize=9)
ax.grid(axis='y', alpha=0.3, which='both')

# 값을 막대 위에 표시
for bar, val in zip(bars, total_memories):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
           f'{val:.1f}GB',
           ha='center', va='bottom', fontsize=10, fontweight='bold')

# 3. 메모리 절감율
ax = axes[1, 0]
reduction_rates = df['절감율 (%)'].tolist()
colors_reduction = ['#d62728', '#ff7f0e', '#ffc933', '#2ca02c', '#17a923']

bars = ax.bar(methods, reduction_rates, color=colors_reduction, alpha=0.8, edgecolor='black', linewidth=1.5)
ax.set_ylabel('Full FT 대비 절감율 (%)', fontsize=11, fontweight='bold')
ax.set_title('메모리 절감 효과', fontsize=12, fontweight='bold')
ax.set_ylim([0, 105])
ax.set_xticklabels(methods, rotation=30, ha='right', fontsize=9)
ax.grid(axis='y', alpha=0.3)
ax.axhline(y=100, color='red', linestyle='--', alpha=0.5, label='100% 절감')

# 값을 막대 위에 표시
for bar, val in zip(bars, reduction_rates):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
           f'{val:.0f}%',
           ha='center', va='bottom', fontsize=10, fontweight='bold')

# 4. GPU 적용 가능성 히트맵
ax = axes[1, 1]

feasibility_matrix = np.array([
    [0 if c == '✗' else 1 for c in feasibility_df.iloc[:, 1:].iloc[i, :]]
    for i in range(len(feasibility_df))
])

im = ax.imshow(feasibility_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
ax.set_xticks(np.arange(len(feasibility_df.columns[1:])))
ax.set_yticks(np.arange(len(feasibility_df)))
ax.set_xticklabels(feasibility_df.columns[1:], fontsize=10)
ax.set_yticklabels(feasibility_df['방법'], fontsize=9)

# 텍스트 추가
for i in range(len(feasibility_df)):
    for j in range(len(feasibility_df.columns[1:])):
        text = ax.text(j, i, feasibility_df.iloc[i, j+1],
                      ha="center", va="center", color="black", fontweight='bold', fontsize=12)

ax.set_title('GPU 메모리별 실행 가능성', fontsize=12, fontweight='bold')
plt.colorbar(im, ax=ax, label='가능', ticks=[0, 1])

plt.tight_layout()
plt.savefig('memory_comparison_comprehensive.png', dpi=150, bbox_inches='tight')
print("✓ 저장: memory_comparison_comprehensive.png")
plt.close()

print()
```

**예상 출력**:
```
==========================================================================================
Step 1: 메모리 비교 종합 분석
==========================================================================================

메모리 사용 비교표
==========================================================================================

                              방법 모델 메모리 (GB) 옵티마이저 상태 (GB)  그래디언트 (GB)  활성화 값 (GB)  총 메모리 (GB)
Full Fine-tuning (이론)              28.0              56.0          28.0            14.0         126.0
Full FT + Gradient Ckpt              28.0              56.0          28.0             4.0         116.0
          LoRA (r=16)                28.0               0.4           0.4             0.4          29.0
     LoRA (r=16) + 4-bit              5.7               0.4           0.4             0.4           6.9
         QLoRA (r=16)                 5.7               0.4           0.4             0.4           6.9

성능 및 파라미터
==========================================================================================

                         방법 학습 가능 파라미터 추론 레이턴시
Full Fine-tuning (이론)       100%          +0%
Full FT + Gradient Ckpt       100%          +0%
          LoRA (r=16)         0.12%         +0%
     LoRA (r=16) + 4-bit      0.12%         +2%
         QLoRA (r=16)         0.12%         +5%

메모리 절감 효과
==========================================================================================

                         방법 총 메모리 (GB)  절감율 (%)  감소 배수
Full Fine-tuning (이론)         126.0         0.0        1.00
Full FT + Gradient Ckpt         116.0         7.9        1.09
          LoRA (r=16)            29.0        77.0        4.34
     LoRA (r=16) + 4-bit          6.9        94.5       18.26
         QLoRA (r=16)             6.9        94.5       18.26

실무 적용 가능성
==========================================================================================

                         방법  8GB GPU  24GB GPU  48GB GPU
Full Fine-tuning (이론)        ✗         ✗        ✗
Full FT + Gradient Ckpt        ✗         ✗        ✗
          LoRA (r=16)          ✗         ✓        ✓
     LoRA (r=16) + 4-bit       ✓         ✓        ✓
         QLoRA (r=16)          ✓         ✓        ✓
```

### 파인튜닝 모델 저장 및 로딩

```python
from peft import AutoPeftModelForCausalLM

print("=" * 80)
print("Step 3: 파인튜닝 모델 저장 및 로딩")
print("=" * 80)
print()

# 모델 저장
output_dir = "./llama7b-lora-finetuned"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"모델 저장 완료: {output_dir}/")
print()

# 저장된 파일 확인
import os
total_size = 0
file_list = []

for root, dirs, files in os.walk(output_dir):
    for file in files:
        file_path = os.path.join(root, file)
        file_size = os.path.getsize(file_path)
        total_size += file_size

        if file_size > 1e5:  # 100KB 이상만 표시
            file_list.append((file, file_size / 1e6))

file_list.sort(key=lambda x: x[1], reverse=True)

print("저장된 파일 목록:")
print(f"{'파일명':<30} {'크기':<15}")
print("-" * 45)

for filename, filesize in file_list:
    print(f"{filename:<30} {filesize:>10.2f} MB")

print(f"\n총 저장 크기: {total_size / 1e6:.2f} MB")
print()

# 원본 모델 크기와 비교
original_model_size = 28 * 1e9 / 1e6  # float32 기준
print(f"원본 모델 크기 (float32): {original_model_size:.2f} MB")
print(f"저장 크기 (LoRA 어댑터만): {total_size / 1e6:.2f} MB")
print(f"저장 공간 절감: {(1 - total_size / (28e9 / 4)) * 100:.2f}%")  # float32를 4바이트로 나눔
print()

# 파인튜닝된 모델 로딩
print("=" * 80)
print("Step 4: 파인튜닝된 모델 로딩 및 추론")
print("=" * 80)
print()

print("파인튜닝 모델 로딩 중...")
inference_model = AutoPeftModelForCausalLM.from_pretrained(
    output_dir,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

print("✓ 파인튜닝 모델 로딩 완료")
print()

# 추론 모드 설정
inference_model.eval()

# 추론 함수
def generate_with_monitoring(prompt, max_new_tokens=100, temperature=0.7, top_p=0.9):
    """파인튜닝된 모델로 텍스트 생성 (메모리 모니터링)"""

    # 메모리 초기화
    torch.cuda.empty_cache()
    allocated_before = torch.cuda.memory_allocated() / 1e9

    # 입력 인코딩
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(inference_model.device)

    # 생성
    with torch.no_grad():
        output_ids = inference_model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    # 생성된 텍스트
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # 메모리 추적
    allocated_after = torch.cuda.memory_allocated() / 1e9
    memory_used = allocated_after - allocated_before

    return generated_text, memory_used

# 테스트 프롬프트
test_prompts = [
    "In the future, artificial intelligence will",
    "The key to learning is",
    "Deep learning models are",
]

print("추론 테스트 (파인튜닝 모델)")
print("=" * 80)
print()

for i, prompt in enumerate(test_prompts, 1):
    print(f"[예제 {i}]")
    print(f"프롬프트: {prompt}")

    generated_text, mem_used = generate_with_monitoring(prompt, max_new_tokens=50)

    print(f"생성 결과: {generated_text}")
    print(f"추가 메모리 사용: {mem_used:.3f} GB")
    print()

print("=" * 80)
print()
```

**예상 출력**:
```
================================================================================
Step 3: 파인튜닝 모델 저장 및 로딩
================================================================================

모델 저장 완료: ./llama7b-lora-finetuned/

저장된 파일 목록:
파일명                           크기
---------------------------------------------------------
adapter_model.bin                         49.45 MB
tokenizer_config.json                      0.70 MB
adapter_config.json                        0.00 MB
tokenizer.model                             0.49 MB
special_tokens_map.json                     0.00 MB

총 저장 크기: 50.64 MB

원본 모델 크기 (float32): 26952000.00 MB
저장 크기 (LoRA 어댑터만): 50.64 MB
저장 공간 절감: 99.81%

================================================================================
Step 4: 파인튜닝된 모델 로딩 및 추론
================================================================================

파인튜닝 모델 로딩 중...
✓ 파인튜닝 모델 로딩 완료

추론 테스트 (파인튜닝 모델)
================================================================================

[예제 1]
프롬프트: In the future, artificial intelligence will
생성 결과: In the future, artificial intelligence will help us solve many problems, such as climate change and disease. With AI's rapid development, we can expect a better world.
추가 메모리 사용: 0.342 GB

[예제 2]
프롬프트: The key to learning is
생성 결과: The key to learning is practice and persistence. Students should study regularly and ask questions when they don't understand. Teachers can provide guidance and support.
추가 메모리 사용: 0.338 GB

[예제 3]
프롬프트: Deep learning models are
생성 결과: Deep learning models are neural networks with many layers. They can learn complex patterns from data. Recent advances in GPU hardware have made training faster.
추가 메모리 사용: 0.339 GB
```

### 핵심 인사이트

**메모리 절감의 실제 효과**:

1. **4-bit 양자화**: 모델 메모리를 28GB → 5.7GB로 감소 (79.6% 절감)
2. **LoRA 어댑터**: 옵티마이저와 그래디언트 메모리를 56GB + 28GB → 0.4GB + 0.4GB로 감소 (99% 절감)
3. **합성 (QLoRA)**: 총 메모리를 126GB → 6.9GB로 감소 (94.5% 절감)

**배포 효율성**:

- 원본 모델 저장: 28GB (float32)
- LoRA 어댑터: 50MB
- 저장 공간 절감: 99.8%

이는 같은 기본 모델(7B, 13B, 70B)을 여러 태스크에 맞춰 파인튜닝할 때, 각 어댑터 50MB만 저장하면 되므로 대규모 서빙 시스템에서 매우 효율적이다.

**실무 적용 가능성**:

- **8GB GPU**: Full FT는 불가능하지만, QLoRA로는 7B 모델 파인튜닝 가능
- **24GB GPU**: LoRA + 4-bit으로 13B 모델까지 가능
- **48GB GPU**: QLoRA로 70B 모델까지 파인튜닝 가능

---

## 흔한 실수 및 해결법

### 실수 1: LoRA rank를 너무 크게 설정

```python
# ❌ 틀림: rank가 너무 크면 메모리 이득이 없다
lora_config = LoraConfig(r=128, ...)  # 메모리 이득 거의 없음

# ✓ 맞음: r=8~32 범위가 대부분 충분
lora_config = LoraConfig(r=16, ...)
```

**영향**: rank=128은 rank=16보다 8배 많은 파라미터를 사용하지만, 성능 향상은 2~3%에 불과하다.

### 실수 2: B를 0이 아닌 값으로 초기화

```python
# ❌ 틀림: 초기화 커스터마이징이 불필요
lora_config = LoraConfig(
    r=16,
    # ... 기본값이 B=0이므로 명시할 필요 없음
)

# 만약 직접 초기화한다면:
# torch.nn.init.normal_(lora_B, 0, 0.1)  # ✗ 이렇게 하면 불안정
```

**영향**: 초기 손실이 매우 크고 학습이 불안정해진다.

### 실수 3: target_modules를 모든 층에 적용

```python
# ❌ 과도한 설정: 모든 선형층에 LoRA 적용
target_modules = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2", "up_proj", "down_proj"]
# → 파라미터 50M 이상, 메모리 절감 효과 감소

# ✓ 균형잡힌 설정: Attention만 선택
target_modules = ["q_proj", "v_proj"]  # 또는 full attention
```

**영향**: 메모리 절감 효과가 줄어들고 학습 시간이 증가한다.

### 실수 4: Gradient Checkpointing 비활성화

```python
# ❌ 메모리 절약 기회 상실
model.gradient_checkpointing_enable()  # 호출 안 함

# ✓ 메모리 제약 상황에서 필수
model.gradient_checkpointing_enable()
```

**영향**: 활성화 값으로 인한 추가 메모리 사용. 계산은 20~30% 증가하지만 메모리는 30~50% 감소.

### 실수 5: 배치 크기를 너무 크게 설정

```python
# ❌ 8GB GPU에서는 배치 크기 4도 위험할 수 있음
training_args = TrainingArguments(
    per_device_train_batch_size=8,  # OOM 발생 가능
)

# ✓ 메모리 부족 시 감소
training_args = TrainingArguments(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,  # 효과적 배치 유지
)
```

**영향**: Out-of-Memory 에러로 학습이 중단된다.

---

## 다양한 모델에 대한 QLoRA 확장

### Llama 13B와 70B 적용 예시

```python
# Llama 13B (24GB GPU 권장)
model_13b = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-13b-hf",
    quantization_config=bnb_config,
    device_map="auto",
)

# 메모리: 약 9GB (4-bit)
# LoRA: 약 0.3GB 추가
# 총합: ~10GB (24GB GPU에서 여유 있음)

# Llama 70B (48GB GPU 권장)
model_70b = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-70b-hf",
    quantization_config=bnb_config,
    device_map="auto",
)

# 메모리: 약 40GB (4-bit)
# LoRA: 약 1.5GB 추가
# 총합: ~42GB (48GB GPU에서 최대 용량)
```

### 다양한 데이터셋 적용

```python
# 1. Instruction-Response 쌍 데이터셋
instruction_dataset = {
    "instruction": ["Explain quantum computing", "What is AI?"],
    "response": ["Quantum computing uses...", "AI is..."]
}

# 2. 요약 태스크
summary_dataset = {
    "text": ["A long document..."],
    "summary": ["Summarized version..."]
}

# 3. Q&A 태스크
qa_dataset = {
    "question": ["What is Python?"],
    "answer": ["Python is a programming language..."]
}
```

---

## 최종 학습 정리

### 10주차 핵심 개념 요약

1. **Full Fine-tuning의 한계**: 70B 모델은 1,260GB 메모리 필요 → 불가능
2. **PEFT**: Parameter-Efficient Fine-Tuning으로 메모리를 90~99% 절감
3. **LoRA**: 파라미터 변화를 저랭크 분해 AB로 표현 → 99.6% 파라미터 생략
4. **초기값**: A는 가우시안, B는 0으로 초기화 → 안정적 학습
5. **Rank 선택**: r=16이 효율성과 성능의 최적값
6. **양자화**: float32 → 4-bit로 메모리 1/8 감소
7. **NF4**: 정규분포 기반 4-bit 표현으로 정보 손실 최소화
8. **Double Quantization**: 스케일값까지 양자화 → 추가 절감
9. **QLoRA**: 양자화 + LoRA 결합 → 70B 모델을 8GB에서 파인튜닝
10. **저장 효율성**: LoRA 어댑터는 50MB만 저장 → 배포 간편

### 메모리 비교 (Llama 7B)

| 방법 | 메모리 | 절감 | 가능한 GPU |
|------|--------|------|-----------|
| Full FT | 126GB | 기준선 | ❌ |
| LoRA | 29GB | 77% | 24GB+ |
| QLoRA | 6.9GB | 95% | 8GB+ |

### 다음 단계로의 연결

11주차 (다음 회차)에서는:
- **Instruction Tuning**: 모델에 명령어를 따르도록 학습
- **RLHF (Reinforcement Learning from Human Feedback)**: 인간의 선호도로 모델 재학습
- **대규모 모델**: 13B, 70B 모델의 실전 파인튜닝

---

**생성일**: 2026-02-25
**대상 독자**: 컴퓨터공학/AI 전공 학부생 (3~4학년)
**난이도**: 중급 (파이썬, 딥러닝 기초, 9주차 Fine-tuning 선수)
