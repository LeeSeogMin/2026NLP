## 10주차 B회차: QLoRA 파인튜닝 및 성능 비교

> **미션**: PEFT 라이브러리로 LoRA를 적용하고, QLoRA로 대형 모델을 효율적으로 파인튜닝하며, Full FT와 성능/메모리를 비교할 수 있다

### 수업 타임라인

| 시간 | 내용 | Copilot 사용 |
|------|------|-------------|
| 00:00~00:05 | 조 편성 + 역할 배분 (조원 A/B) | 사용 안 함 |
| 00:05~00:10 | A회차 핵심 리캡 + 과제 스펙 재확인 | 사용 안 함 |
| 00:10~00:55 | 2인1조 Copilot 구현 (체크포인트 3회) | 적극 사용 |
| 00:55~01:00 | Google Classroom 제출 (조별 1부) | 사용 안 함 |
| 01:00~01:20 | 결과 토론 (메모리·성능 비교 분석) | 사용 안 함 |
| 01:20~01:28 | 핵심 정리 | 사용 안 함 |
| 01:28~01:30 | 다음 주 예고 | 사용 안 함 |

---

### A회차 핵심 리캡

**PEFT와 LoRA의 원리**:
- Full Fine-tuning은 70B 모델 기준 1,000GB 이상의 메모리가 필요하여 일반 워크스테이션에서는 불가능하다
- LoRA는 파라미터 변화 ΔW를 저랭크 분해 AB(r << min(m,n))로 표현하여 99.6% 파라미터를 생략한다
- B를 0으로 초기화하면 초기 ΔW=0이므로 안정적인 학습이 시작된다

**양자화의 가치**:
- float32를 4-bit로 압축하면 메모리가 1/8로 줄어든다
- NF4는 정규분포 기반의 4-bit 표현으로 정보 손실을 최소화한다
- Double Quantization은 스케일값까지 양자화하여 추가 메모리 절감을 달성한다

**QLoRA의 합성**:
- 양자화 + LoRA를 결합하면 70B 모델을 8GB GPU에서 파인튜닝할 수 있다
- 메모리 절감은 Full Fine-tuning 대비 60배 이상이다
- bitsandbytes 라이브러리를 사용하면 간단하게 4-bit 연산을 구현할 수 있다

**실습 연계**:
- 이론에서 배운 LoRA와 QLoRA를 PEFT 라이브러리로 직접 구현한다
- Llama 7B/13B 모델을 파인튜닝하고 메모리 사용량을 실시간 모니터링한다
- 다양한 target_modules 설정이 성능과 메모리에 미치는 영향을 실증적으로 확인한다

---

### 과제 스펙

**과제**: LoRA와 QLoRA를 PEFT로 구현 + 메모리·성능 비교 분석

**제출 형태**: 조별 1부, Google Classroom 업로드

**파일 구성**:
- 구현 코드 파일 (`*.py` 또는 `.ipynb`)
- 메모리 비교 그래프 (3개 이상)
- 분석 리포트 (2-3페이지)

**검증 기준**:
- ✓ Llama 모델을 4-bit로 로딩하고 메모리 확인
- ✓ LoRA 설정 (rank, target_modules, lora_dropout)
- ✓ 파인튜닝 실행 및 학습 로그 수집
- ✓ Full FT 메모리 계산, LoRA, QLoRA 메모리 비교
- ✓ 추론 코드 구현 및 모델 출력 예시

---

### 2인1조 실습

> **Copilot 활용**: 4-bit 모델 로딩 코드를 작성한 후, Copilot에게 "LoRA 설정 구성해줄래?", "학습 루프에서 메모리를 모니터링하는 코드 추가해줄래?", "Full FT와 LoRA의 메모리를 비교하는 표 만들어줄래?" 같이 단계적으로 요청한다. 각 단계에서 Copilot의 제안을 검토하고 수정하면서 QLoRA의 메커니즘을 깊이 있게 이해할 수 있다.

**역할 분담**:
- **조원 A (드라이버)**: 코드 작성 및 실행, 메모리 모니터링, 학습 진행
- **조원 B (네비게이터)**: 구성 검토, Copilot 프롬프트 설계, 수치 해석
- **체크포인트마다 역할 교대**: 두 명 모두 전체 구현을 이해하도록 진행

---

#### 체크포인트 1: 4-bit 모델 로딩 및 메모리 확인 (15분)

**목표**: bitsandbytes로 Llama 모델을 4-bit NF4로 로딩하고, Full Fine-tuning 메모리와 비교한다

**핵심 단계**:

① **필수 라이브러리 설치 및 버전 확인**

```python
import torch
import subprocess
import sys

# 라이브러리 설치 확인
required_packages = {
    'transformers': '4.36.0',
    'peft': '0.7.0',
    'bitsandbytes': '0.41.0',
    'datasets': '2.14.0',
    'accelerate': '0.24.0'
}

print("필수 라이브러리 설치 확인:")
for pkg, version in required_packages.items():
    try:
        module = __import__(pkg)
        installed_version = module.__version__
        print(f"  ✓ {pkg}: {installed_version}")
    except ImportError:
        print(f"  ✗ {pkg} 미설치. 설치 명령: pip install {pkg}=={version}")

# GPU 확인
print(f"\nGPU 정보:")
print(f"  - 이용 가능: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  - 장치명: {torch.cuda.get_device_name(0)}")
    print(f"  - 메모리: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
```

예상 출력:
```
필수 라이브러리 설치 확인:
  ✓ transformers: 4.36.2
  ✓ peft: 0.7.1
  ✓ bitsandbytes: 0.41.1
  ✓ datasets: 2.14.5
  ✓ accelerate: 0.24.1

GPU 정보:
  - 이용 가능: True
  - 장치명: NVIDIA A40
  - 메모리: 48.00 GB
```

② **Full Fine-tuning 메모리 계산**

```python
def calculate_fft_memory(model_params, num_layers=32):
    """Full Fine-tuning의 이론적 메모리 계산"""

    # Assumptions:
    # - float32 = 4 bytes per parameter
    # - Adam optimizer = 2x (momentum + variance)
    # - Gradient = 1x
    # - Activation storage (rough) = 0.5x

    param_bytes = model_params * 4  # float32
    optimizer_bytes = model_params * 4 * 2  # Adam (m, v)
    gradient_bytes = model_params * 4  # gradients
    activation_bytes = model_params * 4 * 0.5  # activations during backward

    total_bytes = param_bytes + optimizer_bytes + gradient_bytes + activation_bytes
    total_gb = total_bytes / 1e9

    return {
        'model_params': model_params,
        'param_memory_gb': param_bytes / 1e9,
        'optimizer_memory_gb': optimizer_bytes / 1e9,
        'gradient_memory_gb': gradient_bytes / 1e9,
        'activation_memory_gb': activation_bytes / 1e9,
        'total_memory_gb': total_gb
    }

# Llama 모델 파라미터 수
models = {
    'Llama-7B': 7e9,
    'Llama-13B': 13e9,
    'Llama-70B': 70e9
}

print("Full Fine-tuning 메모리 추정 (GPU 메모리):")
print("-" * 70)
print(f"{'모델':<15} {'파라미터':<15} {'총 메모리':<15} {'가능성':<10}")
print("-" * 70)

for model_name, params in models.items():
    calc = calculate_fft_memory(params)
    feasible = "❌" if calc['total_memory_gb'] > 80 else "⚠️" if calc['total_memory_gb'] > 24 else "✓"
    print(f"{model_name:<15} {params/1e9:.1f}B{'':<8} {calc['total_memory_gb']:.1f}GB{'':<8} {feasible:<10}")

print("-" * 70)
print("주: ✓=일반 워크스테이션, ⚠️=고사양 워크스테이션, ❌=대규모 클러스터 필요")
```

예상 출력:
```
Full Fine-tuning 메모리 추정 (GPU 메모리):
----------------------------------------------------------------------
모델              파라미터           총 메모리        가능성
----------------------------------------------------------------------
Llama-7B         7.0B             168GB        ❌
Llama-13B        13.0B            312GB        ❌
Llama-70B        70.0B            1680GB       ❌
----------------------------------------------------------------------
주: ✓=일반 워크스테이션, ⚠️=고사양 워크스테이션, ❌=대규모 클러스터 필요
```

③ **4-bit 모델 로딩 및 메모리 모니터링**

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training

# 4-bit 양자화 설정
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",           # NF4 양자화
    bnb_4bit_compute_dtype=torch.bfloat16,  # 연산 정밀도
    bnb_4bit_use_double_quant=True,      # Double Quantization
)

print("BitsAndBytes 4-bit 설정:")
print(f"  - 양자화 타입: {bnb_config.bnb_4bit_quant_type}")
print(f"  - 연산 정밀도: {bnb_config.bnb_4bit_compute_dtype}")
print(f"  - Double Quant: {bnb_config.bnb_4bit_use_double_quant}")
print()

# 모델 로딩 (Llama 7B 사용)
model_name = "meta-llama/Llama-2-7b-hf"  # 또는 7b-chat, 13b 등

print(f"모델 로딩 시작: {model_name}")
print("(첫 로딩 시 약 1-2분 소요)")

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

print("✓ 모델 로딩 완료")
print()

# 메모리 사용량 확인
def print_gpu_memory():
    """GPU 메모리 상태 출력"""
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9

    print(f"GPU 메모리 상태:")
    print(f"  - 할당됨: {allocated:.2f} GB")
    print(f"  - 예약됨: {reserved:.2f} GB")
    print(f"  - 총 용량: {total:.2f} GB")
    print(f"  - 남은 여유: {total - reserved:.2f} GB")
    return allocated, reserved

print_gpu_memory()
print()

# 모델 정보
print(f"모델 파라미터 정보:")
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"  - 전체 파라미터: {total_params:,}")
print(f"  - 학습 가능 파라미터: {trainable_params:,}")
print(f"  - 학습 비율: {100 * trainable_params / total_params:.4f}%")
print()

# 토크나이저 로딩
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("✓ 토크나이저 로딩 완료")
```

예상 출력:
```
BitsAndBytes 4-bit 설정:
  - 양자화 타입: nf4
  - 연산 정밀도: torch.bfloat16
  - Double Quant: True

모델 로딩 시작: meta-llama/Llama-2-7b-hf
(첫 로딩 시 약 1-2분 소요)
✓ 모델 로딩 완료

GPU 메모리 상태:
  - 할당됨: 5.23 GB
  - 예약됨: 5.45 GB
  - 총 용량: 48.00 GB
  - 남은 여유: 42.55 GB

모델 파라미터 정보:
  - 전체 파라미터: 6,738,415,616
  - 학습 가능 파라미터: 6,738,415,616
  - 학습 비율: 100.0000%

✓ 토크나이저 로딩 완료
```

**검증 체크리스트**:
- [ ] BitsAndBytes 설정에서 NF4와 Double Quantization이 활성화되었는가?
- [ ] 모델이 정상적으로 로딩되었는가? (로딩 실패 시 메모리 부족 또는 인터넷 연결 확인)
- [ ] GPU 메모리가 현저히 증가했는가? (float32 280GB 대비 1/8 수준 약 35GB 이상)
- [ ] 토크나이저의 pad_token이 설정되었는가?

**Copilot 프롬프트 1**:
```
"BitsAndBytesConfig를 사용해서 Llama-2-7b-hf를 4-bit NF4로 로딩해줄래?
device_map='auto'도 포함하고, 로딩 후 GPU 메모리를 확인하는 코드도 붙여줘."
```

**Copilot 프롬프트 2**:
```
"Full Fine-tuning이 필요한 GPU 메모리를 계산하는 함수를 짜줄래?
파라미터 수, optimizer 상태, 그래디언트, activation까지 고려해서."
```

---

#### 체크포인트 2: LoRA 설정 및 파인튜닝 (15분)

**목표**: PEFT로 LoRA를 구성하고, 작은 데이터셋으로 파인튜닝을 실행하며 메모리를 모니터링한다

**핵심 단계**:

① **LoRA 설정 (다양한 target_modules 비교)**

```python
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# 모델을 4-bit 학습을 위해 준비
model = prepare_model_for_kbit_training(model)

print("LoRA 설정 비교 (target_modules별):")
print("-" * 80)

# 3가지 설정: Conservative, Balanced, Comprehensive
lora_configs = {
    'Conservative (q, v만)': {
        'target_modules': ['q_proj', 'v_proj']
    },
    'Balanced (Attention 전부)': {
        'target_modules': ['q_proj', 'k_proj', 'v_proj', 'out_proj']
    },
    'Comprehensive (Attention+FFN)': {
        'target_modules': ['q_proj', 'k_proj', 'v_proj', 'out_proj', 'up_proj', 'down_proj']
    }
}

configs_info = {}

for config_name, config_dict in lora_configs.items():
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=config_dict['target_modules'],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # 파라미터 수 계산 (이론적)
    # 각 target module마다: r * (input_dim + output_dim)
    # Llama 7B의 숨은 차원 = 4096, FFN 중간 차원 = 11008

    hidden_dim = 4096
    ffn_dim = 11008
    num_layers = 32

    if 'q_proj' in config_dict['target_modules']:
        attn_params = 4 * 2 * 16 * hidden_dim  # q, k, v, out
    else:
        attn_params = 0

    if 'up_proj' in config_dict['target_modules']:
        ffn_params = 2 * 2 * 16 * max(hidden_dim, ffn_dim)
    else:
        ffn_params = 0

    total_lora_params = (attn_params + ffn_params) * num_layers

    configs_info[config_name] = {
        'config': lora_config,
        'target_modules': config_dict['target_modules'],
        'estimated_params': total_lora_params
    }

    print(f"{config_name}:")
    print(f"  - Target modules: {config_dict['target_modules']}")
    print(f"  - 추정 파라미터: {total_lora_params:,} ({total_lora_params/1e6:.2f}M)")
    print()

# 실제 사용할 설정: Balanced (좋은 성능-효율 트레이드오프)
selected_config_name = 'Balanced (Attention 전부)'
lora_config = configs_info[selected_config_name]['config']

print(f"선택된 설정: {selected_config_name}")
print(f"  - Rank: {lora_config.r}")
print(f"  - Alpha: {lora_config.lora_alpha}")
print(f"  - Dropout: {lora_config.lora_dropout}")
```

예상 출력:
```
LoRA 설정 비교 (target_modules별):
--------------------------------------------------------------------------------
Conservative (q, v만):
  - Target modules: ['q_proj', 'v_proj']
  - 추정 파라미터: 4,194,304 (4.19M)

Balanced (Attention 전부):
  - Target modules: ['q_proj', 'k_proj', 'v_proj', 'out_proj']
  - 추정 파라미터: 8,388,608 (8.39M)

Comprehensive (Attention+FFN):
  - Target modules: ['q_proj', 'k_proj', 'v_proj', 'out_proj', 'up_proj', 'down_proj']
  - 추정 파라미터: 50,331,648 (50.33M)

선택된 설정: Balanced (Attention 전부)
  - Rank: 16
  - Alpha: 32
  - Dropout: 0.05
```

② **LoRA 모델 생성 및 학습 설정**

```python
# LoRA 모델로 변환
model = get_peft_model(model, lora_config)
print("LoRA 적용 후 모델 정보:")
model.print_trainable_parameters()
print()

# Gradient Checkpointing 활성화 (메모리 절약)
model.gradient_checkpointing_enable()

# 학습 설정
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling

training_args = TrainingArguments(
    output_dir="./llama-lora-finetuned",
    num_train_epochs=1,
    per_device_train_batch_size=4,      # 배치 크기
    gradient_accumulation_steps=2,       # 그래디언트 누적
    gradient_checkpointing=True,         # 메모리 절약
    learning_rate=5e-4,
    weight_decay=0.01,
    logging_steps=5,
    save_steps=25,
    save_total_limit=2,
    lr_scheduler_type="cosine",
    warmup_steps=10,
    bf16=True,                          # bfloat16 혼합 정밀도
    max_grad_norm=0.3,
    seed=42,
)

print("학습 설정:")
print(f"  - 배치 크기 (GPU당): {training_args.per_device_train_batch_size}")
print(f"  - 그래디언트 누적: {training_args.gradient_accumulation_steps}")
print(f"  - 효과적 배치: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
print(f"  - 학습률: {training_args.learning_rate}")
print(f"  - Gradient Checkpointing: {training_args.gradient_checkpointing}")
print(f"  - 혼합 정밀도: {training_args.bf16}")
```

예상 출력:
```
trainable params: 8,388,608 || all params: 6,746,804,224 || trainable%: 0.1243

학습 설정:
  - 배치 크기 (GPU당): 4
  - 그래디언트 누적: 2
  - 효과적 배치: 8
  - 학습률: 5e-04
  - Gradient Checkpointing: True
  - 혼합 정밀도: True
```

③ **미니 데이터셋 준비 및 학습 실행**

```python
from datasets import load_dataset
import numpy as np

# Wikitext-2 데이터셋 로딩 (작은 샘플 사용)
print("데이터셋 로딩 시작...")
dataset = load_dataset("wikitext", "wikitext-2", split="train[:1%]")  # 1% 샘플
print(f"데이터셋 크기: {len(dataset)} 샘플")

# 토크나이제이션
def tokenize_function(examples):
    outputs = tokenizer(
        examples["text"],
        truncation=True,
        max_length=512,
        padding="max_length"
    )
    outputs["labels"] = outputs["input_ids"].copy()
    return outputs

print("토크나이제이션 진행 중...")
tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"],
    batch_size=100
)

print(f"토크나이제이션 완료: {len(tokenized_dataset)} 샘플")
print(f"샘플 구조: {tokenized_dataset[0].keys()}")
print()

# 데이터 콜레이터
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# 메모리 상태 (학습 전)
print("학습 시작 전 GPU 메모리:")
print_gpu_memory()
print()

# Trainer 초기화 및 학습
print("Trainer 초기화 및 학습 시작...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

# 학습 시작
train_result = trainer.train()

print("\n학습 완료")
print(f"최종 손실: {train_result.training_loss:.4f}")

# 메모리 상태 (학습 후)
print("학습 완료 후 GPU 메모리:")
print_gpu_memory()
```

예상 출력:
```
데이터셋 로딩 시작...
데이터셋 크기: 183 샘플

토크나이제이션 진행 중...
토크나이제이션 완료: 183 샘플
샘플 구조: dict_keys(['input_ids', 'attention_mask', 'labels'])

학습 시작 전 GPU 메모리:
GPU 메모리 상태:
  - 할당됨: 7.45 GB
  - 예약됨: 7.82 GB
  - 총 용량: 48.00 GB
  - 남은 여유: 40.18 GB

[1/1 00:15<00:00, 15.30it/s, loss=4.2341]

학습 완료
최종 손실: 4.2341

학습 완료 후 GPU 메모리:
GPU 메모리 상태:
  - 할당됨: 5.67 GB
  - 예약됨: 5.89 GB
  - 총 용량: 48.00 GB
  - 남은 여유: 42.11 GB
```

**검증 체크리스트**:
- [ ] LoRA 설정의 target_modules이 올바르게 설정되었는가?
- [ ] LoRA 모델 변환 후 학습 가능 파라미터가 전체의 0.1~0.2% 수준인가?
- [ ] 데이터셋이 정상적으로 로딩되고 토크나이제이션되었는가?
- [ ] 학습이 정상적으로 진행되었는가? (손실이 감소 추세)
- [ ] 메모리 사용량이 원래 4-bit 로딩 상태에서 약간만 증가했는가?

**Copilot 프롬프트 3**:
```
"PEFT의 LoraConfig를 만들어줄래? r=16, lora_alpha=32,
target_modules=['q_proj', 'k_proj', 'v_proj', 'out_proj']로 설정하고,
get_peft_model로 적용해줘."
```

**Copilot 프롬프트 4**:
```
"Wikitext-2 데이터셋을 로딩해서 토크나이저로 처리한 후,
TrainingArguments와 Trainer로 파인튜닝 루프를 만들어줄래?
배치 크기는 4, 그래디언트 누적은 2로."
```

---

#### 체크포인트 3: 메모리·성능 비교 및 분석 (15분)

**목표**: Full FT, LoRA, QLoRA의 메모리/성능을 정량적으로 비교하고, 추론 테스트를 수행한다

**핵심 단계**:

① **메모리 비교 표 및 시각화**

```python
import pandas as pd
import matplotlib.pyplot as plt

# 메모리 및 성능 비교 데이터
comparison_data = {
    '방법': ['Full Fine-tuning (이론)', 'LoRA (r=16)', 'QLoRA (r=16)'],
    '모델 메모리 (GB)': [280, 280, 35],
    '옵티마이저 상태 (GB)': [560, 10, 1.5],
    '그래디언트 (GB)': [280, 10, 1.5],
    '기타 (GB)': [400, 5, 1],
    '총 메모리 (GB)': [1520, 305, 39],
    '학습 가능 파라미터': ['100%', '0.12%', '0.12%'],
    '학습 시간 (상대)': ['100×', '3×', '5×'],
    '모델 저장 (MB)': [280000, 50, 50],
    '추론 레이턴시': ['기준선', '+0%', '+5%']
}

df = pd.DataFrame(comparison_data)

print("Full FT vs LoRA vs QLoRA 비교 (Llama-70B 기준)")
print("=" * 100)
print(df.to_string(index=False))
print("=" * 100)
print()

# 메모리 비교 시각화 (막대 그래프)
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 1. 메모리 구성 비교 (누적 막대)
methods = ['Full FT', 'LoRA', 'QLoRA']
model_mem = [280, 280, 35]
opt_mem = [560, 10, 1.5]
grad_mem = [280, 10, 1.5]
other_mem = [400, 5, 1]

x = range(len(methods))
ax = axes[0]
ax.bar(x, model_mem, label='Model Weights', color='#1f77b4')
ax.bar(x, opt_mem, bottom=model_mem, label='Optimizer State', color='#ff7f0e')
bottom1 = [m+o for m, o in zip(model_mem, opt_mem)]
ax.bar(x, grad_mem, bottom=bottom1, label='Gradients', color='#2ca02c')
bottom2 = [m+o+g for m, o, g in zip(model_mem, opt_mem, grad_mem)]
ax.bar(x, other_mem, bottom=bottom2, label='Others', color='#d62728')

ax.set_ylabel('메모리 (GB)', fontsize=11)
ax.set_title('메모리 사용량 구성', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(methods)
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3)

# 2. 총 메모리 비교
ax = axes[1]
total_mem = [1520, 305, 39]
colors = ['#d62728', '#ff7f0e', '#2ca02c']
bars = ax.bar(methods, total_mem, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
ax.set_ylabel('총 메모리 (GB)', fontsize=11)
ax.set_title('총 메모리 비교', fontsize=12, fontweight='bold')
ax.set_yscale('log')
ax.grid(axis='y', alpha=0.3)

# 값을 막대 위에 표시
for bar, val in zip(bars, total_mem):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
           f'{int(val)}GB',
           ha='center', va='bottom', fontsize=10, fontweight='bold')

# 3. 학습 가능 파라미터 비율
ax = axes[2]
trainable_pct = [100, 0.12, 0.12]
colors = ['#d62728', '#ff7f0e', '#2ca02c']
bars = ax.bar(methods, trainable_pct, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
ax.set_ylabel('학습 가능 파라미터 (%)', fontsize=11)
ax.set_title('학습 가능 파라미터 비율', fontsize=12, fontweight='bold')
ax.set_yscale('log')
ax.grid(axis='y', alpha=0.3)

# 값을 막대 위에 표시
for bar, val in zip(bars, trainable_pct):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
           f'{val:.2f}%',
           ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('memory_comparison.png', dpi=150, bbox_inches='tight')
print("저장: memory_comparison.png")
plt.close()

# 메모리 절감률 계산
print("메모리 절감율 (Full FT 대비):")
print(f"  - LoRA: {(1 - 305/1520) * 100:.1f}% 절감 (5배 감소)")
print(f"  - QLoRA: {(1 - 39/1520) * 100:.1f}% 절감 (39배 감소)")
print()

# 효율성 분석
print("효율성 분석:")
print(f"  - Full FT 메모리-파라미터 비율: {1520 / 100:.1f} GB per 1% trainable")
print(f"  - LoRA 메모리-파라미터 비율: {305 / 0.12:.1f} GB per 1% trainable")
print(f"  - QLoRA 메모리-파라미터 비율: {39 / 0.12:.1f} GB per 1% trainable")
```

예상 출력:
```
Full FT vs LoRA vs QLoRA 비교 (Llama-70B 기준)
====================================================================================================
     방법 모델 메모리 (GB)  옵티마이저 상태 (GB)  그래디언트 (GB)  기타 (GB)  총 메모리 (GB)  학습 가능 파라미터  학습 시간 (상대)  모델 저장 (MB)
Full Fine-tuning (이론)      280              560           280      400        1520              100%          100×       280000
               LoRA (r=16)      280               10            10        5         305             0.12%            3×           50
              QLoRA (r=16)       35              1.5           1.5        1          39             0.12%            5×           50
====================================================================================================

저장: memory_comparison.png

메모리 절감율 (Full FT 대비):
  - LoRA: 79.9% 절감 (5배 감소)
  - QLoRA: 97.4% 절감 (39배 감소)

효율성 분석:
  - Full FT 메모리-파라미터 비율: 15.2 GB per 1% trainable
  - LoRA 메모리-파라미터 비율: 2541.67 GB per 1% trainable
  - QLoRA 메모리-파라미터 비율: 325 GB per 1% trainable
```

② **파인튜닝 모델 저장 및 로딩**

```python
# LoRA 가중치 저장 (원본 모델은 로드하지 않음)
output_dir = "./llama-lora-finetuned"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"모델 저장 완료: {output_dir}")

# 저장된 파일 확인
import os
total_size = 0
for root, dirs, files in os.walk(output_dir):
    for file in files:
        file_path = os.path.join(root, file)
        file_size = os.path.getsize(file_path)
        total_size += file_size
        if file_size > 1e6:  # 1MB 이상만 표시
            print(f"  - {file}: {file_size / 1e6:.2f} MB")

print(f"총 저장 크기: {total_size / 1e6:.2f} MB")
print()

# 파인튜닝된 모델 로딩 (inference)
from peft import AutoPeftModelForCausalLM

print("파인튜닝된 모델 로딩...")
inference_model = AutoPeftModelForCausalLM.from_pretrained(
    output_dir,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

print("✓ 파인튜닝 모델 로딩 완료")
```

③ **추론 테스트 및 성능 확인**

```python
# 추론 설정
inference_model.eval()

def generate_text(prompt, max_new_tokens=100):
    """파인튜닝된 모델으로 텍스트 생성"""
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(inference_model.device)

    with torch.no_grad():
        output_ids = inference_model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )

    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return generated_text

# 테스트 프롬프트
test_prompts = [
    "Natural language processing is",
    "Deep learning models can",
    "The transformer architecture"
]

print("추론 테스트 (파인튜닝 모델):")
print("=" * 70)

for prompt in test_prompts:
    generated = generate_text(prompt, max_new_tokens=50)
    print(f"프롬프트: {prompt}")
    print(f"생성 결과: {generated}")
    print("-" * 70)

print()
print("성능 분석:")
print(f"  - 모델 크기: {total_params / 1e9:.1f}B 파라미터")
print(f"  - 학습 가능 파라미터: {8.39:.2f}M (0.12%)")
print(f"  - 저장 크기: {total_size / 1e6:.2f} MB (원본 대비 0.001%)")
print(f"  - 메모리 사용: ~39GB (4-bit + LoRA)")
print(f"  - 추론 속도: 원본 모델과 동등 (+5% 오버헤드)")
```

**검증 체크리스트**:
- [ ] 메모리 비교 표와 그래프가 생성되었는가?
- [ ] LoRA와 QLoRA의 메모리 절감율이 50% 이상인가?
- [ ] 파인튜닝된 모델이 저장되었는가? (MB 단위)
- [ ] 원본 모델 로드 없이 LoRA 어댑터만으로 추론이 가능한가?
- [ ] 생성된 텍스트가 의미 있는 내용인가?

**Copilot 프롬프트 5**:
```
"Full FT, LoRA, QLoRA의 메모리 사용을 pandas DataFrame으로 정리하고,
matplotlib으로 누적 막대 그래프를 그려줄래?
메모리 구성(모델, 옵티마이저, 그래디언트, 기타)을 각각 다른 색으로 표시해줘."
```

**Copilot 프롬프트 6**:
```
"AutoPeftModelForCausalLM을 사용해서 저장된 LoRA 어댑터를 로딩하고,
generate 함수로 텍스트를 생성해줄래?
프롬프트 3개를 테스트해서 결과를 출력해줘."
```

---

### 제출 안내 (Google Classroom)

**제출 방법**:
- Google Classroom의 "10주차 B회차" 과제에 조별 1부 제출
- 파일명 형식: `group_{조번호}_ch10B.zip` (또는 `.tar.gz`)

**포함할 파일**:
```
group_{조번호}_ch10B/
├── ch10B_qlora_finetuning.py      # 전체 구현 코드 (또는 .ipynb)
├── memory_comparison.png           # 메모리 비교 그래프 (막대 3개)
├── training_loss_curve.png         # 학습 곡선 (손실 vs 스텝)
├── inference_examples.txt          # 추론 결과 예시 (텍스트)
├── llama-lora-finetuned/           # 저장된 LoRA 어댑터 폴더
│   ├── adapter_config.json
│   ├── adapter_model.bin
│   └── tokenizer.model
└── report.md                       # 분석 리포트 (2-3페이지)
```

**리포트 포함 항목** (report.md):
- **체크포인트별 구현 과정** (각 2-3문장)
  - CP1: 4-bit 로딩 및 메모리 확인 결과
  - CP2: LoRA 설정 선택 이유 및 파인튜닝 진행 상황
  - CP3: 메모리/성능 비교 분석

- **메모리 비교 분석** (3-4문장)
  - Full FT vs LoRA vs QLoRA 메모리 차이
  - 어느 방법이 가장 효율적인가? (메모리-성능 관점)
  - 실습 환경(8GB, 24GB, 48GB GPU)에서 각 방법의 가능성

- **target_modules 선택의 영향** (2-3문장)
  - Conservative vs Balanced vs Comprehensive 중 선택 이유
  - 파라미터 수와 성능 간 트레이드오프 분석
  - 실무에서의 권장 설정

- **LoRA 초기값의 중요성** (2-3문장)
  - B=0 초기화가 중요한 이유 (실제 경험 포함)
  - 만약 B를 무작위로 초기화했다면 어떻게 달랐을까?

- **Copilot 활용 경험** (2-3문장)
  - 어떤 프롬프트가 가장 효과적이었는가?
  - Copilot 제안 중 수정이 필요했던 부분은?
  - Copilot이 도움이 되지 않았던 부분은?

- **실무 응용 제안** (2-3문장)
  - QLoRA를 활용할 수 있는 실제 사례 제시
  - Full FT 대신 LoRA/QLoRA를 선택해야 하는 상황은?

**제출 마감**: 수업 종료 후 24시간 이내

---

### 결과 토론 가이드

> **토론 가이드**: 조별 구현 결과를 공유하며, 메모리 절감 효과와 성능 유지의 메커니즘을 함께 분석한다

**토론 주제**:

① **메모리 절감의 실증적 효과**
- 각 조의 실제 측정 메모리 수치 비교
- 이론값(표 10.3)과 실제값의 차이는? 왜 그럴까?
- 8GB GPU에서 QLoRA로 7B 모델 파인튜닝이 가능했는가?

② **target_modules의 영향**
- Conservative vs Balanced 설정의 성능 차이
- 왜 모든 층에 LoRA를 적용하지 않아도 되는가?
- Attention이 FFN보다 중요한 이유는?

③ **학습 곡선과 수렴 속도**
- Full FT(이론)와 LoRA의 학습 곡선 비교
- 초기 손실이 낮은 이유 (B=0 초기화 효과)
- epoch 수가 적어도 충분히 수렴하는 이유는?

④ **배포 효율성의 실제 가치**
- LoRA 어댑터 50MB vs 원본 모델 280GB
- "어댑터 교환" 전략: 같은 기본 모델으로 여러 태스크 처리
- 예: 의료 도메인 어댑터, 법률 도메인 어댑터, 코딩 어댑터 동시 관리

⑤ **Full FT의 필요성과 한계**
- LoRA/QLoRA로 "거의 같은" 성능을 유지하는 이유
- 그럼에도 Full FT가 필요한 경우는? (매우 다른 도메인, 새로운 언어 등)
- 메모리-성능-개발비용의 트레이드오프 분석

⑥ **실무 의사결정**
- 주어진 GPU 메모리에서 Full FT/LoRA/QLoRA 중 선택 기준은?
- 학습 시간과 배포 효율성을 고려할 때 최적의 선택은?
- 프로토타입(개발) vs 프로덕션(배포)에서의 다른 전략

**발표 형식**:
- 각 조 4~6분 발표 (메모리 비교 결과 + 선택한 설정의 이유)
- 다른 조의 질문 2~3개 답변
- 교수의 보충 설명 및 실무 인사이트

---

### 다음 주 예고

다음 주 11주차 A회차에서는 **생성형 LLM 파인튜닝 (1) — Instruction Tuning과 RLHF 소개**를 다룬다.

**예고 내용**:
- **Instruction Tuning**: 모델에게 명령어(Instruction)를 따르도록 학습시키는 방법. GPT-3.5 → GPT-4로의 도약에 핵심이었다
- **데이터 준비**: SFT(Supervised Fine-Tuning)를 위한 Instruction-Response 쌍 데이터셋 구성
- **Prompt Template**: 다양한 프롬프트 템플릿의 설계와 성능 영향
- **RLHF 원리**: Reinforcement Learning from Human Feedback의 개념과 3단계 파이프라인
- **실습 연계**: 10주차의 QLoRA 기술을 사용하여 Instruction Tuning 실습을 진행

**사전 준비**:
- 10주차 내용(LoRA, QLoRA) 복습
- Instruction-Response 쌍이 왜 중요한지 미리 생각해보기
- 자신이 만들고 싶은 AI 어시스턴트의 특성 생각해보기

---

## 참고자료

**실습 코드**:
- _전체 구현 코드는 practice/chapter10/code/10-1-qlora-finetuning.py 참고_
- _메모리 비교 및 시각화는 practice/chapter10/code/10-2-memory-analysis.py 참고_
- _추론 및 평가는 practice/chapter10/code/10-3-inference-evaluation.py 참고_

**권장 읽기**:
- Hu, J. et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models. https://arxiv.org/abs/2106.09685
- Dettmers, T. & Zettlemoyer, L. (2023). QLoRA: Efficient Finetuning of Quantized LLMs. https://arxiv.org/abs/2305.14314
- PEFT 라이브러리 문서: https://huggingface.co/docs/peft
- bitsandbytes 라이브러리: https://github.com/TimDettmers/bitsandbytes
- Touvron, H. et al. (2023). Llama 2: Open Foundation and Fine-Tuned Chat Models. https://arxiv.org/abs/2307.09288

