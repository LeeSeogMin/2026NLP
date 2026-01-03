# 12장 리서치: PEFT와 효율적 튜닝

## 1. LoRA (Low-Rank Adaptation)

### 1.1 핵심 개념
- **원 논문**: Hu et al. (2021), "LoRA: Low-Rank Adaptation of Large Language Models"
- **핵심 아이디어**: 사전 학습된 가중치를 동결하고, 각 Transformer 레이어에 학습 가능한 저랭크 분해 행렬 주입
- **효율성**: GPT-3 175B 대비 학습 파라미터 10,000배 감소, GPU 메모리 3배 감소

### 1.2 수학적 원리
- 기존 가중치 W ∈ ℝ^(d×k)
- 가중치 변화량: ΔW = B × A, where B ∈ ℝ^(d×r), A ∈ ℝ^(r×k)
- 최종 가중치: W' = W + ΔW = W + BA
- Rank r << min(d, k)
- 스케일링 팩터: α/r

### 1.3 주요 하이퍼파라미터
- **r (rank)**: 저랭크 행렬의 차원. 권장값: 4, 8, 16, 32
- **alpha (α)**: 스케일링 팩터. 일반적으로 α = 2r
- **target_modules**: Query, Value 프로젝션이 기본. 최적 성능을 위해 모든 레이어 적용 권장
- **lora_dropout**: 과적합 방지를 위한 드롭아웃

### 1.4 실험 결과
- Rank 4로 Query, Value에 적용 시 최적 성능 달성
- 7B 모델 LoRA 학습에 약 14GB GPU RAM 필요
- Full fine-tuning과 동등한 성능, 파라미터는 0.2% 수준

---

## 2. PEFT (Parameter-Efficient Fine-Tuning)

### 2.1 정의
- 대규모 사전 학습 모델의 소수 파라미터만 학습하여 효율적으로 파인튜닝하는 방법론
- 전이 학습의 확장으로, 새로운 태스크에 적응하면서도 계산 비용 최소화

### 2.2 PEFT 방법론 분류

**Additive (추가형)**:
- Adapter: 병목 구조 레이어 삽입
- Prefix Tuning: 학습 가능한 prefix 벡터 추가 (파라미터 1000배 이상 감소)
- Prompt Tuning: 소프트 프롬프트 학습 (하드 프롬프트보다 우수)

**Selective (선택형)**:
- BitFit: 바이어스 파라미터만 학습 (GLUE에서 Full fine-tuning과 유사 성능)
- Sparse Fine-tuning: 일부 파라미터만 선택적 학습

**Reparameterization (재매개변수화)**:
- LoRA: 저랭크 분해로 가중치 변화량 근사
- 병합 가능하여 추론 시 추가 지연 없음

### 2.3 PEFT 장점
- 작은 체크포인트: Full fine-tuning(40GB) vs PEFT(수 MB)
- 카타스트로픽 포겟팅 방지
- 저데이터 환경에서 더 나은 성능
- Out-of-domain 일반화 개선

---

## 3. QLoRA

### 3.1 개요
- **원 논문**: Dettmers et al. (2023), "QLoRA: Efficient Finetuning of Quantized LLMs"
- 4비트 양자화된 모델에 LoRA 적용
- 65B 모델을 단일 48GB GPU에서 학습 가능

### 3.2 핵심 혁신

**4-bit NormalFloat (NF4)**:
- 정규 분포 가중치에 최적화된 새로운 데이터 타입
- 4-bit Integer/Float보다 우수한 성능

**Double Quantization**:
- 양자화 상수를 다시 양자화
- 파라미터당 약 0.37비트 추가 절감
- 65B 모델에서 약 3GB 메모리 절약

**Paged Optimizers**:
- NVIDIA 통합 메모리 활용
- 그래디언트 체크포인팅 시 메모리 스파이크 관리

### 3.3 메모리 효율성
| 모델 | Full Fine-tuning | QLoRA |
|------|------------------|-------|
| LLaMA-7B | ~28GB | ~10GB |
| LLaMA-33B | - | 24GB (단일 GPU) |
| LLaMA-65B | 780GB+ | 46GB (단일 GPU) |

### 3.4 구현 특징
- 저장 데이터 타입: 4-bit NormalFloat
- 계산 데이터 타입: 16-bit BrainFloat
- bitsandbytes 라이브러리 사용
- Hugging Face PEFT/transformers와 통합

---

## 4. Hugging Face PEFT 라이브러리

### 4.1 기본 사용법
```python
from peft import LoraConfig, get_peft_model, TaskType

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
```

### 4.2 LoraConfig 주요 파라미터
- `r`: 저랭크 차원 (권장: 8-32)
- `lora_alpha`: 스케일링 팩터 (권장: 2×r)
- `target_modules`: LoRA 적용 대상 모듈
- `lora_dropout`: 드롭아웃 비율
- `bias`: 바이어스 학습 방식 (none, all, lora_only)
- `task_type`: CAUSAL_LM, SEQ_CLS 등

### 4.3 모델 저장 및 로드
```python
# 저장 (LoRA 가중치만 저장됨 - 수 MB)
model.save_pretrained("lora_adapter")

# 로드
from peft import PeftModel
model = PeftModel.from_pretrained(base_model, "lora_adapter")

# 병합 (추론 최적화)
merged_model = model.merge_and_unload()
```

---

## 5. 기타 PEFT 기법

### 5.1 Prefix Tuning
- 각 레이어의 Key, Value에 학습 가능한 prefix 추가
- 파라미터 수 1000배 이상 감소

### 5.2 Adapter Layers
- 트랜스포머 레이어 사이에 작은 병목 구조 삽입
- 다중 태스크 설정에 적합

### 5.3 Prompt Tuning / P-tuning
- 입력에 학습 가능한 소프트 프롬프트 추가
- 하드 프롬프트보다 우수한 성능

### 5.4 (IA)³
- Infused Adapter by Inhibiting and Amplifying Inner Activations
- 학습된 벡터로 activation 스케일링

---

## 6. 최신 동향 (2024-2025)

### LoRA-Mini
- LoRA 개선 버전
- 저랭크 행렬을 4개로 분할, 내부 2개만 학습
- 더 높은 파라미터 효율성

### SK-Tuning (Semantic Knowledge Tuning)
- 무작위 토큰 대신 의미 있는 단어 사용
- 더 빠른 학습, 적은 파라미터, 우수한 성능

---

## 참고문헌

- Hu, E., et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models. arXiv:2106.09685
- Dettmers, T., et al. (2023). QLoRA: Efficient Finetuning of Quantized LLMs. arXiv:2305.14314
- Hugging Face PEFT Documentation: https://huggingface.co/docs/peft
- IBM Think: What is LoRA? https://www.ibm.com/think/topics/lora
- Springer: Parameter-efficient fine-tuning in LLMs: a survey (2025)

---

**리서치 완료**: 2026-01-03
