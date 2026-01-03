# 12장 집필계획서: LLM 파인튜닝 (2) - PEFT와 효율적 튜닝

## 개요

**장 제목**: LLM 파인튜닝 (2) - PEFT와 효율적 튜닝
**대상 독자**: 딥러닝과 자연어처리를 처음 배우는 학부생 (3~4학년)
**장 유형**: 핵심 기술 장 (이론:실습 = 50:50)
**예상 분량**: 650-750줄

---

## 학습 목표

이 장을 마치면 다음을 수행할 수 있다:
- Full Fine-tuning의 한계를 이해하고 PEFT의 필요성을 설명할 수 있다
- LoRA의 수학적 원리와 핵심 아이디어를 이해한다
- LoRA 하이퍼파라미터(rank, alpha, target modules)를 적절히 설정할 수 있다
- Hugging Face PEFT 라이브러리를 활용하여 LoRA를 적용할 수 있다
- Full Fine-tuning과 LoRA의 성능을 비교 분석할 수 있다

---

## 절 구성

### 12.1 Full Fine-tuning의 한계 (~60줄)

**핵심 내용**:
- 메모리 요구사항
  - 모델 파라미터 + 그래디언트 + 옵티마이저 상태
  - BERT-Large: ~16GB, LLaMA-7B: ~28GB+
- 계산 비용과 학습 시간
- 모델 저장 및 배포 문제
  - 태스크별로 전체 모델 복사 필요
- 카타스트로픽 포겟팅 (Catastrophic Forgetting)

### 12.2 PEFT 개요 (~80줄)

**핵심 내용**:
- Parameter-Efficient Fine-Tuning 개념
- PEFT의 핵심 아이디어: 소수 파라미터만 학습
- PEFT 방법론 분류
  - Additive: Adapter, Prefix Tuning
  - Selective: BitFit
  - Reparameterization: LoRA
- PEFT의 장점

**다이어그램**: PEFT 방법론 분류

### 12.3 LoRA 심화 (~100줄)

**핵심 내용**:
- LoRA의 핵심 아이디어
  - 가중치 변화량을 저랭크 행렬로 근사
- Low-Rank Matrix Decomposition
  - ΔW = B × A (d×r) × (r×k)
  - Rank r << min(d, k)
- LoRA의 수학적 원리
  - W' = W + ΔW = W + BA
  - Scaling factor: α/r
- 왜 Low-Rank가 효과적인가
  - 언어 모델의 업데이트는 저차원 공간에 있다

**다이어그램**: LoRA 구조

### 12.4 LoRA 하이퍼파라미터 (~80줄)

**핵심 내용**:
- Rank (r)
  - 학습 파라미터 수 결정
  - 권장: 4, 8, 16, 32
- Alpha (α)
  - Scaling factor
  - 일반적으로 α = r 또는 α = 2r
- Target Modules
  - Query, Key, Value, Output Projection
  - Feed-Forward layers
- Dropout
- Bias 처리

**표**: LoRA 하이퍼파라미터 권장값

### 12.5 QLoRA (~80줄)

**핵심 내용**:
- 양자화(Quantization) 개념
- 4-bit Quantization
  - FP32 → INT4
- NormalFloat4 (NF4)
- QLoRA = Quantization + LoRA
  - 4-bit 베이스 모델 + LoRA adapters
- 메모리 효율성 분석
  - LLaMA-65B: 780GB → ~40GB

### 12.6 기타 PEFT 기법 (~60줄)

**핵심 내용**:
- Prefix Tuning
  - 학습 가능한 prefix 벡터
- Adapter Layers
  - 병목 구조 레이어 삽입
- Prompt Tuning / P-tuning
  - 소프트 프롬프트 학습
- (IA)³
  - Learned vector로 activation 스케일링

### 12.7 Hugging Face PEFT 라이브러리 (~100줄)

**핵심 내용**:
- PEFT 라이브러리 설치
- LoraConfig 설정
- get_peft_model() 함수
- print_trainable_parameters()
- 모델 저장 및 로드
- Adapter Merging

**코드**: PEFT 기본 사용법

### 12.8 성능 비교 분석 (~60줄)

**핵심 내용**:
- Full Fine-tuning vs LoRA
  - 정확도, 학습 시간, 메모리, 파라미터 수
- Rank 값에 따른 성능 변화
- Target Modules 선택의 영향
- 실험 결과 분석

**표**: 성능 비교 결과

### 12.9 실습: LoRA 파인튜닝 (~100줄)

**핵심 내용**:
- LoRA Config 설정 및 적용
- 텍스트 분류 모델에 LoRA 적용
- 파라미터 수 비교 (99% 감소 확인)
- 학습 및 평가
- Full Fine-tuning과 비교

---

## 생성할 파일 목록

### 문서
- `schema/chap12.md`: 집필계획서 (본 파일)
- `content/research/ch12-research.md`: 리서치 결과
- `content/drafts/ch12-draft.md`: 초안
- `docs/ch12.md`: 최종 완성본

### 실습 코드
- `practice/chapter12/code/12-3-lora-basics.py`: LoRA 기본 개념
- `practice/chapter12/code/12-7-peft-library.py`: PEFT 라이브러리 사용
- `practice/chapter12/code/12-9-lora-finetuning.py`: LoRA 파인튜닝 실습
- `practice/chapter12/code/requirements.txt`

### 그래픽
- `content/graphics/ch12/fig-12-1-full-vs-peft.mmd`: Full vs PEFT 비교
- `content/graphics/ch12/fig-12-2-peft-methods.mmd`: PEFT 방법론 분류
- `content/graphics/ch12/fig-12-3-lora-architecture.mmd`: LoRA 구조
- `content/graphics/ch12/fig-12-4-qlora.mmd`: QLoRA 개념

---

## 11장과의 연계

- 11장에서 다룬 Full Fine-tuning의 실제 한계 경험
- PEFT/LoRA로 동일 성능을 99% 적은 파라미터로 달성
- 12장 이후 RAG, 프롬프트 엔지니어링으로 연결

---

## 핵심 개념 정리

| 개념 | 설명 |
|------|------|
| PEFT | 소수 파라미터만 학습하여 효율적으로 파인튜닝 |
| LoRA | 가중치 변화량을 저랭크 행렬로 근사 |
| Rank (r) | 저랭크 행렬의 차원, 학습 파라미터 수 결정 |
| Alpha (α) | LoRA의 스케일링 팩터 |
| QLoRA | 양자화 + LoRA로 메모리 효율 극대화 |

---

## 마지막 업데이트

2026-01-03
