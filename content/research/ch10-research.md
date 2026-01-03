# 10장 리서치 결과: LLM 시대 (2) - GPT 아키텍처와 생성 모델

**조사일**: 2026-01-03
**조사 주제**: GPT, Autoregressive LM, Causal Attention, 텍스트 생성 전략, Zero-shot/Few-shot Learning

---

## 1. GPT 개요

### 1.1 GPT란?
- **G**enerative **P**re-trained **T**ransformer
- OpenAI에서 개발한 자기회귀(Autoregressive) 언어 모델
- Decoder-only Transformer 아키텍처
- 2018년 GPT-1부터 시작하여 GPT-4까지 발전

### 1.2 GPT의 혁신성
- Pre-training + Zero/Few-shot 패러다임 확립
- Fine-tuning 없이 다양한 태스크 수행 가능
- 모델 크기 확대에 따른 창발적 능력 발현

### 1.3 GPT 시리즈 발전

| 모델 | 출시 | 파라미터 | 특징 |
|------|------|----------|------|
| GPT-1 | 2018 | 117M | 12층, 사전학습+미세조정 |
| GPT-2 | 2019 | 1.5B | Zero-shot, 1024 토큰 컨텍스트 |
| GPT-3 | 2020 | 175B | In-context Learning, Few-shot |
| GPT-4 | 2023 | ~1.7T (추정) | 멀티모달, 향상된 추론 |

---

## 2. GPT vs BERT

### 2.1 아키텍처 차이

| 특성 | GPT | BERT |
|------|-----|------|
| 구조 | Decoder-only | Encoder-only |
| 어텐션 | Causal (단방향) | Bidirectional (양방향) |
| 학습 목표 | Next Token Prediction | MLM + NSP |
| 주요 용도 | 텍스트 생성 | 텍스트 이해 |

### 2.2 문맥 이해 방식
- **BERT**: 양방향 문맥 활용 (왼쪽 + 오른쪽)
- **GPT**: 단방향 문맥만 활용 (왼쪽만)

---

## 3. GPT 아키텍처

### 3.1 Decoder-Only 구조
- Transformer 디코더만 사용
- 인코더-디코더 간 Cross-Attention 없음
- 단순하면서도 효과적인 구조

### 3.2 Causal Self-Attention
- Masked Self-Attention이라고도 함
- 미래 토큰에 대한 어텐션 차단
- 하삼각 마스크 행렬 적용
- 학습 시 "치팅" 방지

### 3.3 GPT-2 상세 구조

| 모델 | 층 수 | 은닉 차원 | 헤드 수 | 파라미터 |
|------|-------|-----------|---------|----------|
| Small | 12 | 768 | 12 | 124M |
| Medium | 24 | 1024 | 16 | 355M |
| Large | 36 | 1280 | 20 | 774M |
| XL | 48 | 1600 | 25 | 1.5B |

### 3.4 Pre-LN vs Post-LN
- **GPT-1**: Post-LayerNorm (원래 Transformer 방식)
- **GPT-2+**: Pre-LayerNorm (LayerNorm을 먼저 적용)
- Pre-LN이 학습 안정성 향상

---

## 4. 자기회귀 언어 모델링

### 4.1 기본 원리
- P(x₁, x₂, ..., xₙ) = ∏ P(xᵢ | x₁, ..., xᵢ₋₁)
- 이전 토큰들을 조건으로 다음 토큰 예측
- 왼쪽에서 오른쪽으로 순차적 생성

### 4.2 학습 방식
- Teacher Forcing: 학습 시 정답 토큰 제공
- Cross-Entropy Loss 최소화
- 전체 시퀀스에 대해 병렬 학습 가능

### 4.3 생성 방식
- 토큰을 하나씩 순차적으로 생성
- 이전 생성된 토큰을 다음 입력으로 사용
- 종료 토큰(<EOS>) 또는 최대 길이까지 반복

---

## 5. 텍스트 생성 전략

### 5.1 Greedy Search (탐욕 검색)
- 매 단계에서 가장 높은 확률의 토큰 선택
- 장점: 빠르고 단순
- 단점: 반복적, 지루한 텍스트 생성

### 5.2 Beam Search (빔 검색)
- K개의 최고 후보 시퀀스 유지
- 전체 시퀀스 확률 최적화
- Beam Width (K)로 탐색 범위 조절
- 단점: 반복 문제는 여전히 존재

### 5.3 Temperature Sampling
- 소프트맥스 온도 조절
- T < 1: 분포 날카롭게 (더 결정적)
- T > 1: 분포 평탄하게 (더 다양하게)
- T → 0: Greedy Search와 동일

### 5.4 Top-k Sampling
- 상위 k개 토큰에서만 샘플링
- k가 작으면 안전하지만 다양성 감소
- k가 크면 다양하지만 품질 저하 위험

### 5.5 Top-p (Nucleus) Sampling
- 누적 확률이 p를 넘는 최소 토큰 집합에서 샘플링
- 문맥에 따라 후보 수가 동적으로 조절
- 일반적으로 Top-k보다 자연스러운 결과

### 5.6 전략 비교

| 전략 | 다양성 | 품질 | 속도 | 용도 |
|------|--------|------|------|------|
| Greedy | 낮음 | 중간 | 빠름 | 결정적 태스크 |
| Beam | 낮음 | 높음 | 중간 | 번역, 요약 |
| Top-k | 중간 | 중간 | 빠름 | 일반 생성 |
| Top-p | 높음 | 높음 | 빠름 | 창의적 생성 |

---

## 6. Zero-shot / Few-shot Learning

### 6.1 Zero-shot Learning
- 예시 없이 태스크 지시만으로 수행
- 모델의 일반화 능력에 의존
- GPT-4는 Bar Exam에서 90% 백분위 달성

### 6.2 One-shot Learning
- 단일 예시만 제공
- 태스크 형식을 이해하는 데 도움

### 6.3 Few-shot Learning
- 2-5개의 예시 제공
- 모델 크기가 클수록 효과 증대
- 8B 파라미터 이상에서 효과적

### 6.4 In-Context Learning
- 프롬프트 내에서 예시를 통해 학습
- 파라미터 업데이트 없음
- GPT-3의 핵심 능력

### 6.5 창발적 능력 (Emergent Abilities)
- 특정 크기 이상에서 갑자기 나타나는 능력
- Chain-of-Thought 추론
- 산술, 논리적 추론 등

---

## 7. 프롬프트 엔지니어링

### 7.1 기본 원칙
- 명확하고 구체적인 지시
- 맥락과 배경 정보 제공
- 원하는 출력 형식 명시

### 7.2 Chain-of-Thought (CoT) Prompting
- 단계별 추론 유도
- "Let's think step by step" 추가
- 복잡한 문제에서 성능 향상
- 100B+ 모델에서 효과적

### 7.3 Zero-shot CoT
- 예시 없이 추론 유도
- "단계별로 생각해봅시다" 문구 사용

### 7.4 Self-Consistency
- 여러 번 CoT 수행 후 다수결
- 가장 자주 나오는 답변 선택

### 7.5 Tree of Thoughts (ToT)
- CoT의 일반화
- 여러 추론 경로 병렬 탐색
- 백트래킹 가능

---

## 8. 텍스트 생성 평가

### 8.1 Perplexity (혼란도)
- 언어 모델의 기본 평가 지표
- PPL = exp(평균 Cross-Entropy)
- 낮을수록 좋음

### 8.2 BLEU Score
- 기계 번역 평가
- n-gram 정밀도 기반
- 0-100 스케일

### 8.3 ROUGE Score
- 텍스트 요약 평가
- ROUGE-N, ROUGE-L 등
- 재현율 기반

### 8.4 정성적 평가
- 유창성 (Fluency)
- 일관성 (Coherence)
- 관련성 (Relevance)
- 사실 정확성 (Factuality)

---

## 9. Hugging Face 활용

### 9.1 주요 클래스
```python
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline
)
```

### 9.2 텍스트 생성 파이프라인
```python
generator = pipeline("text-generation", model="gpt2")
result = generator("Once upon a time", max_length=50)
```

### 9.3 생성 파라미터
- `max_length`: 최대 생성 길이
- `temperature`: 출력 다양성
- `top_k`: Top-k 샘플링
- `top_p`: Nucleus 샘플링
- `num_beams`: Beam Search
- `do_sample`: 샘플링 활성화

---

## 10. 참고문헌

- Radford, A., et al. (2018). Improving Language Understanding by Generative Pre-Training (GPT-1)
- Radford, A., et al. (2019). Language Models are Unsupervised Multitask Learners (GPT-2)
- Brown, T., et al. (2020). Language Models are Few-Shot Learners (GPT-3). NeurIPS 2020
- Wei, J., et al. (2022). Chain-of-Thought Prompting Elicits Reasoning in Large Language Models
- Holtzman, A., et al. (2020). The Curious Case of Neural Text Degeneration (Nucleus Sampling)
- Hugging Face Transformers: https://huggingface.co/docs/transformers/
