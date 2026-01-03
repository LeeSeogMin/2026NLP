# 11장 집필계획서: LLM 파인튜닝 (1) - 전이 학습과 Full Fine-tuning

## 개요

**장 제목**: LLM 파인튜닝 (1) - 전이 학습과 Full Fine-tuning
**대상 독자**: 딥러닝과 자연어처리를 처음 배우는 학부생 (3~4학년)
**장 유형**: 실습 중심 장 (이론:실습 = 50:50)
**예상 분량**: 600-700줄

---

## 학습 목표

이 장을 마치면 다음을 수행할 수 있다:
- 전이 학습(Transfer Learning)의 개념과 필요성을 설명할 수 있다
- Full Fine-tuning과 Feature Extraction의 차이를 이해한다
- Hugging Face Trainer API를 활용하여 모델을 파인튜닝할 수 있다
- 적절한 하이퍼파라미터를 선택하고 과적합을 방지할 수 있다
- 학습 과정을 모니터링하고 분석할 수 있다

---

## 절 구성

### 11.1 전이 학습의 이해 (~80줄)

**핵심 내용**:
- 전이 학습(Transfer Learning)의 개념
  - 사전학습된 지식의 재활용
  - 도메인 간 지식 전이
- NLP에서의 전이 학습
  - Pre-training → Fine-tuning 패러다임
  - 컴퓨터 비전과의 비교
- 전이 학습의 장점
  - 적은 데이터로 높은 성능
  - 학습 시간 단축
  - 일반화 성능 향상

**다이어그램**: 전이 학습 개념도

### 11.2 파인튜닝 전략 (~80줄)

**핵심 내용**:
- Feature Extraction (특징 추출)
  - 사전학습 가중치 고정
  - 분류 헤드만 학습
- Full Fine-tuning
  - 전체 모델 가중치 업데이트
  - 도메인 특화 학습
- Partial Fine-tuning
  - 일부 레이어만 학습
  - Layer-wise Learning Rate
- 전략 선택 기준

**다이어그램**: 파인튜닝 전략 비교

### 11.3 파인튜닝 태스크 (~60줄)

**핵심 내용**:
- Sequence Classification
  - 문장/문서 분류
  - 감성 분석
- Token Classification
  - NER (개체명 인식)
  - POS Tagging
- Question Answering
  - Extractive QA
  - SQuAD 데이터셋
- Text Generation
  - 요약, 번역

**표**: 태스크별 출력 형식

### 11.4 데이터셋 준비 (~80줄)

**핵심 내용**:
- 데이터 수집 및 정제
- 데이터 포맷 (CSV, JSON)
- Train/Validation/Test 분할
- 데이터 불균형 처리
- Hugging Face Datasets 라이브러리
  - load_dataset()
  - Dataset 클래스

### 11.5 Hugging Face Trainer API (~100줄)

**핵심 내용**:
- Trainer 클래스 구조
- TrainingArguments 설정
  - output_dir, num_train_epochs
  - per_device_train_batch_size
  - learning_rate
  - evaluation_strategy
  - save_strategy
- DataCollator
- compute_metrics 함수 정의
- Trainer 학습 실행

**코드**: Trainer 기본 설정 및 학습

### 11.6 하이퍼파라미터 튜닝 (~80줄)

**핵심 내용**:
- Learning Rate 선택
  - 2e-5 ~ 5e-5 권장 범위
  - Learning Rate Finder
- Batch Size의 영향
  - 메모리 vs 성능 트레이드오프
  - Gradient Accumulation
- Warmup Steps
  - 초기 학습 안정화
- Weight Decay
  - AdamW와 정규화
- Epochs 결정

### 11.7 과적합 방지 (~60줄)

**핵심 내용**:
- Early Stopping
  - patience 설정
- Dropout
- Label Smoothing
- Data Augmentation for NLP
  - 동의어 치환
  - 역번역 (Back-translation)
- Regularization

### 11.8 학습 모니터링 (~60줄)

**핵심 내용**:
- Loss Curve 분석
- Validation Metrics 추적
- TensorBoard 활용
- Weights & Biases 소개
- 학습 로그 해석

**다이어그램**: 학습 곡선 예시

### 11.9 실습: 텍스트 분류 파인튜닝 (~100줄)

**핵심 내용**:
- 감성 분석 데이터셋 로드 (IMDb 또는 한국어 데이터)
- BERT 토크나이저 적용
- 모델 준비 (AutoModelForSequenceClassification)
- Trainer 설정 및 학습
- 평가 및 결과 분석
- 파인튜닝 전후 성능 비교

---

## 생성할 파일 목록

### 문서
- `schema/chap11.md`: 집필계획서 (본 파일)
- `content/research/ch11-research.md`: 리서치 결과
- `content/drafts/ch11-draft.md`: 초안
- `docs/ch11.md`: 최종 완성본

### 실습 코드
- `practice/chapter11/code/11-1-transfer-learning-concept.py`: 전이 학습 개념
- `practice/chapter11/code/11-5-trainer-basics.py`: Trainer API 기본
- `practice/chapter11/code/11-9-text-classification.py`: 텍스트 분류 파인튜닝
- `practice/chapter11/code/requirements.txt`

### 그래픽
- `content/graphics/ch11/fig-11-1-transfer-learning.mmd`: 전이 학습 개념도
- `content/graphics/ch11/fig-11-2-finetuning-strategies.mmd`: 파인튜닝 전략
- `content/graphics/ch11/fig-11-3-trainer-workflow.mmd`: Trainer 워크플로우
- `content/graphics/ch11/fig-11-4-training-curve.mmd`: 학습 곡선

---

## 10장과의 연계

- 10장에서 다룬 GPT의 텍스트 생성과 달리, 파인튜닝을 통한 특화 모델 구축
- 사전학습 모델(BERT, GPT)을 다운스트림 태스크에 적용
- 12장 LoRA/PEFT의 기반이 되는 Full Fine-tuning 이해

---

## 핵심 개념 정리

| 개념 | 설명 |
|------|------|
| Transfer Learning | 사전학습된 모델의 지식을 새로운 태스크에 활용 |
| Fine-tuning | 사전학습 모델을 특정 태스크에 맞게 추가 학습 |
| Feature Extraction | 사전학습 가중치를 고정하고 분류층만 학습 |
| Trainer API | Hugging Face의 학습 자동화 도구 |
| TrainingArguments | 학습 설정을 정의하는 클래스 |
| Early Stopping | 과적합 방지를 위해 학습 조기 종료 |

---

## 마지막 업데이트

2026-01-03
