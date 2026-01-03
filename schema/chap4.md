# 4장 집필계획서: PyTorch 기반 딥러닝 모델 개발 프로세스

## 개요

**장 제목**: PyTorch 기반 딥러닝 모델 개발 프로세스
**대상 독자**: 딥러닝과 자연어처리를 처음 배우는 학부생 (3~4학년)
**장 유형**: 실습 중심 장 (이론:실습 = 40:60)
**예상 분량**: 600-700줄

---

## 학습 목표

이 장을 마치면 다음을 수행할 수 있다:
- nn.Module을 상속하여 커스텀 모델을 정의할 수 있다
- Dataset과 DataLoader를 활용하여 데이터 파이프라인을 구축할 수 있다
- 다양한 옵티마이저와 학습률 스케줄러를 적용할 수 있다
- Training/Validation Loop를 구현하고 과적합을 방지할 수 있다
- 분류 모델의 성능을 다양한 지표로 평가할 수 있다

---

## 절 구성

### 4.1 PyTorch 핵심 구성 요소 (~100줄)

**핵심 내용**:
- nn.Module을 활용한 모델 정의
  - __init__에서 층 정의
  - forward에서 순전파 정의
- torch.nn의 주요 레이어
  - Linear, Conv2d, Embedding
  - Dropout, BatchNorm
- 모델 파라미터 관리
  - model.parameters()
  - 파라미터 초기화 방법

### 4.2 데이터 처리 파이프라인 (~120줄)

**핵심 내용**:
- Dataset 클래스
  - __len__, __getitem__ 구현
  - 커스텀 Dataset 작성
- DataLoader 활용
  - batch_size, shuffle, num_workers
  - collate_fn 커스터마이징
- 데이터 전처리
  - 정규화, 토큰화
  - 데이터 증강 개념
- 배치 처리의 이해

### 4.3 옵티마이저와 학습률 스케줄러 (~100줄)

**핵심 내용**:
- 다양한 옵티마이저
  - SGD, Momentum, Adam, AdamW
  - 옵티마이저 선택 기준
- 학습률의 중요성
  - 학습률이 너무 크거나 작을 때
- 학습률 스케줄러
  - StepLR, ExponentialLR
  - ReduceLROnPlateau
  - CosineAnnealingLR

### 4.4 모델 학습 루프 (~120줄)

**핵심 내용**:
- Training Loop 구현
  - 순전파, 손실 계산, 역전파, 업데이트
  - 에폭과 배치 처리
- Validation Loop 구현
  - model.eval()과 torch.no_grad()
  - 검증 손실 및 정확도 계산
- 과적합과 일반화
  - 학습/검증 손실 비교
  - Early Stopping
- 정규화 기법
  - L2 정규화 (weight_decay)
  - Dropout
  - Batch Normalization

### 4.5 모델 평가 (~80줄)

**핵심 내용**:
- 분류 문제 평가 지표
  - Accuracy, Precision, Recall
  - F1-Score
- Confusion Matrix
  - 해석 방법
  - 시각화
- 학습 과정 시각화
  - Loss Curve
  - Accuracy Curve

### 4.6 실습: 텍스트 분류 (~180줄)

**핵심 내용**:
- 영화 리뷰 감성 분석
  - 데이터셋 로드 및 전처리
  - 간단한 어휘 사전 구축
- 텍스트 벡터화
  - Bag-of-Words
  - 원-핫 인코딩
- MLP 기반 감성 분석 모델
  - 모델 정의
  - 학습 및 평가
- 하이퍼파라미터 튜닝
  - 학습률, 배치 크기, 은닉층 크기

---

## 생성할 파일 목록

### 문서
- `schema/chap4.md`: 집필계획서 (이 파일)
- `content/research/ch4-research.md`: 리서치 결과
- `content/drafts/ch4-draft.md`: 초안
- `content/reviews/ch4-review.md`: Multi-LLM 리뷰 결과
- `docs/ch4.md`: 최종 완성본

### 실습 코드
- `practice/chapter4/code/4-1-모델정의.py`
- `practice/chapter4/code/4-2-데이터로더.py`
- `practice/chapter4/code/4-3-옵티마이저.py`
- `practice/chapter4/code/4-4-학습루프.py`
- `practice/chapter4/code/4-6-텍스트분류.py`
- `practice/chapter4/code/requirements.txt`

### 그래픽
- `content/graphics/ch4/fig-4-1-training-loop.mmd`
- `content/graphics/ch4/fig-4-2-overfitting.mmd`
- `content/graphics/ch4/fig-4-3-lr-scheduler.mmd`

---

## 핵심 키워드

- nn.Module, Dataset, DataLoader
- SGD, Adam, AdamW, Learning Rate Scheduler
- Training Loop, Validation Loop
- Overfitting, Early Stopping, Dropout, BatchNorm
- Accuracy, Precision, Recall, F1-Score, Confusion Matrix
- 텍스트 분류, 감성 분석, Bag-of-Words

---

**작성일**: 2026-01-02
