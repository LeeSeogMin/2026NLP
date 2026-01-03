# 3장 집필계획서: 딥러닝의 핵심 - 신경망과 학습 원리

## 개요

**장 제목**: 딥러닝의 핵심: 신경망과 학습 원리
**대상 독자**: 딥러닝과 자연어처리를 처음 배우는 학부생 (3~4학년)
**장 유형**: 핵심 기술 장 (이론:실습 = 60:40)
**예상 분량**: 600-700줄

---

## 학습 목표

이 장을 마치면 다음을 수행할 수 있다:
- 인공 신경망의 기본 구조와 작동 원리를 설명할 수 있다
- 다양한 활성화 함수의 특징과 사용 시점을 이해한다
- 손실 함수와 경사 하강법의 원리를 설명할 수 있다
- 역전파 알고리즘의 동작 방식을 이해한다
- PyTorch로 간단한 신경망을 구현하고 학습시킬 수 있다

---

## 절 구성

### 3.1 인공 신경망의 기본 구조 (~120줄)

**핵심 내용**:
- 생물학적 뉴런과 인공 뉴런
  - 생물학적 뉴런의 구조 (수상돌기, 세포체, 축삭)
  - 인공 뉴런의 수학적 모델
- 퍼셉트론(Perceptron)의 구조와 한계
  - 퍼셉트론의 구조: 입력, 가중치, 편향, 활성화 함수
  - 단층 퍼셉트론의 한계 (XOR 문제)
- 다층 퍼셉트론(MLP)과 은닉층
  - 은닉층의 역할
  - 비선형성 도입의 중요성
- 가중치(Weights)와 편향(Bias)의 역할
  - 가중치: 입력의 중요도
  - 편향: 활성화 임계값 조정

**다이어그램**:
- 생물학적 뉴런 vs 인공 뉴런 대비도
- 퍼셉트론 구조 다이어그램
- MLP 구조 다이어그램

### 3.2 활성화 함수 (~100줄)

**핵심 내용**:
- 활성화 함수의 필요성
  - 비선형성 도입
  - 선형 변환만으로는 복잡한 패턴 학습 불가
- Sigmoid 함수
  - 수식: σ(x) = 1/(1+e^(-x))
  - 출력 범위: (0, 1)
  - 장점과 단점 (기울기 소실 문제)
- Tanh 함수
  - 수식: tanh(x) = (e^x - e^(-x))/(e^x + e^(-x))
  - 출력 범위: (-1, 1)
- ReLU, Leaky ReLU, GELU
  - ReLU: f(x) = max(0, x)
  - Leaky ReLU: f(x) = max(αx, x)
  - GELU: 최신 Transformer에서 사용
- Softmax (출력층)
  - 다중 클래스 분류에서의 역할
  - 확률 분포 출력

**다이어그램**: 활성화 함수 그래프 비교

### 3.3 손실 함수 (~100줄)

**핵심 내용**:
- 손실 함수의 개념
  - 모델 예측과 실제 값의 차이 측정
  - 최적화의 목표
- 회귀 문제의 손실 함수
  - MSE (Mean Squared Error)
  - MAE (Mean Absolute Error)
  - 각각의 특성과 사용 시점
- 분류 문제의 손실 함수
  - Binary Cross-Entropy
  - Categorical Cross-Entropy
  - 정보 이론적 해석
- 손실 함수와 최적화의 관계
  - 손실 표면(Loss Surface)의 개념
  - 전역 최소점과 지역 최소점

**다이어그램**: 손실 함수 시각화 (3D 손실 표면)

### 3.4 최적화 알고리즘 (~140줄)

**핵심 내용**:
- 경사 하강법(Gradient Descent)의 원리
  - 직관적 설명: 언덕에서 내려가기
  - 수식: θ = θ - η∇L(θ)
  - 학습률(Learning Rate)의 중요성
- Batch GD vs Mini-batch GD vs Stochastic GD
  - 각 방식의 장단점
  - 메모리와 수렴 속도 트레이드오프
- 역전파(Backpropagation) 알고리즘
  - 연쇄 법칙(Chain Rule)
  - 순전파와 역전파의 흐름
  - 각 층의 기울기 계산 과정
- 고급 옵티마이저 소개
  - Momentum
  - Adam (Adaptive Moment Estimation)

**다이어그램**:
- 경사 하강법 시각화
- 역전파 연쇄 법칙 흐름도

### 3.5 실습: PyTorch 기초 (~140줄)

**핵심 내용**:
- Tensor 기본 조작
  - 텐서 생성 (torch.tensor, torch.zeros, torch.randn)
  - 텐서 연산 (덧셈, 곱셈, 행렬 곱)
  - 인덱싱과 슬라이싱
- Autograd를 이용한 자동 미분
  - requires_grad 설정
  - backward() 호출
  - grad 속성 확인
- 간단한 선형 회귀 모델 구현
  - 수동 구현 (forward, backward)
  - nn.Linear 사용
- MLP 모델 직접 설계 및 학습
  - nn.Module 상속
  - forward 메서드 정의
  - 학습 루프 구현

**실습 코드**:
- `3-1-신경망기초.py`: 퍼셉트론과 MLP 시각화
- `3-2-활성화함수.py`: 활성화 함수 비교
- `3-5-pytorch기초.py`: Tensor와 Autograd
- `3-5-선형회귀.py`: 선형 회귀 모델
- `3-5-mlp.py`: MLP 모델 구현

---

## 생성할 파일 목록

### 문서
- `schema/chap3.md`: 집필계획서 (이 파일)
- `content/research/ch3-research.md`: 리서치 결과
- `content/drafts/ch3-draft.md`: 초안
- `content/reviews/ch3-review.md`: Multi-LLM 리뷰 결과
- `docs/ch3.md`: 최종 완성본

### 실습 코드
- `practice/chapter3/code/3-1-신경망기초.py`
- `practice/chapter3/code/3-2-활성화함수.py`
- `practice/chapter3/code/3-5-pytorch기초.py`
- `practice/chapter3/code/3-5-선형회귀.py`
- `practice/chapter3/code/3-5-mlp.py`
- `practice/chapter3/code/requirements.txt`

### 그래픽
- `content/graphics/ch3/fig-3-1-neuron.mmd`: 뉴런 구조
- `content/graphics/ch3/fig-3-2-perceptron.mmd`: 퍼셉트론 구조
- `content/graphics/ch3/fig-3-3-mlp.mmd`: MLP 구조
- `content/graphics/ch3/fig-3-4-activation.mmd`: 활성화 함수
- `content/graphics/ch3/fig-3-5-gradient-descent.mmd`: 경사 하강법
- `content/graphics/ch3/fig-3-6-backprop.mmd`: 역전파 흐름

---

## 7단계 워크플로우 실행 계획

### 1단계: 집필계획서 작성 ✓
- `schema/chap3.md` 작성

### 2단계: 자료 조사
- 인공 신경망의 역사와 발전
- 활성화 함수 특성 및 최신 연구
- 최적화 알고리즘 비교
- PyTorch 공식 문서 참조

### 3단계: 정보 구조화
- 핵심 개념 정리
- 다이어그램 설계
- 실습 시나리오 구성

### 4단계: 구현 및 문서화
- 실습 코드 작성 및 실행
- 본문 초안 작성
- Mermaid 다이어그램 제작

### 5단계: 최적화
- 문체 일관성 검토
- 분량 조정
- 용어 통일

### 6단계: 품질 검증
- Multi-LLM Review (GPT-4o, Grok-3)
- 학부생 눈높이 확인
- 코드 실행 결과 확인
- `docs/ch3.md`로 최종 저장

### 7단계: MS Word 변환
- `npm run convert:chapter 3` 실행

---

## 핵심 키워드

- 인공 뉴런, 퍼셉트론, 다층 퍼셉트론(MLP)
- 가중치, 편향, 은닉층
- 활성화 함수: Sigmoid, Tanh, ReLU, GELU, Softmax
- 손실 함수: MSE, MAE, Cross-Entropy
- 경사 하강법, 역전파, Chain Rule
- PyTorch, Tensor, Autograd

---

**작성일**: 2026-01-02
