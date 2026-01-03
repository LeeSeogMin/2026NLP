# 제3장 리서치 노트: 딥러닝의 핵심 - 신경망과 학습 원리

## 조사 일자: 2026-01-02

---

## 1. 퍼셉트론과 XOR 문제의 역사

### 1.1 퍼셉트론의 탄생

- **1943년**: Warren McCulloch와 Walter Pitts가 인공 뉴런의 수학적 모델 제안
- **1958년**: Frank Rosenblatt이 Cornell Aeronautical Laboratory에서 퍼셉트론 발표
  - 최초의 실용적인 기계학습 알고리즘
  - 학습 가능한 가중치를 통한 패턴 인식
- **1969년**: Minsky와 Papert의 *Perceptrons* 출판
  - 단층 퍼셉트론의 한계 증명 (XOR 문제)

### 1.2 XOR 문제란?

- XOR(배타적 OR)은 두 입력이 다를 때만 1을 출력
- 입력 조합과 출력:
  - (0, 0) → 0
  - (0, 1) → 1
  - (1, 0) → 1
  - (1, 1) → 0
- **문제점**: 2차원 평면에서 하나의 직선으로 분리 불가능 (선형 분리 불가)
- 대각선 위치에 같은 클래스가 분포하여 어떤 직선도 올바르게 분류 불가

### 1.3 AI 겨울 (1970s-1980s)

- *Perceptrons* 출판 이후 신경망 연구 자금 급감
- 단층 퍼셉트론의 한계가 과대 해석됨
- 실제로 Minsky와 Papert는 다층 퍼셉트론이 XOR 해결 가능함을 알고 있었음

### 1.4 다층 퍼셉트론(MLP)의 부활

- **1980년대**: 역전파 알고리즘 재발견
- MLP = 입력층 + 은닉층(들) + 출력층
- 비선형 활성화 함수 + 다층 구조 → 복잡한 패턴 학습 가능
- XOR 문제는 1개의 은닉층만으로도 해결 가능

**출처**: [DEV Community - Demystifying the XOR problem](https://dev.to/jbahire/demystifying-the-xor-problem-1blk), [Sean Trott - Perceptrons and the First AI Winter](https://seantrott.substack.com/p/perceptrons-xor-and-the-first-ai)

---

## 2. 활성화 함수 비교 분석

### 2.1 왜 활성화 함수가 필요한가?

- 선형 변환만으로는 복잡한 패턴 학습 불가
- 여러 선형 층을 쌓아도 결국 하나의 선형 변환과 동일
- 비선형성 도입으로 네트워크의 표현력 증가

### 2.2 Sigmoid 함수

- **수식**: σ(x) = 1 / (1 + e^(-x))
- **출력 범위**: (0, 1)
- **장점**: 확률 해석 가능, 미분 가능
- **단점**:
  - 기울기 소실(Vanishing Gradient) 문제
  - 출력이 0 중심이 아님
  - 지수 연산으로 계산 비용 높음

### 2.3 Tanh 함수

- **수식**: tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
- **출력 범위**: (-1, 1)
- **장점**: 0 중심 출력으로 학습 안정화
- **단점**: 여전히 기울기 소실 문제 존재

### 2.4 ReLU (Rectified Linear Unit)

- **수식**: f(x) = max(0, x)
- **출력 범위**: [0, ∞)
- **장점**:
  - 계산 매우 효율적 (비교 연산만 필요)
  - 기울기 소실 문제 완화
  - 희소 활성화(sparse activation)
- **단점**:
  - Dying ReLU 문제: 음수 입력에서 뉴런이 "죽음"
  - x=0에서 미분 불가
- **현황**: CNN에서 여전히 표준

### 2.5 Leaky ReLU

- **수식**: f(x) = max(αx, x), α ≈ 0.01
- Dying ReLU 문제 완화
- 음수 영역에서도 작은 기울기 유지

### 2.6 GELU (Gaussian Error Linear Unit)

- **수식**: GELU(x) = x × Φ(x), Φ는 표준 정규 분포의 CDF
- **특징**:
  - 확률적 활성화 메커니즘
  - 부드러운 곡선 (x=0에서도 미분 가능)
  - 음수 입력에서도 0이 아닌 기울기 유지
- **성능**:
  - 테스트 손실 0.3685, 정확도 89.52% (실험 비교에서 최고)
  - ReLU 대비 더 낮은 테스트 오류율
- **적용**: GPT, BERT 등 Transformer 모델의 표준
- **단점**: 에러 함수 또는 다항식 근사 필요로 계산 비용 높음

### 2.7 Softmax (출력층)

- **수식**: softmax(xᵢ) = e^xᵢ / Σe^xⱼ
- 다중 클래스 분류의 출력층에 사용
- 모든 출력의 합이 1 (확률 분포)

### 2.8 활성화 함수 선택 가이드

| 용도 | 권장 함수 |
|------|----------|
| CNN | ReLU / Swish |
| Transformer | GELU |
| 출력층 (이진 분류) | Sigmoid |
| 출력층 (다중 분류) | Softmax |
| RNN | Tanh |

**출처**: [IABAC - ReLU 2025 Guide](https://iabac.org/blog/relu-activation-function), [Salt Data Labs - Transformer Activation Functions](https://www.saltdatalabs.com/blog/deep-learning-101-transformer-activation-functions-explainer-relu-leaky-relu-gelu-elu-selu-softmax-and-more), [arXiv - GELU Analysis](https://arxiv.org/abs/2305.12073)

---

## 3. 역전파와 경사 하강법

### 3.1 경사 하강법 (Gradient Descent)

- **목표**: 손실 함수를 최소화하는 파라미터 찾기
- **직관**: "언덕에서 가장 가파른 방향으로 내려가기"
- **수식**: θ = θ - η∇L(θ)
  - θ: 파라미터
  - η: 학습률 (learning rate)
  - ∇L(θ): 손실 함수의 기울기

### 3.2 경사 하강법 변형

| 방식 | 배치 크기 | 장점 | 단점 |
|------|----------|------|------|
| Batch GD | 전체 데이터 | 안정적 수렴 | 느림, 메모리 많이 사용 |
| Stochastic GD | 1개 샘플 | 빠름, 지역 최소점 탈출 | 불안정 |
| Mini-batch GD | 일부 (32, 64 등) | 균형 잡힘 | 배치 크기 튜닝 필요 |

### 3.3 역전파 (Backpropagation) 알고리즘

- **핵심**: 연쇄 법칙(Chain Rule)을 효율적으로 적용
- **과정**:
  1. 순전파(Forward Pass): 입력 → 출력 → 손실 계산
  2. 역전파(Backward Pass): 손실 → 각 층의 기울기 계산
  3. 파라미터 업데이트: 기울기 방향으로 가중치 조정

### 3.4 연쇄 법칙 (Chain Rule)

- 합성 함수의 미분: dz/dx = dz/dy × dy/dx
- 신경망은 여러 함수의 합성 → 연쇄 법칙으로 각 층의 기울기 계산
- "뒤에서 앞으로" 계산하여 중복 계산 방지 (효율성)

### 3.5 역전파의 문제점

- **기울기 소실 (Vanishing Gradient)**: 깊은 네트워크에서 기울기가 매우 작아짐
  - Sigmoid, Tanh에서 주로 발생
  - 해결: ReLU, 잔차 연결(Residual Connection)
- **기울기 폭주 (Exploding Gradient)**: 기울기가 폭발적으로 증가
  - 해결: Gradient Clipping
- **지역 최소점**: 전역 최소점 도달 보장 불가

**출처**: [IBM - What is Backpropagation](https://www.ibm.com/think/topics/backpropagation), [GeeksforGeeks - Backpropagation](https://www.geeksforgeeks.org/machine-learning/backpropagation-in-neural-network/), [University of Toronto Lecture Notes](https://www.cs.toronto.edu/~rgrosse/courses/csc321_2017/readings/L06%20Backpropagation.pdf)

---

## 4. PyTorch Autograd

### 4.1 Autograd란?

- PyTorch의 자동 미분 엔진
- 계산 그래프(DAG)를 자동으로 구축하여 기울기 계산
- 역방향 자동 미분(Reverse-mode Automatic Differentiation)

### 4.2 핵심 개념

- **requires_grad=True**: 기울기 추적 활성화
- **backward()**: 역전파 실행
- **.grad**: 계산된 기울기 저장
- **torch.no_grad()**: 추론 시 기울기 계산 비활성화 (메모리 절약)

### 4.3 주의사항

- 기울기는 기본적으로 누적됨 → 매 반복마다 .zero_grad() 호출 필요
- 계산 그래프는 동적으로 생성 (define-by-run)

### 4.4 Jacobian-Vector Product

- Autograd는 야코비안-벡터 곱(JVP)을 사용하여 효율적으로 기울기 계산
- 스칼라 출력의 경우 벡터 기울기 직접 계산

**출처**: [PyTorch Official Tutorial - Autograd](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html), [PyTorch Autograd Mechanics](https://docs.pytorch.org/docs/stable/notes/autograd.html)

---

## 5. 손실 함수

### 5.1 회귀 문제

- **MSE (Mean Squared Error)**: L = (1/n) Σ(yᵢ - ŷᵢ)²
  - 이상치에 민감
  - 큰 오차에 더 큰 패널티
- **MAE (Mean Absolute Error)**: L = (1/n) Σ|yᵢ - ŷᵢ|
  - 이상치에 덜 민감
  - 미분 불연속점 존재

### 5.2 분류 문제

- **Binary Cross-Entropy**: L = -[y log(p) + (1-y) log(1-p)]
  - 이진 분류에 사용
  - Sigmoid 출력과 함께 사용
- **Categorical Cross-Entropy**: L = -Σyᵢ log(pᵢ)
  - 다중 클래스 분류
  - Softmax 출력과 함께 사용

### 5.3 손실 표면

- 고차원 공간에서의 손실 함수 지형
- 전역 최소점(global minimum)과 지역 최소점(local minimum)
- 안장점(saddle point)의 존재

---

## 6. 교재 구성 시사점

### 6.1 학부생 난이도 조절

- 수식은 직관적 해석과 함께 제시
- XOR 문제를 통한 퍼셉트론 한계 시각화
- 활성화 함수 그래프로 차이점 명확화
- 역전파는 단계별로 설명 (순전파 → 손실 → 역전파 → 업데이트)

### 6.2 실습 포인트

- PyTorch 텐서 기본 조작
- requires_grad와 backward() 실습
- 간단한 선형 회귀로 학습 루프 이해
- MLP로 XOR 문제 해결 (단층 → 다층 비교)

### 6.3 시각화 필수 항목

- 생물학적 뉴런 vs 인공 뉴런
- 퍼셉트론 구조
- MLP 구조 (입력층 → 은닉층 → 출력층)
- 활성화 함수 그래프 비교
- 경사 하강법 시각화 (등고선 위의 화살표)
- 역전파 흐름도

---

## 참고문헌

1. Rosenblatt, F. (1958). The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain. *Psychological Review*.
2. Minsky, M. & Papert, S. (1969). *Perceptrons*. MIT Press.
3. Rumelhart, D. et al. (1986). Learning representations by back-propagating errors. *Nature*.
4. Hendrycks, D. & Gimpel, K. (2016). Gaussian Error Linear Units (GELUs). *arXiv*.
5. PyTorch Documentation. (2025). Autograd: Automatic Differentiation. https://docs.pytorch.org/
