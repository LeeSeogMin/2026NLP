# 7장 집필계획서: 중간 점검 - 핵심 개념 복습과 실전 문제

## 개요

**장 제목**: 중간 점검: 핵심 개념 복습과 실전 문제
**대상 독자**: 딥러닝과 자연어처리를 처음 배우는 학부생 (3~4학년)
**장 유형**: 복습 및 평가 장 (이론 복습:문제 풀이 = 60:40)
**예상 분량**: 500-600줄

---

## 학습 목표

이 장을 마치면 다음을 수행할 수 있다:
- 1-6장에서 학습한 핵심 개념을 체계적으로 정리할 수 있다
- AI, 머신러닝, 딥러닝의 관계를 명확히 설명할 수 있다
- RNN/LSTM/GRU의 구조와 차이점을 비교 설명할 수 있다
- Transformer와 Attention 메커니즘의 핵심 원리를 설명할 수 있다
- 실전 문제를 통해 학습 내용의 이해도를 점검할 수 있다

---

## 절 구성

### 7.1 딥러닝 기초 복습 (~80줄)

**핵심 내용**:
- AI, 머신러닝, 딥러닝의 관계
  - 포함 관계와 각 분야의 특징
  - 학습 패러다임 (지도/비지도/강화 학습)
- 신경망의 기본 구조
  - 퍼셉트론과 다층 퍼셉트론(MLP)
  - 활성화 함수 (Sigmoid, ReLU, GELU, Softmax)
  - 손실 함수와 최적화
- 역전파 알고리즘
  - 경사 하강법의 원리
  - Chain Rule과 자동 미분

**핵심 수식**:
- Cross-Entropy Loss: L = -Σyᵢlog(ŷᵢ)
- Gradient Descent: θ = θ - α∇L(θ)

### 7.2 자연어처리 기초 복습 (~80줄)

**핵심 내용**:
- 언어 모델의 발전
  - 통계 기반 N-gram 모델
  - 신경망 기반 언어 모델
  - Perplexity 평가 지표
- 단어 임베딩
  - Word2Vec (CBOW, Skip-gram)
  - 임베딩 공간의 의미적 특성
- 텍스트 전처리
  - 토큰화, 정제, 불용어 제거
  - Subword Tokenization

**핵심 수식**:
- Perplexity: PP(W) = P(w₁, w₂, ..., wₙ)^(-1/n)

### 7.3 순환 신경망 복습 (~90줄)

**핵심 내용**:
- RNN의 구조와 원리
  - Hidden State의 개념
  - BPTT (Backpropagation Through Time)
  - 장기 의존성 문제
- LSTM 아키텍처
  - Cell State와 Hidden State
  - Forget/Input/Output Gate
  - 게이트 메커니즘의 역할
- GRU 아키텍처
  - Reset Gate와 Update Gate
  - LSTM vs GRU 비교

**비교표**: RNN vs LSTM vs GRU

### 7.4 Transformer 아키텍처 복습 (~100줄)

**핵심 내용**:
- Attention 메커니즘
  - Query, Key, Value의 의미
  - Scaled Dot-Product Attention
  - Self-Attention의 장점
- Multi-Head Attention
  - 다중 관점에서의 Attention
  - 헤드 수와 차원 분할
- Positional Encoding
  - Sinusoidal vs Learned PE
- Transformer 전체 구조
  - Encoder-Decoder 구조
  - Residual Connection과 Layer Normalization

**핵심 수식**:
- Attention(Q, K, V) = softmax(QKᵀ/√dₖ)V

### 7.5 PyTorch 핵심 정리 (~70줄)

**핵심 내용**:
- 텐서 기본 연산
  - 생성, 인덱싱, 형태 변환
- 자동 미분 (Autograd)
- 모델 정의 (nn.Module)
- 학습 루프 구현
  - Forward → Loss → Backward → Update
- 주요 레이어와 함수
  - nn.Linear, nn.Embedding, nn.LSTM
  - F.softmax, F.cross_entropy

### 7.6 핵심 개념 비교 정리 (~50줄)

**핵심 내용**:
- 모델 아키텍처 비교
  - MLP vs RNN vs Transformer
- 어텐션 유형 비교
  - Self-Attention vs Cross-Attention vs Masked Attention
- 학습 기법 비교
  - SGD vs Adam vs AdamW

**비교표 3개**: 아키텍처/어텐션/옵티마이저 비교

### 7.7 실전 문제 (~130줄)

**핵심 내용**:
- 객관식 문제 (10문제)
  - 개념 이해 확인
- 단답형 문제 (5문제)
  - 용어 및 수식 작성
- 서술형 문제 (3문제)
  - RNN vs Transformer 비교
  - Self-Attention 계산 과정 설명
  - LSTM 게이트 역할 설명
- 코딩 문제 (2문제)
  - PyTorch 기본 구현
  - Attention 점수 계산

---

## 생성할 파일 목록

### 문서
- `schema/chap7.md`: 집필계획서 (현재 파일)
- `content/research/ch7-research.md`: 리서치 결과 (1-6장 요약)
- `content/drafts/ch7-draft.md`: 초안
- `docs/ch7.md`: 최종 완성본

### 실습 코드
- `practice/chapter7/code/7-1-review-exercises.py`
- `practice/chapter7/code/requirements.txt`

### 그래픽
- `content/graphics/ch7/fig-7-1-architecture-comparison.mmd`
- `content/graphics/ch7/fig-7-2-attention-types.mmd`
- `content/graphics/ch7/fig-7-3-lstm-gates.mmd`

---

## 핵심 개념 요약

1. **딥러닝 3요소**: 모델(네트워크) + 손실 함수 + 최적화 알고리즘

2. **RNN의 핵심 문제**: 장기 의존성, 기울기 소실/폭주

3. **LSTM 게이트**:
   - Forget Gate: 과거 정보 중 버릴 것 결정
   - Input Gate: 새 정보 중 저장할 것 결정
   - Output Gate: 출력할 정보 결정

4. **Transformer 혁신**:
   - 병렬 처리 가능 (vs RNN의 순차 처리)
   - Self-Attention으로 장거리 의존성 해결
   - O(1) 경로 길이로 정보 전달

5. **Attention 수식**:
   Attention(Q, K, V) = softmax(QKᵀ/√dₖ)V

---

## 7단계 워크플로우 실행 계획

### 1단계: 집필계획서 작성 ✓
- `schema/chap7.md` 작성 완료

### 2단계: 자료 조사
- 1-6장 핵심 내용 요약
- 중간고사 출제 범위 정리
- 실전 문제 유형 조사

### 3단계: 정보 구조화
- 핵심 개념 정리
- 비교표 설계
- 문제 유형별 분류

### 4단계: 구현 및 문서화
- 복습 자료 작성
- 실전 문제 제작
- Mermaid 다이어그램 제작

### 5단계: 최적화
- 문체 일관성 검토
- 난이도 적절성 확인
- 용어 통일

### 6단계: 품질 검증
- Multi-LLM Review (GPT-4o + grok-4-1-fast-reasoning)
- `docs/ch7.md`로 최종 저장

### 7단계: MS Word 변환
- `npm run convert:chapter 7` 실행

---

## 참고문헌

- 1-6장 교재 내용
- Vaswani, A., et al. (2017). Attention is All You Need.
- Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory.
- Cho, K., et al. (2014). Learning Phrase Representations using RNN Encoder-Decoder.
