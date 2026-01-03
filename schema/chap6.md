# 6장 집필계획서: 혁신의 중심 - Transformer 아키텍처

## 개요

**장 제목**: 혁신의 중심: Transformer 아키텍처
**대상 독자**: 딥러닝과 자연어처리를 처음 배우는 학부생 (3~4학년)
**장 유형**: 핵심 기술 장 (이론:실습 = 60:40)
**예상 분량**: 600-700줄

---

## 학습 목표

이 장을 마치면 다음을 수행할 수 있다:
- RNN/LSTM의 한계와 Transformer의 등장 배경을 설명할 수 있다
- Attention 메커니즘의 Query, Key, Value 개념을 이해한다
- Self-Attention의 계산 과정을 단계별로 설명할 수 있다
- Multi-Head Attention의 필요성과 구조를 이해한다
- Positional Encoding의 역할과 구현 방법을 설명할 수 있다
- Transformer Encoder/Decoder 블록의 구조를 이해한다
- PyTorch로 Self-Attention과 Transformer Encoder를 구현할 수 있다

---

## 절 구성

### 6.1 Transformer 등장 배경 (~80줄)

**핵심 내용**:
- RNN/LSTM의 한계
  - 순차 처리로 인한 병렬화 어려움
  - 긴 시퀀스에서 정보 손실
  - 학습 시간 문제
- "Attention is All You Need" 논문 소개 (Vaswani et al., 2017)
- Transformer의 혁신성
  - 순환 구조 완전 제거
  - 병렬 처리 가능
  - 장거리 의존성 직접 모델링

**다이어그램**: RNN vs Transformer 비교 다이어그램

### 6.2 Attention 메커니즘 (~100줄)

**핵심 내용**:
- Attention의 직관적 이해
  - 번역할 때 어디를 볼지 선택
  - 인간의 주의 집중 메커니즘
- Query, Key, Value의 개념
  - Query: 질문 (현재 처리하는 단어)
  - Key: 참조할 후보들의 라벨
  - Value: 실제 정보
- Attention Score 계산
  - 유사도 측정 (dot product)
  - Softmax로 정규화
- Scaled Dot-Product Attention
  - 수식: Attention(Q, K, V) = softmax(QK^T / √d_k) V
  - 스케일링의 필요성 (√d_k)

**다이어그램**: Scaled Dot-Product Attention 흐름도

### 6.3 Self-Attention (~100줄)

**핵심 내용**:
- Self-Attention의 개념
  - 같은 시퀀스 내 단어들 간의 관계
  - 입력 시퀀스가 Q, K, V 모두 생성
- Self-Attention 계산 과정 (단계별)
  1. Q, K, V 행렬 생성 (Linear projection)
  2. Attention Score 계산
  3. Softmax 적용
  4. Value와 가중합
- Self-Attention의 장점
  - 모든 위치 직접 연결 (O(1) path length)
  - 장거리 의존성 효과적 학습
  - 병렬 처리 가능
- 계산 복잡도: O(n² · d)

**다이어그램**: Self-Attention 계산 과정 시각화

### 6.4 Multi-Head Attention (~90줄)

**핵심 내용**:
- Multi-Head Attention의 필요성
  - 다양한 관점에서 관계 파악
  - 주어-동사, 수식어-명사 등 다른 패턴
- Multi-Head Attention 구조
  - h개의 독립적인 Attention Head
  - 각 Head는 서로 다른 표현 공간 학습
- Concatenation과 Linear Projection
  - 모든 Head 출력 연결
  - 최종 Linear layer로 차원 복원
- Head 수 결정
  - BERT-base: 12 heads
  - GPT-2: 12 heads
  - d_model / num_heads = d_k

**다이어그램**: Multi-Head Attention 구조도

### 6.5 Positional Encoding (~80줄)

**핵심 내용**:
- 위치 정보의 필요성
  - Attention은 순서 정보 없음
  - "나는 너를 좋아해" vs "너는 나를 좋아해"
- Sinusoidal Positional Encoding
  - PE(pos, 2i) = sin(pos / 10000^(2i/d))
  - PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
  - 다양한 주파수의 sin/cos 조합
- Learned Positional Encoding
  - 학습 가능한 임베딩
  - 최대 길이 제한
- Rotary Position Embedding (RoPE) 간단 소개

**다이어그램**: Positional Encoding 시각화 (히트맵)

### 6.6 Transformer 구조 (~100줄)

**핵심 내용**:
- Encoder 블록 구조
  - Multi-Head Self-Attention
  - Position-wise Feed-Forward Network
  - Add & Norm (Residual Connection + Layer Normalization)
- Decoder 블록 구조
  - Masked Multi-Head Self-Attention
  - Encoder-Decoder Attention (Cross-Attention)
  - Feed-Forward Network
  - Add & Norm
- Masking의 종류
  - Padding Mask: PAD 토큰 무시
  - Look-ahead Mask (Causal Mask): 미래 정보 차단
- Feed-Forward Network
  - 2개의 Linear layer + ReLU
  - 차원 확장 후 축소 (d → 4d → d)

**다이어그램**: 전체 Transformer 아키텍처

### 6.7 실습: Transformer 구현 (~150줄)

**핵심 내용**:
- Self-Attention 메커니즘 단계별 구현
- Attention Weights 시각화
- Positional Encoding 구현 및 시각화
- Transformer Encoder 블록 구현
- 간단한 텍스트 분류 적용

**실습 코드**:
- `6-2-attention.py`: Attention 메커니즘 구현
- `6-5-positional-encoding.py`: Positional Encoding 구현 및 시각화
- `6-6-transformer-encoder.py`: Transformer Encoder 블록 구현

---

## 생성할 파일 목록

### 문서
- `schema/chap6.md`: 집필계획서 (현재 파일)
- `content/research/ch6-research.md`: 리서치 결과
- `content/drafts/ch6-draft.md`: 초안
- `docs/ch6.md`: 최종 완성본

### 실습 코드
- `practice/chapter6/code/6-2-attention.py`
- `practice/chapter6/code/6-5-positional-encoding.py`
- `practice/chapter6/code/6-6-transformer-encoder.py`
- `practice/chapter6/code/requirements.txt`

### 그래픽
- `content/graphics/ch6/fig-6-1-rnn-vs-transformer.mmd`
- `content/graphics/ch6/fig-6-2-scaled-dot-product.mmd`
- `content/graphics/ch6/fig-6-3-self-attention.mmd`
- `content/graphics/ch6/fig-6-4-multi-head.mmd`
- `content/graphics/ch6/fig-6-5-transformer-architecture.mmd`

---

## 핵심 수식

1. **Scaled Dot-Product Attention**:
   Attention(Q, K, V) = softmax(QK^T / √d_k) V

2. **Multi-Head Attention**:
   MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
   where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)

3. **Positional Encoding**:
   PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
   PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

4. **Feed-Forward Network**:
   FFN(x) = ReLU(xW_1 + b_1)W_2 + b_2

---

## 7단계 워크플로우 실행 계획

### 1단계: 집필계획서 작성 ✓
- `schema/chap6.md` 작성 완료

### 2단계: 자료 조사
- "Attention is All You Need" 논문 핵심 내용
- Transformer 구현 가이드
- Positional Encoding 연구
- Multi-Head Attention 분석

### 3단계: 정보 구조화
- 핵심 개념 정리
- 수식과 직관적 설명 연결
- 다이어그램 설계

### 4단계: 구현 및 문서화
- 실습 코드 작성 및 실행
- 본문 초안 작성
- Mermaid 다이어그램 제작

### 5단계: 최적화
- 문체 일관성 검토
- 5장(RNN/LSTM)과의 연결성 확인
- 용어 통일

### 6단계: 품질 검증
- Multi-LLM Review (GPT-4o + grok-4-1-fast-reasoning)
- `docs/ch6.md`로 최종 저장

### 7단계: MS Word 변환
- `npm run convert:chapter 6` 실행

---

## 참고문헌

- Vaswani, A., et al. (2017). Attention is All You Need. NeurIPS.
- Devlin, J., et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers. NAACL.
- The Illustrated Transformer (Jay Alammar) - https://jalammar.github.io/illustrated-transformer/
- PyTorch Documentation: Transformer Layers
