# 3장 집필계획서: 시퀀스 모델에서 Transformer로

## 개요

**장 제목**: 시퀀스 모델에서 Transformer로
**대상 독자**: 딥러닝과 자연어처리를 처음 배우는 학부생 (3~4학년)
**장 유형**: 핵심 기술 장 (이론:실습 = 60:40)
**예상 분량**: 600-700줄

> **미션**: 수업이 끝나면 Attention이 문장의 어디에 집중하는지 시각화한다

---

## 학습 목표

이 장을 마치면 다음을 수행할 수 있다:

1. 단어 임베딩의 원리와 Word2Vec의 학습 방식을 설명할 수 있다
2. RNN/LSTM/GRU의 구조와 장기 의존성 문제를 이해한다
3. Attention 메커니즘의 Query, Key, Value 개념을 설명할 수 있다
4. Self-Attention과 Multi-Head Attention을 구현할 수 있다
5. Attention Weight를 시각화하고 해석할 수 있다

---

## 3교시 구조

| 시간 | 구분 | 내용 |
|------|------|------|
| 00:00~00:50 | **1교시** | 임베딩 + RNN/LSTM 개념 |
| 00:50~01:00 | 쉬는시간 | |
| 01:00~01:50 | **2교시** | Attention + Self-Attention + Multi-Head |
| 01:50~02:00 | 쉬는시간 | |
| 02:00~02:50 | **3교시** | Attention 구현 + 시각화 실습 + 과제 |

---

## 절 구성

### 3.1 순차 데이터와 단어 임베딩 (~120줄) — 1교시

**핵심 내용**:
- 순차 데이터의 특성: 순서가 의미를 가지는 데이터
- One-hot Encoding의 한계: 고차원, 의미 관계 표현 불가
- 분포 가설: "비슷한 문맥의 단어는 비슷한 의미를 갖는다"
- Word2Vec 원리:
  - CBOW: 주변 단어로 중심 단어 예측
  - Skip-gram: 중심 단어로 주변 단어 예측
- 임베딩 공간의 의미적 특성: "왕 - 남자 + 여자 = 여왕"
- 사전학습 임베딩: GloVe, FastText

**직관적 비유**: "왕 - 남자 + 여자 = 여왕"이 성립하는 마법 같은 공간

**다이어그램**:
- fig-3-1-embedding-space.mmd: 임베딩 공간 시각화 (2D)
- fig-3-2-word2vec.mmd: Word2Vec (CBOW/Skip-gram) 구조

### 3.2 RNN/LSTM/GRU (개념 중심) (~120줄) — 1교시

**핵심 내용**:
- RNN의 구조: 이전 은닉 상태를 현재 입력과 함께 처리
- 장기 의존성 문제: 긴 문장에서 앞부분 정보 소실
- LSTM: Cell State + 3개 Gate (Forget, Input, Output)
- GRU: LSTM 간소화 (2개 Gate: Reset, Update)
- Seq2Seq 모델과 Encoder-Decoder 구조
- RNN 계열의 한계: 순차 처리(병렬화 불가), 긴 문맥 어려움

**직관적 비유**: RNN은 "기억력 있는 신경망", LSTM은 "선택적 기억장치"

**다이어그램**:
- fig-3-3-rnn-unrolled.mmd: RNN 펼친 구조
- fig-3-4-lstm-cell.mmd: LSTM 셀 구조

### 3.3 Attention 메커니즘 (~130줄) — 2교시

**핵심 내용**:
- Attention의 동기: 모든 정보를 같은 비중으로 볼 수 없다
- Query, Key, Value 개념
- Scaled Dot-Product Attention 수식과 단계별 풀이
  - Attention(Q, K, V) = softmax(QKᵀ / √dₖ) · V
- Attention Weight의 의미와 해석
- √dₖ로 나누는 이유: 내적 값이 커지면 softmax가 극단적이 됨

**직관적 비유**: 시험 공부할 때 중요한 부분에 밑줄 긋고 집중하는 것
Query = "질문", Key = "후보 답의 라벨", Value = "실제 답의 내용"

**다이어그램**:
- fig-3-5-attention-mechanism.mmd: Attention 계산 흐름

### 3.4 Self-Attention과 Multi-Head Attention (~130줄) — 2교시

**핵심 내용**:
- Self-Attention: 문장 안에서 단어들이 서로를 바라보는 것
- Q, K, V가 모두 같은 입력에서 나오는 구조
- Multi-Head Attention: 여러 관점에서 동시에 바라보기
  - 각 Head가 다른 관계를 포착 (문법적, 의미적)
- Concatenation과 Linear Projection
- Attention vs Self-Attention vs Cross-Attention 비교

**직관적 비유**: "나는 은행에서 돈을 찾았다"에서 "은행"이 "돈"에 높은 Attention → 금융기관

**다이어그램**:
- fig-3-6-self-attention.mmd: Self-Attention 계산 과정
- fig-3-7-multi-head.mmd: Multi-Head Attention 구조

### 3.5 실습 (~100줄) — 3교시

**핵심 내용**:
- 사전학습 임베딩 로드 및 단어 유사도 측정
- Scaled Dot-Product Attention 직접 구현
- Self-Attention 모듈 단계별 구현
- Multi-Head Attention으로 확장
- Attention Weight 히트맵 시각화

**과제**: Self-Attention 모듈 구현 + Attention 시각화 리포트

---

## 생성할 파일 목록

### 문서
- `schema/chap3.md`: 집필계획서 (이 파일)
- `content/drafts/ch3-draft.md`: 초안
- `content/reviews/ch3-review.md`: Multi-LLM 리뷰 결과
- `docs/ch3.md`: 최종 완성본

### 실습 코드
- `practice/chapter3/code/3-1-임베딩.py`: Word2Vec 임베딩 실습
- `practice/chapter3/code/3-3-어텐션.py`: Attention 메커니즘 구현
- `practice/chapter3/code/3-5-실습.py`: 통합 실습 (Self-Attention + 시각화)
- `practice/chapter3/code/requirements.txt`

### 그래픽
- `content/graphics/ch3/fig-3-1-embedding-space.mmd`
- `content/graphics/ch3/fig-3-2-word2vec.mmd`
- `content/graphics/ch3/fig-3-3-rnn-unrolled.mmd`
- `content/graphics/ch3/fig-3-4-lstm-cell.mmd`
- `content/graphics/ch3/fig-3-5-attention-mechanism.mmd`
- `content/graphics/ch3/fig-3-6-self-attention.mmd`
- `content/graphics/ch3/fig-3-7-multi-head.mmd`

---

**작성일**: 2026-02-08
