# 5장 집필계획서: 순차 데이터 처리 - RNN과 LSTM/GRU

## 개요

**장 제목**: 순차 데이터 처리: RNN과 LSTM/GRU
**대상 독자**: 딥러닝과 자연어처리를 처음 배우는 학부생 (3~4학년)
**장 유형**: 핵심 기술 장 (이론:실습 = 60:40)
**예상 분량**: 650-750줄

---

## 학습 목표

이 장을 마치면 다음을 수행할 수 있다:
- 순차 데이터의 특성과 기존 신경망의 한계를 이해한다
- RNN의 구조와 작동 원리를 설명할 수 있다
- 장기 의존성 문제와 기울기 소실/폭주 현상을 이해한다
- LSTM과 GRU의 게이트 메커니즘을 설명할 수 있다
- PyTorch로 RNN/LSTM/GRU 기반 언어 모델을 구현할 수 있다

---

## 절 구성

### 5.1 순차 데이터의 이해 (~80줄)

**핵심 내용**:
- 순차 데이터(Sequential Data)의 정의와 특성
  - 시계열 데이터, 자연어, 오디오 신호
  - 순서(Order)가 중요한 이유
- Feedforward 네트워크의 한계
  - 고정 길이 입력의 제약
  - 시간적 관계를 포착하지 못함
- 순환 구조의 필요성

### 5.2 순환 신경망(RNN) (~120줄)

**핵심 내용**:
- RNN의 기본 구조
  - 입력, 은닉 상태, 출력
  - 가중치 공유(Weight Sharing)
- Hidden State의 개념
  - 과거 정보의 저장소
  - 시간에 따른 정보 전달
- RNN의 순전파
  - h_t = tanh(W_hh * h_{t-1} + W_xh * x_t + b)
  - 시간 단계별 계산
- BPTT (Backpropagation Through Time)
  - 시간 축을 펼친 역전파
  - 계산 비용

**다이어그램**: RNN 구조도 (펼친 형태)

### 5.3 RNN의 문제점 (~80줄)

**핵심 내용**:
- 장기 의존성(Long-term Dependency) 문제
  - 먼 과거 정보의 소실
  - 실제 예시: 긴 문장에서의 문맥 이해
- 기울기 소실(Vanishing Gradient)
  - tanh 함수의 미분 특성
  - 연쇄 곱셈의 영향
- 기울기 폭주(Exploding Gradient)
  - 기울기의 기하급수적 증가
  - Gradient Clipping 해결책

### 5.4 LSTM (Long Short-Term Memory) (~150줄)

**핵심 내용**:
- LSTM의 핵심 아이디어
  - Cell State: 장기 기억 저장소
  - 게이트 메커니즘으로 정보 흐름 제어
- 세 가지 게이트
  - Forget Gate: 버릴 정보 결정
  - Input Gate: 저장할 정보 결정
  - Output Gate: 출력할 정보 결정
- LSTM의 정보 흐름
  - Cell State 업데이트 과정
  - Hidden State 계산
- LSTM이 장기 의존성을 해결하는 방법
  - 덧셈 연산으로 기울기 흐름 유지
  - 게이트를 통한 선택적 기억/망각

**다이어그램**: LSTM 셀 구조 (게이트 포함)

### 5.5 GRU (Gated Recurrent Unit) (~80줄)

**핵심 내용**:
- GRU의 구조와 LSTM과의 차이
  - Reset Gate, Update Gate
  - Cell State 없음 (단순화)
- GRU의 정보 흐름
  - Reset Gate: 과거 정보 리셋
  - Update Gate: 업데이트 비율 결정
- LSTM vs GRU 비교
  - 파라미터 수
  - 성능 차이
  - 사용 시나리오

**표**: LSTM과 GRU 비교표

### 5.6 Sequence-to-Sequence 모델 (~80줄)

**핵심 내용**:
- Encoder-Decoder 구조
  - Encoder: 입력 시퀀스 압축
  - Decoder: 출력 시퀀스 생성
- 기계 번역에의 적용
  - 가변 길이 입출력 처리
  - Context Vector의 역할
- Seq2Seq의 한계점
  - 고정 길이 Context Vector의 병목
  - Attention의 필요성 (6장 예고)

**다이어그램**: Seq2Seq 구조도

### 5.7 실습 (~160줄)

**핵심 내용**:
- PyTorch로 RNN/LSTM/GRU 구현
  - nn.RNN, nn.LSTM, nn.GRU 사용법
  - 입출력 shape 이해
- Character-level 언어 모델
  - 문자 단위 텍스트 생성
  - 데이터 전처리 (문자 → 인덱스)
- 모델 학습 및 텍스트 생성
  - Temperature Sampling
  - 생성 결과 분석

---

## 생성할 파일 목록

### 문서
- `schema/chap5.md`: 집필계획서 (이 파일)
- `content/research/ch5-research.md`: 리서치 결과
- `content/drafts/ch5-draft.md`: 초안
- `content/reviews/ch5-review.md`: Multi-LLM 리뷰 결과
- `docs/ch5.md`: 최종 완성본

### 실습 코드
- `practice/chapter5/code/5-2-rnn기초.py`
- `practice/chapter5/code/5-4-lstm.py`
- `practice/chapter5/code/5-5-gru.py`
- `practice/chapter5/code/5-7-문자단위언어모델.py`
- `practice/chapter5/code/requirements.txt`

### 그래픽
- `content/graphics/ch5/fig-5-1-rnn-unfolded.mmd`
- `content/graphics/ch5/fig-5-2-lstm-cell.mmd`
- `content/graphics/ch5/fig-5-3-seq2seq.mmd`

---

## 핵심 키워드

- Sequential Data, Time Series, RNN
- Hidden State, BPTT
- Vanishing Gradient, Exploding Gradient
- Long-term Dependency
- LSTM, Cell State, Forget Gate, Input Gate, Output Gate
- GRU, Reset Gate, Update Gate
- Sequence-to-Sequence, Encoder-Decoder
- Character-level Language Model

---

**작성일**: 2026-01-02
