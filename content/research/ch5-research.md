# 제5장 리서치 노트: 순차 데이터 처리 - RNN과 LSTM/GRU

## 조사 일자: 2026-01-02

---

## 1. RNN의 기울기 소실 문제

### 1.1 문제의 원인
- RNN에서 역전파 시 시간 축을 따라 기울기가 연쇄 곱셈됨
- tanh 함수의 미분값이 (0, 1) 범위
- 긴 시퀀스에서 기울기가 기하급수적으로 감소

### 1.2 장기 의존성 문제
- 먼 과거의 정보가 현재에 영향을 미치지 못함
- 예: "The cat, which already ate a lot of fish, was full" - cat과 was의 관계

---

## 2. LSTM의 게이트 메커니즘

### 2.1 핵심 아이디어
- Cell State: 컨베이어 벨트처럼 정보를 보존하는 전용 경로
- 덧셈 연산으로 기울기 흐름 유지 (곱셈이 아닌 덧셈)
- 게이트로 정보 흐름을 선택적으로 제어

### 2.2 세 가지 게이트
- **Forget Gate**: 버릴 정보 결정 (σ 함수 → 0~1)
- **Input Gate**: 저장할 새 정보 결정
- **Output Gate**: 출력할 정보 결정

### 2.3 LSTM이 기울기 소실을 해결하는 방법
- Cell State의 덧셈 업데이트: C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t
- 미분 시 덧셈으로 인해 기울기가 직접 흐름
- 네트워크가 언제 기울기를 유지/소실시킬지 학습

**출처**: [Baeldung - LSTM Vanishing Gradient Prevention](https://www.baeldung.com/cs/lstm-vanishing-gradient-prevention)

---

## 3. GRU vs LSTM 비교

### 3.1 구조적 차이
| 항목 | LSTM | GRU |
|------|------|-----|
| 게이트 수 | 3개 (input, forget, output) | 2개 (reset, update) |
| Cell State | 있음 | 없음 |
| 파라미터 수 | 더 많음 | ~25% 적음 |

### 3.2 성능 비교
- **학습 속도**: GRU가 20-40% 빠름
- **메모리 사용량**: GRU가 ~25% 적음
- **모델 크기**: LSTM 42MB vs GRU 31MB (26% 감소)

### 3.3 태스크별 선택 가이드
- **단순한 시퀀스**: GRU가 우수하거나 비슷
- **복잡한 시퀀스**: LSTM이 우수
- **소형 데이터셋 (<10K)**: GRU (과적합 위험 낮음)
- **대형 데이터셋 (>100K)**: LSTM (복잡한 패턴)
- **장기 의존성**: LSTM이 우수

### 3.4 NLP에서의 성능
- 중간 길이 시퀀스 (20-100 토큰): GRU와 LSTM 비슷
- 긴 문서 분석: LSTM이 유리

**출처**: [Analytics Vidhya - When to Use GRUs Over LSTMs](https://www.analyticsvidhya.com/blog/2025/03/lstms-and-grus/)

---

## 4. PyTorch RNN/LSTM 구현

### 4.1 입력 텐서 형태
- **input**: (seq_len, batch, input_size)
- **h_0**: (num_layers, batch, hidden_size)
- **c_0**: (num_layers, batch, hidden_size) - LSTM만

### 4.2 주요 클래스
```python
nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
```

### 4.3 batch_first 옵션
- True: (batch, seq_len, features)
- False: (seq_len, batch, features) - 기본값

### 4.4 가변 길이 시퀀스 처리
- `pack_padded_sequence()`: 패딩된 시퀀스 압축
- `pad_packed_sequence()`: 압축 해제

**출처**: [PyTorch Documentation - LSTM](https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html)

---

## 5. Sequence-to-Sequence 모델

### 5.1 Encoder-Decoder 구조
- **Encoder**: 입력 시퀀스를 고정 길이 Context Vector로 압축
- **Decoder**: Context Vector로부터 출력 시퀀스 생성

### 5.2 한계점
- 고정 길이 Context Vector가 병목
- 긴 시퀀스에서 정보 손실
- → Attention 메커니즘으로 해결 (6장)

**출처**: [GitHub - pytorch-seq2seq](https://github.com/bentrevett/pytorch-seq2seq)

---

## 6. Character-level Language Model

### 6.1 개념
- 문자 단위로 다음 문자 예측
- 어휘 크기가 작음 (영어: ~100, 한국어: ~2000+)
- 미등록 단어(OOV) 문제 없음

### 6.2 텍스트 생성 전략
- **Greedy**: 항상 최고 확률 문자 선택
- **Temperature Sampling**: 확률 분포 조정
  - T < 1: 더 결정적 (보수적)
  - T > 1: 더 랜덤 (창의적)
- **Top-k Sampling**: 상위 k개 후보 중 샘플링

---

## 참고문헌

1. Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. *Neural Computation*, 9(8), 1735-1780.
2. Cho, K., et al. (2014). Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation. *EMNLP*.
3. PyTorch Documentation. (2025). Sequence Models and LSTM Networks. https://docs.pytorch.org/tutorials/
4. Sutskever, I., Vinyals, O., & Le, Q. V. (2014). Sequence to Sequence Learning with Neural Networks. *NeurIPS*.
