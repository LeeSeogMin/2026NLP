# 제3장 리서치 노트: 시퀀스 모델에서 Transformer로

## 조사 일자: 2026-02-08

---

## 1. Word2Vec: 단어 임베딩의 혁명

### 1.1 원논문 정보

Word2Vec은 2013년 Google의 Tomas Mikolov 연구팀이 두 편의 논문으로 발표하였다.

**논문 1**: Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). Efficient Estimation of Word Representations in Vector Space. *Proceedings of the Workshop at ICLR 2013*. https://arxiv.org/abs/1301.3781

**논문 2**: Mikolov, T., Sutskever, I., Chen, K., Corrado, G., & Dean, J. (2013). Distributed Representations of Words and Phrases and their Compositionality. *Advances in Neural Information Processing Systems (NeurIPS) 2013*. https://arxiv.org/abs/1310.4546

### 1.2 핵심 아이디어: 분포 가설 (Distributional Hypothesis)

- "비슷한 문맥에 등장하는 단어는 비슷한 의미를 갖는다" (J.R. Firth, 1957: "You shall know a word by the company it keeps")
- Word2Vec은 이 직관을 신경망으로 구현한 것이다
- 단어를 고정 크기의 실수 벡터(dense vector)로 매핑하여 의미적 관계를 벡터 공간에서 표현한다

### 1.3 CBOW (Continuous Bag-of-Words) 아키텍처

- **목표**: 주변 단어(context words)로부터 중심 단어(center word)를 예측
- **구조**:
  - 입력층: 윈도우 내 주변 단어들의 one-hot 벡터
  - 투사층(Projection Layer): 주변 단어 벡터들의 평균 (또는 합)
  - 출력층: softmax를 통해 중심 단어 확률 분포 예측
- **특징**:
  - 빈번한 단어(frequent words)의 표현에 더 우수
  - Skip-gram보다 학습 속도가 빠름
  - 문맥 정보를 평균하여 사용하므로 순서 정보 소실

### 1.4 Skip-gram 아키텍처

- **목표**: 중심 단어로부터 주변 단어들을 예측
- **구조**:
  - 입력층: 중심 단어의 one-hot 벡터
  - 투사층(Projection Layer): 중심 단어의 임베딩 벡터
  - 출력층: softmax를 통해 주변 단어 확률 분포 예측
- **특징**:
  - 희소한 단어(infrequent words)의 표현에 더 우수
  - 작은 데이터셋에서 효과적
  - 각 단어-문맥 쌍을 개별 학습 사례로 처리

### 1.5 CBOW vs Skip-gram 구조 비교

| 항목 | CBOW | Skip-gram |
|------|------|-----------|
| 입력 | 주변 단어들 | 중심 단어 |
| 출력 | 중심 단어 | 주변 단어들 |
| 학습 속도 | 빠름 | 느림 |
| 빈번한 단어 | 우수 | 보통 |
| 희소한 단어 | 보통 | 우수 |
| 소량 데이터 | 보통 | 우수 |

**핵심**: 두 아키텍처는 구조적으로 동일하며, 목적 함수(objective function)만 다르다.

### 1.6 임베딩 차원 (Embedding Dimensions)

원논문 및 후속 실험에서 사용된 차원:
- 원논문 실험: 300차원 (Google News 사전학습 모델)
- NNLM 비교 실험: 100, 640, 1000차원
- 논문 2 (phrase analogy): 1000차원 (hierarchical softmax)
- **실무 일반**: 50~300차원이 가장 보편적
  - 50차원: 소규모 작업, 빠른 학습
  - 100~200차원: 범용적 용도
  - 300차원: 대규모 코퍼스, 높은 정확도 필요 시

### 1.7 학습 기법

**Negative Sampling**:
- 전체 어휘에 대한 softmax 대신, k개의 "부정 샘플"만 사용
- 부정 샘플은 단어 빈도의 3/4 제곱에 비례하여 추출: P(w) = f(w)^(3/4) / Z
- k = 5~20 (소규모 데이터셋), k = 2~5 (대규모 데이터셋)
- 빈번한 단어와 저차원 벡터에 효과적

**Hierarchical Softmax**:
- 이진 허프만 트리(binary Huffman tree)를 사용하여 softmax 계산 가속
- 계산 복잡도: O(V) -> O(log V) (V = 어휘 크기)
- 희소한 단어에 효과적

**Window Size**:
- 일반적으로 5~10 (양쪽 합)
- 작은 윈도우: 문법적(syntactic) 관계 포착
- 큰 윈도우: 의미적(semantic) 관계 포착

### 1.8 "King - Man + Woman = Queen" 유추 (Analogy)

**수학적 기반**:
- 단어 벡터 공간에서 의미적 관계가 **벡터 차이**로 인코딩된다
- vec("king") - vec("man") + vec("woman") ≈ vec("queen")
- 이는 "왕과 남자의 관계"가 "여왕과 여자의 관계"와 동일한 방향 벡터로 표현됨을 의미한다
- 수학적으로: vec("king") - vec("man") ≈ vec("queen") - vec("woman")
  - 즉, "성별" 방향 벡터가 공간에 존재

**작동 원리**:
- 분포 가설에 의해 "king"과 "queen"은 유사한 문맥에 등장하되, "man/woman" 관련 문맥에서 체계적 차이를 보인다
- 이 체계적 차이가 벡터 공간에서 일관된 방향(direction)으로 학습된다
- 연구에 따르면, 이는 조건부 PMI(Pointwise Mutual Information) 값과 수학적으로 연결된다

**주의 사항**:
- 실제 구현에서는 입력 단어(king, man, woman)를 결과 후보에서 제외한다
- 제외하지 않으면 결과가 "king" 자체가 되는 경우가 발생한다
- 이 유추는 항상 정확하지 않으며, 편향(bias) 문제도 내포한다

**다른 유추 예시**:
- vec("Paris") - vec("France") + vec("Italy") ≈ vec("Rome") (국가-수도)
- vec("bigger") - vec("big") + vec("small") ≈ vec("smaller") (비교급)

---

## 2. GloVe와 FastText: Word2Vec의 발전

### 2.1 GloVe (Global Vectors for Word Representation)

**논문**: Pennington, J., Socher, R., & Manning, C.D. (2014). GloVe: Global Vectors for Word Representation. *Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP)*, pp. 1532-1543. https://aclanthology.org/D14-1162/

**핵심 아이디어**:
- 전체 코퍼스의 **단어-단어 동시 출현 행렬(co-occurrence matrix)**을 먼저 구성
- 이 행렬을 분해(factorization)하여 단어 벡터를 학습
- **동시 출현 확률의 비율(ratio)**이 의미적 관계를 인코딩한다는 핵심 통찰

**Word2Vec과의 핵심 차이점**:

| 항목 | Word2Vec | GloVe |
|------|----------|-------|
| 방법론 | 예측 기반 (Predictive) | 카운트 기반 (Count-based) |
| 학습 데이터 | 지역적 문맥 윈도우 | 전역적 동시 출현 통계 |
| 행렬 사용 | 사용 안 함 | 동시 출현 행렬 분해 |
| 메모리 | 적음 | 동시 출현 행렬에 큰 메모리 필요 |
| 학습 시간 | Word2Vec이 일반적으로 빠름 | 대규모 코퍼스에서 효율적 |
| 전역 통계 활용 | 간접적 | 직접적 |

**사전학습 모델**:
- Wikipedia 2014 + Gigaword 5 (6B 토큰)으로 학습
- 50d, 100d, 200d, 300d 차원 제공
- Stanford NLP 그룹에서 무료 배포

### 2.2 FastText

**논문**: Bojanowski, P., Grave, E., Joulin, A., & Mikolov, T. (2017). Enriching Word Vectors with Subword Information. *Transactions of the Association for Computational Linguistics (TACL)*, Vol. 5, pp. 135-146. https://arxiv.org/abs/1607.04606

**핵심 아이디어**:
- Word2Vec의 Skip-gram을 확장하여 **서브워드(subword) 정보**를 활용
- 각 단어를 **문자 n-gram의 집합(bag of character n-grams)**으로 표현
- 단어 벡터 = 단어 자체 벡터 + 구성 n-gram 벡터들의 합

**문자 n-gram 예시** (n = 3~6):
- "where" → `<wh`, `whe`, `her`, `ere`, `re>`, `<whe`, `wher`, `here`, `ere>`, `<wher`, `where`, `here>`, `<where>`
- `<` 와 `>` 는 단어 경계 표시

**Word2Vec/GloVe와의 핵심 차이점**:

| 항목 | Word2Vec/GloVe | FastText |
|------|---------------|----------|
| 단위 | 전체 단어 | 단어 + 문자 n-gram |
| OOV 처리 | 불가 (알 수 없는 단어) | 가능 (n-gram 조합으로 추론) |
| 형태소 풍부한 언어 | 약함 | 강함 (예: 한국어, 터키어) |
| NER 성능 | 보통 | 우수 |
| n-gram 범위 | 해당 없음 | 3~6 (기본값) |

**OOV(Out-of-Vocabulary) 처리**:
- 학습 데이터에 없는 단어도 구성 n-gram 벡터들의 합으로 벡터 생성 가능
- 예: "unlikelihood"를 학습하지 않았어도 "un", "like", "li", "hood" 등의 n-gram에서 의미 추론
- 이는 형태소가 풍부한 언어(한국어, 터키어, 핀란드어 등)에서 특히 유용

---

## 3. RNN / LSTM / GRU: 순환 신경망 계열

### 3.1 RNN (Recurrent Neural Network)

**기원**: Elman, J.L. (1990). Finding Structure in Time. *Cognitive Science*, 14(2), 179-211.

**구조**:
- 은닉 상태(hidden state) hₜ가 이전 시점의 은닉 상태 hₜ₋₁과 현재 입력 xₜ를 함께 처리
- **수식**: hₜ = tanh(Wₕₕ · hₜ₋₁ + Wₓₕ · xₜ + b)
- 출력: yₜ = Wₕᵧ · hₜ

**특징**:
- 순차 데이터의 시간적 패턴을 학습
- 가변 길이 입력 처리 가능
- 파라미터 공유: 모든 시점에서 동일한 가중치 사용

### 3.2 기울기 소실 문제 (Vanishing Gradient Problem)

**핵심 논문**: Bengio, Y., Simard, P., & Frasconi, P. (1994). Learning Long-Term Dependencies with Gradient Descent Is Difficult. *IEEE Transactions on Neural Networks*, 5(2), 157-166. https://doi.org/10.1109/72.279181

또한 Hochreiter의 1991년 졸업논문(diplom thesis)에서 최초로 공식적으로 규명되었다.

**수학적 원인**:
- RNN의 역전파 시 기울기는 다음과 같이 전파된다:
  - ∂L/∂hₜ = ∂L/∂hₜ₊₁ · ∂hₜ₊₁/∂hₜ
  - t 시점에서 k 시점까지의 기울기: ∏ᵢ₌ₖᵗ (∂hᵢ₊₁/∂hᵢ)
- 각 ∂hᵢ₊₁/∂hᵢ는 야코비안 행렬(Jacobian matrix)이며, 이것이 **같은 가중치 행렬 Wₕₕ의 반복 곱셈**을 포함한다
- **Wₕₕ의 스펙트럼 반지름(spectral radius) ρ에 의해 결정**:
  - ρ < 1: 기울기가 지수적으로 감소 (소실) → 장기 의존성 학습 불가
  - ρ > 1: 기울기가 지수적으로 증가 (폭주) → 학습 불안정
  - ρ = 1: 이론적으로 이상적이지만 실무에서 달성 어려움

**직관적 설명**:
- 0.9를 50번 곱하면: 0.9^50 ≈ 0.005 (거의 소실)
- 1.1을 50번 곱하면: 1.1^50 ≈ 117.4 (폭발)
- 따라서 RNN은 약 10~20 시점 이상의 장기 의존성을 학습하기 어렵다

**기울기 폭주의 해결: Gradient Clipping**
- 기울기 노름(norm)이 임계값을 초과하면 스케일링하여 제한
- 일반적 임계값: 1.0 ~ 10.0
- 수식: if ||g|| > threshold then g = threshold × g / ||g||

### 3.3 LSTM (Long Short-Term Memory)

**핵심 논문**: Hochreiter, S. & Schmidhuber, J. (1997). Long Short-Term Memory. *Neural Computation*, 9(8), 1735-1780. https://doi.org/10.1162/neco.1997.9.8.1735

**참고**: 망각 게이트(Forget Gate)는 원래 1997년 논문에는 없었으며, 이후 Gers et al. (2000)이 추가하였다.
- Gers, F.A., Schmidhuber, J., & Cummins, F. (2000). Learning to Forget: Continual Prediction with LSTM. *Neural Computation*, 12(10), 2451-2471.

**핵심 구성 요소**:

#### (1) Cell State (셀 상태) Cₜ
- LSTM의 핵심 메모리 채널
- 전체 시퀀스를 관통하는 "컨베이어 벨트"
- RNN의 은닉 상태와 달리, 매 시점마다 완전히 덮어쓰이지 않고 **선택적으로 수정**됨
- 정보가 최소한의 변형으로 전달될 수 있음

#### (2) Forget Gate (망각 게이트) fₜ
- **역할**: 이전 셀 상태에서 어떤 정보를 **버릴지** 결정
- **수식**: fₜ = σ(Wf · [hₜ₋₁, xₜ] + bf)
- **출력**: 0~1 사이 값 (0 = 완전히 잊음, 1 = 완전히 기억)
- **비유**: "이 정보가 아직 유효한가?"를 판단하는 필터

#### (3) Input Gate (입력 게이트) iₜ + 후보 값 C̃ₜ
- **역할**: 어떤 **새로운 정보**를 셀 상태에 저장할지 결정
- **수식**:
  - iₜ = σ(Wi · [hₜ₋₁, xₜ] + bi)  ← 얼마나 저장할지
  - C̃ₜ = tanh(Wc · [hₜ₋₁, xₜ] + bc)  ← 저장할 후보 값
- **비유**: "이 새 정보가 중요한가?"를 판단하는 필터

#### (4) Cell State 업데이트
- **수식**: Cₜ = fₜ ⊙ Cₜ₋₁ + iₜ ⊙ C̃ₜ
  - ⊙ = 원소별 곱셈 (Hadamard product)
- **의미**: 이전 기억의 일부를 잊고(fₜ ⊙ Cₜ₋₁), 새 정보의 일부를 추가(iₜ ⊙ C̃ₜ)

#### (5) Output Gate (출력 게이트) oₜ
- **역할**: 셀 상태에서 어떤 정보를 **출력**할지 결정
- **수식**:
  - oₜ = σ(Wo · [hₜ₋₁, xₜ] + bo)
  - hₜ = oₜ ⊙ tanh(Cₜ)
- **비유**: "지금 이 순간 어떤 정보를 내보낼 것인가?"

#### 기울기 소실 해결 원리
- Cell State 업데이트가 **덧셈 연산**(Cₜ = fₜ ⊙ Cₜ₋₁ + ...)이므로 기울기 경로에서 곱셈의 반복이 완화된다
- Forget Gate의 값이 1에 가까우면, 기울기가 거의 손실 없이 먼 과거까지 전달된다

### 3.4 GRU (Gated Recurrent Unit)

**핵심 논문**: Cho, K., van Merrienboer, B., Gulcehre, C., Bahdanau, D., Bougares, F., Schwenk, H., & Bengio, Y. (2014). Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation. *Proceedings of EMNLP 2014*, pp. 1724-1734. https://aclanthology.org/D14-1179/

**성능 비교 논문**: Chung, J., Gulcehre, C., Cho, K., & Bengio, Y. (2014). Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling. *arXiv:1412.3555*. https://arxiv.org/abs/1412.3555

**핵심 아이디어**: LSTM의 3개 게이트를 2개로 간소화

#### (1) Reset Gate (리셋 게이트) rₜ
- **역할**: 이전 은닉 상태를 **얼마나 무시할지** 결정
- **수식**: rₜ = σ(Wr · [hₜ₋₁, xₜ] + br)
- **특징**: 단기 의존성(short-term dependencies) 포착에 유리

#### (2) Update Gate (업데이트 게이트) zₜ
- **역할**: 이전 은닉 상태를 **얼마나 유지할지** 결정 (LSTM의 Forget Gate + Input Gate 역할을 합침)
- **수식**: zₜ = σ(Wz · [hₜ₋₁, xₜ] + bz)
- **특징**: 장기 의존성(long-term dependencies) 포착에 유리

#### (3) 은닉 상태 업데이트
- **후보 은닉 상태**: h̃ₜ = tanh(W · [rₜ ⊙ hₜ₋₁, xₜ] + b)
- **최종 은닉 상태**: hₜ = (1 - zₜ) ⊙ hₜ₋₁ + zₜ ⊙ h̃ₜ

### 3.5 LSTM vs GRU 비교

| 항목 | LSTM | GRU |
|------|------|-----|
| 게이트 수 | 3 (Forget, Input, Output) | 2 (Reset, Update) |
| 별도 셀 상태 | 있음 (Cₜ) | 없음 (은닉 상태만 사용) |
| 파라미터 수 | 더 많음 | 더 적음 (~25% 절감) |
| 계산 속도 | 상대적으로 느림 | 상대적으로 빠름 |
| 성능 | 대규모 데이터에서 약간 우위 | 소규모~중규모 데이터에서 효율적 |
| 일반적 결론 | 대부분의 태스크에서 비슷한 성능 | 대부분의 태스크에서 비슷한 성능 |

### 3.6 구체적 수치: 일반적 하이퍼파라미터

| 항목 | 일반적 값 |
|------|----------|
| 은닉 차원 (hidden_size) | 128, 256, 512, 1024 |
| 층 수 (num_layers) | 1~4 (2가 가장 보편적) |
| Gradient Clipping | max_norm = 1.0 ~ 5.0 |
| 드롭아웃 (Dropout) | 0.2 ~ 0.5 |
| 학습률 (Learning Rate) | 0.001 (Adam), 0.01 (SGD) |

### 3.7 RNN 계열의 근본적 한계

1. **순차 처리 (Sequential Processing)**: 시점 t의 계산은 시점 t-1의 결과에 의존 → GPU 병렬화 불가
2. **장기 의존성 한계**: LSTM/GRU도 수백 토큰 이상의 매우 긴 의존성은 어려움
3. **정보 병목 (Bottleneck)**: Seq2Seq의 Encoder 마지막 은닉 상태에 모든 정보 압축 → 긴 문장에서 정보 손실
4. **학습 속도**: 순차 처리로 인해 Transformer 대비 학습이 매우 느림

---

## 4. Attention 메커니즘

### 4.1 Bahdanau Attention (Additive Attention)

**논문**: Bahdanau, D., Cho, K., & Bengio, Y. (2015). Neural Machine Translation by Jointly Learning to Align and Translate. *Proceedings of ICLR 2015*. https://arxiv.org/abs/1409.0473 (2014년 9월 arXiv 게재, 2015년 ICLR 발표)

**동기**:
- Seq2Seq 모델의 정보 병목 문제 해결
- Encoder의 마지막 은닉 상태만 사용하는 대신, **Decoder가 Encoder의 모든 은닉 상태를 선택적으로 참조**

**스코어 함수 (Additive / Concat)**:
- score(sₜ, hᵢ) = vₐᵀ · tanh(Wₐ · [sₜ; hᵢ])
  - sₜ: Decoder의 은닉 상태 (이전 시점)
  - hᵢ: Encoder의 i번째 은닉 상태
  - Wₐ, vₐ: 학습 가능한 파라미터
- 단층 피드포워드 신경망으로 정렬 스코어(alignment score) 계산
- tanh 비선형성을 포함하여 더 유연한 관계 학습 가능

**특징**:
- Decoder의 **이전 시점** 은닉 상태 sₜ₋₁을 사용하여 attention 계산
- 파라미터가 많아 과적합 위험 존재 (특히 소량 데이터)
- 계산 비용이 상대적으로 높음

### 4.2 Luong Attention (Multiplicative Attention)

**논문**: Luong, T., Pham, H., & Manning, C.D. (2015). Effective Approaches to Attention-based Neural Machine Translation. *Proceedings of EMNLP 2015*, pp. 1412-1421. https://arxiv.org/abs/1508.04025

**스코어 함수 변형**:

| 변형 | 수식 | 설명 |
|------|------|------|
| Dot Product | score(sₜ, hᵢ) = sₜᵀ · hᵢ | 가장 단순, 추가 파라미터 없음 |
| General | score(sₜ, hᵢ) = sₜᵀ · Wₐ · hᵢ | 학습 가능한 가중치 행렬 |
| Concat | score(sₜ, hᵢ) = vₐᵀ · tanh(Wₐ · [sₜ; hᵢ]) | Bahdanau 방식과 동일 |

**특징**:
- Decoder의 **현재 시점** 은닉 상태 sₜ를 사용
- 내적(dot product) 기반으로 계산이 더 효율적
- Global Attention (전체 소스 참조) vs Local Attention (소스의 일부만 참조) 구분 제안

### 4.3 Bahdanau vs Luong 비교

| 항목 | Bahdanau (2014) | Luong (2015) |
|------|----------------|-------------|
| 스코어 함수 | Additive (신경망) | Multiplicative (내적 기반) |
| Decoder 상태 | 이전 시점 sₜ₋₁ | 현재 시점 sₜ |
| 추가 파라미터 | 많음 (Wₐ, vₐ) | 적음 (Wₐ 또는 없음) |
| 계산 효율 | 낮음 | 높음 |
| Attention 범위 | Global | Global + Local |

### 4.4 Scaled Dot-Product Attention

**논문**: Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser, L., & Polosukhin, I. (2017). Attention Is All You Need. *Advances in Neural Information Processing Systems (NeurIPS) 2017*. https://arxiv.org/abs/1706.03762

**수식**:
```
Attention(Q, K, V) = softmax(QKᵀ / sqrt(dₖ)) · V
```

**단계별 계산**:
1. Q와 K의 내적 계산: QKᵀ → 유사도 행렬 (n × n)
2. sqrt(dₖ)로 스케일링: QKᵀ / sqrt(dₖ)
3. Softmax 적용: 각 행(query)에 대해 확률 분포 생성
4. V와 가중 합: attention weights × V → 최종 출력

**sqrt(dₖ)로 나누는 이유 (CRITICAL)**:
- Q와 K의 각 원소가 평균 0, 분산 1인 독립 확률변수라고 가정
- dₖ차원 벡터의 내적 결과: 평균 = 0, **분산 = dₖ**
- dₖ가 클 경우 (예: 64, 512), 내적 값의 크기가 매우 커짐
- 큰 값이 softmax에 입력되면:
  - softmax 출력이 극단적으로 편중됨 (거의 one-hot에 가까움)
  - softmax의 기울기가 **극도로 작아짐** (vanishing gradient)
  - 학습이 불안정해지거나 매우 느려짐
- sqrt(dₖ)로 나누면 분산이 1로 정규화되어 softmax가 적절한 분포를 유지

**구체적 수치 예시**:
- dₖ = 64일 때, sqrt(64) = 8로 나눔
- 스케일링 없는 내적: 분산 = 64 → 값의 범위가 [-16, +16] 정도
- 스케일링 후: 분산 = 1 → 값의 범위가 [-2, +2] 정도
- softmax(-16, +16)은 거의 (0, 1)로 포화 → 기울기 ≈ 0
- softmax(-2, +2)은 적절한 확률 분포 유지 → 기울기 안정

---

## 5. Self-Attention

### 5.1 핵심 개념

- **일반 Attention**: Encoder와 Decoder 사이의 관계 (Cross-Attention)
- **Self-Attention**: 같은 시퀀스 **내부**에서 단어 간의 관계를 모델링

### 5.2 Q, K, V가 같은 입력에서 유도되는 과정

입력 시퀀스 X = [x₁, x₂, ..., xₙ] (각 xᵢ는 d차원 벡터)에 대해:

1. 세 개의 학습 가능한 가중치 행렬을 사용:
   - Q = X · Wq  (Wq: d × dₖ)
   - K = X · Wk  (Wk: d × dₖ)
   - V = X · Wv  (Wv: d × dᵥ)

2. **동일한 입력 X**에서 서로 다른 가중치 행렬을 통해 Q, K, V 생성
   - Q (Query): "나는 무엇을 찾고 있는가?"
   - K (Key): "나는 어떤 정보를 제공할 수 있는가?"
   - V (Value): "나의 실제 내용은 무엇인가?"

3. 각 단어가 다른 모든 단어와의 관계를 계산:
   - Attention(Q, K, V) = softmax(QKᵀ / sqrt(dₖ)) · V

### 5.3 계산 복잡도

| 모델 | 레이어당 계산 복잡도 | 순차 연산 | 최대 경로 길이 |
|------|-------------------|---------|-----------    |
| Self-Attention | O(n² · d) | O(1) | O(1) |
| RNN | O(n · d²) | O(n) | O(n) |
| CNN (커널 k) | O(k · n · d²) | O(1) | O(logₖ(n)) |

- n: 시퀀스 길이, d: 표현 차원

**Self-Attention의 O(n² · d)**:
- QKᵀ 계산에서 모든 쌍 비교: n × n 행렬 → O(n² · d)
- 시퀀스가 길어지면(n > 512) 메모리와 연산이 급격히 증가
- 이것이 Transformer의 주요 병목이며, 후속 연구(Linformer, Performer 등)의 동기

### 5.4 RNN 대비 장기 의존성 처리의 우월성

**RNN의 경우**:
- 시점 1의 정보가 시점 100에 도달하려면 99번의 순차 처리를 거쳐야 함
- 최대 경로 길이 = O(n): 정보가 n번의 비선형 변환을 통과
- 매 변환마다 정보 손실 발생 가능

**Self-Attention의 경우**:
- 시점 1과 시점 100이 **직접 연결** (단 1번의 Attention 연산)
- 최대 경로 길이 = O(1): 어떤 두 위치도 단일 연산으로 연결
- 장거리 의존성을 효과적으로 학습

**병렬화**:
- RNN: 시점 t의 계산은 시점 t-1 결과에 의존 → 순차 처리 필수 → GPU 활용 제한
- Self-Attention: 모든 위치가 동시에 계산 가능 → 완전 병렬화 → GPU/TPU 효율적 활용

---

## 6. Multi-Head Attention

### 6.1 핵심 아이디어

- 단일 Attention으로는 **하나의 관점**만 포착
- 여러 개의 Attention을 병렬로 수행하여 **다양한 유형의 관계** 동시 포착
- 각 "Head"가 서로 다른 부분 공간(subspace)에서 독립적으로 Attention 학습

### 6.2 구조

```
MultiHead(Q, K, V) = Concat(head₁, head₂, ..., headₕ) · W^O

headᵢ = Attention(Q · Wᵢᑫ, K · Wᵢᴷ, V · Wᵢⱽ)
```

**과정**:
1. Q, K, V를 h개의 헤드에 대해 각각 선형 변환 (투사)
   - 각 헤드: dₖ = dᵥ = d_model / h 차원으로 축소
2. 각 헤드에서 독립적으로 Scaled Dot-Product Attention 수행
3. h개의 Attention 출력을 **Concatenation** (연결)
4. 연결된 벡터에 **선형 변환(W^O)**을 적용하여 원래 차원(d_model)으로 복원

### 6.3 Transformer 원논문의 구체적 수치

| 항목 | Base Model | Big Model |
|------|-----------|-----------|
| d_model (모델 차원) | 512 | 1024 |
| h (헤드 수) | **8** | **16** |
| dₖ = dᵥ = d_model/h | 64 | 64 |
| d_ff (FFN 내부 차원) | 2048 | 4096 |
| N (인코더/디코더 층 수) | 6 | 6 |
| 총 파라미터 | ~65M | ~213M |
| 학습 시간 | 12시간 (100K steps) | 3.5일 (300K steps) |
| Step 시간 | 0.4초 | 1.0초 |
| 하드웨어 | 8 NVIDIA P100 GPU | 8 NVIDIA P100 GPU |

**계산 비용 보존**:
- 전체 계산 비용은 단일 헤드 (d_model 차원)와 거의 동일
- 이유: 각 헤드가 d_model/h 차원에서 작동하므로, h개 헤드의 총 연산량 ≈ 1개 헤드의 d_model 차원 연산량

### 6.4 각 헤드가 포착하는 관계의 다양성

연구에 따르면 각 헤드는 서로 다른 언어적 관계를 학습한다:
- **문법적 관계**: 주어-동사 일치, 수식어-피수식어
- **의미적 관계**: 동의어, 반의어, 관련 개념
- **위치적 관계**: 인접 단어, 특정 거리의 단어
- **구문적 관계**: 구(phrase) 구조, 절(clause) 경계

예시 ("The animal didn't cross the street because it was too tired"):
- 한 Head: "it" → "animal" (대명사 참조 해결)
- 다른 Head: "didn't" → "cross" (부정-동사 관계)
- 또 다른 Head: "tired" → "animal" (형용사-주어 관계)

### 6.5 Concatenation + Linear Projection의 역할

1. **Concatenation**: h개 헤드의 출력 (각 dᵥ 차원)을 연결 → h × dᵥ = d_model 차원
2. **Linear Projection (W^O)**: 연결된 벡터를 d_model 차원으로 변환
   - 여러 헤드의 정보를 **혼합(mixing)하고 통합**
   - 각 헤드가 독립적으로 학습한 다양한 관점을 하나의 표현으로 종합

---

## 7. Seq2Seq와 Cross-Attention (보충)

### 7.1 Seq2Seq 모델

**논문**: Sutskever, I., Vinyals, O., & Le, Q.V. (2014). Sequence to Sequence Learning with Neural Networks. *NeurIPS 2014*. https://arxiv.org/abs/1409.3215

- Encoder-Decoder 구조: 입력 시퀀스를 고정 크기 벡터로 압축 → 출력 시퀀스 생성
- 번역, 요약 등 시퀀스-투-시퀀스 태스크의 기초
- **한계**: 긴 문장에서 정보 병목 → Attention으로 해결

### 7.2 Cross-Attention vs Self-Attention

| 항목 | Self-Attention | Cross-Attention |
|------|---------------|----------------|
| Q 출처 | 입력 자체 | Decoder |
| K, V 출처 | 입력 자체 | Encoder |
| 용도 | 시퀀스 내부 관계 | Encoder-Decoder 간 관계 |
| 예시 | Encoder 내부, Decoder 내부 | Decoder가 Encoder 참조 |

---

## 8. 교재 구성 시사점

### 8.1 학부생 난이도 조절 핵심

- Word2Vec은 "왕 - 남자 + 여자 = 여왕" 예시로 시작하여 동기 부여
- RNN은 "기억력 있는 신경망" 비유, LSTM은 "선택적 기억장치" 비유 활용
- 기울기 소실은 "0.9^50 ≈ 0.005" 같은 구체적 숫자로 직관 제공
- Attention은 "시험 공부 시 중요한 부분에 밑줄 긋기" 비유 사용
- sqrt(dₖ)로 나누는 이유는 "softmax 포화" 현상을 수치 예시로 설명

### 8.2 필수 시각화 항목

1. **임베딩 공간 시각화**: 2D t-SNE/PCA로 단어 간 관계 표현
2. **Word2Vec 아키텍처**: CBOW vs Skip-gram 구조 비교 다이어그램
3. **RNN 펼친 구조(Unrolled)**: 시간축을 따라 펼친 RNN
4. **LSTM 셀 구조**: Cell State, 3개 Gate의 흐름
5. **Attention 계산 흐름**: Q, K, V → 스코어 → softmax → 가중합
6. **Self-Attention 과정**: 단어 간 Attention Weight 히트맵
7. **Multi-Head 구조**: 여러 Head의 병렬 처리 → Concat → Linear

### 8.3 핵심 흐름

RNN의 한계 → Attention으로 해결 → Self-Attention (모든 위치를 동시에 참조) → Multi-Head (다양한 관점) → **Transformer의 핵심 구성 요소 완성**

이 장은 "왜 Transformer가 필요한가?"에 대한 답을 구축하는 과정이다.

---

## 검증된 참고문헌 (Verified References)

### 핵심 논문

1. Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). Efficient Estimation of Word Representations in Vector Space. *ICLR 2013 Workshop*. https://arxiv.org/abs/1301.3781
2. Mikolov, T., Sutskever, I., Chen, K., Corrado, G., & Dean, J. (2013). Distributed Representations of Words and Phrases and their Compositionality. *NeurIPS 2013*. https://arxiv.org/abs/1310.4546
3. Pennington, J., Socher, R., & Manning, C.D. (2014). GloVe: Global Vectors for Word Representation. *EMNLP 2014*, pp. 1532-1543. https://aclanthology.org/D14-1162/
4. Bojanowski, P., Grave, E., Joulin, A., & Mikolov, T. (2017). Enriching Word Vectors with Subword Information. *TACL*, Vol. 5, pp. 135-146. https://arxiv.org/abs/1607.04606
5. Elman, J.L. (1990). Finding Structure in Time. *Cognitive Science*, 14(2), 179-211.
6. Bengio, Y., Simard, P., & Frasconi, P. (1994). Learning Long-Term Dependencies with Gradient Descent Is Difficult. *IEEE Transactions on Neural Networks*, 5(2), 157-166. https://doi.org/10.1109/72.279181
7. Hochreiter, S. & Schmidhuber, J. (1997). Long Short-Term Memory. *Neural Computation*, 9(8), 1735-1780. https://doi.org/10.1162/neco.1997.9.8.1735
8. Gers, F.A., Schmidhuber, J., & Cummins, F. (2000). Learning to Forget: Continual Prediction with LSTM. *Neural Computation*, 12(10), 2451-2471.
9. Cho, K., et al. (2014). Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation. *EMNLP 2014*, pp. 1724-1734. https://aclanthology.org/D14-1179/
10. Chung, J., Gulcehre, C., Cho, K., & Bengio, Y. (2014). Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling. *arXiv:1412.3555*. https://arxiv.org/abs/1412.3555
11. Sutskever, I., Vinyals, O., & Le, Q.V. (2014). Sequence to Sequence Learning with Neural Networks. *NeurIPS 2014*. https://arxiv.org/abs/1409.3215
12. Bahdanau, D., Cho, K., & Bengio, Y. (2015). Neural Machine Translation by Jointly Learning to Align and Translate. *ICLR 2015*. https://arxiv.org/abs/1409.0473
13. Luong, T., Pham, H., & Manning, C.D. (2015). Effective Approaches to Attention-based Neural Machine Translation. *EMNLP 2015*, pp. 1412-1421. https://arxiv.org/abs/1508.04025
14. Vaswani, A., et al. (2017). Attention Is All You Need. *NeurIPS 2017*. https://arxiv.org/abs/1706.03762

### 참고 자료 (설명/교육용)

15. Colah's Blog. (2015). Understanding LSTM Networks. https://colah.github.io/posts/2015-08-Understanding-LSTMs/
16. Dive into Deep Learning. (2024). Chapter 10-11: Recurrent Neural Networks and Attention Mechanisms. https://d2l.ai/
17. Rong, X. (2014). word2vec Parameter Learning Explained. https://arxiv.org/abs/1402.3722
18. Pascanu, R., Mikolov, T., & Bengio, Y. (2013). On the difficulty of training Recurrent Neural Networks. *ICML 2013*. https://arxiv.org/abs/1211.5063
