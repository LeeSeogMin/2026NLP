# 제2장 리서치 결과

## 조사 일자: 2026-01-02

---

## 1. 언어 모델 기초

### 언어 모델의 정의
- 텍스트의 확률 분포를 학습한 모델
- P(w₁, w₂, ..., wₙ): 단어 시퀀스의 결합 확률
- 연쇄 법칙: P(w₁:ₙ) = ∏P(wᵢ|w₁:ᵢ₋₁)

### Perplexity (혼잡도)
- 언어 모델 평가의 표준 지표
- 수식: PPL = 2^(-1/N × Σlog₂P(wᵢ|context))
- 직관적 해석: "다음 단어를 예측할 때 평균적으로 몇 개의 선택지 중에서 고민하는가"
- 낮을수록 좋은 모델 (더 확신을 가지고 예측)

출처: [Stanford NLP 3rd Ed. Ch.3](https://web.stanford.edu/~jurafsky/slp3/3.pdf)

---

## 2. N-gram 모델

### N-gram 정의
- 연속된 N개 단어의 시퀀스
- 마르코프 가정: P(wₙ|w₁:ₙ₋₁) ≈ P(wₙ|wₙ₋ₖ:ₙ₋₁)

### N-gram 종류
| N | 이름 | 고려 문맥 |
|---|------|----------|
| 1 | Unigram | 없음 (독립 가정) |
| 2 | Bigram | 바로 앞 1단어 |
| 3 | Trigram | 앞 2단어 |

### 확률 계산 (MLE)
- P(wₙ|wₙ₋₁) = Count(wₙ₋₁, wₙ) / Count(wₙ₋₁)
- 예: P("먹었다"|"밥을") = Count("밥을 먹었다") / Count("밥을")

### 희소성 문제 (Sparsity)
- 학습 데이터에 없는 N-gram → 확률 0
- 제로 확률 문제: 전체 문장 확률이 0이 됨

### 스무딩 기법
1. **Add-k Smoothing (Laplace)**
   - P(wₙ|wₙ₋₁) = (Count(wₙ₋₁, wₙ) + k) / (Count(wₙ₋₁) + k×V)
   - k=1: Laplace Smoothing
   - 단순하지만 확률 분포 왜곡 가능

2. **Backoff**
   - 높은 차수 N-gram 없으면 낮은 차수로 후퇴
   - Trigram → Bigram → Unigram

3. **Interpolation**
   - 여러 차수 N-gram 확률 가중 합
   - P = λ₁P₁ + λ₂P₂ + λ₃P₃

4. **Kneser-Ney Smoothing**
   - 가장 효과적인 스무딩 기법
   - 절대 할인 + 연속성 고려

---

## 3. Word2Vec

### 개요
- Mikolov et al. (2013) Google
- 단어를 밀집 벡터(Dense Vector)로 표현
- 유사한 문맥 → 유사한 벡터

### 두 가지 아키텍처

#### CBOW (Continuous Bag of Words)
- **목표**: 주변 단어(context) → 중심 단어(target) 예측
- **입력**: 윈도우 내 문맥 단어들 (one-hot)
- **출력**: 중심 단어 확률 분포
- **특징**: 빠른 학습, 빈번한 단어에 효과적
- **권장 윈도우**: 5

#### Skip-gram
- **목표**: 중심 단어(target) → 주변 단어(context) 예측
- **입력**: 중심 단어 (one-hot)
- **출력**: 문맥 단어 확률 분포
- **특징**: 희귀 단어에 효과적, 작은 데이터셋에 유리
- **권장 윈도우**: 10

### Negative Sampling
- Softmax 계산 비용 절감
- 정답 단어 + 무작위 오답 단어 샘플링
- 이진 분류 문제로 변환

### 핵심 차이점
| 항목 | CBOW | Skip-gram |
|------|------|-----------|
| 예측 방향 | Context → Target | Target → Context |
| 학습 속도 | 빠름 | 느림 |
| 희귀 단어 | 약함 | 강함 |
| 데이터 크기 | 대규모 적합 | 소규모도 가능 |

출처: [Baeldung](https://www.baeldung.com/cs/word-embeddings-cbow-vs-skip-gram), [Kaggle Tutorial](https://www.kaggle.com/code/ftaham/understanding-word2vec-cbow-skip-gram)

---

## 4. GloVe (Global Vectors)

### 개요
- Pennington et al. (2014) Stanford
- 전역 동시 출현(Co-occurrence) 행렬 활용
- Word2Vec과 달리 전체 코퍼스 통계 사용

### 핵심 아이디어
- 동시 출현 행렬 X 구축
- Xᵢⱼ: 단어 i 문맥에서 단어 j 출현 빈도
- 목표: wᵢᵀw̃ⱼ + bᵢ + b̃ⱼ = log(Xᵢⱼ)

### Word2Vec vs GloVe
| 항목 | Word2Vec | GloVe |
|------|----------|-------|
| 학습 방식 | 지역 문맥 (윈도우) | 전역 통계 (행렬) |
| 모델 타입 | 예측 모델 | 카운트 기반 |
| 메모리 | 효율적 | 동시출현 행렬 필요 |

---

## 5. FastText

### 개요
- Bojanowski et al. (2016) Facebook AI
- 서브워드(Subword) 정보 활용
- OOV(Out-of-Vocabulary) 문제 해결

### 핵심 아이디어
- 단어를 문자 N-gram으로 분해
- 예: "where" (n=3) → "<wh", "whe", "her", "ere", "re>"
- 단어 벡터 = 서브워드 벡터들의 합

### 장점
- **OOV 처리**: 처음 보는 단어도 임베딩 생성 가능
- **형태소 풍부한 언어**: 한국어, 터키어 등에 효과적
- **오타/신조어 처리**: SNS 데이터에 강점

### 성능 비교 (AG News 데이터셋)
| 모델 | 정확도 |
|------|--------|
| BERT | 90.88% |
| FastText | 86.91% |
| Skip-gram | 85.82% |
| CBOW | 86.15% |
| GloVe | 80.86% |

출처: [Medium Analytics Vidhya](https://medium.com/analytics-vidhya/word-embeddings-in-nlp-word2vec-glove-fasttext-24d4d4286a73)

---

## 6. 임베딩 공간의 특성

### 벡터 연산
- "왕 - 남자 + 여자 ≈ 여왕"
- "파리 - 프랑스 + 한국 ≈ 서울"
- 의미적 관계가 벡터 연산으로 표현

### 유사도 측정
- 코사인 유사도: cos(u, v) = (u·v) / (||u|| × ||v||)
- 범위: -1 ~ 1 (1에 가까울수록 유사)

### 시각화
- t-SNE: 고차원 → 2D/3D 변환
- UMAP: 더 빠르고 전역 구조 보존

---

## 7. 한국어 Word2Vec

### 사전학습 모델
- **Gensim**: 한국어 Word2Vec 모델 제공
- **KoNLPy**: 한국어 형태소 분석기
- **Korean Word2Vec**: 위키피디아 기반 학습 모델

### 한국어 특수성
- 형태소 분석 필수 (교착어)
- 띄어쓰기 불규칙
- FastText가 더 효과적일 수 있음

---

## 참고문헌

1. Jurafsky, D. & Martin, J.H. (2024). Speech and Language Processing (3rd ed.). Chapter 3. https://web.stanford.edu/~jurafsky/slp3/3.pdf
2. Mikolov, T. et al. (2013). Efficient Estimation of Word Representations in Vector Space. https://arxiv.org/abs/1301.3781
3. Pennington, J. et al. (2014). GloVe: Global Vectors for Word Representation. https://nlp.stanford.edu/pubs/glove.pdf
4. Bojanowski, P. et al. (2017). Enriching Word Vectors with Subword Information. https://arxiv.org/abs/1607.04606
