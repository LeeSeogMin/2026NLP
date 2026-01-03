# 제2장 집필계획서: 언어 모델의 진화 - 통계에서 신경망까지

## 개요

| 항목 | 내용 |
|------|------|
| **장 제목** | 언어 모델의 진화: 통계에서 신경망까지 |
| **대상 독자** | 딥러닝과 자연어처리를 처음 배우는 학부생 (3~4학년) |
| **장 유형** | 핵심 기술 장 |
| **이론:실습 비율** | 60:40 |
| **예상 분량** | 600-700줄 (약 35쪽) |

---

## 학습 목표

이 장을 마치면 다음을 수행할 수 있다:

- 언어 모델의 수학적 정의와 목적을 설명할 수 있다
- N-gram 모델을 직접 구현하고 텍스트를 생성할 수 있다
- Perplexity를 사용하여 언어 모델의 성능을 평가할 수 있다
- Word2Vec의 원리를 이해하고 사전학습 모델을 활용할 수 있다

---

## 절 구성

### 2.1 언어 모델의 기초 (~100줄)

**핵심 내용**:

1. **언어 모델의 역할과 목적**
   - 언어 모델이란: 텍스트의 확률 분포를 학습한 모델
   - 활용 분야: 텍스트 생성, 자동 완성, 기계 번역, 음성 인식

2. **조건부 확률과 언어 모델**
   - 결합 확률: P(w₁, w₂, ..., wₙ)
   - 연쇄 법칙(Chain Rule): P(w₁:ₙ) = ∏P(wᵢ|w₁:ᵢ₋₁)
   - 마르코프 가정(Markov Assumption)

3. **언어 모델 평가 지표: Perplexity**
   - Perplexity의 정의와 직관적 해석
   - 수식: PPL = 2^(-1/N × Σlog₂P(wᵢ|context))
   - 낮은 Perplexity = 더 좋은 모델

**필수 다이어그램**: 언어 모델의 확률 계산 흐름

---

### 2.2 통계 기반 언어 모델 (~140줄)

**핵심 내용**:

1. **N-gram 모델의 원리**
   - N-gram: 연속된 N개 단어의 시퀀스
   - 마르코프 가정 적용: P(wₙ|w₁:ₙ₋₁) ≈ P(wₙ|wₙ₋ₖ:ₙ₋₁)
   - 최대우도추정(MLE)으로 확률 계산

2. **Unigram, Bigram, Trigram 모델**
   - Unigram (N=1): 단어 독립 가정
   - Bigram (N=2): 바로 앞 단어만 고려
   - Trigram (N=3): 앞 두 단어 고려
   - N 증가에 따른 트레이드오프

3. **희소성 문제(Sparsity Problem)**
   - 학습 데이터에 없는 N-gram → 확률 0
   - 제로 확률 문제의 심각성

4. **스무딩(Smoothing) 기법**
   - Add-k Smoothing (Laplace Smoothing)
   - Backoff와 Interpolation
   - Kneser-Ney Smoothing 소개

5. **N-gram 모델의 한계점**
   - 고정된 문맥 길이
   - 의미적 유사성 무시
   - 대규모 N-gram 저장 문제

**필수 다이어그램**: N-gram 확률 계산 예시

---

### 2.3 신경망 기반 언어 모델로의 전환 (~80줄)

**핵심 내용**:

1. **신경망 언어 모델의 등장 배경**
   - N-gram의 한계 극복 필요성
   - Bengio et al. (2003) 신경망 언어 모델

2. **분산 표현(Distributed Representation)**
   - 원-핫 인코딩의 한계
   - 밀집 벡터(Dense Vector)의 장점
   - 의미적 유사성 표현 가능

3. **Word Embedding의 필요성**
   - 단어를 고정 차원 벡터로 변환
   - 유사한 단어 → 유사한 벡터
   - 임베딩 공간에서의 연산

---

### 2.4 단어 임베딩 (~140줄)

**핵심 내용**:

1. **Word2Vec (Mikolov et al., 2013)**
   - CBOW (Continuous Bag of Words): 주변 단어로 중심 단어 예측
   - Skip-gram: 중심 단어로 주변 단어 예측
   - Negative Sampling 기법
   - 학습 과정과 하이퍼파라미터

2. **GloVe (Global Vectors, 2014)**
   - 전역 동시 출현 행렬 활용
   - Word2Vec과의 차이점
   - 장단점 비교

3. **FastText (Facebook, 2016)**
   - 서브워드(Subword) 정보 활용
   - OOV(Out-of-Vocabulary) 문제 해결
   - 형태소가 풍부한 언어에 효과적

4. **임베딩 공간의 의미적 특성**
   - 벡터 연산: "왕 - 남자 + 여자 ≈ 여왕"
   - 유사도 측정: 코사인 유사도
   - 시각화: t-SNE, UMAP

**필수 다이어그램**: Word2Vec CBOW vs Skip-gram 구조

---

### 2.5 실습 (~140줄)

**핵심 내용**:

1. **N-gram 모델 직접 구현**
   - 토큰화 및 N-gram 생성
   - 빈도 기반 확률 계산
   - Add-k Smoothing 적용

2. **텍스트 생성 실습**
   - N-gram 기반 다음 단어 예측
   - 확률적 샘플링
   - 문장 생성

3. **텍스트 데이터 전처리**
   - 토큰화 (Tokenization)
   - 정제 (Cleaning)
   - 불용어 제거 (Stopword Removal)

4. **사전 학습된 Word2Vec 모델 활용**
   - Gensim 라이브러리 사용
   - 한국어 Word2Vec 모델 로드
   - 단어 유사도 측정 및 시각화

**실습 코드**:
- `2-2-ngram모델.py`: N-gram 모델 구현 및 텍스트 생성
- `2-4-word2vec실습.py`: Word2Vec 활용 및 유사도 측정
- `2-5-전처리.py`: 텍스트 전처리 파이프라인

---

## 생성할 파일 목록

### 문서
| 파일 경로 | 설명 |
|-----------|------|
| `schema/chap2.md` | 본 집필계획서 |
| `content/research/ch2-research.md` | 리서치 결과 |
| `content/drafts/ch2-draft.md` | 초안 |
| `content/reviews/ch2-review.md` | Multi-LLM 리뷰 결과 |
| `docs/ch2.md` | 최종 완성본 |

### 실습 코드
| 파일 경로 | 설명 |
|-----------|------|
| `practice/chapter2/code/2-2-ngram모델.py` | N-gram 구현 |
| `practice/chapter2/code/2-4-word2vec실습.py` | Word2Vec 활용 |
| `practice/chapter2/code/2-5-전처리.py` | 텍스트 전처리 |
| `practice/chapter2/code/requirements.txt` | 의존성 목록 |

### 그래픽 (Mermaid)
| 파일 경로 | 설명 |
|-----------|------|
| `content/graphics/ch2/fig-2-1-lm-probability.mmd` | 언어 모델 확률 계산 |
| `content/graphics/ch2/fig-2-2-ngram.mmd` | N-gram 예시 |
| `content/graphics/ch2/fig-2-3-word2vec.mmd` | Word2Vec 구조 |

---

## 참고문헌 (검증 필요)

1. Jurafsky, D. & Martin, J.H. (2024). Speech and Language Processing (3rd ed.). Chapter 3: N-gram Language Models.
2. Mikolov, T. et al. (2013). Efficient Estimation of Word Representations in Vector Space. *arXiv*.
3. Pennington, J. et al. (2014). GloVe: Global Vectors for Word Representation. *EMNLP*.
4. Bojanowski, P. et al. (2017). Enriching Word Vectors with Subword Information. *TACL*.
5. Bengio, Y. et al. (2003). A Neural Probabilistic Language Model. *JMLR*.

---

**작성일**: 2026-01-02
**상태**: 작성 중
