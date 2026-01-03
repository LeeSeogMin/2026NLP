# 10장 집필계획서: LLM 시대 (2) - GPT 아키텍처와 생성 모델

## 개요

**장 제목**: LLM 시대 (2) - GPT 아키텍처와 생성 모델
**대상 독자**: 딥러닝과 자연어처리를 처음 배우는 학부생 (3~4학년)
**장 유형**: 핵심 기술 장 (이론:실습 = 60:40)
**예상 분량**: 600-700줄

---

## 학습 목표

이 장을 마치면 다음을 수행할 수 있다:
- 자기회귀(Autoregressive) 언어 모델의 원리를 이해한다
- GPT의 아키텍처와 Causal Self-Attention을 설명할 수 있다
- 다양한 텍스트 생성 전략(Greedy, Beam, Sampling)을 비교할 수 있다
- GPT-2를 활용하여 텍스트를 생성할 수 있다
- Zero-shot, Few-shot Learning의 개념을 이해한다

---

## 절 구성

### 10.1 자기회귀 언어 모델 (~80줄)

**핵심 내용**:
- Autoregressive Language Modeling의 개념
- Next Token Prediction (다음 토큰 예측)
- GPT의 등장 배경과 의의
- BERT vs GPT: Encoder-only vs Decoder-only

**다이어그램**: Autoregressive vs Masked LM 비교

### 10.2 GPT 아키텍처 (~100줄)

**핵심 내용**:
- Decoder-only 구조의 의미
- Causal Self-Attention (Masked Self-Attention)
  - 미래 토큰 마스킹의 필요성
  - 하삼각 마스크 행렬
- GPT-1, GPT-2, GPT-3의 발전 과정
- 모델 크기 비교

**다이어그램**: GPT 아키텍처 구조

### 10.3 GPT 아키텍처 상세 (~100줄)

**핵심 내용**:
- Transformer Decoder 블록 구성
  - Masked Multi-Head Attention
  - Feed-Forward Network
  - LayerNorm 위치 (Pre-LN vs Post-LN)
- GPT-2 구조 분석
  - 모델 버전별 파라미터 (124M, 355M, 774M, 1.5B)
  - 층 수, 은닉 차원, 어텐션 헤드
- 입력 임베딩 (Token + Position)

**표**: GPT-2 모델 크기별 비교

### 10.4 텍스트 생성 메커니즘 (~120줄)

**핵심 내용**:
- Greedy Search (탐욕 검색)
  - 가장 높은 확률의 토큰 선택
  - 한계점: 반복, 다양성 부족
- Beam Search (빔 검색)
  - 여러 후보 유지
  - Beam Width의 영향
- Sampling 기법
  - Temperature Sampling
  - Top-k Sampling
  - Top-p (Nucleus) Sampling
- 생성 전략 비교 및 선택 가이드

**다이어그램**: 텍스트 생성 전략 비교

### 10.5 GPT의 능력 (~80줄)

**핵심 내용**:
- Zero-shot Learning
  - 예시 없이 태스크 수행
- Few-shot Learning
  - 몇 개의 예시만으로 학습
- In-Context Learning
  - 프롬프트 내에서 학습
- Emergent Abilities (창발적 능력)
  - 모델 크기에 따른 새로운 능력 출현

### 10.6 프롬프트 엔지니어링 기초 (~80줄)

**핵심 내용**:
- 프롬프트의 중요성
- 효과적인 프롬프트 작성법
  - 명확한 지시
  - 맥락 제공
  - 출력 형식 지정
- Instruction Following
- Chain-of-Thought Prompting 소개

### 10.7 텍스트 생성 평가 (~60줄)

**핵심 내용**:
- Perplexity (혼란도)
- BLEU Score (기계 번역 평가)
- ROUGE Score (요약 평가)
- 정성적 평가의 중요성

### 10.8 실습: GPT-2 텍스트 생성 (~100줄)

**핵심 내용**:
- GPT-2 모델 로드
- 다양한 디코딩 전략 비교
  - Greedy vs Beam vs Sampling
- Temperature, Top-k, Top-p 파라미터 조정
- Zero-shot, Few-shot 프롬프팅 실습
- 텍스트 완성 예제

---

## 생성할 파일 목록

### 문서
- `schema/chap10.md`: 집필계획서 (본 파일)
- `content/research/ch10-research.md`: 리서치 결과
- `content/drafts/ch10-draft.md`: 초안
- `docs/ch10.md`: 최종 완성본

### 실습 코드
- `practice/chapter10/code/10-1-gpt-basics.py`: GPT 기본 사용법
- `practice/chapter10/code/10-4-generation-strategies.py`: 생성 전략 비교
- `practice/chapter10/code/10-8-gpt-applications.py`: GPT 응용 실습
- `practice/chapter10/code/requirements.txt`

### 그래픽
- `content/graphics/ch10/fig-10-1-autoregressive-vs-masked.mmd`: AR vs MLM 비교
- `content/graphics/ch10/fig-10-2-gpt-architecture.mmd`: GPT 아키텍처
- `content/graphics/ch10/fig-10-3-causal-attention.mmd`: Causal Attention 마스크
- `content/graphics/ch10/fig-10-4-generation-strategies.mmd`: 생성 전략 비교

---

## 9장과의 연계

- 9장에서 다룬 BERT(Encoder-only)와 대비하여 GPT(Decoder-only) 설명
- Transformer 아키텍처 기반이지만 목적과 구조가 다름을 강조
- 양방향(Bidirectional) vs 단방향(Unidirectional) 문맥 이해 비교

---

## 핵심 개념 정리

| 개념 | 설명 |
|------|------|
| Autoregressive | 이전 토큰들을 조건으로 다음 토큰 예측 |
| Causal Attention | 미래 토큰을 볼 수 없도록 마스킹 |
| Temperature | 출력 분포의 예리함/평탄함 조절 |
| Top-k Sampling | 상위 k개 토큰에서만 샘플링 |
| Top-p Sampling | 누적 확률 p까지의 토큰에서 샘플링 |
| Zero-shot | 예시 없이 태스크 수행 |
| Few-shot | 소수 예시로 태스크 수행 |

---

## 마지막 업데이트

2026-01-03
