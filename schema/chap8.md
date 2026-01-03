# 8장 집필계획서: 텍스트 속 숨겨진 주제 찾기 - 임베딩 기반 토픽 모델링

## 개요

**장 제목**: 텍스트 속 숨겨진 주제 찾기: 임베딩 기반 토픽 모델링
**대상 독자**: 딥러닝과 자연어처리를 처음 배우는 학부생 (3~4학년)
**장 유형**: 실습 중심 장 (이론:실습 = 40:60)
**예상 분량**: 550-650줄

---

## 학습 목표

이 장을 마치면 다음을 수행할 수 있다:
- 토픽 모델링의 개념과 활용 분야를 이해한다
- 전통적 LDA와 BERTopic의 차이점을 설명할 수 있다
- BERTopic의 5단계 파이프라인을 이해한다
- UMAP, HDBSCAN, c-TF-IDF의 역할을 설명할 수 있다
- BERTopic으로 실제 데이터의 토픽을 추출하고 시각화할 수 있다
- Dynamic Topic Modeling으로 시간에 따른 토픽 변화를 분석할 수 있다

---

## 절 구성

### 8.1 토픽 모델링 개요 (~80줄)

**핵심 내용**:
- 토픽 모델링이란?
  - 대량의 문서에서 숨겨진 주제 발견
  - 비지도 학습 기반 문서 분류
- 토픽 모델링 활용 분야
  - 뉴스 기사 분류
  - 학술 논문 트렌드 분석
  - 소셜 미디어 여론 분석
  - 고객 피드백 분석
- 전통적 LDA (Latent Dirichlet Allocation)
  - 확률적 생성 모델
  - 문서-토픽, 토픽-단어 분포
- LDA의 한계점
  - Bag-of-Words 기반 (어순/문맥 무시)
  - 토픽 수 사전 지정 필요
  - 짧은 문서에서 성능 저하

**다이어그램**: LDA vs BERTopic 비교도

### 8.2 BERTopic 소개 (~70줄)

**핵심 내용**:
- BERTopic의 등장 배경
  - Transformer 기반 임베딩 활용
  - 의미론적 유사성 반영
- BERTopic vs LDA 비교
  - 임베딩 vs Bag-of-Words
  - 자동 토픽 수 결정 vs 사전 지정
  - 문맥 이해 vs 단어 빈도
- BERTopic의 강점
  - 짧은 문서에서도 우수한 성능
  - 직관적인 토픽 표현
  - 다양한 시각화 지원
  - 모듈화된 파이프라인

### 8.3 BERTopic 아키텍처 (~100줄)

**핵심 내용**:
- 5단계 파이프라인
  1. **Document Embedding**: Sentence-BERT로 문서 벡터화
  2. **Dimensionality Reduction**: UMAP으로 차원 축소
  3. **Clustering**: HDBSCAN으로 밀도 기반 군집화
  4. **Topic Representation**: c-TF-IDF로 토픽 키워드 추출
  5. **Fine-tuning**: (선택) 토픽 표현 개선
- 각 단계의 역할과 연결

**다이어그램**: BERTopic 5단계 파이프라인

### 8.4 주요 구성 요소 심화 (~100줄)

**핵심 내용**:
- **Sentence Transformers**
  - all-MiniLM-L6-v2: 경량, 빠름
  - paraphrase-multilingual: 다국어 지원
  - 한국어 모델: KoSimCSE, Ko-Sentence-BERT
- **UMAP (Uniform Manifold Approximation and Projection)**
  - t-SNE보다 빠르고 글로벌 구조 보존
  - 주요 하이퍼파라미터: n_neighbors, n_components, min_dist
- **HDBSCAN (Hierarchical Density-Based Spatial Clustering)**
  - 밀도 기반 클러스터링
  - 노이즈 포인트 자동 처리
  - 클러스터 수 자동 결정
  - min_cluster_size, min_samples
- **c-TF-IDF (Class-based TF-IDF)**
  - 토픽(클래스)별 중요 단어 추출
  - TF-IDF의 클래스 버전
  - 수식: c-TF-IDF = TF × log(1 + A/f)

### 8.5 토픽 표현 및 해석 (~80줄)

**핵심 내용**:
- 토픽별 주요 키워드 추출
  - get_topic_info()
  - get_topic(topic_id)
- 토픽 레이블링
  - 자동 레이블 vs 수동 레이블
  - LLM 기반 레이블 생성
- Outlier 처리
  - Topic -1 (노이즈)
  - reduce_outliers() 메서드
- 토픽 간 유사도 분석
  - 토픽 임베딩
  - 계층적 토픽 구조

### 8.6 고급 기능 (~70줄)

**핵심 내용**:
- **Dynamic Topic Modeling**
  - 시간에 따른 토픽 변화 추적
  - topics_over_time()
- **Guided Topic Modeling**
  - 시드 단어로 토픽 유도
  - seed_topic_list 파라미터
- **Hierarchical Topic Modeling**
  - 토픽 간 계층 구조
  - hierarchical_topics()
- **토픽 병합 및 축소**
  - reduce_topics()
  - merge_topics()

### 8.7 실습: BERTopic 활용 (~150줄)

**핵심 내용**:
- BERTopic 설치 및 기본 사용법
- 한국어 뉴스 데이터로 토픽 모델링
- 토픽 시각화
  - Intertopic Distance Map
  - Topic Word Scores
  - Topic Hierarchy
- 시간별 토픽 트렌드 분석
- 모델 저장 및 로드

**실습 코드**:
- `8-1-bertopic-basic.py`: 기본 토픽 모델링
- `8-5-topic-visualization.py`: 토픽 시각화
- `8-6-dynamic-topics.py`: 시간별 토픽 분석

---

## 생성할 파일 목록

### 문서
- `schema/chap8.md`: 집필계획서 (현재 파일)
- `content/research/ch8-research.md`: 리서치 결과
- `content/drafts/ch8-draft.md`: 초안
- `docs/ch8.md`: 최종 완성본

### 실습 코드
- `practice/chapter8/code/8-1-bertopic-basic.py`
- `practice/chapter8/code/8-5-topic-visualization.py`
- `practice/chapter8/code/8-6-dynamic-topics.py`
- `practice/chapter8/code/requirements.txt`

### 그래픽
- `content/graphics/ch8/fig-8-1-lda-vs-bertopic.mmd`
- `content/graphics/ch8/fig-8-2-bertopic-pipeline.mmd`
- `content/graphics/ch8/fig-8-3-ctfidf.mmd`

---

## 핵심 개념

1. **토픽 모델링**: 문서 집합에서 숨겨진 주제를 자동으로 발견하는 기법

2. **BERTopic 파이프라인**:
   - Embedding → UMAP → HDBSCAN → c-TF-IDF

3. **c-TF-IDF**:
   c-TF-IDF(t,c) = TF(t,c) × log(1 + A / f(t))
   - t: 단어, c: 클래스(토픽)
   - A: 전체 문서 수, f(t): 단어 t가 등장한 문서 수

4. **HDBSCAN 장점**:
   - 클러스터 수 자동 결정
   - 다양한 밀도의 클러스터 탐지
   - 노이즈 포인트 처리

---

## 7단계 워크플로우 실행 계획

### 1단계: 집필계획서 작성 ✓
- `schema/chap8.md` 작성 완료

### 2단계: 자료 조사
- BERTopic 공식 문서 및 논문
- UMAP, HDBSCAN 알고리즘
- c-TF-IDF 원리
- 한국어 토픽 모델링 사례

### 3단계: 정보 구조화
- 핵심 개념 정리
- 파이프라인 다이어그램 설계
- 실습 시나리오 구성

### 4단계: 구현 및 문서화
- 실습 코드 작성 및 실행
- 본문 초안 작성
- Mermaid 다이어그램 제작

### 5단계: 최적화
- 문체 일관성 검토
- 6장(Transformer)과의 연결성 확인
- 용어 통일

### 6단계: 품질 검증
- Multi-LLM Review (GPT-4o + grok-4-1-fast-reasoning)
- `docs/ch8.md`로 최종 저장

### 7단계: MS Word 변환
- `npm run convert:chapter 8` 실행

---

## 참고문헌

- Grootendorst, M. (2022). BERTopic: Neural topic modeling with a class-based TF-IDF procedure. arXiv:2203.05794.
- BERTopic Documentation: https://maartengr.github.io/BERTopic/
- McInnes, L., Healy, J., & Melville, J. (2018). UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction.
- Campello, R. J., et al. (2013). Density-Based Clustering Based on Hierarchical Density Estimates.
