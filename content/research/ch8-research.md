# 8장 리서치 결과: 텍스트 속 숨겨진 주제 찾기 - 임베딩 기반 토픽 모델링

**조사일**: 2026-01-03
**조사 주제**: BERTopic, UMAP, HDBSCAN, c-TF-IDF, 토픽 모델링

---

## 1. 토픽 모델링 개요

### 1.1 토픽 모델링이란?
- 대량의 문서 집합에서 숨겨진 주제(토픽)를 자동으로 발견하는 비지도 학습 기법
- 문서 분류, 정보 검색, 추천 시스템 등에 활용
- 주요 응용: 뉴스 기사 분류, 학술 논문 트렌드 분석, 소셜 미디어 분석, 고객 피드백 분석

### 1.2 전통적 LDA (Latent Dirichlet Allocation)
- Blei et al. (2003)이 제안한 확률적 생성 모델
- 문서-토픽 분포와 토픽-단어 분포를 동시에 학습
- 변분 추론(Variational Inference) 또는 깁스 샘플링(Gibbs Sampling)으로 추정

### 1.3 LDA의 한계점
1. **토픽 수 사전 지정 필요**: k (토픽 수)를 미리 지정해야 함
2. **Bag-of-Words 기반**: 어순과 문맥 무시
3. **짧은 문서에서 성능 저하**: 데이터 희소성 문제
4. **하이퍼파라미터 튜닝**: α, β 파라미터 조정 필요
5. **계산 비용**: 대규모 데이터셋에서 시간 소요
6. **토픽 해석 어려움**: 인간의 직관적 분류와 일치하지 않을 수 있음

---

## 2. BERTopic 소개

### 2.1 등장 배경
- Grootendorst (2022)가 제안
- Transformer 기반 사전학습 언어모델의 문맥적 임베딩 활용
- 의미론적 유사성을 반영한 토픽 클러스터링

### 2.2 BERTopic vs LDA 비교

| 특성 | LDA | BERTopic |
|------|-----|----------|
| 문서 표현 | Bag-of-Words | Transformer 임베딩 |
| 토픽 수 | 사전 지정 필요 | 자동 결정 (HDBSCAN) |
| 문맥 이해 | 불가 | 가능 |
| 짧은 문서 | 성능 저하 | 우수 |
| 일관성(Coherence) | ~0.38 | ~0.76 (약 2배) |
| 노이즈 처리 | 없음 | Topic -1로 처리 |

### 2.3 BERTopic의 강점
- 짧은 문서에서도 우수한 성능
- 직관적인 토픽 키워드 표현
- 다양한 시각화 지원 (Plotly 기반)
- 모듈화된 파이프라인 (각 단계 교체 가능)
- Dynamic Topic Modeling 지원

---

## 3. BERTopic 5단계 파이프라인

### 3.1 파이프라인 개요
```
문서 → [1. Embedding] → [2. UMAP] → [3. HDBSCAN] → [4. c-TF-IDF] → 토픽
```

### 3.2 각 단계 설명

**1단계: Document Embedding (문서 임베딩)**
- Sentence-BERT (SBERT)로 문서를 고차원 벡터로 변환
- 의미적으로 유사한 문서는 벡터 공간에서 가까이 위치
- 기본 모델: `all-MiniLM-L6-v2` (384차원, 빠름)

**2단계: Dimensionality Reduction (차원 축소)**
- UMAP으로 고차원 임베딩을 저차원(기본 5차원)으로 축소
- 클러스터링 효율성 향상 및 노이즈 제거
- 로컬/글로벌 구조 모두 보존

**3단계: Clustering (군집화)**
- HDBSCAN으로 밀도 기반 클러스터링
- 클러스터 수 자동 결정
- 노이즈 포인트는 Topic -1로 할당

**4단계: Topic Representation (토픽 표현)**
- c-TF-IDF로 각 토픽의 대표 키워드 추출
- 클래스(토픽) 단위로 TF-IDF 계산

**5단계: Fine-tuning (선택)**
- MMR, KeyBERT, LLM 등을 활용한 토픽 표현 개선
- 토픽 병합, 축소, 레이블링

---

## 4. UMAP (Uniform Manifold Approximation and Projection)

### 4.1 개요
- McInnes, Healy, Melville (2018) 제안
- 비선형 차원 축소 알고리즘
- 리만 기하학과 위상 데이터 분석에 기반

### 4.2 UMAP vs t-SNE

| 특성 | t-SNE | UMAP |
|------|-------|------|
| 속도 | 느림 | 빠름 |
| 글로벌 구조 보존 | 약함 | 강함 |
| 로컬 구조 보존 | 강함 | 강함 |
| 확장성 | 낮음 | 높음 |
| 거리 함수 | 제한적 | 다양 지원 |

### 4.3 핵심 하이퍼파라미터
- **n_neighbors**: 로컬 구조 결정 (기본 15)
- **n_components**: 출력 차원 (BERTopic 기본 5)
- **min_dist**: 포인트 간 최소 거리 (기본 0.0)
- **metric**: 거리 함수 (cosine, euclidean 등)

### 4.4 작동 원리
1. 각 데이터 포인트의 최근접 이웃 탐색
2. 로컬 거리 스케일 계산 (밀도 적응적)
3. 퍼지 심플리셜 집합(fuzzy simplicial set) 구성
4. 저차원에서 유사한 구조 최적화 (교차 엔트로피 최소화)

---

## 5. HDBSCAN (Hierarchical Density-Based Spatial Clustering)

### 5.1 개요
- Campello et al. (2013, 2015) 제안
- DBSCAN의 계층적 확장
- 다양한 밀도의 클러스터 탐지 가능

### 5.2 HDBSCAN vs DBSCAN

| 특성 | DBSCAN | HDBSCAN |
|------|--------|---------|
| 클러스터 수 | 자동 결정 | 자동 결정 |
| 다양한 밀도 | 어려움 | 가능 |
| 파라미터 | eps, min_samples | min_cluster_size, min_samples |
| 노이즈 처리 | 포함 | 포함 |

### 5.3 핵심 하이퍼파라미터
- **min_cluster_size**: 최소 클러스터 크기 (가장 중요)
- **min_samples**: 밀도 추정용 샘플 수
- **cluster_selection_epsilon**: 클러스터 선택 임계값
- **cluster_selection_method**: 'eom' (과잉 질량) 또는 'leaf'

### 5.4 작동 원리
1. 상호 도달 거리(mutual reachability distance) 계산
2. 최소 신장 트리(minimum spanning tree) 구성
3. 클러스터 계층 구조 생성
4. 클러스터 응축(condensing) 및 안정성 기반 선택

---

## 6. c-TF-IDF (Class-based TF-IDF)

### 6.1 개요
- BERTopic의 핵심 구성 요소
- 클래스(토픽) 단위로 중요 단어 추출
- 각 토픽을 대표하는 키워드 생성

### 6.2 수식

**전통적 TF-IDF**:
```
TF-IDF(t, d) = TF(t, d) × IDF(t)
IDF(t) = log(N / df(t))
```

**c-TF-IDF**:
```
c-TF-IDF(t, c) = TF(t, c) × log(1 + A / f(t))
```
- t: 단어
- c: 클래스(토픽)
- TF(t, c): 클래스 c 내 단어 t의 빈도
- A: 전체 문서 수 (평균 단어 수)
- f(t): 단어 t가 등장한 문서 수

### 6.3 작동 방식
1. 각 클러스터(토픽)의 모든 문서를 하나의 "슈퍼 문서"로 결합
2. 각 슈퍼 문서에 대해 TF-IDF 계산
3. 토픽별로 가장 높은 c-TF-IDF 점수를 가진 단어 = 토픽 키워드

### 6.4 특징
- 높은 c-TF-IDF: 해당 토픽에서 중요하고 다른 토픽에서는 드문 단어
- 낮은 c-TF-IDF: 여러 토픽에서 공통적이거나 중요하지 않은 단어

---

## 7. 고급 기능

### 7.1 Dynamic Topic Modeling
- 시간에 따른 토픽 변화 추적
- `topics_over_time()` 메서드
- 각 시점별로 c-TF-IDF 재계산

**사용법**:
```python
topics_over_time = topic_model.topics_over_time(docs, timestamps)
topic_model.visualize_topics_over_time(topics_over_time)
```

### 7.2 Guided Topic Modeling
- 시드 단어로 토픽 유도
- `seed_topic_list` 파라미터
- 특정 주제에 대한 토픽 형성 유도

### 7.3 Hierarchical Topic Modeling
- 토픽 간 계층 구조 파악
- `hierarchical_topics()` 메서드
- 상위/하위 토픽 관계 분석

### 7.4 Outlier 처리
- Topic -1: HDBSCAN이 노이즈로 분류한 문서
- `reduce_outliers()`: 아웃라이어를 가장 가까운 토픽에 재할당

### 7.5 토픽 조작
- `reduce_topics(nr_topics)`: 토픽 수 축소
- `merge_topics(topics_to_merge)`: 특정 토픽 병합

---

## 8. 한국어 BERTopic 적용

### 8.1 토크나이저 수정
- 기본 띄어쓰기 토큰화 → 형태소 분석기 사용
- Mecab, Komoran, Okt 등 활용
- CountVectorizer에 커스텀 토크나이저 적용

### 8.2 한국어 Sentence Transformer
- `paraphrase-multilingual-MiniLM-L12-v2`: 다국어 지원
- `xlm-r-100langs-bert-base-nli-stsb-mean-tokens`: 100개 언어
- `ko-sbert-nli`: 한국어 특화 모델
- KoSimCSE, Ko-Sentence-BERT 등

### 8.3 KoBERTopic
- ukairia777/KoBERTopic: 한국어 최적화 버전
- Mecab + 다국어 SBERT 조합
- 불용어 처리 추가 시 성능 향상

---

## 9. 시각화 기능

### 9.1 Intertopic Distance Map
- 토픽 간 거리와 크기 시각화
- `visualize_topics()`
- 2D 공간에서 토픽 분포 확인

### 9.2 Topic Word Scores
- 토픽별 상위 키워드와 점수
- `visualize_barchart()`
- 막대 그래프로 키워드 중요도 표현

### 9.3 Topic Hierarchy
- 토픽 계층 구조
- `visualize_hierarchy()`
- 덴드로그램 형태로 토픽 관계 표시

### 9.4 Document Visualization
- 문서-토픽 분포
- `visualize_documents()`
- 각 문서의 토픽 할당 시각화

### 9.5 Topics over Time
- 시간별 토픽 트렌드
- `visualize_topics_over_time()`
- Dynamic Topic Modeling 결과 시각화

---

## 10. 참고문헌

### 핵심 논문
1. Grootendorst, M. (2022). BERTopic: Neural topic modeling with a class-based TF-IDF procedure. arXiv:2203.05794. https://arxiv.org/abs/2203.05794
2. McInnes, L., Healy, J., & Melville, J. (2018). UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction. arXiv:1802.03426. https://arxiv.org/abs/1802.03426
3. Campello, R. J., Moulavi, D., & Sander, J. (2013). Density-Based Clustering Based on Hierarchical Density Estimates. PAKDD 2013.
4. Blei, D. M., Ng, A. Y., & Jordan, M. I. (2003). Latent Dirichlet Allocation. Journal of Machine Learning Research.

### 공식 문서
- BERTopic Documentation: https://maartengr.github.io/BERTopic/
- UMAP Documentation: https://umap-learn.readthedocs.io/
- HDBSCAN Documentation: https://hdbscan.readthedocs.io/
- Scikit-learn HDBSCAN: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.HDBSCAN.html

### 한국어 자료
- KoBERTopic: https://github.com/ukairia777/KoBERTopic
- 한국어 CTM: https://wikidocs.net/161414
- ko-sentence-transformers: https://pypi.org/project/ko-sentence-transformers/

### 최근 연구 (2024-2025)
- Mutsaddi et al. (2025): BERTopic C_v coherence ~0.76 vs LDA ~0.38
- Kaur et al. (2024): BERTopic의 온라인 토론 분석 우수성
- Koterwa et al. (2025): 불용어 제거로 coherence/diversity 향상

---

## 11. 실습 데이터 및 코드 계획

### 11.1 데이터셋
- 한국어 뉴스 기사 데이터 (네이버 뉴스 등)
- 시간 정보 포함하여 Dynamic Topic Modeling 가능

### 11.2 실습 코드 구성
1. `8-1-bertopic-basic.py`: 기본 토픽 모델링
2. `8-5-topic-visualization.py`: 토픽 시각화
3. `8-6-dynamic-topics.py`: 시간별 토픽 분석

### 11.3 필수 라이브러리
- bertopic
- sentence-transformers
- umap-learn
- hdbscan
- plotly
- scikit-learn
