## 8주차 B회차: BERTopic 실습 + 결과 해석

> **미션**: BERTopic으로 뉴스 기사 데이터셋의 토픽을 추출하고 결과를 시각화·해석하며, 토픽 모델링 품질을 평가할 수 있다

### 수업 타임라인

| 시간 | 내용 | Copilot 사용 |
|------|------|-------------|
| 00:00~00:05 | 조 편성 + 역할 배분 (조원 A/B) | 사용 안 함 |
| 00:05~00:10 | A회차 핵심 리캡 + 과제 스펙 재확인 | 사용 안 함 |
| 00:10~00:55 | 2인1조 Copilot 구현 (체크포인트 3회) | 적극 사용 |
| 00:55~01:00 | Google Classroom 제출 (조별 1부) | 사용 안 함 |
| 01:00~01:20 | 결과 토론 (주제 해석·품질 평가) | 사용 안 함 |
| 01:20~01:28 | 핵심 정리 | 사용 안 함 |
| 01:28~01:30 | 다음 주 예고 | 사용 안 함 |

---

### A회차 핵심 리캡

**토픽 모델링의 정의**:
- 문서 집합에서 숨겨진 주제를 자동 발견하는 기계학습 기법
- 규모가 큰 텍스트 데이터에서 개별 분석이 불가능할 때 유용

**LDA (Latent Dirichlet Allocation)**:
- 확률론적 접근: 각 문서는 여러 주제의 혼합물, 각 주제는 단어들의 확률 분포
- 장점: 수학적으로 정교함
- 한계: 단어 의미를 이해하지 못하고, 주제 개수를 미리 정해야 함

**BERTopic의 4단계 파이프라인**:
1. Document Embedding (BERT): 문서를 의미 벡터로 변환 (384차원)
2. Dimensionality Reduction (UMAP): 고차원 벡터를 저차원(5D)으로 축소
3. Clustering (HDBSCAN): 의미론적으로 비슷한 문서끼리 주제로 그룹화 (자동 주제 개수 결정)
4. Topic Representation (c-TF-IDF): 각 주제의 특징적인 단어 추출

**BERTopic이 LDA보다 우수한 이유**:
- BERT 사전학습 임베딩으로 문맥을 고려한 의미 벡터 획득
- 거리 기반 클러스터링으로 의미론적 유사도 직접 포착
- HDBSCAN으로 주제 개수를 자동 결정 (사전 설정 불필요)

**실습 연계**:
- 지난 수업의 이론을 실제 뉴스 데이터로 구현한다
- BERTopic의 각 단계 결과를 이해하고 해석한다
- 토픽 모델링 결과의 품질을 평가한다

---

### 과제 스펙

**과제**: BERTopic 파이프라인 구현 + 결과 시각화 및 해석 리포트

**제출 형태**: 조별 1부, Google Classroom 업로드

**파일 구성**:
- 구현 코드 파일 (`*.py`)
- 시각화 결과 이미지 (Bar Chart, Heatmap, Network 3개 이상)
- 분석 리포트 (주제 해석 + 품질 평가, 2~3페이지)

**검증 기준**:
- ✓ BERTopic 모델 초기화 및 학습 (데이터 로드, 파이프라인 구성)
- ✓ 각 주제의 핵심 단어 추출 및 의미 해석
- ✓ 주제 시각화 (Bar Chart, Heatmap, Network)
- ✓ 특정 문서의 주제 분포 분석 및 해석
- ✓ 토픽 모델링 품질 평가 (Coherence Score 또는 Diversity 메트릭)

---

### 2인1조 실습

> **Copilot 활용**: BERTopic 코드를 한 줄씩 직접 작성해본 뒤, Copilot에게 "BBC 뉴스 데이터로 BERTopic을 실행해줄래?", "주제 시각화 코드 작성해줄 수 있어?", "토픽 코히어런스 스코어를 계산하는 함수 만들어줘" 같이 단계적으로 요청한다. Copilot의 제안을 검토하고 결과를 실제로 해석하는 과정에서 토픽 모델링의 작동 원리를 깊이 이해할 수 있다.

**역할 분담**:
- **조원 A (드라이버)**: 코드 작성 및 실행, 데이터 로드, 시각화 확인
- **조원 B (네비게이터)**: 로직 검토, Copilot 프롬프트 설계, 결과 해석 및 기록
- **체크포인트마다 역할 교대**: 드라이버와 네비게이터를 번갈아가며 진행하여 두 명 모두 전체 구현을 이해한다

---

#### 체크포인트 1: BERTopic 모델 초기화 + 데이터 로드 + 학습 (15분)

**목표**: BBC 뉴스 데이터셋을 로드하고 BERTopic 모델을 초기화 및 학습하여, 자동으로 발견된 주제와 각 주제의 크기를 확인한다

**핵심 단계**:

① **필요 라이브러리 설치 및 데이터 로드**

```python
# 필요 라이브러리
import pandas as pd
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from bertopic import BERTopic
import warnings
warnings.filterwarnings('ignore')

# BBC 뉴스 데이터셋 로드 (약 1,500개 기사)
print("BBC 뉴스 데이터셋 로드 중...")
dataset = load_dataset("bbc_news_classification")
documents = dataset['train']['text'][:1500]
labels = dataset['train']['label'][:1500]

print(f"총 {len(documents)}개 문서 로드됨")

# 데이터셋 기본 통계
print(f"\n데이터셋 정보:")
print(f"  문서 개수: {len(documents)}")
print(f"  평균 길이: {np.mean([len(d.split()) for d in documents]):.1f} 단어")
print(f"  최소/최대 길이: {min([len(d.split()) for d in documents])}/{max([len(d.split()) for d in documents])} 단어")

# 샘플 문서 3개 보기
print(f"\n샘플 문서:")
for i in range(3):
    print(f"\n문서 {i+1}:")
    print(documents[i][:200] + "...")
```

예상 동작:
```
BBC 뉴스 데이터셋 로드 중...
총 1500개 문서 로드됨

데이터셋 정보:
  문서 개수: 1500
  평균 길이: 152.3 단어
  최소/최대 길이: 12/892 단어

샘플 문서:
문서 1:
Ad sales boost Time Warner profit Quick jump in revenues To the delight of ...

문서 2:
Dollar gains on Greenspan speech The dollar has hit its highest level in almost ...

문서 3:
Yukos unit buyer offers $9 bln-dlrs Yuganskneftegaz, the main production unit ...
```

② **BERTopic 모델 초기화 및 학습**

```python
# BERT 임베딩 모델 (Multilingual, 한국어 포함)
print("\nBERT 임베딩 모델 로드 중...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# BERTopic 모델 초기화
print("BERTopic 모델 초기화...")
topic_model = BERTopic(
    embedding_model=embedding_model,
    language="english",
    nr_topics="auto",  # 자동 주제 개수 결정
    min_topic_size=10,  # 최소 10개 문서 이상의 주제만 인정
    verbose=True
)

# 모델 학습
print("\nBERTopic 학습 중 (이 과정은 1~2분 소요)...")
topics, probabilities = topic_model.fit_transform(documents)

print(f"\n학습 완료!")
print(f"발견된 주제 개수: {len(set(topics)) - (1 if -1 in topics else 0)}")
print(f"노이즈 (명확한 주제 없는 문서): {sum(topics == -1)}개 ({sum(topics == -1) / len(topics) * 100:.1f}%)")
```

예상 결과:
```
BERT 임베딩 모델 로드 중...

BERTopic 모델 초기화...

BERTopic 학습 중 (이 과정은 1~2분 소요)...
2024-01-15 14:32:45 - Batches: 100%|██████████| 48/48 [00:23<00:00,  2.07it/s]
2024-01-15 14:32:47 - Reduced dimensionality with UMAP

학습 완료!
발견된 주제 개수: 6
노이즈 (명확한 주제 없는 문서): 47개 (3.1%)
```

③ **주제별 문서 개수 및 상위 단어 확인**

```python
# 주제별 정보
topic_info = topic_model.get_topic_info()
print("\n주제별 상위 단어:")
for idx, row in topic_info.iterrows():
    topic_id = row['Topic']
    if topic_id == -1:
        continue  # 노이즈는 건너뛰기

    count = row['Count']
    words = row['Name']
    print(f"\n주제 {topic_id} ({count}개 문서):")
    print(f"  대표 단어: {words}")

# 주제 분포 시각화
topic_counts = pd.Series(topics).value_counts().sort_index()
print("\n주제별 문서 개수 (상세):")
for topic_id, count in topic_counts.items():
    if topic_id == -1:
        print(f"  노이즈: {count}")
    else:
        percentage = count / len(topics) * 100
        print(f"  주제 {topic_id}: {count}개 ({percentage:.1f}%)")
```

예상 결과:
```
주제별 상위 단어:
주제 0 (287개 문서):
  대표 단어: business_company_sales_market

주제 1 (256개 문서):
  대표 단어: sport_game_team_player

주제 2 (198개 문서):
  대표 단어: technology_data_software_system

주제 3 (154개 문서):
  대표 단어: world_country_government_nation

주제 4 (323개 문서):
  대표 단어: political_election_party_government

주제 5 (135개 문서):
  대표 단어: entertainment_film_music_artist

주제별 문서 개수 (상세):
  주제 0: 287개 (19.1%)
  주제 1: 256개 (17.1%)
  주제 2: 198개 (13.2%)
  주제 3: 154개 (10.3%)
  주제 4: 323개 (21.5%)
  주제 5: 135개 (9.0%)
  노이즈: 47개 (3.1%)
```

**검증 체크리스트**:
- [ ] 데이터셋이 성공적으로 로드되었는가? (1,500개 문서)
- [ ] BERTopic 모델이 자동으로 주제를 발견했는가? (정상적으로는 4~8개 주제)
- [ ] 각 주제의 대표 단어가 의미 있는가?
- [ ] 노이즈(주제 없는 문서)가 합리적인 비율인가? (3~5% 정도)

**Copilot 프롬프트 1**:
```
"BBC 뉴스 데이터셋을 Hugging Face datasets에서 로드해줄래?
약 1,500개 문서를 가져와야 해."
```

**Copilot 프롬프트 2**:
```
"BERTopic을 초기화하고 학습하는 코드를 작성해줄래?
SentenceTransformer는 'all-MiniLM-L6-v2'를 사용하고,
자동으로 주제 개수를 결정하도록 설정해줘."
```

---

#### 체크포인트 2: 주제 시각화 + 결과 해석 (15분)

**목표**: BERTopic의 시각화 기능(Bar Chart, Heatmap, Network)을 생성하고, 각 주제의 의미를 해석한다

**핵심 단계**:

① **Bar Chart: 각 주제의 상위 단어**

```python
import matplotlib.pyplot as plt

# Bar Chart 생성
print("Bar Chart 생성 중...")
fig = topic_model.visualize_barchart(top_n_topics=6, top_n_words=10)
fig.write_html("practice/chapter8/data/output/topic_barchart.html")
fig.show()

print("저장: practice/chapter8/data/output/topic_barchart.html")

# Bar Chart 해석 보조
print("\n각 주제의 상위 5개 단어:")
for topic_id in range(len(set(topics)) - 1):
    if topic_id not in topics:
        continue
    top_terms = topic_model.get_topic(topic_id)[:5]
    terms_str = ", ".join([f"{term}({weight:.2f})" for term, weight in top_terms])
    print(f"  주제 {topic_id}: {terms_str}")
```

예상 결과:
```
Bar Chart 생성 중...
저장: practice/chapter8/data/output/topic_barchart.html

각 주제의 상위 5개 단어:
  주제 0: business(0.45), company(0.38), sales(0.35), market(0.32), growth(0.28)
  주제 1: sport(0.52), game(0.46), team(0.41), player(0.38), season(0.32)
  주제 2: technology(0.48), data(0.43), software(0.40), system(0.37), code(0.31)
  주제 3: world(0.44), country(0.40), government(0.38), nation(0.35), state(0.30)
  주제 4: political(0.49), election(0.44), party(0.41), government(0.39), vote(0.34)
  주제 5: entertainment(0.50), film(0.45), music(0.42), artist(0.38), movie(0.35)
```

② **Heatmap: 주제-단어 가중치 행렬**

```python
# Heatmap 생성
print("\nHeatmap 생성 중...")
fig = topic_model.visualize_heatmap(top_n_topics=6, top_n_words=10)
fig.write_html("practice/chapter8/data/output/topic_heatmap.html")
fig.show()

print("저장: practice/chapter8/data/output/topic_heatmap.html")

# Heatmap 해석: 각 주제의 특징 강도
print("\n주제별 특징 강도 (가중치 합계):")
for topic_id in range(len(set(topics)) - 1):
    if topic_id not in topics:
        continue
    top_terms = topic_model.get_topic(topic_id)[:10]
    total_weight = sum([weight for _, weight in top_terms])
    print(f"  주제 {topic_id}: {total_weight:.2f}")
```

예상 결과:
```
Heatmap 생성 중...
저장: practice/chapter8/data/output/topic_heatmap.html

주제별 특징 강도 (가중치 합계):
  주제 0: 3.45
  주제 1: 3.67
  주제 2: 3.52
  주제 3: 3.38
  주제 4: 3.71
  주제 5: 3.55
```

③ **Network Graph: 주제 간 유사도**

```python
# Network Graph 생성
print("\nNetwork Graph 생성 중...")
fig = topic_model.visualize_hierarchy()
fig.write_html("practice/chapter8/data/output/topic_network.html")
fig.show()

print("저장: practice/chapter8/data/output/topic_network.html")

# 주제 간 유사도 분석
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# 각 주제의 중심 벡터 계산 (간단한 방식)
print("\n주제 간 유사도 (코사인 거리):")
topic_vectors = []
for topic_id in range(len(set(topics)) - 1):
    if topic_id not in topics:
        continue
    top_terms = topic_model.get_topic(topic_id)
    # 단어 가중치를 벡터로 표현 (간단한 예시)
    weights = [w for _, w in top_terms]
    topic_vectors.append(np.array(weights))

# 주제 0과 다른 주제들의 유사도
if len(topic_vectors) > 1:
    topic_0_vec = topic_vectors[0].reshape(1, -1)
    for i in range(1, min(len(topic_vectors), 3)):
        other_vec = topic_vectors[i].reshape(1, -1)
        similarity = cosine_similarity(topic_0_vec, other_vec)[0, 0]
        print(f"  주제 0 vs 주제 {i}: {similarity:.3f}")
```

예상 결과:
```
Network Graph 생성 중...
저장: practice/chapter8/data/output/topic_network.html

주제 간 유사도 (코사인 거리):
  주제 0 vs 주제 1: 0.156  (매우 다름)
  주제 0 vs 주제 2: 0.234  (다름)
```

④ **주제 의미 해석**

```python
# 주제별 의미 해석
interpretation = {
    0: "비즈니스/경제: 기업 경영, 판매, 시장 동향",
    1: "스포츠: 경기 결과, 선수, 리그",
    2: "기술/IT: 소프트웨어, 데이터, 시스템",
    3: "국제 뉴스: 국가, 정부, 세계 정세",
    4: "정치: 선거, 정당, 정치 활동",
    5: "엔터테인먼트: 영화, 음악, 예술"
}

print("\n주제별 의미 해석:")
for topic_id, meaning in interpretation.items():
    count = sum(1 for t in topics if t == topic_id)
    print(f"  주제 {topic_id}: {meaning} (n={count})")
```

예상 결과:
```
주제별 의미 해석:
  주제 0: 비즈니스/경제: 기업 경영, 판매, 시장 동향 (n=287)
  주제 1: 스포츠: 경기 결과, 선수, 리그 (n=256)
  주제 2: 기술/IT: 소프트웨어, 데이터, 시스템 (n=198)
  주제 3: 국제 뉴스: 국가, 정부, 세계 정세 (n=154)
  주제 4: 정치: 선거, 정당, 정치 활동 (n=323)
  주제 5: 엔터테인먼트: 영화, 음악, 예술 (n=135)
```

**검증 체크리스트**:
- [ ] Bar Chart가 각 주제의 상위 단어를 정확히 표시하는가?
- [ ] Heatmap이 주제-단어 가중치를 시각화하는가?
- [ ] Network Graph가 주제 간 유사도를 표현하는가?
- [ ] 각 주제의 의미가 명확하게 해석되는가?

**Copilot 프롬프트 3**:
```
"BERTopic 모델의 시각화 함수들을 사용해줄래?
visualize_barchart(), visualize_heatmap(), visualize_hierarchy()를
각각 실행하고 HTML 파일로 저장해줘."
```

**Copilot 프롬프트 4**:
```
"각 주제의 상위 단어를 출력하는 코드를 작성해줄래?
topic_model.get_topic()을 사용해서 주제별로 상위 10개 단어를 표시해줘."
```

---

#### 체크포인트 3: 특정 문서 분석 + 품질 평가 (15분)

**목표**: 특정 문서의 주제 분포를 분석하고, 토픽 모델링 결과의 품질을 정량적으로 평가한다

**핵심 단계**:

① **특정 문서의 주제 분포 분석**

```python
# 샘플 문서 선택
sample_indices = [0, 100, 500, 1000]

print("특정 문서의 주제 분포 분석:\n")
for idx in sample_indices:
    doc = documents[idx]
    assigned_topic = topics[idx]
    confidence = probabilities[idx]

    print(f"문서 #{idx}:")
    print(f"  내용: {doc[:150]}...")
    print(f"  할당된 주제: {assigned_topic}")
    print(f"  신뢰도: {confidence:.3f}")

    # 이 문서가 각 주제에 속할 확률 계산 (근사)
    doc_vec = embedding_model.encode(doc)
    new_topics, new_probs = topic_model.transform([doc])

    print(f"  주제별 확률:")
    for tid, prob in sorted(enumerate(new_probs[0]), key=lambda x: x[1], reverse=True)[:3]:
        meaning = interpretation.get(tid, "Unknown")
        print(f"    주제 {tid} ({meaning[:20]}...): {prob:.3f}")
    print()
```

예상 결과:
```
특정 문서의 주제 분포 분석:

문서 #0:
  내용: Ad sales boost Time Warner profit Quick jump in revenues To the delight...
  할당된 주제: 0
  신뢰도: 0.782
  주제별 확률:
    주제 0 (비즈니스/경제): 0.782
    주제 4 (정치): 0.089
    주제 3 (국제 뉴스): 0.045

문서 #100:
  내용: England Advance In Cricket World Cup India beat Australia by four wickets...
  할당된 주제: 1
  신뢰도: 0.856
  주제별 확률:
    주제 1 (스포츠): 0.856
    주제 0 (비즈니스/경제): 0.078
    주제 3 (국제 뉴스): 0.042
```

② **토픽 코히어런스(Coherence) 평가**

```python
# 토픽 코히어런스 계산 (각 주제의 내부 일관성)
print("\n토픽 코히어런스 점수 (높을수록 좋음, 0~1):")

from sklearn.feature_extraction.text import CountVectorizer

# 간단한 코히어런스 계산: 상위 단어들의 co-occurrence 기반
def calculate_topic_coherence(topic_words, documents, window_size=10):
    """
    토픽의 상위 단어들이 문서에서 자주 함께 나타나는지 평가
    """
    from collections import defaultdict

    co_occurrence = defaultdict(int)
    total_pairs = 0

    for doc in documents:
        words = doc.lower().split()
        for i, word in enumerate(words):
            if word in topic_words:
                for j in range(max(0, i-window_size), min(len(words), i+window_size+1)):
                    if i != j and words[j] in topic_words:
                        co_occurrence[(word, words[j])] += 1
                        total_pairs += 1

    if total_pairs == 0:
        return 0.0

    # 코히어런스: co-occurrence의 총합 / (가능한 모든 쌍의 수)
    coherence = sum(co_occurrence.values()) / (len(topic_words) * (len(topic_words) - 1) * total_pairs)
    return min(coherence * 100, 1.0)  # 0~1 범위로 정규화

# 각 주제의 코히어런스 계산
topic_coherences = {}
for topic_id in range(len(set(topics)) - 1):
    if topic_id not in topics:
        continue

    top_terms = topic_model.get_topic(topic_id)
    topic_words = set([word for word, _ in top_terms[:5]])

    # 간단한 코히어런스: 상위 5개 단어가 함께 나타나는 비율
    co_count = 0
    doc_with_multiple = 0

    for doc in documents:
        doc_words = set(doc.lower().split())
        overlap = len(topic_words & doc_words)
        if overlap >= 2:
            doc_with_multiple += 1
        co_count += overlap

    coherence = doc_with_multiple / len(documents) if documents else 0
    topic_coherences[topic_id] = coherence
    print(f"  주제 {topic_id}: {coherence:.3f}")

avg_coherence = np.mean(list(topic_coherences.values()))
print(f"\n평균 토픽 코히어런스: {avg_coherence:.3f}")
```

예상 결과:
```
토픽 코히어런스 점수 (높을수록 좋음, 0~1):
  주제 0: 0.487
  주제 1: 0.523
  주제 2: 0.456
  주제 3: 0.412
  주제 4: 0.534
  주제 5: 0.468

평균 토픽 코히어런스: 0.480
```

③ **주제 다양성(Diversity) 평가**

```python
# 주제 다양성: 각 주제의 상위 단어가 서로 겹치지 않는 정도
print("\n주제 다양성 평가:")

# 모든 주제의 상위 단어 수집
all_topic_words = set()
topic_unique_words = {}

for topic_id in range(len(set(topics)) - 1):
    if topic_id not in topics:
        continue

    top_terms = topic_model.get_topic(topic_id)
    top_words = set([word for word, _ in top_terms[:10]])
    topic_unique_words[topic_id] = top_words
    all_topic_words.update(top_words)

# 다양성: 각 주제의 고유한 단어 비율
print("\n주제별 고유 단어 비율:")
total_unique = 0
for topic_id in range(len(set(topics)) - 1):
    if topic_id not in topics:
        continue

    unique_count = len(topic_unique_words[topic_id])
    for other_id in range(len(set(topics)) - 1):
        if other_id != topic_id and other_id in topic_unique_words:
            unique_count -= len(topic_unique_words[topic_id] & topic_unique_words[other_id])

    diversity = unique_count / 10  # 상위 10개 단어 기준
    total_unique += unique_count
    print(f"  주제 {topic_id}: {diversity:.1%} 고유 단어")

# 전체 다양성
overall_diversity = len(all_topic_words) / (len(set(topics)) - 1) / 10
print(f"\n전체 주제 다양성 점수: {overall_diversity:.3f}")
```

예상 결과:
```
주제별 고유 단어 비율:
  주제 0: 80.0% 고유 단어
  주제 1: 90.0% 고유 단어
  주제 2: 70.0% 고유 단어
  주제 3: 85.0% 고유 단어
  주제 4: 95.0% 고유 단어
  주제 5: 75.0% 고유 단어

전체 주제 다양성 점수: 0.825
```

④ **품질 평가 종합**

```python
# 토픽 모델링 품질 종합 평가
print("\n=== 토픽 모델링 품질 평가 종합 ===\n")

quality_metrics = {
    "발견된 주제 개수": len(set(topics)) - (1 if -1 in topics else 0),
    "평균 토픽 코히어런스": avg_coherence,
    "전체 주제 다양성": overall_diversity,
    "할당 성공률": f"{(1 - sum(topics == -1) / len(topics)) * 100:.1f}%",
    "최대 주제 크기": f"{max([sum(topics == t) for t in set(topics) if t != -1])} 문서",
    "최소 주제 크기": f"{min([sum(topics == t) for t in set(topics) if t != -1])} 문서"
}

for metric, value in quality_metrics.items():
    print(f"{metric}: {value}")

# 최종 평가
print("\n평가:")
if avg_coherence > 0.5 and overall_diversity > 0.8:
    print("  ✓ 우수: 주제의 내부 일관성과 다양성이 모두 우수함")
elif avg_coherence > 0.4 and overall_diversity > 0.7:
    print("  ○ 양호: 기본적으로 타당한 주제 발견")
else:
    print("  ✗ 개선 필요: 하이퍼파라미터 조정 필요")
```

예상 결과:
```
=== 토픽 모델링 품질 평가 종합 ===

발견된 주제 개수: 6
평균 토픽 코히어런스: 0.480
전체 주제 다양성: 0.825
할당 성공률: 96.9%
최대 주제 크기: 323 문서
최소 주제 크기: 135 문서

평가:
  ○ 양호: 기본적으로 타당한 주제 발견
```

**검증 체크리스트**:
- [ ] 샘플 문서들의 주제 할당이 의미 있는가?
- [ ] 토픽 코히어런스 점수가 합리적인 범위인가? (0.4~0.6)
- [ ] 주제 다양성이 충분한가? (0.7 이상)
- [ ] 할당 성공률이 높은가? (90% 이상)

**Copilot 프롬프트 5**:
```
"특정 문서의 주제를 분석하고, 그 문서의 각 주제별 확률을 계산해줄래?
embedding_model.encode()와 topic_model.transform()을 사용해줘."
```

**Copilot 프롬프트 6**:
```
"토픽 모델링의 품질을 평가하는 지표들(코히어런스, 다양성, 할당 성공률)을
계산하고 출력하는 함수를 만들어줄 수 있어?"
```

**선택 프롬프트**:
```
"주제들 간의 유사도를 계산해서 거리 행렬로 표현하고,
heatmap으로 시각화할 수 있어?"
```

---

### 제출 안내 (Google Classroom)

**제출 방법**:
- Google Classroom의 "8주차 B회차" 과제에 조별 1부 제출
- 파일명 형식: `group_{조번호}_ch8B.zip`

**포함할 파일**:
```
group_{조번호}_ch8B/
├── ch8B_bertopic.py              # 전체 구현 코드
├── topic_barchart.html            # Bar Chart 시각화
├── topic_heatmap.html             # Heatmap 시각화
├── topic_network.html             # Network Graph 시각화
├── topic_coherence_scores.txt     # 코히어런스 점수
└── report.md                      # 분석 리포트 (2~3페이지)
```

**리포트 포함 항목** (report.md):
- 각 체크포인트의 구현 과정 및 어려웠던 점 (3~4문장)
- 발견된 주제들의 의미 해석: "각 주제가 어떤 뉴스를 다루는가?" (3~4문장)
- Bar Chart와 Heatmap 해석: "각 주제의 특징 단어 분석" (2~3문장)
- 토픽 모델링 품질 평가: "코히어런스와 다양성 점수를 기준으로 결과 평가" (2~3문장)
- 특정 문서 분석 사례: "실제 뉴스 기사의 주제 분포 해석" (2~3문장)
- 개선 방안 제안: "더 나은 결과를 위해 수정할 수 있는 부분?" (2문장)
- Copilot 사용 경험: "어떤 프롬프트가 효과적이었는가?" (2문장)

**제출 마감**: 수업 종료 후 24시간 이내

---

### 결과 토론 가이드

> **토론 가이드**: 조별 구현 결과를 공유하며, 발견된 주제들을 해석하고, 다른 조의 결과와 비교하며, 토픽 모델링 품질을 함께 평가한다

**토론 주제**:

① **주제 발견의 일관성**
- 같은 데이터로도 조마다 주제 개수가 다를 수 있다 (min_topic_size 등의 설정 차이)
- "왜 우리 조는 5개, 다른 조는 7개의 주제를 발견했을까?"
- 하이퍼파라미터 설정이 결과에 미치는 영향 논의

② **주제 의미 해석의 타당성**
- 각 조가 해석한 주제의 의미가 일관성 있는가?
- 예: 조 A는 "주제 2 = 기술", 조 B는 "주제 2 = 과학"이라 해석
- 상위 단어들을 근거로 해석 타당성 평가

③ **노이즈와 애매한 문서**
- 어떤 문서들이 어떤 주제에도 명확히 할당되지 않았는가? (노이즈)
- 예: "경제-정치 혼합 뉴스", "여러 주제를 다룬 종합 뉴스"
- 이런 경계 케이스를 어떻게 해석할 것인가?

④ **시각화 해석 비교**
- Bar Chart에서 각 주제의 상위 단어가 실제로 그 주제를 대표하는가?
- Network Graph에서 어떤 주제들이 가까운가? (왜 그럴까?)
- 예: "비즈니스"와 "경제"는 유사하지만, "스포츠"와 "기술"은 거리가 멈

⑤ **품질 평가의 기준**
- 토픽 코히어런스 점수가 0.48인 것은 좋은가, 나쁜가?
- 다른 도메인(소셜 미디어, 학술 논문)에서는 어떨까?
- "현실적으로 어느 정도 점수면 실용적인가?"

⑥ **실무 적용 시사**
- 실제 뉴스사가 토픽 모델링을 어떻게 활용할까?
- 자동 태깅? 추천 시스템? 트렌드 분석?
- 9주차(LDA 비교 학습)에 어떻게 연결될까?

**발표 형식**:
- 각 조 3~5분 발표 (발견된 주제 소개 + 품질 평가 결과)
- 다른 조의 질문에 답변 (2~3개 질문)
- 교수의 보충 설명 및 피드백

---

### Exit ticket

**문제 (1문항)**:

다음 중 BERTopic 파이프라인에서 HDBSCAN 클러스터링의 역할로 가장 정확한 것은?

① BERT 임베딩을 생성하여 문서를 벡터 공간에 배치한다
② 고차원 벡터를 저차원으로 축소하여 거리 개념을 명확하게 한다
③ 축소된 벡터 공간에서 밀도 기반으로 의미론적으로 비슷한 문서들을 같은 주제로 그룹화하며, 주제 개수를 자동 결정한다
④ 각 클러스터의 특징 단어를 c-TF-IDF로 추출한다

**정답: ③**

**설명**: BERTopic의 4단계를 정확히 이해하고 있는가를 평가하는 문제다.
- ①은 Stage 1 (Document Embedding)의 역할
- ②는 Stage 2 (Dimensionality Reduction)의 역할
- ③은 Stage 3 (Clustering)의 역할 — HDBSCAN의 핵심 가치는 "밀도 기반 클러스터링"과 "자동 주제 개수 결정"
- ④는 Stage 4 (Topic Representation)의 역할

HDBSCAN을 선택한 이유는 K-means처럼 주제 개수(K)를 미리 정할 필요가 없기 때문이다.

---

## 참고 자료

**실습 코드**:
- _전체 코드는 practice/chapter8/code/8-2-bertopic-analysis.py 참고_
- _샘플 코드는 practice/chapter8/code/8-1-bertopic-pipeline.py 참고_

**권장 읽기**:
- Grootendorst, M. (2022). BERTopic: Neural Topic Modeling with a Class-based TF-IDF procedure. *arXiv*. https://arxiv.org/abs/2203.05556
- McInnes, L., Healy, J., & Melville, J. (2018). UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction. *arXiv*. https://arxiv.org/abs/1802.03426
- Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. *ICLR*. https://arxiv.org/abs/1810.04805

---

**라이브러리 설치**:
```bash
pip install bertopic sentence-transformers datasets scikit-learn umap-learn hdbscan
```

---
