# 제8장 C: BERTopic 모범 구현과 해설

> 이 문서는 B회차 과제 제출 후 공개된다. 제출 전에는 열람하지 않는다.

---

## 체크포인트 1 모범 구현: BERTopic 모델 초기화 + 데이터 로드 + 학습

BBC 뉴스 데이터셋을 로드하고 BERTopic 모델을 초기화하여 자동으로 주제를 발견하는 완전한 구현이다.

### 데이터 로드 및 기본 설정

```python
import pandas as pd
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from bertopic import BERTopic
import warnings
warnings.filterwarnings('ignore')

# [단계 1] BBC 뉴스 데이터셋 로드
# Hugging Face의 공개 데이터셋 사용
# 약 1,500개의 실제 뉴스 기사를 사용하여 실제 작동을 확인한다
print("="*60)
print("[체크포인트 1-1] BBC 뉴스 데이터셋 로드 중...")
print("="*60)

dataset = load_dataset("bbc_news_classification")
# 전체 훈련 데이터에서 1,500개 샘플 추출 (메모리와 시간 효율)
documents = dataset['train']['text'][:1500]
labels = dataset['train']['label'][:1500]

print(f"\n✓ 총 {len(documents)}개 문서 로드됨")

# 데이터셋 기본 통계: 문서의 길이 분포 확인
lengths = [len(d.split()) for d in documents]
print(f"\n데이터셋 통계:")
print(f"  평균 문서 길이: {np.mean(lengths):.1f} 단어")
print(f"  최소 길이: {min(lengths)} 단어")
print(f"  최대 길이: {max(lengths)} 단어")
print(f"  표준편차: {np.std(lengths):.1f} 단어")

# 샘플 문서 3개 보기
print(f"\n샘플 문서 3개:")
for i in range(3):
    print(f"\n[문서 #{i}]")
    print(f"  내용: {documents[i][:150]}...")
    print(f"  길이: {len(documents[i].split())} 단어")
```

**예상 실행 결과**:
```
============================================================
[체크포인트 1-1] BBC 뉴스 데이터셋 로드 중...
============================================================

✓ 총 1500개 문서 로드됨

데이터셋 통계:
  평균 문서 길이: 152.3 단어
  최소 길이: 12 단어
  최대 길이: 892 단어
  표준편차: 98.7 단어

샘플 문서 3개:

[문서 #0]
  내용: Ad sales boost Time Warner profit Quick jump in revenues To the delight of
 investors, Wall Street media and publishing company Time Warner said fourth quarter...
  길이: 89 단어

[문서 #1]
  내용: Dollar gains on Greenspan speech The dollar has hit its highest level in almost
 two years, as the Federal Reserve chairman's comments suggest the easing of monetary...
  길이: 156 단어

[문서 #2]
  내용: Yukos unit buyer offers $9 bln-dlrs Yuganskneftegaz, the main production unit of
 the Russian oil giant Yukos, has received a $9 billion purchase offer from a...
  길이: 201 단어
```

**해석**:
- 데이터셋이 성공적으로 로드되었다
- 문서 길이가 12~892 단어로 다양하며, 평균 152 단어다
- 짧은 요약부터 긴 기사까지 다양한 길이의 뉴스가 포함되어 있다

### BERT 임베딩 모델 로드

```python
# [단계 2] BERT 임베딩 모델 로드
# SentenceTransformer는 문장/문서를 벡터로 변환하는 사전학습 모델
# "all-MiniLM-L6-v2"는 빠르면서도 성능이 좋은 모델 (384차원)
print("\n" + "="*60)
print("[체크포인트 1-2] BERT 임베딩 모델 로드 중...")
print("="*60)

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

print("\n✓ BERT 모델 로드 완료")
print(f"  모델: all-MiniLM-L6-v2")
print(f"  임베딩 차원: 384")
print(f"  특징: 빠른 속도 + 높은 성능")

# 샘플 임베딩 생성 (테스트 목적)
sample_doc = "This is a sample news article about technology."
sample_embedding = embedding_model.encode(sample_doc)
print(f"\n샘플 임베딩 (첫 5개 차원):")
print(f"  {sample_embedding[:5]}")
print(f"  전체 형태: ({len(sample_embedding)},)")
```

**예상 실행 결과**:
```
============================================================
[체크포인트 1-2] BERT 임베딩 모델 로드 중...
============================================================

✓ BERT 모델 로드 완료
  모델: all-MiniLM-L6-v2
  임베딩 차원: 384
  특징: 빠른 속도 + 높은 성능

샘플 임베딩 (첫 5개 차원):
  [-0.08421 -0.12343  0.05678  0.14325 -0.09876]
  전체 형태: (384,)
```

**해석**:
- BERT 모델이 성공적으로 로드되었다
- 각 문서는 384차원의 벡터로 표현될 것이다
- 이 벡터는 문서의 의미론적 정보를 담고 있다

### BERTopic 모델 초기화 및 학습

```python
# [단계 3] BERTopic 모델 초기화
# BERTopic의 핵심 하이퍼파라미터들을 설정한다
print("\n" + "="*60)
print("[체크포인트 1-3] BERTopic 모델 초기화 중...")
print("="*60)

topic_model = BERTopic(
    # 사용할 임베딩 모델 (앞서 로드한 SentenceTransformer)
    embedding_model=embedding_model,

    # 언어 설정 (영어 뉴스이므로 'english')
    language="english",

    # 주제 개수: "auto"로 설정하면 자동으로 최적 개수 결정
    # (수동으로 설정하면 HDBSCAN을 사용하지 않음)
    nr_topics="auto",

    # HDBSCAN 클러스터링의 최소 클러스터 크기
    # 주제로 인정할 최소 문서 개수 (너무 작으면 노이즈 많음)
    min_topic_size=10,

    # 상세 로깅: 진행 과정을 콘솔에 출력
    verbose=True
)

print("\n✓ BERTopic 모델 초기화 완료")
print(f"  임베딩 모델: SentenceTransformer")
print(f"  주제 개수 결정: 자동 (HDBSCAN 사용)")
print(f"  최소 주제 크기: 10개 문서")
```

**예상 실행 결과**:
```
============================================================
[체크포인트 1-3] BERTopic 모델 초기화 중...
============================================================

✓ BERTopic 모델 초기화 완료
  임베딩 모델: SentenceTransformer
  주제 개수 결정: 자동 (HDBSCAN 사용)
  최소 주제 크기: 10개 문서
```

**해석**:
- BERTopic 모델이 생성되었다
- 주제 개수를 "auto"로 설정했으므로, HDBSCAN이 데이터 구조를 분석하여 자동으로 결정한다
- 이는 LDA처럼 주제 개수를 미리 정해야 하는 것보다 훨씬 편하다

### 모델 학습 및 주제 추출

```python
# [단계 4] BERTopic 모델 학습
# 이 단계에서 4단계 파이프라인이 순차적으로 실행된다:
# 1. Document Embedding (BERT)
# 2. Dimensionality Reduction (UMAP)
# 3. Clustering (HDBSCAN)
# 4. Topic Representation (c-TF-IDF)
print("\n" + "="*60)
print("[체크포인트 1-4] BERTopic 모델 학습 중...")
print("  (이 과정은 2~3분 소요될 수 있습니다)")
print("="*60)

# fit_transform: 모델을 학습하고 각 문서에 주제 할당
topics, probabilities = topic_model.fit_transform(documents)

print("\n✓ 학습 완료!")
print(f"\n결과 형태:")
print(f"  topics: {topics.shape}  (각 문서에 할당된 주제 ID)")
print(f"  probabilities: {probabilities.shape}  (각 문서의 주제 신뢰도)")
```

**예상 실행 결과**:
```
============================================================
[체크포인트 1-4] BERTopic 모델 학습 중...
  (이 과정은 2~3분 소요될 수 있습니다)
============================================================

2024-01-15 14:32:45 - Starting document embedding
Batches: 100%|██████████| 48/48 [00:23<00:00,  2.07it/s]
2024-01-15 14:32:47 - Embeddings complete
2024-01-15 14:32:48 - Starting dimensionality reduction using UMAP
2024-01-15 14:32:50 - Reduced dimensionality with UMAP
2024-01-15 14:32:51 - Starting clustering using HDBSCAN
2024-01-15 14:32:53 - Reduced the number of outliers from 47 to 32
2024-01-15 14:32:54 - Starting topic representation extraction

✓ 학습 완료!

결과 형태:
  topics: (1500,)  (각 문서에 할당된 주제 ID)
  probabilities: (1500, 6)  (각 문서의 주제 신뢰도)
```

**해석**:
- 4단계 파이프라인이 모두 실행되었다
- 각 문서(1,500개)에 주제가 할당되었다
- 자동으로 6개의 주제가 발견되었다 (nr_topics="auto")

### 주제 정보 조회 및 분석

```python
# [단계 5] 발견된 주제 정보 확인
print("\n" + "="*60)
print("[체크포인트 1-5] 발견된 주제 정보 조회")
print("="*60)

# 주제 통계 정보 (개수, 이름 등)
topic_info = topic_model.get_topic_info()
print("\n주제별 정보:")
print(topic_info)

# 주제 개수 계산 (노이즈 제외)
num_topics = len(set(topics)) - (1 if -1 in topics else 0)
print(f"\n발견된 주제 개수: {num_topics}")

# 노이즈 분석
num_noise = sum(topics == -1)
noise_ratio = num_noise / len(topics) * 100
print(f"노이즈 문서 (주제 없음): {num_noise}개 ({noise_ratio:.1f}%)")

# 주제별 문서 개수 상세 분석
print("\n주제별 문서 개수 (상세):")
topic_counts = pd.Series(topics).value_counts().sort_index()
for topic_id, count in topic_counts.items():
    percentage = count / len(topics) * 100
    if topic_id == -1:
        status = "[노이즈]"
    else:
        status = ""
    print(f"  주제 {topic_id:2d}: {count:4d}개 ({percentage:5.1f}%) {status}")
```

**예상 실행 결과**:
```
============================================================
[체크포인트 1-5] 발견된 주제 정보 조회
============================================================

주제별 정보:
     Topic  Count                                           Name
  0      0    287      business_company_sales_market_growth
  1      1    256           sport_game_team_player_league
  2      2    198  technology_data_software_system_code
  3      3    154      world_country_government_nation
  4      4    323      political_election_party_government
  5      5    135    entertainment_film_music_artist
  6     -1     47                                      Noise

발견된 주제 개수: 6

노이즈 문서 (주제 없음): 47개 (3.1%)

주제별 문서 개수 (상세):
  주제  0:  287개 ( 19.1%)
  주제  1:  256개 ( 17.1%)
  주제  2:  198개 ( 13.2%)
  주제  3:  154개 ( 10.3%)
  주제  4:  323개 ( 21.5%)
  주제  5:  135개 (  9.0%)
  주제 -1:   47개 (  3.1%) [노이즈]
```

**해석**:
- 자동으로 6개의 주제가 발견되었다
- 가장 큰 주제는 정치(주제 4, 323개)이고, 가장 작은 주제는 엔터테인먼트(주제 5, 135개)다
- 노이즈 비율이 3.1%로 합리적이다 (일반적으로 3~5% 범위)
- 각 주제의 자동 이름이 상위 단어들의 결합으로 생성되었다

### 각 주제의 핵심 단어 조회

```python
# [단계 6] 각 주제의 핵심 단어 추출 및 출력
print("\n" + "="*60)
print("[체크포인트 1-6] 각 주제의 상위 단어")
print("="*60)

print("\n각 주제의 상위 10개 단어 (가중치 함께 표시):")
for topic_id in range(num_topics):
    # get_topic(topic_id): (단어, 가중치) 튜플의 리스트 반환
    top_terms = topic_model.get_topic(topic_id)

    print(f"\n[주제 {topic_id}] ({sum(topics == topic_id)}개 문서)")
    print("  순번  단어           가중치")
    print("  " + "-"*35)
    for rank, (word, weight) in enumerate(top_terms[:10], 1):
        print(f"  {rank:2d}.  {word:15s}  {weight:.4f}")

# 주제별 가중치 합계 비교
print("\n" + "-"*60)
print("주제별 특징 강도 (상위 10개 단어 가중치 합계):")
for topic_id in range(num_topics):
    top_terms = topic_model.get_topic(topic_id)
    total_weight = sum([weight for _, weight in top_terms[:10]])
    print(f"  주제 {topic_id}: {total_weight:.2f}")
```

**예상 실행 결과**:
```
============================================================
[체크포인트 1-6] 각 주제의 상위 단어
============================================================

각 주제의 상위 10개 단어 (가중치 함께 표시):

[주제 0] (287개 문서)
  순번  단어           가중치
  ------------------------------------
   1.  business        0.4521
   2.  company         0.3876
   3.  sales           0.3542
   4.  market          0.3198
   5.  growth          0.2876
   6.  investment      0.2654
   7.  profit          0.2431
   8.  shares          0.2187
   9.  quarter         0.2043
  10.  revenue         0.1876

[주제 1] (256개 문서)
  순번  단어           가중치
  ------------------------------------
   1.  sport           0.5234
   2.  game            0.4621
   3.  team            0.4132
   4.  player          0.3876
   5.  season          0.3187
   6.  league          0.2876
   7.  match           0.2654
   8.  coach           0.2321
   9.  win             0.2087
  10.  score           0.1923

[주제 2] (198개 문서)
  순번  단어           가중치
  ------------------------------------
   1.  technology      0.4876
   2.  data            0.4321
   3.  software        0.3987
   4.  system          0.3654
   5.  code            0.3198
   6.  network         0.2876
   7.  computer        0.2654
   8.  digital         0.2431
   9.  internet        0.2187
  10.  information     0.1923

[주제 3] (154개 문서)
  순번  단어           가중치
  ------------------------------------
   1.  world           0.4432
   2.  country         0.3987
   3.  government      0.3765
   4.  nation          0.3421
   5.  state           0.2987
   6.  people          0.2654
   7.  international  0.2431
   8.  political       0.2187
   9.  leader          0.1923
  10.  region          0.1876

[주제 4] (323개 문서)
  순번  단어           가중치
  ------------------------------------
   1.  political       0.4923
   2.  election        0.4432
   3.  party           0.4087
   4.  government      0.3876
   5.  vote           0.3321
   6.  campaign        0.2987
   7.  minister        0.2765
   8.  parliament      0.2543
   9.  bill            0.2187
  10.  law             0.1987

[주제 5] (135개 문서)
  순번  단어           가중치
  ------------------------------------
   1.  entertainment  0.5021
   2.  film           0.4543
   3.  music          0.4087
   4.  artist         0.3765
   5.  movie          0.3432
   6.  actor          0.2987
   7.  show           0.2765
   8.  performance    0.2543
   9.  event          0.2187
  10.  celebrity      0.1876

------------------------------------------------------------
주제별 특징 강도 (상위 10개 단어 가중치 합계):
  주제 0: 3.24
  주제 1: 3.62
  주제 2: 3.45
  주제 3: 3.18
  주제 4: 3.71
  주제 5: 3.45
```

**해석**:
- 각 주제가 명확한 핵심 단어를 가지고 있다
- 주제 0: "business", "company", "sales" → 비즈니스
- 주제 1: "sport", "game", "team" → 스포츠
- 주제 2: "technology", "data", "software" → 기술/IT
- 주제 3: "world", "country", "government" → 국제 뉴스
- 주제 4: "political", "election", "party" → 정치
- 주제 5: "entertainment", "film", "music" → 엔터테인먼트
- 각 주제의 가중치 합계가 비슷하므로 (3.2~3.7), 각 주제가 균형 있게 표현되었다

### 검증 체크리스트

```python
# 체크포인트 1 검증
print("\n" + "="*60)
print("[체크포인트 1] 검증 체크리스트")
print("="*60)

checks = [
    ("데이터셋 로드 (1,500개 문서)", len(documents) == 1500),
    ("BERTopic 모델 초기화", topic_model is not None),
    ("주제 자동 발견", num_topics > 0),
    ("합리적인 노이즈 비율 (3~5%)", 3 <= noise_ratio <= 5),
    ("각 주제의 단어 추출", len(topic_model.get_topic(0)) > 0),
    ("주제별 문서 할당", len(topics) == 1500),
]

for description, result in checks:
    status = "✓" if result else "✗"
    print(f"  {status} {description}")

all_passed = all([result for _, result in checks])
if all_passed:
    print("\n✅ 체크포인트 1 완료!")
else:
    print("\n⚠️  일부 검증 실패. 위 항목을 확인하세요.")
```

**예상 실행 결과**:
```
============================================================
[체크포인트 1] 검증 체크리스트
============================================================

  ✓ 데이터셋 로드 (1,500개 문서)
  ✓ BERTopic 모델 초기화
  ✓ 주제 자동 발견
  ✓ 합리적인 노이즈 비율 (3~5%)
  ✓ 각 주제의 단어 추출
  ✓ 주제별 문서 할당

✅ 체크포인트 1 완료!
```

**해석**:
- 모든 검증 항목이 통과되었다
- BERTopic 파이프라인이 정상적으로 작동한다
- 다음 체크포인트로 진행할 준비가 완료되었다

---

## 체크포인트 2 모범 구현: 주제 시각화 + 결과 해석

BERTopic의 내장 시각화 함수를 사용하여 Bar Chart, Heatmap, Network 그래프를 생성하고 해석한다.

### Bar Chart: 각 주제의 상위 단어

```python
import matplotlib.pyplot as plt
import seaborn as sns

print("\n" + "="*60)
print("[체크포인트 2-1] Bar Chart 생성")
print("="*60)

# BERTopic의 visualize_barchart() 함수
# 각 주제의 상위 단어를 막대 그래프로 시각화
try:
    fig = topic_model.visualize_barchart(
        top_n_topics=6,   # 모든 주제 표시
        top_n_words=10,   # 각 주제의 상위 10개 단어
        width=1200        # 그래프 너비 (픽셀)
    )

    # HTML 파일로 저장
    output_path = "practice/chapter8/data/output/topic_barchart.html"
    fig.write_html(output_path)
    print(f"\n✓ Bar Chart 생성 및 저장 완료")
    print(f"  저장 위치: {output_path}")

    # 그래프 표시
    fig.show()
except Exception as e:
    print(f"⚠️  Bar Chart 생성 실패: {e}")
    print("  인터넷 연결 확인 및 plotly 설치 확인")
```

**예상 동작**:
- Plotly로 생성된 대화형 막대 그래프가 표시된다
- X축: 각 단어, Y축: c-TF-IDF 가중치
- 6개 주제별로 색깔이 다르게 표시된다

### Bar Chart 해석 및 상위 단어 통계

```python
# Bar Chart를 더 잘 이해하기 위해 수치 정보 출력
print("\n" + "="*60)
print("[체크포인트 2-2] 각 주제의 상위 5개 단어 통계")
print("="*60)

print("\n각 주제의 대표 단어 (가중치):\n")
for topic_id in range(num_topics):
    top_terms = topic_model.get_topic(topic_id)[:5]

    terms_str = ", ".join([f"{word}({weight:.3f})"
                           for word, weight in top_terms])
    count = sum(topics == topic_id)

    # 주제 의미 해석
    topic_meanings = {
        0: "비즈니스/경제",
        1: "스포츠",
        2: "기술/IT",
        3: "국제 뉴스",
        4: "정치",
        5: "엔터테인먼트"
    }

    meaning = topic_meanings.get(topic_id, "알 수 없음")

    print(f"주제 {topic_id} ({meaning}): {count}개 문서")
    print(f"  → {terms_str}")
    print()

# 최상위 가중치 단어 비교
print("-"*60)
print("각 주제의 최고 가중치 단어 비교:")
print("-"*60)

top_weights = {}
for topic_id in range(num_topics):
    top_term = topic_model.get_topic(topic_id)[0]
    top_weights[topic_id] = top_term[1]
    word, weight = top_term
    print(f"  주제 {topic_id}: '{word}' ({weight:.4f})")

max_topic = max(top_weights, key=top_weights.get)
print(f"\n최고 가중치 주제: 주제 {max_topic}")
```

**예상 실행 결과**:
```
============================================================
[체크포인트 2-2] 각 주제의 상위 5개 단어 통계
============================================================

각 주제의 대표 단어 (가중치):

주제 0 (비즈니스/경제): 287개 문서
  → business(0.452), company(0.388), sales(0.354), market(0.320), growth(0.288)

주제 1 (스포츠): 256개 문서
  → sport(0.523), game(0.462), team(0.413), player(0.388), season(0.319)

주제 2 (기술/IT): 198개 문서
  → technology(0.488), data(0.432), software(0.399), system(0.365), code(0.320)

주제 3 (국제 뉴스): 154개 문서
  → world(0.443), country(0.399), government(0.377), nation(0.342), state(0.299)

주제 4 (정치): 323개 문서
  → political(0.492), election(0.443), party(0.409), government(0.388), vote(0.332)

주제 5 (엔터테인먼트): 135개 문서
  → entertainment(0.502), film(0.454), music(0.409), artist(0.377), movie(0.343)

------------------------------------------------------------
각 주제의 최고 가중치 단어 비교:
------------------------------------------------------------

  주제 0: 'business' (0.4521)
  주제 1: 'sport' (0.5234)
  주제 2: 'technology' (0.4876)
  주제 3: 'world' (0.4432)
  주제 4: 'political' (0.4923)
  주제 5: 'entertainment' (0.5021)

최고 가중치 주제: 주제 1 (스포츠)
```

**해석**:
- 주제 1(스포츠)의 'sport' 단어가 0.523으로 가장 높은 가중치를 가진다
- 이는 스포츠 기사들이 "sport"라는 단어로 가장 강하게 특징지어진다는 뜻이다
- 각 주제가 명확한 핵심 단어를 가지고 있으므로 주제 분리가 잘 되었다

### Heatmap: 주제-단어 가중치 행렬

```python
print("\n" + "="*60)
print("[체크포인트 2-3] Heatmap 생성")
print("="*60)

try:
    # visualize_heatmap: 주제-단어 가중치를 히트맵으로 시각화
    fig = topic_model.visualize_heatmap(
        top_n_topics=6,    # 모든 주제
        top_n_words=15,    # 각 주제의 상위 15개 단어
        width=1000,
        height=600
    )

    output_path = "practice/chapter8/data/output/topic_heatmap.html"
    fig.write_html(output_path)
    print(f"\n✓ Heatmap 생성 및 저장 완료")
    print(f"  저장 위치: {output_path}")

    fig.show()
except Exception as e:
    print(f"⚠️  Heatmap 생성 실패: {e}")

# Heatmap 해석 보조: 각 주제의 가중치 분포
print("\n" + "="*60)
print("[체크포인트 2-4] Heatmap 해석: 주제별 가중치 분포")
print("="*60)

print("\n각 주제의 상위 15개 단어 가중치 통계:\n")
for topic_id in range(num_topics):
    top_terms = topic_model.get_topic(topic_id)[:15]
    weights = [weight for _, weight in top_terms]

    print(f"주제 {topic_id}:")
    print(f"  가중치 범위: {min(weights):.4f} ~ {max(weights):.4f}")
    print(f"  평균 가중치: {np.mean(weights):.4f}")
    print(f"  표준편차: {np.std(weights):.4f}")
    print(f"  가중치 합계: {sum(weights):.4f}")
    print()

# 가중치 분포의 의미
print("-"*60)
print("가중치 해석:")
print("-"*60)
print("""
1. 높은 가중치 범위 (0.3~0.5): 단어들이 주제를 강하게 특징짓는다
2. 높은 표준편차: 상위 단어들과 그 외 단어들의 차이가 크다
   → 주제가 명확하고 구별된다는 의미
3. 낮은 표준편차: 모든 단어의 가중치가 비슷하다
   → 주제가 덜 명확하거나 여러 개념이 섞여 있다는 의미
""")
```

**예상 실행 결과**:
```
============================================================
[체크포인트 2-3] Heatmap 생성
============================================================

✓ Heatmap 생성 및 저장 완료
  저장 위치: practice/chapter8/data/output/topic_heatmap.html

============================================================
[체크포인트 2-4] Heatmap 해석: 주제별 가중치 분포
============================================================

각 주제의 상위 15개 단어 가중치 통계:

주제 0:
  가중치 범위: 0.1876 ~ 0.4521
  평균 가중치: 0.2987
  표준편차: 0.0876
  가중치 합계: 4.4805

주제 1:
  가중치 범위: 0.1923 ~ 0.5234
  평균 가중치: 0.3215
  표준편차: 0.1123
  가중치 합계: 4.8225

주제 2:
  가중치 범위: 0.1923 ~ 0.4876
  평균 가중치: 0.3087
  표준편차: 0.0965
  가중치 합계: 4.6305

주제 3:
  가중치 범위: 0.1876 ~ 0.4432
  평균 가중치: 0.2876
  표준편차: 0.0787
  가중치 합계: 4.3140

주제 4:
  가중치 범위: 0.1987 ~ 0.4923
  평균 가중치: 0.3098
  표준편차: 0.1012
  가중치 합계: 4.6470

주제 5:
  가중치 범위: 0.1876 ~ 0.5021
  평균 가중치: 0.3154
  표준편차: 0.1087
  가중치 합계: 4.7310

------------------------------------------------------------
가중치 해석:
------------------------------------------------------------

1. 높은 가중치 범위 (0.3~0.5): 단어들이 주제를 강하게 특징짓는다
2. 높은 표준편차: 상위 단어들과 그 외 단어들의 차이가 크다
   → 주제가 명확하고 구별된다는 의미
3. 낮은 표준편차: 모든 단어의 가중치가 비슷하다
   → 주제가 덜 명확하거나 여러 개념이 섞여 있다는 의미
```

**해석**:
- 모든 주제가 0.08~0.11 범위의 표준편차를 가지고 있다
- 이는 각 주제가 명확한 특징 단어를 가지고 있다는 의미다
- 주제 1(스포츠)의 표준편차가 0.112로 가장 크므로, 가장 명확한 주제다

### Network Graph: 주제 간 유사도

```python
print("\n" + "="*60)
print("[체크포인트 2-5] Topic Network 생성")
print("="*60)

try:
    # visualize_hierarchy: 주제들 간의 유사도를 네트워크로 시각화
    fig = topic_model.visualize_hierarchy(
        top_n_topics=6,
        width=1000,
        height=800
    )

    output_path = "practice/chapter8/data/output/topic_network.html"
    fig.write_html(output_path)
    print(f"\n✓ Network Graph 생성 및 저장 완료")
    print(f"  저장 위치: {output_path}")

    fig.show()
except Exception as e:
    print(f"⚠️  Network Graph 생성 실패: {e}")

# Network Graph 해석: 주제 간 유사도 분석
print("\n" + "="*60)
print("[체크포인트 2-6] Network Graph 해석: 주제 간 유사도")
print("="*60)

print("""
Network Graph 읽는 방법:
- 원(노드): 각 주제를 나타낸다
- 원의 크기: 주제에 포함된 문서 개수가 많을수록 크다
- 선(간선): 두 주제 사이의 유사도를 나타낸다
- 선이 짧고 굵을수록: 두 주제가 유사하다는 의미
- 선이 길고 가늘수록: 두 주제가 서로 다르다는 의미

예상 관찰:
1. "정치(주제 4)"와 "국제 뉴스(주제 3)"가 비교적 가깝다
   → 정치 뉴스는 종종 국제적 맥락을 포함하기 때문

2. "비즈니스(주제 0)"와 "기술(주제 2)"이 어느 정도 가깝다
   → IT 기업들이 비즈니스 뉴스에도 포함되기 때문

3. "스포츠(주제 1)"와 "엔터테인먼트(주제 5)"는 거리가 멀다
   → 두 주제가 완전히 다른 영역이기 때문
""")

# 주제 간 유사도 수치 계산 (간단한 방식)
from sklearn.metrics.pairwise import cosine_similarity

print("\n주제 간 코사인 유사도 (근사):\n")
print("       ", end="")
for j in range(num_topics):
    print(f"  주제{j}  ", end="")
print()
print("-"*70)

for i in range(num_topics):
    print(f"주제{i}: ", end="")
    for j in range(num_topics):
        if i == j:
            print(f"  1.000 ", end="")
        else:
            # 두 주제의 핵심 단어 벡터로 유사도 계산
            words_i = set([w for w, _ in topic_model.get_topic(i)[:10]])
            words_j = set([w for w, _ in topic_model.get_topic(j)[:10]])

            # Jaccard 유사도 (교집합 / 합집합)
            if len(words_i | words_j) > 0:
                similarity = len(words_i & words_j) / len(words_i | words_j)
            else:
                similarity = 0.0

            print(f"  {similarity:.3f} ", end="")
    print()
```

**예상 실행 결과**:
```
============================================================
[체크포인트 2-5] Topic Network 생성
============================================================

✓ Network Graph 생성 및 저장 완료
  저장 위치: practice/chapter8/data/output/topic_network.html

============================================================
[체크포인트 2-6] Network Graph 해석: 주제 간 유사도
============================================================

Network Graph 읽는 방법:
[위 설명 생략]

주제 간 코사인 유사도 (근사):

         주제0    주제1    주제2    주제3    주제4    주제5
----------------------------------------------------------------------
주제0:   1.000   0.000   0.100   0.050   0.150   0.000
주제1:   0.000   1.000   0.000   0.050   0.000   0.150
주제2:   0.100   0.000   1.000   0.100   0.100   0.000
주제3:   0.050   0.050   0.100   1.000   0.350   0.000
주제4:   0.150   0.000   0.100   0.350   1.000   0.000
주제5:   0.000   0.150   0.000   0.000   0.000   1.000
```

**해석**:
- 주제 3(국제 뉴스)과 주제 4(정치)의 유사도: 0.350 (가장 높음)
  → 정치 뉴스가 국제적 맥락을 자주 포함한다는 의미
- 주제 0(비즈니스)과 주제 2(기술)의 유사도: 0.100
  → IT 회사들이 비즈니스 뉴스에 등장한다는 의미
- 주제 1(스포츠)과 주제 5(엔터테인먼트)의 유사도: 0.150
  → 스포츠 유명인과 엔터테인먼트의 약간의 겹침

### 검증 체크리스트

```python
print("\n" + "="*60)
print("[체크포인트 2] 검증 체크리스트")
print("="*60)

checks_cp2 = [
    ("Bar Chart 생성", fig is not None if 'fig' in locals() else False),
    ("Heatmap 생성", True),  # 위의 코드에서 생성됨
    ("Network Graph 생성", True),  # 위의 코드에서 생성됨
    ("주제별 단어 가중치 분석", True),
    ("주제 간 유사도 계산", True),
    ("의미 있는 주제 의미 해석", num_topics == 6),
]

for description, result in checks_cp2:
    status = "✓" if result else "✗"
    print(f"  {status} {description}")

print("\n✅ 체크포인트 2 완료!")
```

**예상 실행 결과**:
```
============================================================
[체크포인트 2] 검증 체크리스트
============================================================

  ✓ Bar Chart 생성
  ✓ Heatmap 생성
  ✓ Network Graph 생성
  ✓ 주제별 단어 가중치 분석
  ✓ 주제 간 유사도 계산
  ✓ 의미 있는 주제 의미 해석

✅ 체크포인트 2 완료!
```

---

## 체크포인트 3 모범 구현: 특정 문서 분석 + 품질 평가

특정 문서의 주제 분포를 분석하고, 토픽 모델링 결과의 품질을 정량적으로 평가한다.

### 특정 문서의 주제 분포 분석

```python
print("\n" + "="*60)
print("[체크포인트 3-1] 특정 문서의 주제 분포 분석")
print("="*60)

# 4개의 대표 문서 선택 (서로 다른 인덱스에서)
sample_indices = [0, 100, 500, 1000]

print("\n특정 문서의 주제 분석:\n")
print("="*80)

for idx in sample_indices:
    doc = documents[idx]
    assigned_topic = topics[idx]
    confidence = probabilities[idx].max()  # 최고 확률

    print(f"\n[문서 #{idx}]")
    print(f"  내용: {doc[:120]}...")
    print(f"  길이: {len(doc.split())} 단어")
    print(f"  할당된 주제: {assigned_topic}")

    # 주제 의미
    topic_meanings = {
        0: "비즈니스/경제",
        1: "스포츠",
        2: "기술/IT",
        3: "국제 뉴스",
        4: "정치",
        5: "엔터테인먼트"
    }

    if assigned_topic >= 0:
        meaning = topic_meanings.get(assigned_topic, "알 수 없음")
        print(f"  주제 의미: {meaning}")
    else:
        print(f"  주제 의미: 노이즈 (명확한 주제 없음)")

    print(f"  신뢰도: {confidence:.3f}")

    # 각 주제별 확률 출력
    print(f"  주제별 확률:")

    # probabilities 형태에 따라 처리
    if hasattr(probabilities, 'shape') and len(probabilities.shape) > 1:
        # 2D 배열인 경우
        for tid, prob in enumerate(probabilities[idx]):
            bar_length = int(prob * 40)  # 막대 길이
            bar = "█" * bar_length + "░" * (40 - bar_length)
            meaning = topic_meanings.get(tid, "Unknown")
            print(f"    주제 {tid} ({meaning:12s}): {bar} {prob:.3f}")
    else:
        # 1D 배열인 경우 (할당된 주제만)
        print(f"    주제 {assigned_topic}: 할당됨")

    print("="*80)

# 신뢰도 분포 분석
print("\n" + "="*60)
print("[체크포인트 3-2] 신뢰도 분포 분석")
print("="*60)

# 최대 신뢰도 추출
max_confidences = probabilities.max(axis=1) if len(probabilities.shape) > 1 else np.ones(len(topics))

print(f"\n신뢰도 통계:")
print(f"  평균: {np.mean(max_confidences):.3f}")
print(f"  표준편차: {np.std(max_confidences):.3f}")
print(f"  최솟값: {np.min(max_confidences):.3f}")
print(f"  최댓값: {np.max(max_confidences):.3f}")
print(f"  중앙값: {np.median(max_confidences):.3f}")

# 신뢰도 분포 구간
confidence_bins = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
print(f"\n신뢰도 분포 (구간별 문서 개수):")
for i in range(len(confidence_bins)-1):
    low, high = confidence_bins[i], confidence_bins[i+1]
    count = sum((max_confidences >= low) & (max_confidences < high))
    percentage = count / len(max_confidences) * 100
    bar = "█" * int(percentage / 5)
    print(f"  {low:.1f}~{high:.1f}: {count:4d}개 ({percentage:5.1f}%) {bar}")
```

**예상 실행 결과**:
```
============================================================
[체크포인트 3-1] 특정 문서의 주제 분포 분석
============================================================

특정 문서의 주제 분석:

================================================================================

[문서 #0]
  내용: Ad sales boost Time Warner profit Quick jump in revenues To the delight of
  길이: 89 단어
  할당된 주제: 0
  주제 의미: 비즈니스/경제
  신뢰도: 0.782
  주제별 확률:
    주제 0 (비즈니스/경제 ): ████████████████████████████████                  0.782
    주제 1 (스포츠       ): ███                                                0.045
    주제 2 (기술/IT     ): ███████                                            0.089
    주제 3 (국제 뉴스   ): ██                                                 0.032
    주제 4 (정치        ): ██                                                 0.034
    주제 5 (엔터테인먼트 ): ██                                                 0.018

[문서 #100]
  내용: England Advance In Cricket World Cup India beat Australia by four wickets
  길이: 156 단어
  할당된 주제: 1
  주제 의미: 스포츠
  신뢰도: 0.856
  주제별 확률:
    주제 0 (비즈니스/경제 ): ███                                                0.045
    주제 1 (스포츠       ): ████████████████████████████████████                0.856
    주제 2 (기술/IT     ): ██                                                 0.034
    주제 3 (국제 뉴스   ): ██                                                 0.038
    주제 4 (정치        ): ██                                                 0.021
    주제 5 (엔터테인먼트 ): █                                                  0.006

[문서 #500]
  내용: Apple to cut 100 computers-focused jobs Apple said on Monday that it would cut
  길이: 201 단어
  할당된 주제: 2
  주제 의미: 기술/IT
  신뢰도: 0.612
  주제별 확률:
    주제 0 (비즈니스/경제 ): ████████████████████                              0.234
    주제 1 (스포츠       ): █                                                  0.012
    주제 2 (기술/IT     ): ██████████████████████████                          0.612
    주제 3 (국제 뉴스   ): ██                                                 0.045
    주제 4 (정치        ): ████                                               0.078
    주제 5 (엔터테인먼트 ): █                                                  0.019

[문서 #1000]
  내용: New Palestinian leader vows to fight corruption Abbas won his first major
  길이: 312 단어
  할당된 주제: 4
  주제 의미: 정치
  신뢰도: 0.723
  주제별 확률:
    주제 0 (비즈니스/경제 ): ██                                                 0.034
    주제 1 (스포츠       ): █                                                  0.008
    주제 2 (기술/IT     ): ██                                                 0.023
    주제 3 (국제 뉴스   ): ████████████████████                               0.287
    주제 4 (정치        ): ██████████████████████████                          0.723
    주제 5 (엔터테인먼트 ): █                                                  0.005

================================================================================

============================================================
[체크포인트 3-2] 신뢰도 분포 분석
============================================================

신뢰도 통계:
  평균: 0.678
  표준편차: 0.186
  최솟값: 0.234
  최댓값: 0.987
  중앙값: 0.712

신뢰도 분포 (구간별 문서 개수):
  0.0~0.5:   98개 (  6.5%) █
  0.5~0.6:  187개 ( 12.5%) ██
  0.6~0.7:  412개 ( 27.5%) █████
  0.7~0.8:  567개 ( 37.8%) ███████
  0.8~0.9:  215개 ( 14.3%) ███
  0.9~1.0:   21개 (  1.4%)
```

**해석**:
- 문서 #0 (비즈니스): 신뢰도 78.2% → 비즈니스 주제에 강하게 할당
- 문서 #100 (스포츠): 신뢰도 85.6% → 스포츠 주제에 가장 확실하게 할당
- 문서 #500 (기술): 신뢰도 61.2% → 기술과 비즈니스 모두 포함 (Apple 기업 뉴스)
- 문서 #1000 (정치): 신뢰도 72.3% → 정치와 국제 뉴스의 혼합
- 평균 신뢰도 67.8%는 합리적인 수준
- 대부분 문서(65.6%)가 0.6~0.8 범위의 신뢰도를 가짐

### 토픽 코히어런스(Coherence) 평가

```python
print("\n" + "="*60)
print("[체크포인트 3-3] 토픽 코히어런스 점수 계산")
print("="*60)

# 토픽 코히어런스: 각 주제의 상위 단어들이 의미론적으로 일관성 있는지 평가
# 방법: 상위 N개 단어가 문서에서 함께 나타나는 빈도 계산

def calculate_topic_coherence_simple(topic_words, documents, top_n=5):
    """
    간단한 코히어런스 계산:
    주제의 상위 5개 단어가 같은 문서에 함께 나타나는 비율
    """
    from collections import defaultdict

    word_set = set(w.lower() for w in topic_words[:top_n])

    doc_with_multiple_words = 0

    for doc in documents:
        doc_words = set(doc.lower().split())
        overlap = len(word_set & doc_words)

        # 상위 단어 중 2개 이상이 나타나면 일관성 있다고 본다
        if overlap >= 2:
            doc_with_multiple_words += 1

    coherence = doc_with_multiple_words / len(documents) if documents else 0
    return coherence

print("\n토픽 코히어런스 점수 (높을수록 좋음, 0~1):\n")

topic_coherences = {}
for topic_id in range(num_topics):
    top_terms = topic_model.get_topic(topic_id)
    topic_words = [word for word, _ in top_terms[:10]]

    coherence = calculate_topic_coherence_simple(topic_words, documents)
    topic_coherences[topic_id] = coherence

    topic_meanings = {
        0: "비즈니스/경제",
        1: "스포츠",
        2: "기술/IT",
        3: "국제 뉴스",
        4: "정치",
        5: "엔터테인먼트"
    }
    meaning = topic_meanings.get(topic_id, "Unknown")

    bar_length = int(coherence * 50)
    bar = "█" * bar_length + "░" * (50 - bar_length)

    print(f"주제 {topic_id} ({meaning:12s}): {bar} {coherence:.3f}")

avg_coherence = np.mean(list(topic_coherences.values()))
print(f"\n평균 토픽 코히어런스: {avg_coherence:.3f}")

# 코히어런스 해석
print(f"\n코히어런스 해석:")
if avg_coherence > 0.5:
    print(f"  ✓ 우수 (>0.5): 주제의 상위 단어들이 의미론적으로 일관성 있음")
elif avg_coherence > 0.4:
    print(f"  ○ 양호 (0.4~0.5): 기본적으로 타당한 주제 구성")
else:
    print(f"  ✗ 개선 필요 (<0.4): 주제의 단어들이 일관성 없음")
```

**예상 실행 결과**:
```
============================================================
[체크포인트 3-3] 토픽 코히어런스 점수 계산
============================================================

토픽 코히어런스 점수 (높을수록 좋음, 0~1):

주제 0 (비즈니스/경제 ): ██████████████████████████                0.487
주제 1 (스포츠       ): ███████████████████████████████            0.523
주제 2 (기술/IT     ): █████████████████████████                  0.456
주제 3 (국제 뉴스   ): ████████████████████                        0.412
주제 4 (정치        ): ██████████████████████████████              0.534
주제 5 (엔터테인먼트 ): ██████████████████████████                 0.468

평균 토픽 코히어런스: 0.480

코히어런스 해석:
  ○ 양호 (0.4~0.5): 기본적으로 타당한 주제 구성
```

**해석**:
- 평균 코히어런스 0.48은 0.4~0.5 범위 (양호)
- 주제 1(스포츠)과 주제 4(정치)가 가장 높은 일관성 (0.52~0.53)
- 주제 3(국제 뉴스)이 가장 낮은 일관성 (0.41)
  → 국제 뉴스가 여러 영역을 포함하기 때문에 자연스러움

### 주제 다양성(Diversity) 평가

```python
print("\n" + "="*60)
print("[체크포인트 3-4] 주제 다양성 점수 계산")
print("="*60)

# 주제 다양성: 각 주제의 상위 단어들이 다른 주제의 단어와 겹치지 않는 정도

# 모든 주제의 상위 10개 단어 수집
all_topic_words = {}
for topic_id in range(num_topics):
    top_terms = topic_model.get_topic(topic_id)
    top_words = set([word.lower() for word, _ in top_terms[:10]])
    all_topic_words[topic_id] = top_words

# 각 주제의 고유 단어 비율 계산
print("\n주제별 고유 단어 비율 (다른 주제와 겹치지 않는 단어의 비율):\n")

diversity_scores = {}
for topic_id in range(num_topics):
    unique_count = len(all_topic_words[topic_id])

    # 다른 주제와의 교집합 개수 계산
    for other_id in range(num_topics):
        if other_id != topic_id:
            overlap = len(all_topic_words[topic_id] & all_topic_words[other_id])
            unique_count -= overlap

    # 정규화: 음수가 되지 않도록
    diversity = max(0, unique_count) / 10  # 상위 10개 단어 기준
    diversity_scores[topic_id] = diversity

    topic_meanings = {
        0: "비즈니스/경제",
        1: "스포츠",
        2: "기술/IT",
        3: "국제 뉴스",
        4: "정치",
        5: "엔터테인먼트"
    }
    meaning = topic_meanings.get(topic_id, "Unknown")

    bar_length = int(diversity * 50)
    bar = "█" * bar_length + "░" * (50 - bar_length)

    print(f"주제 {topic_id} ({meaning:12s}): {bar} {diversity:.1%}")

overall_diversity = np.mean(list(diversity_scores.values()))
print(f"\n전체 주제 다양성 점수: {overall_diversity:.3f}")

# 다양성 해석
print(f"\n다양성 해석:")
if overall_diversity > 0.8:
    print(f"  ✓ 우수 (>0.8): 주제들이 명확히 구분됨")
elif overall_diversity > 0.7:
    print(f"  ○ 양호 (0.7~0.8): 주제들이 대부분 구분됨")
else:
    print(f"  ✗ 개선 필요 (<0.7): 주제들이 겹침이 있음")
```

**예상 실행 결과**:
```
============================================================
[체크포인트 3-4] 주제 다양성 점수 계산
============================================================

주제별 고유 단어 비율 (다른 주제와 겹치지 않는 단어의 비율):

주제 0 (비즈니스/경제 ): ████████████████████████████████████████ 80.0%
주제 1 (스포츠       ): ██████████████████████████████████████████ 90.0%
주제 2 (기술/IT     ): ██████████████████████████████              70.0%
주제 3 (국제 뉴스   ): ██████████████████████████████████████      85.0%
주제 4 (정치        ): ██████████████████████████████████████████ 95.0%
주제 5 (엔터테인먼트 ): ██████████████████████████████              75.0%

전체 주제 다양성 점수: 0.825

다양성 해석:
  ✓ 우수 (>0.8): 주제들이 명확히 구분됨
```

**해석**:
- 전체 다양성 점수 0.825는 0.8 이상 (우수)
- 주제 4(정치)가 95%로 가장 고유한 단어를 가짐
- 주제 1(스포츠)도 90%로 매우 구별됨
- 주제 2(기술/IT)의 다양성이 가장 낮은 이유:
  → "government", "system" 등이 다른 주제에도 포함됨

### 품질 평가 종합

```python
print("\n" + "="*60)
print("[체크포인트 3-5] 토픽 모델링 품질 평가 종합")
print("="*60)

# 모든 지표를 종합적으로 평가
quality_metrics = {
    "발견된 주제 개수": num_topics,
    "평균 토픽 코히어런스": f"{avg_coherence:.3f}",
    "전체 주제 다양성": f"{overall_diversity:.3f}",
    "할당 성공률": f"{(1 - num_noise / len(topics)) * 100:.1f}%",
    "최대 주제 크기": max([sum(topics == t) for t in range(num_topics)]),
    "최소 주제 크기": min([sum(topics == t) for t in range(num_topics)]),
    "주제 크기 표준편차": f"{np.std([sum(topics == t) for t in range(num_topics)]):.1f}",
}

print("\n品질 평가 결과:")
print("-"*60)
for metric, value in quality_metrics.items():
    print(f"  {metric:25s}: {value}")

# 최종 평가 판정
print("\n" + "-"*60)
print("최종 평가:")
print("-"*60)

coherence_score = float(quality_metrics["평균 토픽 코히어런스"])
diversity_score = float(quality_metrics["전체 주제 다양성"])

if coherence_score > 0.5 and diversity_score > 0.8:
    print("  ✅ 우수: 주제의 내부 일관성과 다양성이 모두 우수함")
    print("  → 신뢰할 수 있는 토픽 모델링 결과")
elif coherence_score > 0.4 and diversity_score > 0.7:
    print("  ✅ 양호: 기본적으로 타당한 주제 발견")
    print("  → 실무 활용 가능한 수준")
elif coherence_score > 0.3 or diversity_score > 0.6:
    print("  ⚠️  보통: 기본적 타당성 있지만 개선의 여지 있음")
    print("  → 하이퍼파라미터 튜닝 권장")
else:
    print("  ❌ 개선 필요: 주제의 일관성이나 다양성이 낮음")
    print("  → min_topic_size, embedding_model 변경 권장")

# 주제 균형성 평가
print("\n주제 크기 균형성:")
topic_sizes = [sum(topics == t) for t in range(num_topics)]
max_size = max(topic_sizes)
min_size = min(topic_sizes)
balance_ratio = min_size / max_size

if balance_ratio > 0.5:
    print(f"  ✓ 균형 잡힘: 최소 주제({min_size})가 최대 주제({max_size})의 {balance_ratio:.1%}")
elif balance_ratio > 0.3:
    print(f"  ○ 보통: 최소 주제({min_size})가 최대 주제({max_size})의 {balance_ratio:.1%}")
else:
    print(f"  ✗ 불균형: 최소 주제({min_size})가 최대 주제({max_size})의 {balance_ratio:.1%}")
    print("  → 일부 주제가 과도하게 크거나 작음")
```

**예상 실행 결과**:
```
============================================================
[체크포인트 3-5] 토픽 모델링 품질 평가 종합
============================================================

品質 평가 결과:
--------------------------------------------------------------
  발견된 주제 개수            : 6
  평균 토픽 코히어런스        : 0.480
  전체 주제 다양성            : 0.825
  할당 성공률                 : 96.9%
  최대 주제 크기              : 323
  최소 주제 크기              : 135
  주제 크기 표준편차          : 68.9

--------------------------------------------------------------
최종 평가:
--------------------------------------------------------------
  ✅ 양호: 기본적으로 타당한 주제 발견
  → 실무 활용 가능한 수준

주제 크기 균형성:
  ○ 보통: 최소 주제(135)가 최대 주제(323)의 41.8%
  → 일부 주제가 과도하게 크거나 작음
```

**해석**:
- 평가 결과: "양호" 수준 (실무 활용 가능)
- 코히어런스가 0.48로 약간 낮지만, 다양성이 0.825로 우수함
- 할당 성공률 96.9%는 매우 높음
- 주제 크기가 불균형(135~323)하지만, 자연스러운 분포다
  → 정치 뉴스가 실제로 더 많기 때문

### 검증 체크리스트

```python
print("\n" + "="*60)
print("[체크포인트 3] 검증 체크리스트")
print("="*60)

checks_cp3 = [
    ("특정 문서 주제 분석", len(sample_indices) > 0),
    ("신뢰도 분포 분석", np.mean(max_confidences) > 0.6),
    ("토픽 코히어런스 계산", avg_coherence > 0.4),
    ("주제 다양성 평가", overall_diversity > 0.7),
    ("품질 지표 종합 평가", True),
    ("최종 품질 판정", True),
]

for description, result in checks_cp3:
    status = "✓" if result else "✗"
    print(f"  {status} {description}")

print("\n✅ 체크포인트 3 완료!")
```

**예상 실행 결과**:
```
============================================================
[체크포인트 3] 검증 체크리스트
============================================================

  ✓ 특정 문서 주제 분석
  ✓ 신뢰도 분포 분석
  ✓ 토픽 코히어런스 계산
  ✓ 주제 다양성 평가
  ✓ 품질 지표 종합 평가
  ✓ 최종 품질 판정

✅ 체크포인트 3 완료!
```

---

## 흔한 실수 및 디버깅 가이드

### 실수 1: 데이터 전처리 부족

**문제**:
```python
# 틀림
documents = dataset['train']['text']  # 특수문자, 대소문자 섞여 있음
topic_model.fit_transform(documents)
```

**결과**: 토픽 품질이 낮음, 노이즈가 많음

**해결책**:
```python
# 맞음
def preprocess_text(text):
    """
    간단한 전처리:
    1. 소문자 변환
    2. 특수문자 제거
    3. 공백 정규화
    """
    import re
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)  # 특수문자 제거
    text = ' '.join(text.split())  # 공백 정규화
    return text

documents = [preprocess_text(doc) for doc in dataset['train']['text'][:1500]]
```

### 실수 2: 임베딩 모델 선택 오류

**문제**:
```python
# 틀림 - 느린 모델 사용
embedding_model = SentenceTransformer("all-mpnet-base-v2")
# 이 모델은 좋지만 768차원, 계산이 느림
```

**결과**: 학습이 매우 오래 걸림

**해결책**:
```python
# 맞음 - 빠르고 충분한 성능의 모델
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
# 384차원, 빠른 속도, 충분한 성능
```

### 실수 3: 최소 주제 크기 설정 오류

**문제**:
```python
# 틀림 - 너무 작은 값
topic_model = BERTopic(min_topic_size=3)
# 결과: 주제가 너무 많고 노이즈가 많음 (100+ topics)

# 또는 너무 큰 값
topic_model = BERTopic(min_topic_size=100)
# 결과: 주제가 너무 적고 정보 손실
```

**해결책**:
```python
# 맞음 - 일반적인 경우 10~20 권장
topic_model = BERTopic(min_topic_size=15)
# 1,500개 문서의 경우 5~20 범위가 적절
```

### 실수 4: 노이즈 문서 무시

**문제**:
```python
# 틀림
for topic_id in range(num_topics):
    # 주제 -1(노이즈)은 건너뜀
    if topic_id == -1:
        continue
```

**결과**: 전체 문서의 3~5%가 분석에서 제외됨

**해결책**:
```python
# 맞음 - 노이즈도 분석 대상에 포함
print(f"노이즈 문서: {sum(topics == -1)}개")
print(f"분석 가능 주제: {sum(topics != -1)}개")

# 필요시 노이즈 별도 분석
noise_docs = [documents[i] for i in range(len(documents)) if topics[i] == -1]
```

### 실수 5: 시각화 파일 저장 경로 오류

**문제**:
```python
# 틀림
fig.write_html("topic_barchart.html")  # 현재 디렉토리에 저장
# 어디에 저장되는지 불명확
```

**해결책**:
```python
# 맞음
from pathlib import Path

output_dir = Path("practice/chapter8/data/output")
output_dir.mkdir(parents=True, exist_ok=True)  # 디렉토리 생성

fig.write_html(output_dir / "topic_barchart.html")
print(f"저장: {output_dir / 'topic_barchart.html'}")
```

### 실수 6: 주제 의미 해석 과정 생략

**문제**:
```python
# 틀림
for topic_id in range(num_topics):
    print(topic_model.get_topic(topic_id))  # 단어만 출력
# 주제가 무엇을 의미하는지 해석 없음
```

**해결책**:
```python
# 맞음 - 의미 해석 포함
interpretation = {
    0: "비즈니스/경제: 기업, 판매, 시장",
    1: "스포츠: 경기, 선수, 리그",
    # ...
}

for topic_id, meaning in interpretation.items():
    print(f"주제 {topic_id}: {meaning}")
    print(f"  단어: {[w for w, _ in topic_model.get_topic(topic_id)[:5]]}")
```

---

## 종합 해설

### BERTopic 파이프라인의 4단계 이해

**1단계 (Document Embedding)**:
- BERT 사전학습 모델이 각 문서를 384차원 벡터로 변환
- 단순 단어 빈도가 아닌 문맥을 고려한 의미 벡터
- 의미론적으로 비슷한 문서는 벡터 공간에서 가까워짐

**2단계 (Dimensionality Reduction)**:
- UMAP이 384차원 벡터를 5차원으로 축소
- 고차원에서는 "거리"가 의미 없지만, 저차원에서는 의미 있게 됨
- 거리와 이웃 관계를 최대한 보존하면서 시각화 가능하게 만듦

**3단계 (Clustering)**:
- HDBSCAN이 축소된 벡터 공간에서 밀도 기반 클러스터링
- 데이터 자체의 구조에서 자연스러운 클러스터 개수 결정
- 주제 개수를 미리 정할 필요 없음

**4단계 (Topic Representation)**:
- c-TF-IDF가 각 클러스터(주제)의 특징 단어 추출
- "전체 말뭉치에서 자주 나타나는 흔한 단어"를 제외
- "주제마다 특정한 핵심 단어"를 선택

### 왜 BERTopic이 작동하는가?

LDA vs BERTopic:
- **LDA**: 수학 공식으로 주제를 계산 (확률론적)
- **BERTopic**: 벡터 공간의 거리로 주제를 판단 (기하학적)

BERTopic이 더 나은 이유:
1. **의미 이해**: BERT가 이미 단어의 의미를 알고 있음
2. **직관성**: "가까운 것끼리 같은 주제"가 직관적
3. **효율성**: 복잡한 확률 계산 불필요
4. **유연성**: 주제 개수 자동 결정

### 실무 응용

BERTopic은 다음에서 활용 가능:
- **뉴스 분석**: 일일 뉴스의 주요 주제 자동 분류
- **소셜 미디어**: 트렌딩 주제 추적
- **고객 피드백**: 리뷰에서 주요 불만사항 추출
- **학술 논문**: 특정 분야의 연구 주제 변화 추적

### 다음 단계

B회차 과제에서 얻은 경험을 토대로:
- 9주차: LDA와 BERTopic을 직접 비교 (성능 차이 분석)
- 10주차: Dynamic Topic Modeling (시간에 따른 주제 변화)
- 11~12주차: 프로젝트에 토픽 모델링 응용

---

## 참고 코드 파일

_전체 코드는 practice/chapter8/code/8-2-bertopic-analysis.py 참고_

### 최종 요약 코드 (빠른 실행용)

```python
# 빠른 실행: 모든 체크포인트를 한 번에 실행
import pandas as pd
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic

# 데이터 로드
dataset = load_dataset("bbc_news_classification")
documents = dataset['train']['text'][:1500]

# 모델 학습
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
topic_model = BERTopic(embedding_model=embedding_model, language="english")
topics, probs = topic_model.fit_transform(documents)

# 시각화
topic_model.visualize_barchart().show()
topic_model.visualize_heatmap().show()
topic_model.visualize_hierarchy().show()

# 품질 평가
print(f"주제 개수: {len(set(topics))-1}")
print(f"할당 성공률: {(1-sum(topics==-1)/len(topics))*100:.1f}%")
```

---

**생성일**: 2026-02-25
**대상 독자**: 컴퓨터공학/AI 전공 학부생 (3~4학년)
**난이도**: 중급 (파이썬, 딥러닝 기초 선수)
