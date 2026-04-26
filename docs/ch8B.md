## 8주차 B회차: BERTopic 다듬기 + 동적·멀티모달 확장

> **미션**: 수업이 끝나면 BERTopic 결과를 노이즈 처리·토픽 병합·자동 튜닝으로 다듬을 수 있고, 시간에 따른 주제 변화와 텍스트+이미지 통합 분석까지 직접 실행할 수 있다

### 학습목표

이 회차를 마치면 다음을 수행할 수 있다:

1. `reduce_outliers()`로 노이즈(-1) 문서를 가장 유사한 토픽에 재할당하고, 적용 전후 토픽별 문서 수 변화를 비교할 수 있다
2. `reduce_topics()`로 유사한 토픽들을 자동 병합하고, c-TF-IDF 코사인 유사도 기반의 병합 원리를 설명할 수 있다
3. **Optuna 다목적 최적화**로 일관성(coherence)은 최대화하고 노이즈 비율은 최소화하는 파라미터를 자동으로 찾고, 파레토 프론트(trade-off 곡선)를 해석할 수 있다
4. `topics_over_time()`을 사용해 시간에 따른 토픽 변화를 추출하고, 추세 5가지 패턴(급상승·점진상승·급락·주기·안정)을 분류할 수 있다
5. 토픽 간 시계열 **상관관계 행렬**을 계산해 함께 움직이는 주제를 식별할 수 있다
6. 각 토픽의 **출현(emergence) → 최고점(peak) → 소멸(decline)** 시점을 자동 식별하는 생명 주기 분석 함수를 구현하고, 결과를 비즈니스 시나리오와 연결해 해석할 수 있다
7. (선택) **CLIP** 기반 멀티모달 토픽 모델링의 두 시나리오(텍스트+이미지 결합 / 이미지 단독)를 시연할 수 있다

### 수업 타임라인

| 시간        | 내용                                                              | Copilot 활용     |
| ----------- | ----------------------------------------------------------------- | ---------------- |
| 00:00~00:05 | A회차 회고 + 오늘의 미션                                          | 사용 안 함       |
| 00:05~00:30 | 체크포인트 1: 평가지표 → `reduce_outliers` → `reduce_topics`       | 보조용으로 가능  |
| 00:30~00:55 | 체크포인트 2: Optuna 다목적 튜닝 + 파레토 해석                     | 보조용으로 가능  |
| 00:55~01:20 | 체크포인트 3: `topics_over_time` + 시계열 상관 + 토픽 생명 주기    | 보조용으로 가능  |
| 01:20~01:25 | (선택) 체크포인트 4: CLIP 멀티모달 시연                            | 보조용으로 가능  |
| 01:25~01:30 | 핵심 정리 + 제출물 안내                                            |                  |

> **A회차와의 관계**: A회차에서 만든 BERTopic 모델을 그대로 받아서 다듬는다. 노트북은 **셀 단위로 누적**되므로, A회차 마지막 모델(`topic_model`, `documents`, `topics`, `probabilities`)을 그대로 이어 쓴다.

---

### 오늘의 질문

**오늘의 질문**: "BERTopic이 자동으로 토픽을 뽑아줬다. 그런데 *너무 많은 토픽이 노이즈로 빠지고*, *비슷한 토픽이 따로따로 잡혀* 있다. 그리고 *시간이 지나면 어떤 주제가 뜨고 지는지* 알고 싶다. 이걸 데이터에 손대지 않고 도구로 풀 수 있는가?"

**대답**: 그럴 수 있다. BERTopic은 결과를 다시 만지작거릴 수 있는 **후처리 메서드 3종 세트**(`reduce_outliers`, `reduce_topics`, Optuna 튜닝)와 **동적 분석 API**(`topics_over_time`)를 갖춘다. 오늘은 이 도구들을 손에 익힌다.

---

### 8.6 BERTopic 다듬기 — "기본 결과를 실무 결과로"

A회차에서 만든 모델을 그대로 받아서 세 가지 도구로 다듬는다.

#### 8.6.1 노이즈 처리 — `reduce_outliers()`

##### 직관

HDBSCAN은 어디에도 명확히 속하지 않는 문서를 **-1(노이즈)**로 분리한다. 이는 안전한 기본 동작이지만, 노이즈가 너무 많으면 **분석에서 빠지는 데이터가 너무 많아져** 손실이 크다.

> **쉽게 말해서**: "이 사람은 어느 그룹에 속할지 애매해서 일단 빼두자"고 한 사람들에게, "그래도 가장 가까운 그룹은 어디인가?"를 다시 물어보는 것이다.

##### 처리 전략 3가지

1. **그대로 둔다(제외)**: 노이즈가 적고(< 5%) 분석 정확성이 우선일 때. 가장 단순하지만 정보 손실이 있다.
2. **재할당(`reduce_outliers`)**: 각 노이즈 문서를 **가장 유사한 토픽**에 다시 배정. 일반적으로 가장 권장.
3. **별도 검토**: 노이즈 문서가 새로운 패턴을 담고 있는지 사람이 직접 본다. 새 토픽을 발견할 때 사용.

##### 코드

```python
# strategy 옵션:
#   "distributions" : 토픽 확률 분포로 가장 가까운 토픽에 할당 (권장)
#   "embeddings"    : 임베딩 코사인 유사도로 할당
#   "c-tf-idf"      : 키워드 유사도로 할당
new_topics = topic_model.reduce_outliers(
    documents=documents,
    topics=topics,
    strategy="distributions",
)

# BERTopic 내부 토픽 정보 갱신 (이후 시각화/평가에 반영)
topic_model.update_topics(documents, topics=new_topics)
```

##### 적용 전후 비교 (예시)

**표 8.4** `reduce_outliers` 적용 전후 토픽별 문서 수 변화 (가상 결과)

| 토픽 | 자동 레이블        | 적용 전 | 적용 후 | 변화량 |
| ---- | ------------------ | ------- | ------- | ------ |
| -1   | (노이즈)           | 218     | 2       | -216   |
| 0    | Computer Graphics  | 528     | 544     | +16    |
| 1    | Sports/Baseball    | 462     | 468     | +6     |
| 2    | Medical/Health     | 411     | 511     | +100   |
| 3    | Politics           | 275     | 369     | +94    |
| 합계 |                    | 1894    | 1894    |        |

> **주의**: 재할당은 "**가장 가까운**" 토픽을 찾는 것이지 "**확실히 맞는**" 토픽을 찾는 것이 아니다. 적용 후에는 반드시 대표 문서를 다시 읽어 검증한다.

#### 8.6.2 토픽 축소·병합 — `reduce_topics()`

##### 직관

HDBSCAN은 데이터의 밀도 구조를 보고 토픽 수를 **자동으로** 결정한다. 그래서 종종 **너무 잘게 쪼개진** 결과가 나온다. "프로야구"와 "축구"가 별도 토픽이지만, 보고서에는 **"스포츠"** 한 줄로 충분할 수 있다.

`reduce_topics(nr_topics=K)`는 토픽 임베딩(c-TF-IDF 벡터) 사이의 **코사인 유사도**가 큰 순서로 토픽을 병합해 K개로 줄여준다. 키워드는 병합 후 **재계산**된다.

##### 코드

```python
# 현재 토픽 개수 확인
before = topic_model.get_topic_info()
print(f"원래 토픽 수: {len(before) - 1}")  # -1 노이즈 제외

# 토픽 수를 4개로 축소
topic_model.reduce_topics(documents, nr_topics=4)

after = topic_model.get_topic_info()
print(f"축소 후 토픽 수: {len(after) - 1}")
print(after.head())
```

##### 어떤 K를 고를까?

`reduce_topics`로 무작정 줄이지 말고, **`visualize_hierarchy()`로 덴드로그램을 먼저 보고** 자연스러운 절단 높이를 결정하자. 가지가 길게 떨어진 곳을 자르면 의미 있는 그룹화가 유지된다.

> **언제 쓰나**: (1) 토픽이 너무 많아 보고서로 옮기기 어렵다, (2) 비슷한 키워드 집합이 두세 개로 쪼개져 있다, (3) 상위 수준의 주제 구조를 먼저 파악하고 싶다.

#### 8.6.3 Optuna 다목적 최적화 — "일관성↑ × 노이즈↓"

##### 직관

UMAP·HDBSCAN의 파라미터를 손으로 돌리는 건 비효율적이다. **Optuna**는 여러 트라이얼을 **베이지안 최적화**로 똑똑하게 탐색한다. 그리고 토픽 모델링은 본질적으로 **두 가지 목표가 충돌**한다.

- **일관성(coherence)** ↑ : 토픽 키워드가 의미적으로 잘 뭉치게
- **노이즈 비율** ↓ : -1로 빠지는 문서를 줄이게

이 두 목표를 동시에 만족시키기 어렵다. 그래서 **다목적 최적화(multi-objective)**를 쓰고, 결과로 받는 것은 **파레토 프론트(Pareto front)**다.

> **파레토 프론트란**: "한쪽을 더 좋게 하려면 다른 쪽이 반드시 나빠지는 지점들의 집합". 즉, 더 이상 일방적으로 개선할 수 없는 후보 해(解)들의 모음. 분석가는 이 곡선 위에서 **취향대로** 한 점을 고른다("일관성 0.55를 포기하더라도 노이즈를 5% 미만으로 잡고 싶다" 식으로).

##### 코드 — 작은 trial로 데모 가능

```python
import optuna
from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN

def objective(trial):
    n_neighbors      = trial.suggest_int("n_neighbors", 5, 25)
    n_components     = trial.suggest_int("n_components", 2, 10)
    min_cluster_size = trial.suggest_int("min_cluster_size", 5, 30)
    min_samples      = trial.suggest_int("min_samples", 3, 20)

    umap_model = UMAP(
        n_neighbors=n_neighbors, n_components=n_components,
        min_dist=0.0, metric="cosine", random_state=42,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=min_cluster_size, min_samples=min_samples,
        metric="euclidean", cluster_selection_method="eom",
    )
    model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model, hdbscan_model=hdbscan_model,
        calculate_probabilities=False, verbose=False,
    )
    topics, _ = model.fit_transform(documents)

    coherence      = npmi_coherence(model, documents)            # ↑
    outlier_ratio  = sum(1 for t in topics if t == -1) / len(topics)  # ↓
    return coherence, outlier_ratio

study = optuna.create_study(directions=["maximize", "minimize"])
study.optimize(objective, n_trials=10, show_progress_bar=False)

# 파레토 프론트 후보 출력
for t in study.best_trials:
    print(f"NPMI={t.values[0]:.3f}  noise={t.values[1]:.3f}  params={t.params}")
```

##### 결과 해석 가이드

- **파레토 프론트 위의 한 점**을 분석 목적에 맞춰 고른다.
- 노이즈 비율이 0인데 일관성이 너무 낮다면 → 토픽이 거대한 한두 덩어리로 뭉쳐졌을 가능성. `min_cluster_size`를 줄여 본다.
- 일관성은 좋은데 노이즈가 30% 이상이면 → `reduce_outliers()`로 후처리한다(둘은 충돌이 아니라 보완 관계).

> **데모용 trial 수**: 강의용으로는 `n_trials=10` 정도면 충분히 직관을 보여준다. 실무에서는 50~100회를 돌린다.

---

### 8.7 동적 토픽 모델링 — 시간이 흐르면 주제는 어떻게 변하는가

#### 8.7.1 무엇을 푸는가

정적 토픽 모델은 "전체 기간에 걸친 큰 주제"는 보여주지만, **언제 그 주제가 떠올랐고 언제 사라졌는지**는 알려주지 않는다. 동적 토픽 모델링(Blei & Lafferty, 2006)은 문서에 **타임스탬프**를 부여해 시간대별 토픽 빈도를 추적한다.

**활용 예시**:

- 정책 시행 효과 모니터링 ("재택근무 의무화 이후 '화상회의' 토픽 빈도 변화")
- 위기 대응 타이밍 ("코로나 확진 토픽이 언제 정점에 도달했는가")
- 신제품 반응 추적 ("새 모델 출시 후 부정 키워드의 부상 시점")
- 학술 트렌드 ("어떤 연구 주제가 부상하고 있는가")

#### 8.7.2 `topics_over_time()` — 한 줄 호출, 시계열 결과

##### 코드

```python
import pandas as pd

timestamps = pd.to_datetime(df["date"]).tolist()  # 문서별 타임스탬프

topics_over_time = topic_model.topics_over_time(
    docs=documents,
    timestamps=timestamps,
    nr_bins=20,             # 시간을 몇 구간으로 나눌지
    global_tuning=True,     # 토픽 단어를 전역 기준으로 보정
    evolution_tuning=True,  # 인접 시점 단어 분포를 부드럽게
)

topic_model.visualize_topics_over_time(
    topics_over_time, top_n_topics=5,
).show()
```

##### `nr_bins` 가이드

- 1년치 일별 데이터 → `nr_bins=12` (월별)
- 3년치 데이터 → `nr_bins=36` (월별) 또는 `12` (분기별)
- 데이터 양이 적은 시점이 있으면 빈이 비어 보일 수 있다 → bin을 더 넓게 잡는다.

#### 8.7.3 추세 5가지 패턴 — 무엇을 보아야 하는가

**표 8.5** 동적 토픽의 추세 패턴

| 패턴            | 모양                            | 해석 / 비즈니스 시나리오                          |
| --------------- | ------------------------------- | ------------------------------------------------- |
| **급격한 상승** | 짧은 기간에 빈도가 폭발적으로 ↑ | 새 이슈 발생, 위기, 바이럴 — 즉시 대응 필요         |
| **점진적 상승** | 느리지만 꾸준히 ↑               | 장기 트렌드, 신규 시장 — 투자/기획 신호            |
| **급격한 하락** | 짧은 기간에 빠르게 ↓            | 이슈 해결, 관심 이동 — 자원 재배분 신호             |
| **주기적 변동** | 규칙적으로 ↑↓ 반복             | 계절성, 정기 이벤트 — 캘린더 기반 마케팅            |
| **안정 (상시)** | 일정한 빈도 유지                | 지속 관심 주제 — "언제나 챙겨야 하는 주제"          |

#### 8.7.4 토픽 간 시계열 상관 — 함께 움직이는 주제 찾기

##### 직관

"코로나" 토픽이 오를 때 "재택근무" 토픽도 같이 오른다면, 둘 사이에 **양의 상관**이 있다. 이런 신호는 단독 토픽 분석으로는 잡히지 않는다.

##### 코드

```python
# 토픽×시간 피벗 → 시계열 상관 행렬
pivot_df = topics_over_time.pivot(
    index="Timestamp", columns="Topic", values="Frequency",
).fillna(0)

correlation_matrix = pivot_df.corr()
print(correlation_matrix.round(2))
```

> **해석 팁**: 절대값이 0.7 이상이면 강한 동행/역행 관계. 0.3 미만은 거의 무관.

#### 8.7.5 토픽 생명 주기 — 출현 → 최고점 → 소멸

##### 직관

각 토픽이 **언제 처음 등장했고**(출현), **언제 가장 활발했고**(최고점), **언제 사라졌는지**(소멸)를 자동으로 찾는다. 마케팅·정책·연구에서 매우 강력한 인사이트.

##### 자동 식별 함수

```python
def analyze_topic_lifecycle(topics_over_time, topic_model, threshold=0.01):
    """토픽의 출현·최고점·소멸 시점 자동 식별."""
    lifecycle = {}
    unique_topics = [t for t in topics_over_time["Topic"].unique() if t != -1]

    for topic in unique_topics:
        td = topics_over_time[topics_over_time["Topic"] == topic]

        # 출현: 빈도가 임계값을 넘는 첫 시점
        above = td[td["Frequency"] > threshold]
        emergence = above.iloc[0]["Timestamp"] if len(above) > 0 else None

        # 최고점: 빈도가 최대인 시점
        peak_idx = td["Frequency"].idxmax()
        peak_t   = td.loc[peak_idx, "Timestamp"]
        peak_f   = td.loc[peak_idx, "Frequency"]

        # 소멸: 최고점 이후 임계값 밑으로 떨어진 첫 시점
        after = td.loc[peak_idx:]
        below = after[after["Frequency"] < threshold]
        decline = below.iloc[0]["Timestamp"] if len(below) > 0 else None

        lifecycle[topic] = {
            "emergence": emergence,
            "peak": peak_t,
            "peak_frequency": peak_f,
            "decline": decline,
            "keywords": [w for w, _ in topic_model.get_topic(topic)[:5]],
        }
    return lifecycle
```

##### 결과 예시

**표 8.6** 토픽 생명 주기 분석 결과 (예시)

| 토픽         | 키워드            | 출현    | 최고점  | 소멸    | 패턴 해석                            |
| ------------ | ----------------- | ------- | ------- | ------- | ------------------------------------ |
| 코로나(확진) | 코로나, 방역, 확진 | 2020-01 | 2020-03 | 2021-06 | 팬데믹 초기 급부상 후 점진적 감소     |
| 백신접종     | 백신, 접종, 면역  | 2020-11 | 2021-04 | -       | 백신 발표 후 지속적 관심              |
| 화상회의     | 화상, 플랫폼, 재택 | 2020-03 | 2020-05 | 2021-03 | 거리두기 기간의 한시적 급증           |
| 스마트폰     | 제품, 스마트폰    | -       | 2019-07 | -       | 안정적 상시 토픽                      |

> **임계값 `threshold=0.01`**: 데이터 규모와 빈도 분포에 맞춰 조정. 빈도가 모두 작은 데이터에서는 더 낮춘다.

---

### 8.8 (선택) 멀티모달 토픽 모델링 — 텍스트 + 이미지

> **이 절은 선택 학습이다**. CLIP 모델 다운로드(약 600MB)와 추가 의존성(`bertopic[vision]`)이 필요하다. 노트북에서는 옵션 셀로 분기되어 있어, 데이터·환경이 준비되지 않으면 자동으로 스킵된다.

#### 8.8.1 무엇을 푸는가

소셜 미디어 게시물, 전자상거래 상품, 뉴스 기사 — 현실의 데이터는 **텍스트와 이미지가 짝**으로 존재한다. 멀티모달 토픽 모델링은 이 둘을 **같은 벡터 공간**에 두고 함께 분석한다.

핵심은 **CLIP**(Radford et al., 2021)이다. CLIP은 텍스트와 이미지를 동일한 512차원 공간에 투영해 의미적으로 유사한 텍스트와 이미지가 가까운 위치에 오도록 학습되었다. "해변에서 서핑하는 사람"이라는 텍스트와 실제 서핑 이미지는 같은 동네에 모인다.

#### 8.8.2 두 가지 시나리오

**표 8.7** BERTopic 멀티모달 시나리오

| 시나리오          | 입력           | 임베딩 방식                              | 활용 사례         |
| ----------------- | -------------- | ---------------------------------------- | ----------------- |
| 텍스트 + 이미지    | 문서-이미지 쌍  | CLIP 결합 임베딩(텍스트·이미지 평균)      | 소셜 미디어, 상품 |
| 이미지 단독        | 이미지만        | CLIP 이미지 임베딩 + 자동 캡션           | 의료 영상, 위성   |

##### 시나리오 A — 텍스트 + 이미지 결합

```python
from bertopic import BERTopic
from bertopic.backend import MultiModalBackend
from bertopic.representation import VisualRepresentation

embedding_model = MultiModalBackend("clip-ViT-B-32", batch_size=32)
representation_model = {"Visual_Aspect": VisualRepresentation()}

topic_model = BERTopic(
    embedding_model=embedding_model,
    representation_model=representation_model,
    verbose=True,
)
topics, probs = topic_model.fit_transform(
    documents=captions,    # 텍스트
    images=image_paths,    # 이미지 경로 리스트
)
```

> **결과의 직관**: "강아지와 산책" 캡션과 실제 산책 이미지가 같은 공간에서 가까워지므로, 텍스트만으로는 흐릿했던 "반려동물 일상" 토픽이 더 또렷하게 잡힌다.

##### 시나리오 B — 이미지 단독 + 자동 캡션

```python
representation_model = {
    "Visual_Aspect": VisualRepresentation(
        image_to_text_model="nlpconnect/vit-gpt2-image-captioning",
    )
}
topic_model = BERTopic(
    embedding_model=embedding_model,
    representation_model=representation_model,
)
topics, probs = topic_model.fit_transform(
    documents=None,    # 텍스트 없음
    images=images,
)
```

이미지로 군집을 만든 뒤, 각 토픽의 대표 이미지에 **자동 캡션**을 붙여 사람이 읽을 수 있는 토픽 설명을 생성한다.

#### 8.8.3 실무 적용 가이드

- **데이터**: 텍스트–이미지가 1:1로 정렬되어야 한다. 이미지를 224×224로 사전 정규화하면 처리 속도가 크게 빨라진다.
- **모델 선택**: 일반 용도는 `clip-ViT-B/32` (가벼움), 정확도가 더 필요하면 `clip-ViT-L/14` (무거움). 패션·의료 등 도메인 특화 CLIP 모델도 존재한다.
- **성능 최적화**: 임베딩을 미리 계산해서 디스크에 저장해두면 반복 분석 시 시간을 크게 줄일 수 있다. 이미지 I/O가 병목이면 병렬 처리.
- **언제 쓸까?**: 텍스트만으로 토픽이 혼동될 때(예: "leather"가 재킷·신발·가방에 모두 나오는 경우) 시각적 형태가 토픽을 깔끔하게 분리해 준다.

---

### B회차 90분 실습 가이드

#### 체크포인트 1 (25분) — 평가 → 노이즈 처리 → 토픽 병합

**수행 순서**:

1. **NPMI** + **Topic Diversity** 계산 (A회차 라이브 코딩의 함수 그대로)
2. `topic_model.reduce_outliers(documents, topics, strategy="distributions")` 실행
3. 적용 전후 토픽별 문서 수 변화 표 생성 (CSV로 저장)
4. `topic_model.visualize_hierarchy()` 로 덴드로그램 확인
5. `topic_model.reduce_topics(documents, nr_topics=4)` 적용 후 키워드 변화 출력

**산출물**:

- `data/output/ch8B_outlier_reduction.csv` (전후 비교)
- `data/output/ch8B_reduced_topic_keywords.json` (병합 후 키워드)

**리포트 (1문단)**: 노이즈 재할당 후 어떤 토픽이 가장 많이 늘었는가? 병합 후 토픽 키워드가 더 명료해졌는가, 흐려졌는가?

#### 체크포인트 2 (25분) — Optuna 다목적 튜닝

**수행 순서**:

1. `objective(trial)` 함수 작성 (NPMI 최대화, 노이즈 비율 최소화)
2. `optuna.create_study(directions=["maximize", "minimize"])`로 study 생성
3. `n_trials=10`으로 시범 최적화 (실무는 50~100, 강의는 10이면 충분)
4. 파레토 프론트의 trial들을 출력하고, 그 중 한 점을 선택
5. 선택한 파라미터로 BERTopic을 재학습한 결과의 평가지표 계산

**산출물**:

- `data/output/ch8B_optuna_pareto.csv` (파레토 프론트)
- `data/output/ch8B_tuned_topic_info.csv`

**리포트 (1문단)**: 어떤 파라미터를 선택했고, 그 이유는? trade-off에서 어떤 가치(일관성/포괄성)를 우선했는가?

#### 체크포인트 3 (25분) — 동적 토픽 + 생명 주기

**수행 순서**:

1. `topic_model.topics_over_time(documents, timestamps, nr_bins=12)` 실행
2. `topic_model.visualize_topics_over_time(...)` HTML 저장
3. 토픽 간 시계열 상관 행렬 계산 후, 절대값 0.5 이상인 토픽 쌍 찾기
4. `analyze_topic_lifecycle()` 함수로 출현·최고점·소멸 식별
5. 생명 주기 결과를 표 8.6 형식의 표로 정리

**산출물**:

- `data/output/ch8B_topics_over_time.csv`
- `data/output/ch8B_topic_correlation.csv`
- `data/output/ch8B_topic_lifecycle.csv`
- `data/output/ch8B_topics_over_time.html`

**리포트 (1문단)**: 가장 인상적인 추세 패턴(5가지 중 어떤 것)은 어느 토픽에서 관찰되는가? 그 패턴이 데이터의 도메인 사건과 어떻게 연결되는가?

#### (선택) 체크포인트 4 (15분) — CLIP 멀티모달 시연

조건: `bertopic[vision]` 설치 가능 + 디스크 ~1GB 여유 + 텍스트-이미지 쌍 데이터.

**수행 순서**:

1. `MultiModalBackend("clip-ViT-B-32")` 초기화
2. 텍스트+이미지 결합 시나리오로 BERTopic 학습
3. 토픽별 대표 키워드 + 대표 이미지 출력

**리포트 (1문단)**: 텍스트 단독 분석 대비 토픽이 어떻게 달라졌는가?

---

### 제출물 체크리스트

- [ ] 노트북(`practice/chapter8/8A-topic-modeling-assignment.ipynb`) 모든 셀 실행 완료
- [ ] `data/output/`에 산출물 CSV/JSON/HTML 저장
- [ ] 분석 리포트 (체크포인트별 1문단씩, 총 3~4문단)
- [ ] (선택) CLIP 시연 결과 캡처

### Copilot 활용 가이드

A회차와 달리 B회차에서는 보조 도구로 Copilot을 사용해도 좋다. 단, **결과 해석은 본인이 직접** 작성한다.

- 기본: "이 노이즈 토픽들을 가장 가까운 토픽에 재할당해줘"
- 심화: "Optuna 다목적 최적화의 파레토 프론트를 matplotlib로 시각화해줘"
- 검증: "이 토픽 간 상관관계가 통계적으로 유의한지 확인하는 코드를 추가해줘"

### 평가 루브릭 (참고)

| 기준                  | 우수                                              | 보통                              | 미흡             |
| --------------------- | ------------------------------------------------- | --------------------------------- | ---------------- |
| 노이즈 처리/병합 적용  | 전후 변화를 표로 비교 + 해석                      | 적용만 함                         | 미적용           |
| Optuna 튜닝           | 파레토 프론트 분석 + 선택 근거 제시               | 단일 trial 결과만 보고             | 실행 실패        |
| 동적/생명주기 분석     | 패턴 분류 + 도메인 해석 + 상관관계 발견            | 빈도 그래프만 제시                | 그래프 누락      |
| 산출물 정리            | 모든 CSV/HTML/리포트 완비                         | 일부 누락                         | 다수 누락        |
| 리포트                | 결과를 정량적·정성적으로 모두 해석                 | 결과만 나열                       | 해석 없음        |

---

### Exit ticket

**문제 (1문항)**:

다음 중 BERTopic 후처리 메서드에 대한 설명으로 **잘못된** 것은?

① `reduce_outliers(strategy="distributions")`는 -1로 분류된 문서를 토픽 확률 분포 기준으로 가장 유사한 토픽에 재할당한다
② `reduce_topics(nr_topics=K)`는 c-TF-IDF 벡터 간 코사인 유사도가 큰 토픽들을 우선 병합한다
③ Optuna 다목적 최적화는 일관성과 노이즈 비율 둘 다를 *동시에 같은 방향으로 최대화*한다
④ 파레토 프론트의 한 점을 고른다는 것은 두 목표 사이의 trade-off 중 하나를 명시적으로 선택한다는 뜻이다

정답: **③**

**설명**: 일관성은 **최대화**(`maximize`), 노이즈 비율은 **최소화**(`minimize`) — 두 목표는 방향이 다르고, 흔히 서로 충돌한다. Optuna는 두 목표를 동시에 같은 방향으로 끌어당기는 게 아니라, 어느 한쪽을 양보하지 않고서는 다른 쪽을 더 개선할 수 없는 후보들의 집합(파레토 프론트)을 보여준다. 분석가는 이 곡선 위에서 자신의 목적에 맞는 점을 고른다(④).

---

## 더 알아보기

- Grootendorst, M. (2022). _BERTopic: Neural Topic Modeling with a Class-based TF-IDF procedure_. arXiv. https://arxiv.org/abs/2203.05556
- Akiba, T. et al. (2019). _Optuna: A Next-generation Hyperparameter Optimization Framework_. KDD. https://arxiv.org/abs/1907.10902
- Blei, D. M., & Lafferty, J. D. (2006). _Dynamic Topic Models_. ICML. https://dl.acm.org/doi/10.1145/1143844.1143859
- Radford, A. et al. (2021). _Learning Transferable Visual Models From Natural Language Supervision (CLIP)_. ICML. https://arxiv.org/abs/2103.00020
- BERTopic 공식 문서 (시각화·동적·멀티모달 가이드): https://maartengr.github.io/BERTopic/

---

## 참고문헌

1. Grootendorst, M. (2022). BERTopic: Neural Topic Modeling with a Class-based TF-IDF procedure. _arXiv preprint_. https://arxiv.org/abs/2203.05556
2. Blei, D. M., & Lafferty, J. D. (2006). Dynamic Topic Models. _ICML_, 113-120.
3. Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019). Optuna: A Next-generation Hyperparameter Optimization Framework. _KDD_, 2623-2631.
4. Radford, A. et al. (2021). Learning Transferable Visual Models From Natural Language Supervision. _ICML_, 8748-8763.
5. Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. _EMNLP_. https://arxiv.org/abs/1908.10084
6. Röder, M., Both, A., & Hinneburg, A. (2015). Exploring the Space of Topic Coherence Measures. _WSDM_, 399-408.
