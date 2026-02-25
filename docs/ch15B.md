## 15주차 B회차: 기말고사 및 종강

> **미션**: 8~13주차 핵심 개념을 종합 평가하고 학기를 마무리한다

### 수업 타임라인

| 시간 | 내용 |
|------|------|
| 00:00~01:20 | 기말고사 (객관식, 80분) |
| 01:20~01:30 | 채점 및 종강 안내 |

---

## 기말고사 안내

### 시험 개요

- **시험 시간**: 80분 (10:00~11:20)
- **문항 수**: 40문제
- **형식**: 객관식 4지선다
- **범위**: 8주차(토픽 모델링)~13주차(배포 최적화)
- **배점**: 1문제 2.5점 (총 100점)
- **사용 도구**: 금지 — Copilot, 인터넷 접속, 노트북 계산 불가
- **유의 사항**:
  - 교재 및 강의노트는 **참고 자료로 제시되지 않음** (실시간 참고 불가)
  - 개인 손전등/조명 사용 불가
  - 휴대전화, 스마트워치 등 전자기기 제출 필수
  - 부정행위 시 해당 과목 F학점 + 학칙상 징계

### 주차별 출제 비중

| 주차 | 개념 | 문제 수 | 비중 |
|:---:|------|:---:|:---:|
| 8 | 토픽 모델링, BERTopic | 7 | 17.5% |
| 9 | Full Fine-tuning, Trainer API | 7 | 17.5% |
| 10 | PEFT, LoRA, QLoRA | 8 | 20% |
| 11 | RAG 시스템 아키텍처 | 7 | 17.5% |
| 12 | AI Agent, LangGraph | 7 | 17.5% |
| 13 | FastAPI, Docker 배포, 평가 | 4 | 10% |
| **합계** | | **40** | **100%** |

### 출제 방향

1. **개념 이해** (40%): 핵심 원리를 설명하는 문제
   - "왜 필요한가", "어떤 문제를 해결하는가"
   - 직관적 비유로 원리를 이해했는가

2. **코드 해석** (35%): 코드 단편을 읽고 결과 예측
   - Hugging Face Transformers API 활용
   - PyTorch 코드의 논리 추적
   - 출력값 또는 동작 결과 판단

3. **실무 적용** (25%): 상황 제시 후 기술 선택
   - "이런 상황에서 어떤 방법을 쓸까?"
   - 하이퍼파라미터 설정 및 트레이드오프
   - 성능 최적화 전략

---

## 예시 문제은행

### 8주차: 토픽 모델링과 BERTopic

**문제 8-1**
다음 중 토픽 모델링(Topic Modeling)의 가장 근본적인 목적으로 옳은 것은?

① 문서의 모든 단어를 의미별로 분류하기
② 대규모 문서 컬렉션에서 숨은 주제 구조를 발견하기
③ 각 문서의 감정을 분류하기
④ 문장 내 문법 오류를 찾기

**정답**: ② — 토픽 모델링은 문서 집합에서 반복되는 의미 패턴(주제)을 자동으로 추출하는 기법이다.

---

**문제 8-2**
전통적인 LDA(Latent Dirichlet Allocation)와 비교할 때, BERTopic이 갖는 가장 큰 개선점은?

① 더 빠른 학습 속도
② 사전 학습된 언어 모델(BERT)의 의미 이해를 활용하여 의미 있는 토픽 추출
③ 더 적은 메모리 사용
④ 자동으로 토픽 개수를 결정

**정답**: ② — BERTopic은 문장 임베딩(Sentence Transformers)으로 문서 표현을 만들고, 클러스터링을 통해 더 일관된 토픽을 찾는다.

---

**문제 8-3**
다음 코드의 실행 결과로 옳은 것은?

```python
from bertopic import BERTopic
from datasets import load_dataset

docs = ["machine learning models", "deep neural networks", "food taste"]

model = BERTopic(language="english", calculate_probabilities=False)
topics, probs = model.fit_transform(docs)

print(topics)
```

① `[0, 0, 1]` (첫 두 문서는 같은 토픽, 세 번째는 다른 토픽)
② `[-1, -1, -1]` (모든 문서가 이상치로 판단)
③ `[0, 0, 0]` (모든 문서가 한 토픽으로 합쳐짐)
④ `[0, 1, 2]` (각 문서가 서로 다른 토픽)

**정답**: ① — 첫 두 문서는 의미적으로 유사(AI/ML 관련)하여 같은 클러스터에 배정되고, "food taste"는 전혀 다른 토픽으로 분류된다. 단, 문서 3개로는 BERTopic이 기본 클러스터링을 수행할 때 작은 클러스터는 병합될 수 있으므로 실제 동작은 하이퍼파라미터에 따라 달라질 수 있다.

---

**문제 8-4**
BERTopic에서 `get_topic()`이 반환하는 값은 무엇인가?

① 토픽 이름 (자동 생성된 문자열)
② 토픽에 속한 상위 N개 단어와 각 단어의 중요도 점수 (Word-Topic 분포)
③ 토픽에 속한 문서들의 원본 텍스트
④ 토픽의 크기 (문서 개수)

**정답**: ② — `get_topic(topic_id)`는 `[(word, score), ...]` 형태로 해당 토픽의 대표 단어와 중요도를 반환한다. 이를 통해 토픽이 어떤 의미를 갖는지 해석할 수 있다.

---

**문제 8-5**
BERTopic 모델 학습 후, 새로운 문서에 대해 토픽을 예측할 때 사용하는 메서드는?

① `fit_transform()`
② `transform()` 또는 `predict()`
③ `fit()`
④ `get_topics()`

**정답**: ② — 이미 학습된 모델에 대해 새 문서의 토픽을 예측할 때는 `transform()` 또는 `predict()`를 사용한다. `fit_transform()`은 학습과 변환을 동시에 하므로 새 데이터 예측에는 부적절하다.

---

**문제 8-6**
BERTopic에서 임베딩 모델로 Sentence Transformers 대신 OpenAI Embedding API를 사용하려고 한다. 다음 중 올바른 코드는?

```python
# ①
model = BERTopic(embedding_model="text-embedding-ada-002")

# ②
from sentence_transformers import SentenceTransformer
embedding_model = SentenceTransformer("text-embedding-ada-002")
model = BERTopic(embedding_model=embedding_model)

# ③
from langchain.embeddings.openai import OpenAIEmbeddings
embedding_model = OpenAIEmbeddings()
model = BERTopic(embedding_model=embedding_model)

# ④
model = BERTopic(use_openai_embedding=True)
```

① ② ③ ④

**정답**: ③ (또는 현재 BERTopic 버전에 따라 조정 가능) — BERTopic은 embedding_model 파라미터로 외부 임베딩 모델을 받을 수 있으며, LangChain의 OpenAIEmbeddings 클래스 같은 호환 가능한 객체를 전달할 수 있다. (정확한 API는 버전에 따라 다를 수 있으므로 공식 문서 확인 필수)

---

**문제 8-7**
토픽 모델의 성능을 평가할 때, Coherence Score의 의미로 가장 옳은 것은?

① 토픽별 문서 개수의 균형 정도
② 추출된 상위 단어들이 의미적으로 얼마나 일관성 있는지 (높을수록 좋음)
③ 토픽과 토픽 간의 구분 정도
④ 토픽 학습에 소요된 시간

**정답**: ② — Coherence Score (높은 값: 0.5~1.0)는 토픽 내 상위 단어들이 의미적으로 얼마나 관련 있는지 측정한다. 높을수록 더 일관성 있는(해석 가능한) 토픽을 의미한다.

---

### 9주차: Full Fine-tuning과 Trainer API

**문제 9-1**
Full Fine-tuning의 정의로 가장 올바른 것은?

① 모델의 마지막 레이어만 학습시키는 기법
② 사전 학습된 모델의 **모든 파라미터**를 목표 작업에 맞게 업데이트하는 기법
③ 모델의 일부 레이어만 동결하고 나머지만 학습
④ 모델을 처음부터 훈련하는 기법

**정답**: ② — Full Fine-tuning은 BERT 같은 사전 학습 모델의 모든 가중치를 업데이트한다. 메모리와 계산이 많이 필요하지만, 작은 데이터셋에서도 충분히 학습할 수 있다.

---

**문제 9-2**
다음 중 Hugging Face `Trainer`의 **가장 큰 장점**으로 옳은 것은?

① 학습 루프를 수동으로 작성하지 않아도 자동으로 처리
② GPU 메모리 관리를 완전 자동화
③ 모든 모델에 대해 동일하게 작동 보장
④ 학습 속도를 항상 최대화

**정답**: ① — Trainer는 forward pass, backward pass, 옵티마이저 스텝, 평가 루프 등을 자동으로 처리하여 개발자가 고수준의 설정(learning rate, epochs, batch size 등)만 지정하면 된다.

---

**문제 9-3**
다음 코드의 실행 결과를 예측하시오:

```python
from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification
from datasets import load_dataset

model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    evaluation_strategy="epoch",
    save_strategy="epoch"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=None,  # 학습 데이터 미지정
    eval_dataset=None    # 평가 데이터 미지정
)

trainer.train()
```

① 에러 없이 3 에포크 학습 완료
② `ValueError: Training data is required` 에러 발생
③ 자동으로 기본 데이터셋 로드하여 학습 진행
④ `per_device_eval_batch_size`만 사용되어 학습 스킵

**정답**: ② — Trainer는 train_dataset이 필수이다. None을 전달하면 에러가 발생한다.

---

**문제 9-4**
Full Fine-tuning 시 메모리 부족 오류를 경험했다. 다음 중 가장 효과적인 해결 방법은?

① 더 강력한 GPU 구매
② Batch size를 줄이고 gradient accumulation 사용
③ 모델을 더 간단한 것으로 변경
④ 학습 에포크를 줄이기

**정답**: ② — batch size를 8에서 4로 줄이고 `gradient_accumulation_steps=2`를 설정하면, 메모리 사용은 절반 이상 줄이면서 같은 유효 배치 크기(effective batch size = 8)를 유지할 수 있다.

---

**문제 9-5**
다음 `TrainingArguments` 설정 중, 모델이 **과적합되기 쉬운** 조건은?

```python
# ①
TrainingArguments(
    learning_rate=5e-5,
    num_train_epochs=1,
    weight_decay=0.01,  # L2 정규화
    warmup_steps=100
)

# ②
TrainingArguments(
    learning_rate=1e-3,  # 매우 높음
    num_train_epochs=10,
    weight_decay=0.0,    # 정규화 없음
    warmup_steps=0
)

# ③
TrainingArguments(
    learning_rate=5e-5,
    num_train_epochs=5,
    weight_decay=0.01,
    warmup_steps=50
)
```

①② ②③ ②만 ①③

**정답**: ②만 — 높은 학습율 + 많은 에포크 + 정규화 없음 + 준비 단계 없음 = 과적합 위험이 매우 높다.

---

**문제 9-6**
BERT를 감정 분류 작업에 파인튜닝할 때, 적절한 학습율 범위는?

① 0.01 ~ 0.1 (매우 높음)
② 1e-4 ~ 1e-3 (적절함)
③ 1e-6 ~ 1e-5 (매우 낮음)
④ 학습율은 상관없고 배치 크기만 중요

**정답**: ② — BERT 같은 사전 학습 모델 파인튜닝에서는 보통 1e-5 ~ 5e-5가 권장된다. 너무 높으면 사전 학습된 지식이 파괴되고, 너무 낮으면 학습이 지나치게 느리다.

---

**문제 9-7**
Trainer의 `save_strategy="epoch"`과 `eval_strategy="epoch"`을 모두 설정했을 때의 동작은?

① 매 에포크마다 평가를 수행하고 결과를 저장
② 매 에포크마다 평가를 수행하고, 각 에포크 후 모델 체크포인트 저장
③ 평가만 수행하고 모델은 저장하지 않음
④ 모델만 저장하고 평가는 수행하지 않음

**정답**: ② — 두 설정이 모두 "epoch"이면, Trainer는 매 에포크마다 validation set에 대해 평가를 수행하고, 동시에 그 시점의 모델 상태를 체크포인트로 저장한다.

---

### 10주차: PEFT, LoRA, QLoRA

**문제 10-1**
PEFT(Parameter-Efficient Fine-Tuning)의 핵심 아이디어를 가장 잘 설명한 것은?

① 사전 학습 모델의 모든 파라미터를 업데이트하되 빠르게
② 모든 파라미터를 고정하고 매우 작은 수의 **추가 파라미터**만 학습
③ 모델의 정확도를 희생하고 속도만 높이기
④ 더 작은 모델을 사용하기

**정답**: ② — PEFT는 사전 학습 모델을 동결하고, LoRA 같은 저랭크 어댑터를 추가하여 전체 파라미터의 1~5%만 학습한다.

---

**문제 10-2**
LoRA(Low-Rank Adaptation)에서 두 개의 저랭크 행렬 A와 B의 곱(A × B)이 의미하는 바는?

① 원본 가중치의 대체 행렬
② 원본 가중치에 더하는 **작은 보정값**(어댑터)
③ 학습 불안정을 막기 위한 정규화
④ 원본 가중치와 동일한 크기의 새 행렬

**정답**: ② — LoRA에서 W_new = W_original + ΔW (= A × B)이다. 이 보정값(A × B)은 원본 가중치보다 훨씬 작은 저랭크 행렬이므로, 메모리와 계산량을 크게 줄일 수 있다.

---

**문제 10-3**
다음 코드에서 LoRA의 랭크(r)를 16에서 8로 줄이면 어떤 변화가 생기는가?

```python
from peft import get_peft_model, LoraConfig

lora_config = LoraConfig(
    r=16,  # 랭크
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05
)

model = get_peft_model(model, lora_config)
```

① 학습 파라미터 수 증가, 메모리 사용량 감소
② 학습 파라미터 수 감소, 메모리 사용량 감소 (표현 능력도 소폭 감소)
③ 학습 파라미터 수 증가, 메모리 사용량 증가
④ 모델의 크기가 증가

**정답**: ② — 랭크를 줄이면 A와 B의 크기가 작아져 학습해야 할 파라미터가 줄어든다. 대신 표현 능력이 제한될 수 있으므로 작은 데이터셋에서는 r=8, 큰 데이터셋에서는 r=16 이상이 권장된다.

---

**문제 10-4**
QLoRA의 특징을 설명한 문장 중 **틀린 것**은?

① 4비트 양자화(4-bit Quantization)를 사용하여 메모리를 극적으로 절감
② LoRA 어댑터는 32비트 또는 16비트로 유지하여 학습 안정성 확보
③ 70B 이상의 대형 모델도 단일 GPU에서 파인튜닝 가능
④ 정확도는 Full Fine-tuning과 동일하게 유지됨

**정답**: ④ — QLoRA는 Full Fine-tuning보다 메모리 효율이 좋지만, **정확도는 약간 낮아질 수 있다**. 양자화로 인한 정보 손실이 있기 때문이다.

---

**문제 10-5**
LoRA와 Full Fine-tuning을 같은 조건(데이터, 하이퍼파라미터, 학습시간)에서 비교했을 때, 일반적으로 기대할 수 있는 결과는?

① LoRA가 Full Fine-tuning보다 항상 더 높은 정확도
② Full Fine-tuning이 더 높은 정확도를 보일 가능성이 높음 (더 많은 파라미터 업데이트)
③ 두 방법의 정확도가 거의 같음
④ 정확도는 모델 크기에만 의존

**정답**: ② — Full Fine-tuning은 모든 파라미터를 업데이트하므로 더 높은 학습 용량을 가진다. 다만 LoRA는 메모리 효율과 계산 속도에서 우수하므로, 리소스 제약이 있을 때는 LoRA가 실용적인 선택이다.

---

**문제 10-6**
다음 중 LoRA의 `target_modules`에서 BERT 모델의 경우 **반드시 포함해야 하는** 모듈은?

```python
lora_config = LoraConfig(
    r=8,
    target_modules=[???]
)
```

① 모든 레이어의 모든 모듈
② `"query"`, `"value"` 프로젝션
③ `"q_proj"`, `"v_proj"` 또는 `"query"`, `"value"`
④ `"output"` 프로젝션만

**정답**: ③ — BERT의 Self-Attention 모듈에서 Q, K, V 프로젝션이 가장 효과적인 어댑터 위치이다. `q_proj`, `v_proj`만 적용해도 충분한 경우가 많다. (K 프로젝션은 때에 따라 생략 가능)

---

**문제 10-7**
QLoRA 학습 중 메모리 부족 오류가 발생했다. 다음 중 **가장 효과적인** 해결 방법은?

① batch size 감소
② `load_in_4bit=True`에서 `load_in_8bit=True`로 변경
③ LoRA 랭크(r) 감소
④ 모델을 더 작은 것으로 변경

**정답**: ③ — LoRA 랭크를 줄이는 것이 가장 직접적이다. r=8 → r=4로 줄이면 LoRA 파라미터가 절반이 되어 메모리를 많이 절감할 수 있다. (①도 도움이 되지만 ③이 더 근본적)

---

**문제 10-8**
LoRA 어댑터를 저장하고 나중에 로드하는 코드는?

```python
# 저장
# ①
lora_model.save_pretrained("./my_lora")

# ②
model.save_pretrained("./my_lora")

# ③
torch.save(lora_model.state_dict(), "./my_lora.pt")

# 로드 (로드 코드 별도)
from peft import AutoPeftModelForSequenceClassification
model = AutoPeftModelForSequenceClassification.from_pretrained("./my_lora")
```

①만 가능 ②만 가능 ①③ 모두 가능 ①③ 모두 불가능

**정답**: ①만 가능 (또는 ①③ 모두 가능, 버전 및 설정에 따라 다름) — PEFT 패키지의 `save_pretrained()`가 표준이며, ③의 `torch.save()`도 동작하지만 로드할 때 추가 처리가 필요할 수 있다.

---

### 11주차: RAG 시스템 아키텍처

**문제 11-1**
RAG(Retrieval-Augmented Generation)의 핵심 아이디어는?

① 모든 답을 외우는 대신, 필요할 때 **문서에서 찾아 답하기**
② LLM을 더 크게 만들기
③ 검색 엔진과 언어 모델을 순차적으로 연결
④ 사용자 입력을 여러 번 재해석하기

**정답**: ① — RAG는 "오픈북 시험"처럼 작동한다. 모델이 모든 정보를 외우지 않아도, 검색기에서 관련 문서를 먼저 가져온 뒤 그를 바탕으로 답을 생성한다.

---

**문제 11-2**
RAG 파이프라인의 **순서**로 옳은 것은?

① 문서 검색 → 임베딩 → LLM 생성
② 쿼리 임베딩 → 문서 검색 → LLM으로 답 생성
③ LLM으로 답 생성 → 문서 검색 → 다시 생성
④ 문서 전처리 → 검색 → 생성

**정답**: ② — 사용자 쿼리를 임베딩 모델로 벡터화 → 유사도 검색으로 관련 문서 추출 → LLM에 "문서: ... / 질문: ..."을 입력하여 답 생성

---

**문제 11-3**
Vector Database의 역할로 **가장 부정확한** 것은?

① 문서를 임베딩 벡터로 변환하여 저장
② 쿼리 벡터와의 유사도 검색을 빠르게 수행
③ 모든 문서의 원본 텍스트를 저장하고 관리
④ 대규모 문서 컬렉션에서 고속 근사 최근린 검색(ANN) 제공

**정답**: ③ — Vector DB는 주로 임베딩 벡터를 관리하고, 원본 텍스트는 별도의 메타데이터로 참조만 저장한다. 전체 텍스트 저장은 비효율적이므로 문서 ID와 텍스트를 매핑하는 방식이 일반적이다.

---

**문제 11-4**
다음 중 RAG에서 검색(Retrieval) 단계의 문제로 **가장 심각한** 것은?

① 검색된 문서에 답이 없어도 LLM이 답을 만들어냄 (할루시네이션)
② 관련 문서를 찾지 못하고 무관한 문서를 반환
③ 검색 속도가 느림
④ 문서가 너무 많아서 저장 공간 부족

**정답**: ② — 검색 단계가 실패하면 아무리 좋은 LLM도 도움이 될 수 없다. 이를 "Garbage In, Garbage Out"이라 부른다. ①의 할루시네이션은 생성 단계의 문제로, 상대적으로 덜 심각할 수 있다.

---

**문제 11-5**
RAG 시스템에서 임베딩 모델(예: Sentence Transformers)을 변경할 때 고려해야 할 사항은?

① 기존 벡터 DB의 모든 임베딩을 다시 계산해야 함
② 새 모델로 쿼리도 동일한 방식으로 임베딩해야 함
③ 임베딩 차원이 달라지면 호환되지 않을 수 있음
④ 위의 모든 것

**정답**: ④ — 임베딩 모델을 바꾸면 기존 벡터는 무의미해지므로 재구성이 필수이고, 쿼리와 문서 인코더가 일관되어야 하며, 차원 변화도 데이터베이스 스키마 변경을 필요로 한다.

---

**문제 11-6**
다음 코드에서 RAG의 **검색** 단계 결과를 분석하려면?

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings

documents = [doc1, doc2, doc3]  # 원본 문서
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(documents, embeddings)

# 쿼리
results = vectorstore.similarity_search("What is machine learning?", k=3)
```

`results`에는 무엇이 담기는가?

① 3개의 임베딩 벡터
② 3개의 문서 객체 (원본 텍스트 포함)
③ 3개의 유사도 점수만
④ 3개의 쿼리 벡터

**정답**: ② — `similarity_search()`는 유사도가 높은 상위 k개의 문서 객체를 반환한다. 원본 텍스트, 메타데이터, 유사도 정보 등이 포함된다.

---

**문제 11-7**
RAG 시스템에서 "컨텍스트 윈도우 초과" 에러를 피하려면?

① 검색 문서 수(k)를 줄이기
② 각 문서를 더 짧은 청크로 분할
③ LLM의 max_tokens를 줄이기
④ 위의 모든 것이 도움이 될 수 있음

**정답**: ④ — k를 3에서 1로 줄이거나, chunk_size를 1000에서 500으로 줄이거나, max_tokens를 조정하는 모든 방법이 토큰 수를 줄일 수 있다.

---

### 12주차: AI Agent와 LangGraph

**문제 12-1**
AI Agent의 정의로 가장 옳은 것은?

① 자율적으로 목표를 세우고 실행하는 엔티티
② 사용자 지시에 따라 **순차적 또는 반복적으로 행동하며** 작업을 완수하는 시스템
③ LLM에 더 많은 매개변수를 추가한 것
④ 모든 프롬프트 기반 시스템

**정답**: ② — Agent는 단순한 패턴 매칭이 아니라, 관찰(Observe) → 생각(Think) → 행동(Act) → 반복 사이클을 통해 복잡한 작업을 해결한다.

---

**문제 12-2**
ReAct(Reasoning + Acting) 패턴에서 "Thought" 단계의 역할은?

① 다음 액션을 실행하기
② 지금까지의 상태를 분석하고 **다음 액션을 결정** 하기
③ 사용자와 대화하기
④ 외부 도구 API를 호출하기

**정답**: ② — ReAct는 Thought → Action → Observation → Thought (반복) 사이클을 따른다. Thought 단계에서 LLM이 논리적으로 다음 스텝을 계획한다.

---

**문제 12-3**
LangGraph의 **Graph**와 **Node**의 관계는?

① Graph = 여러 Node와 Edge로 이루어진 **작업 흐름**
② Node = 단일 작업 (함수 호출, LLM 쿼리 등)
③ Graph = 원형, Node = 정점
④ ①과 ②가 모두 맞음

**정답**: ④ — LangGraph에서 각 Node는 독립적인 작업을 수행하고, Graph는 이 Node들을 특정 순서와 조건에 따라 연결하여 복잡한 워크플로우를 구성한다.

---

**문제 12-4**
다음 LangGraph 코드의 실행 흐름을 예측하시오:

```python
from langgraph.graph import Graph

graph = Graph()
graph.add_node("A", func_a)
graph.add_node("B", func_b)
graph.add_edge("A", "B")
graph.set_start_node("A")

result = graph.invoke({"input": "test"})
```

① Node A만 실행되고 B는 실행 안 됨
② A 실행 → B 실행 → 종료 (순차 실행)
③ A와 B가 동시에 실행
④ B가 실행된 후 A가 실행

**정답**: ② — start node가 A이고, A→B의 엣지가 있으므로 A 실행 후 결과가 B로 전달되고, B가 실행된다.

---

**문제 12-5**
Agent 설계 시 **무한 루프를 방지**하기 위한 가장 일반적인 방법은?

① Agent를 사용하지 않기
② 최대 반복 횟수(max_iterations) 설정
③ 특정 조건에서 루프를 종료하는 **조건부 엣지** 추가
④ ②와 ③이 모두 권장됨

**정답**: ④ — 반복 횟수 제한과 종료 조건(예: 목표 달성 시)을 함께 설정하는 것이 안전하다.

---

**문제 12-6**
Tool 호출 기반 Agent에서, LLM이 사용할 수 있는 도구로 등록해야 할 정보는?

① 도구 이름 (name)
② 도구의 기능 설명 (description)
③ 입력 파라미터 정의 (parameter schema)
④ 위의 모든 것

**정답**: ④ — LLM이 어떤 도구를 언제 사용할지 판단할 수 있으려면, 도구의 이름, 설명, 입력 형식이 모두 필요하다.

---

**문제 12-7**
다음 중 LangGraph의 **조건부 분기**를 구현하는 올바른 방법은?

```python
# ①
graph.add_edge("A", "B")
graph.add_edge("A", "C")

# ②
graph.add_conditional_edges("A", decide_next_node)
# decide_next_node는 "B" 또는 "C"를 반환

# ③
if some_condition:
    graph.add_edge("A", "B")
else:
    graph.add_edge("A", "C")

# ④
graph.add_decision_node("A", [("B", condition1), ("C", condition2)])
```

①② ②③ ②만 ①②③

**정답**: ②만 (또는 ②③) — `add_conditional_edges()`를 사용하여 런타임에 조건에 따라 다음 노드를 결정하는 것이 LangGraph의 표준이다. ③도 동작하지만, 동적 조건을 표현하기에는 부족하다.

---

### 13주차: FastAPI와 배포 최적화

**문제 13-1**
FastAPI의 **가장 큰 장점**으로 옳은 것은?

① 가장 빠른 프레임워크
② Python 타입 힌트 기반 자동 문서화 (Swagger, ReDoc)
③ Django보다 더 강력한 ORM
④ 가장 많은 사용자

**정답**: ② — FastAPI는 타입 힌트로부터 요청/응답 스키마를 자동으로 생성하여 Swagger UI를 제공한다.

---

**문제 13-2**
다음 FastAPI 엔드포인트의 역할은?

```python
from fastapi import FastAPI

app = FastAPI()

@app.post("/predict")
async def predict(text: str):
    return {"text": text, "label": "positive"}
```

① GET 요청으로 예측 결과 반환
② POST 요청으로 텍스트를 받아 **감정 예측 결과 반환**
③ 텍스트를 데이터베이스에 저장
④ 모든 POST 요청을 거부

**정답**: ② — `@app.post("/predict")`는 POST 메서드를 정의하고, text 파라미터를 받아 예측 결과를 딕셔너리로 반환한다.

---

**문제 13-3**
FastAPI에서 **배경 작업(Background Task)**을 사용하는 이유는?

① HTTP 응답 속도 향상 (시간 소요 작업을 비동기로 처리)
② 모든 작업을 더 빠르게 만들기
③ 데이터베이스 쿼리 최적화
④ 보안 강화

**정답**: ① — Background Task를 사용하면, 사용자에게 빠르게 응답하고(예: "작업 큐에 추가됨"), 무거운 작업(예: 모델 학습)은 백그라운드에서 처리할 수 있다.

---

**문제 13-4**
Docker 컨테이너의 역할로 **가장 정확한** 것은?

① 애플리케이션과 **모든 의존성을 패키징하여** 어떤 환경에서든 동일하게 실행
② 보안 향상만을 위한 기술
③ 클라우드에서만 사용 가능
④ GPU 가속을 제공

**정답**: ① — Docker는 코드, 라이브러리, 런타임, 환경 변수 등을 모두 이미지에 담아, "내 컴퓨터에서는 되는데..."를 해결한다.

---

**문제 13-5**
다음 Dockerfile의 문제점은?

```dockerfile
FROM python:3.10
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
CMD ["python", "app.py"]
```

① GPU를 사용하지 못함
② 조건문이 없어 실행 불가능
③ 이미지 크기가 불필요하게 클 수 있음 (캐시 레이어 최적화 미부족)
④ 포트를 노출하지 않음

**정답**: ③④ — ③: `pip install` 캐시를 최적화하려면 requirements.txt를 먼저 COPY하고 설치한 후 소스 코드를 COPY해야 한다. ④: `EXPOSE 8000`을 추가하면 명확성이 높아진다.

---

**문제 13-6**
NLP 모델 배포 시 **가장 중요한 성능 지표(metric)**는?

① 응답 시간(Latency)
② 처리량(Throughput)
③ 메모리 사용량(Memory)
④ 위의 모든 것을 균형 있게 고려

**정답**: ④ — 이상적 배포는 빠른 응답(낮은 latency), 많은 요청 처리(높은 throughput), 합리적 리소스 사용을 모두 만족해야 한다.

---

**문제 13-7**
모델 양자화(Quantization)를 적용했을 때 기대 효과는?

① 모델 크기 감소 + 추론 속도 향상 (메모리 footprint 감소)
② 정확도 향상
③ 학습 속도 가속
④ 배포 난이도 감소

**정답**: ① — 양자화는 32비트 부동소수를 8비트 정수로 변환하여 모델 크기를 1/4로 줄이고, 추론을 가속화한다. 정확도는 약간 저하될 수 있다.

---

### 추가 통합 문제 (선택 심화)

**문제 14-1 (통합)**
8~13주차 전체 기술을 활용한 최종 NLP 시스템을 설계한다면, **기술 선택의 올바른 근거**는?

① 토픽 모델링(8주) → 문서 분류 → LoRA 파인튜닝(10주) → RAG 시스템(11주) → Agent로 자동화(12주) → FastAPI로 배포(13주)
② Full Fine-tuning(9주) → BERTopic(8주) → 배포 → Agent 추가
③ 9주, 10주, 11주, 12주, 13주만 사용
④ 모든 기술을 동시에 사용

**정답**: ① — 전형적인 NLP 프로젝트 파이프라인이다. 문서 이해 → 모델 개선 → 지식 통합 → 자동화 → 서빙 순으로 진행한다.

---

**문제 14-2 (통합)**
메모리 제약 환경(단일 GPU, 4GB VRAM)에서 BERT 모델을 활용해야 한다면?

① Full Fine-tuning 사용
② LoRA + QLoRA 사용
③ 전혀 사용 불가
④ 토픽 모델링만 사용

**정답**: ② — LoRA로 학습 파라미터를 줄이고, QLoRA로 4비트 양자화를 적용하면 단일 GPU에서도 가능하다.

---

## 정답 및 해설

### 정답표

| 문제 | 정답 | 주차 |
|:---:|:---:|:---:|
| 8-1 | ② | 토픽 모델링 |
| 8-2 | ② | BERTopic |
| 8-3 | ① | BERTopic 클러스터링 |
| 8-4 | ② | 토픽 해석 |
| 8-5 | ② | 토픽 예측 |
| 8-6 | ③ | 임베딩 모델 |
| 8-7 | ② | 평가 지표 |
| 9-1 | ② | Fine-tuning 정의 |
| 9-2 | ① | Trainer 장점 |
| 9-3 | ② | Trainer 학습 |
| 9-4 | ② | 메모리 최적화 |
| 9-5 | ②만 | 과적합 방지 |
| 9-6 | ② | 학습율 범위 |
| 9-7 | ② | 평가 및 저장 |
| 10-1 | ② | PEFT 정의 |
| 10-2 | ② | LoRA 원리 |
| 10-3 | ② | 랭크 감소 영향 |
| 10-4 | ④ | QLoRA 정확도 |
| 10-5 | ② | LoRA vs Full FT |
| 10-6 | ③ | Target modules |
| 10-7 | ③ | 메모리 해결 |
| 10-8 | ①만 가능 | LoRA 저장 |
| 11-1 | ① | RAG 정의 |
| 11-2 | ② | RAG 파이프라인 |
| 11-3 | ③ | Vector DB 역할 |
| 11-4 | ② | 검색 문제 |
| 11-5 | ④ | 임베딩 변경 |
| 11-6 | ② | 검색 결과 |
| 11-7 | ④ | 컨텍스트 윈도우 |
| 12-1 | ② | Agent 정의 |
| 12-2 | ② | ReAct 패턴 |
| 12-3 | ④ | LangGraph 구조 |
| 12-4 | ② | 순차 실행 |
| 12-5 | ④ | 무한 루프 방지 |
| 12-6 | ④ | Tool 등록 정보 |
| 12-7 | ②만 | 조건부 분기 |
| 13-1 | ② | FastAPI 장점 |
| 13-2 | ② | POST 엔드포인트 |
| 13-3 | ① | Background Task |
| 13-4 | ① | Docker 역할 |
| 13-5 | ③④ | Dockerfile 최적화 |
| 13-6 | ④ | 배포 성능 지표 |
| 13-7 | ① | 양자화 효과 |

---

## 종강 안내

### 학기 성과 요약

이 수업을 통해 여러분은 다음을 경험했습니다:

**1단계: 기초 다지기 (1~4주)**
- AI, 딥러닝, NLP의 기본 원리 이해
- PyTorch로 신경망 구현 및 Transformer 아키텍처 학습

**2단계: 사전 학습 모델 활용 (5~7주)**
- BERT, GPT 같은 실제 LLM 활용 및 파인튜닝
- 프롬프트 엔지니어링 기초

**3단계: 실무 기술 습득 (8~13주)**
- 토픽 분석, 효율적 학습(LoRA), 지식 통합(RAG)
- AI Agent로 자동화, FastAPI로 배포

**4단계: 개별 역량 검증 (14~15주)**
- 개인 프로젝트로 처음부터 끝까지 시스템 구축 경험
- 기말고사로 개념적 이해도 검증

### 다음 단계 (추천 학습 방향)

1. **심화 학습**
   - Multimodal 모델 (Vision-Language 통합)
   - 강화 학습으로 LLM 정렬 (RLHF)
   - 대규모 모델 배포 최적화 (vLLM, TensorRT)

2. **실무 프로젝트**
   - 팀 기반 NLP 프로젝트 (스타트업 인턴십 수준)
   - 오픈소스 기여 (Hugging Face, PyTorch)

3. **산업 자격증**
   - AWS Certified Machine Learning
   - Google Cloud Professional ML Engineer

### 교수자 연락처

- **메일**: [교수 이메일]
- **오피스 아워**: [시간]
- **조교**: [조교명] ([조교 이메일])

### 설문 및 피드백

강의 평가 설문에 성실히 응해주시기 바랍니다:
- 링크: [Google Form 또는 학사관리시스템 링크]
- 마감: [마감일]
- 여러분의 소중한 의견은 더 나은 강의를 만드는 원동력입니다

### 종강 축사

"ChatGPT가 등장한 지 1년 반이 지난 지금, AI는 더 이상 먼 미래의 기술이 아닙니다. 이 수업에서 여러분이 배운 Transformer, Fine-tuning, RAG, Agent는 모두 현재 진행 중인 산업 혁신의 기반입니다.

여러분이 배운 기술과 사고 방식을 갖춘다면, AI 시대의 프론티어가 될 수 있습니다. 자신의 분야에서 '자동화할 수 있는 것'과 '인간만이 할 수 있는 것'을 구분하고, 전자에 AI를 활용하는 엔지니어가 되어주시기 바랍니다.

여름 방학 동안 개인 프로젝트를 계속 발전시키고, 필요하면 메일로 언제든 질문해주세요. 여러분의 성장을 응원합니다. 고생 많았습니다."

---

## 종강 체크리스트

- [ ] 기말고사 응시
- [ ] 개인 프로젝트 최종 제출 (Google Classroom)
- [ ] 강의 평가 설문 응답
- [ ] 성적 확인 (강의평가 후 2주 내)
- [ ] 오픈 오피스 아워: [이후 특정 날짜/시간]

---

**2026학년도 1학기 종강**

"감사합니다. 다음 학기에 또 만나요!"
