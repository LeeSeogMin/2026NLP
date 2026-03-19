## 5주차 B회차: BERT/GPT 활용 실습

> **미션**: BERT 기반 개체명 인식 모델과 GPT-2 텍스트 생성 모델을 직접 구현하고, 다양한 디코딩 전략을 비교하며 두 모델의 강점과 한계를 실무적으로 이해한다

### 수업 타임라인

| 시간 | 내용 | Copilot 사용 |
|------|------|-------------|
| 00:00~00:05 | 조 편성 + 역할 배분 (조원 A/B) | 사용 안 함 |
| 00:05~00:10 | A회차 핵심 리캡 + 과제 스펙 재확인 | 사용 안 함 |
| 00:10~00:55 | 2인1조 Copilot 구현 (체크포인트 3회) | 적극 사용 |
| 00:55~01:00 | Google Classroom 제출 (조별 1부) | 사용 안 함 |
| 01:00~01:20 | 결과 토론 (생성 전략 비교·해석) | 사용 안 함 |
| 01:20~01:28 | 핵심 정리 | 사용 안 함 |
| 01:28~01:30 | 다음 주 예고 | 사용 안 함 |

---

### A회차 핵심 리캡

**사전학습 패러다임**:
- Pre-training은 대규모 레이블 없는 텍스트로 "언어 기초" 학습, Fine-tuning은 소규모 레이블 데이터로 특정 과제에 맞춤
- Transfer Learning을 통해 소량 데이터로도 높은 성능을 달성할 수 있다

**BERT의 양방향 이해**:
- Encoder 기반, MLM으로 학습되어 양방향(앞+뒤) 문맥을 동시에 활용
- 빈칸 채우기(다의어 해소, 깊은 이해)에 강점
- WordPiece 토크나이저로 텍스트를 서브워드로 분해

**GPT의 자기회귀 생성**:
- Decoder 기반, 다음 토큰 예측으로 학습되어 왼쪽 문맥만 활용
- 텍스트 생성에 강점, 실제 생성 상황과 일치
- Causal Self-Attention으로 미래 토큰 마스킹

**디코딩 전략의 중요성**:
- Greedy: 최고 확률 토큰만 선택 (빠르지만 반복적)
- Beam Search: K개 경로 동시 추적 (품질 높음)
- Temperature: 분포의 날카로움 조절 (T<1 보수적, T>1 창의적)
- Top-k: 상위 k개 중 샘플링 (다양성과 품질 균형)
- Top-p (Nucleus): 누적확률 p까지 샘플링 (더 유연한 다양성)

**실습 연계**:
- Hugging Face Pipeline과 AutoModel의 차이 이해
- Pipeline으로 빠른 프로토타입, AutoModel로 세밀한 제어
- 이번 실습에서는 BERT NER + GPT-2 생성을 조합하여 실무적 파이프라인 구축

---

### 과제 스펙

**과제**: BERT 기반 개체명 인식(NER) 모델과 GPT-2 텍스트 생성기의 통합 시스템 구현

**제출 형태**: 조별 1부, Google Classroom 업로드

**파일 구성**:
- 구현 코드 파일 (`*.py`)
- 생성 결과 비교 리포트 (Markdown 형식, 1-2페이지)
- 시각화 이미지 (개체 추출 예시, 생성 전략별 결과)

**검증 기준**:
- ✓ BERT Tokenizer 작동 확인 및 임베딩 추출
- ✓ BERT NER 모델로 텍스트에서 Person, Organization, Location 개체 추출
- ✓ 추출된 개체를 프롬프트에 반영하여 GPT-2로 텍스트 생성
- ✓ 세 가지 디코딩 전략(Greedy, Top-p, Temperature) 비교
- ✓ 결과 해석: 각 전략의 강점과 약점 분석

**제출 마감**: 수업 종료 후 24시간 이내

---

### 2인1조 실습

> **Copilot 활용**: BERT NER 기본 코드를 먼저 직접 작성한 뒤, Copilot에게 "이 개체를 추출해서 GPT-2 프롬프트에 넣는 코드를 작성해줄래?"라고 요청한다. GPT-2 생성 시 "다른 디코딩 전략으로 비교하는 코드도 추가해줘"로 단계적으로 확장한다. Copilot의 제안을 검토하고 수정하는 과정에서 Transfer Learning과 생성 전략의 원리를 깊이 있게 이해할 수 있다.

**역할 분담**:
- **조원 A (드라이버)**: 코드 작성 및 실행, 결과 확인, 생성된 텍스트 평가
- **조원 B (네비게이터)**: 로직 검토, Copilot 프롬프트 설계, 오류 해석, 결과 해석
- **체크포인트마다 역할 교대**: 드라이버와 네비게이터를 번갈아가며 진행하여 두 명 모두 전체 구현을 이해한다

---

#### 체크포인트 1: BERT Tokenizer + NER 모델 (15분)

**목표**: Hugging Face의 사전학습된 BERT NER 모델과 Pipeline을 사용하여 텍스트에서 개체명을 추출하고, 토크나이저 동작을 이해한다

**핵심 단계**:

① **BERT Tokenizer 작동 원리 이해**

```python
from transformers import AutoTokenizer

# BERT-Base 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# 문장을 토크나이즈
sentence = "John works at Google in California"
tokens = tokenizer.tokenize(sentence)
print(f"토큰: {tokens}")

# 토큰 ID로 변환
token_ids = tokenizer.convert_tokens_to_ids(tokens)
print(f"토큰 ID: {token_ids}")

# 역변환: ID → 토큰
recovered_tokens = tokenizer.convert_ids_to_tokens(token_ids)
print(f"복원된 토큰: {recovered_tokens}")

# 임베딩 ID로 변환 (CLS, SEP 토큰 자동 추가)
encoded = tokenizer.encode(sentence, return_tensors="pt")
print(f"인코딩 결과 (CLS/SEP 포함): {encoded}")
print(f"형태: {encoded.shape}")
```

예상 결과:
```
토큰: ['john', 'works', 'at', 'google', 'in', 'california']
토큰 ID: [2062, 2573, 1037, 1043, 1999, 1029]
복원된 토큰: ['john', 'works', 'at', 'google', 'in', 'california']
인코딩 결과 (CLS/SEP 포함): tensor([[  101,  2062,  2573,  1037,  1043,  1999,  1029,   102]])
형태: torch.Size([1, 8])

[CLS] 토큰(101)과 [SEP] 토큰(102)이 자동으로 추가됨
```

② **Pipeline을 사용한 NER (개체명 인식)**

```python
from transformers import pipeline

# NER Pipeline 로드
ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")

# 샘플 텍스트
text = "John Smith works at Google in Mountain View. He studied at Stanford University."

# NER 수행
entities = ner_pipeline(text)

print(f"추출된 개체:")
for entity in entities:
    print(f"  단어: {entity['word']}, 태그: {entity['entity']}, 확신도: {entity['score']:.4f}")
```

예상 결과:
```
추출된 개체:
  단어: John, 태그: B-PER, 확신도: 0.9991
  단어: Smith, 태그: I-PER, 확신도: 0.9989
  단어: Google, 태그: B-ORG, 확신도: 0.9987
  단어: Mountain, 태그: B-LOC, 확신도: 0.9925
  단어: View, 태그: I-LOC, 확신도: 0.9897
  단어: Stanford, 태그: B-ORG, 확신도: 0.9967
  단어: University, 태그: I-ORG, 확신도: 0.9873

(B- = Begin, I- = Inside: 개체명의 시작과 연속)
(PER = Person, ORG = Organization, LOC = Location)
```

③ **개체 그룹화 및 정제**

```python
from typing import List, Dict

def extract_entities_grouped(text: str, ner_pipeline) -> Dict[str, List[str]]:
    """개체명을 태그별로 그룹화하는 함수"""
    entities = ner_pipeline(text)

    # 그룹화
    grouped = {"PER": [], "ORG": [], "LOC": []}
    current_entity = ""
    current_type = None

    for token in entities:
        tag = token["entity"]
        word = token["word"]

        # ##로 시작하면 이전 토큰과 합치기 (WordPiece)
        if word.startswith("##"):
            current_entity += word[2:]
        else:
            # 이전 개체 저장
            if current_entity and current_type:
                grouped[current_type].append(current_entity)
            # 새 개체 시작
            current_entity = word
            entity_type = tag.split("-")[1]  # B-PER → PER
            current_type = entity_type

    # 마지막 개체 저장
    if current_entity and current_type:
        grouped[current_type].append(current_entity)

    return grouped

# 사용
text = "John Smith works at Google. Maria Garcia is from Barcelona."
grouped = extract_entities_grouped(text, ner_pipeline)

print("개체별 분류:")
for entity_type, entities in grouped.items():
    if entities:
        print(f"  {entity_type}: {', '.join(entities)}")
```

예상 결과:
```
개체별 분류:
  PER: John Smith, Maria Garcia
  ORG: Google
  LOC: Barcelona
```

**검증 체크리스트**:
- [ ] Tokenizer가 문장을 올바르게 토큰화하는가?
- [ ] CLS, SEP 토큰이 자동으로 추가되는가?
- [ ] NER Pipeline이 개체명을 정확히 추출하는가?
- [ ] B-/I- 태그의 의미를 이해했는가?
- [ ] 개체를 태그별로 올바르게 그룹화했는가?

**Copilot 프롬프트 1**:
```
"BERT 토크나이저로 텍스트를 토큰화하고 ID로 변환하는 코드를 작성해줄래?
AutoTokenizer.from_pretrained('bert-base-uncased')를 사용하고
encode, tokenize, convert_ids_to_tokens를 모두 시연해줘."
```

**Copilot 프롬프트 2**:
```
"Hugging Face의 NER Pipeline으로 '예시 텍스트'에서 Person, Organization, Location을 추출하는 코드를 만들어줘.
결과를 태그별로 그룹화해서 출력해야 해."
```

---

#### 체크포인트 2: GPT-2 텍스트 생성 기본 (20분)

**목표**: GPT-2를 사용하여 프롬프트를 입력했을 때 다양한 디코딩 전략으로 텍스트를 생성하고, 각 전략의 특성을 비교한다

**핵심 단계**:

① **Pipeline을 사용한 기본 생성**

```python
from transformers import pipeline

# GPT-2 Pipeline 로드
generator = pipeline("text-generation", model="gpt2", device=0)

# 간단한 프롬프트
prompt = "John Smith works at Google. He"

# 기본 생성 (Greedy Search)
result = generator(prompt, max_length=30, num_return_sequences=1, do_sample=False)
print(f"Greedy 생성 결과:")
print(f"  {result[0]['generated_text']}")
```

예상 결과:
```
Greedy 생성 결과:
  John Smith works at Google. He is a software engineer and has been working
```

② **디코딩 전략 1: Greedy Search**

```python
# Greedy: 매번 최고 확률 토큰 선택
def generate_greedy(prompt: str, generator, max_length: int = 30) -> str:
    """Greedy Decoding으로 텍스트 생성"""
    result = generator(
        prompt,
        max_length=max_length,
        num_return_sequences=1,
        do_sample=False,  # Greedy 옵션
        temperature=1.0   # 무시됨 (do_sample=False)
    )
    return result[0]["generated_text"]

prompt = "The future of artificial intelligence is"
text_greedy = generate_greedy(prompt, generator)
print(f"Greedy:\n  {text_greedy}\n")
```

예상 결과:
```
Greedy:
  The future of artificial intelligence is very bright. We will see many new
```

③ **디코딩 전략 2: Top-p (Nucleus) Sampling**

```python
def generate_top_p(prompt: str, generator, p: float = 0.9, max_length: int = 30) -> str:
    """Top-p Sampling으로 텍스트 생성"""
    result = generator(
        prompt,
        max_length=max_length,
        num_return_sequences=1,
        do_sample=True,     # 샘플링 활성화
        top_p=p,            # 누적확률 p까지만 고려
        temperature=1.0     # 표준 확률 분포
    )
    return result[0]["generated_text"]

prompt = "The future of artificial intelligence is"
text_top_p = generate_top_p(prompt, generator, p=0.9)
print(f"Top-p (p=0.9):\n  {text_top_p}\n")
```

예상 결과:
```
Top-p (p=0.9):
  The future of artificial intelligence is dependent on how we approach the ethical
```

④ **디코딩 전략 3: Temperature Sampling**

```python
def generate_with_temperature(prompt: str, generator, temperature: float = 0.7,
                             max_length: int = 30) -> str:
    """Temperature 조절로 텍스트 생성"""
    result = generator(
        prompt,
        max_length=max_length,
        num_return_sequences=1,
        do_sample=True,
        temperature=temperature,
        top_p=1.0           # Top-p 비활성화
    )
    return result[0]["generated_text"]

prompt = "The future of artificial intelligence is"

# 다양한 temperature로 생성
print("Temperature별 생성 결과:")
for temp in [0.3, 0.7, 1.5]:
    text = generate_with_temperature(prompt, generator, temperature=temp)
    print(f"  T={temp}: {text[:60]}...")
```

예상 결과:
```
Temperature별 생성 결과:
  T=0.3: The future of artificial intelligence is very important and the...
  T=0.7: The future of artificial intelligence is going to be a key part of...
  T=1.5: The future of artificial intelligence is perhaps not just a...
```

⑤ **전략별 비교 및 시각화**

```python
import matplotlib.pyplot as plt
import numpy as np

# 여러 프롬프트에 대해 전략별로 생성
prompts = [
    "John Smith works at Google.",
    "The project deadline is",
    "Natural language processing"
]

strategies = {
    "Greedy": {"do_sample": False},
    "Top-p (0.9)": {"do_sample": True, "top_p": 0.9, "temperature": 1.0},
    "Temperature (0.7)": {"do_sample": True, "top_p": 1.0, "temperature": 0.7},
}

print("디코딩 전략별 비교:")
print("=" * 80)
for prompt in prompts[:1]:  # 첫 번째 프롬프트만
    print(f"\n프롬프트: {prompt}\n")
    for strategy_name, params in strategies.items():
        result = generator(prompt, max_length=25, num_return_sequences=1, **params)
        generated = result[0]["generated_text"][len(prompt):]  # 프롬프트 제거
        print(f"{strategy_name}:")
        print(f"  → {generated}\n")
```

예상 결과:
```
프롬프트: John Smith works at Google.

Greedy:
  → He is a software engineer who has been working

Top-p (0.9):
  → They are very happy with his performance in

Temperature (0.7):
  → He is responsible for building new services
```

**검증 체크리스트**:
- [ ] Greedy Search가 매번 같은 결과를 생성하는가? (결정적)
- [ ] Top-p Sampling이 다양한 결과를 생성하는가?
- [ ] Temperature를 낮출수록 보수적이 되는가? (T=0.3)
- [ ] Temperature를 높일수록 창의적이 되는가? (T=1.5)
- [ ] 모든 전략의 출력이 문법적으로 올바른가?

**Copilot 프롬프트 3**:
```
"GPT-2로 텍스트를 생성하는 코드를 만들어줄래?
Greedy Search, Top-p Sampling, Temperature Sampling 세 가지를 모두 시연해야 해."
```

**Copilot 프롬프트 4**:
```
"같은 프롬프트에 대해 세 가지 디코딩 전략으로 생성한 결과를 비교 출력하는 함수를 만들어줘.
각 전략의 특성이 명확하게 드러나야 해."
```

---

#### 체크포인트 3: BERT NER + GPT-2 통합 (20분)

**목표**: BERT NER로 추출한 개체명을 GPT-2 프롬프트에 포함시켜 이를 바탕으로 텍스트를 생성하는 통합 시스템을 구축하고, 생성 결과를 해석한다

**핵심 단계**:

① **입력 텍스트 → 개체 추출**

```python
from transformers import pipeline

# BERT NER + GPT-2 Pipeline 로드
ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")
generator = pipeline("text-generation", model="gpt2", device=0)

# 입력 텍스트
input_text = "Sarah Johnson is the CEO of Apple Inc., based in Cupertino, California."

# 개체 추출
entities = ner_pipeline(input_text)

# 개체 그룹화
def parse_entities(text: str, ner_result) -> dict:
    """NER 결과를 정리하여 태그별 개체명 추출"""
    grouped = {"PERSON": [], "ORG": [], "LOC": []}
    current = {"text": "", "tag": None}

    for token in ner_result:
        word = token["word"]
        tag = token["entity"].split("-")[1]

        if word.startswith("##"):
            current["text"] += word[2:]
        else:
            if current["text"] and current["tag"]:
                tag_map = {"PER": "PERSON", "ORG": "ORG", "LOC": "LOC"}
                grouped[tag_map[current["tag"]]].append(current["text"])
            current = {"text": word, "tag": tag}

    if current["text"] and current["tag"]:
        tag_map = {"PER": "PERSON", "ORG": "ORG", "LOC": "LOC"}
        grouped[tag_map[current["tag"]]].append(current["text"])

    return grouped

parsed = parse_entities(input_text, entities)
print("추출된 개체:")
for entity_type, names in parsed.items():
    if names:
        print(f"  {entity_type}: {', '.join(names)}")
```

예상 결과:
```
추출된 개체:
  PERSON: Sarah Johnson
  ORG: Apple Inc.
  LOC: Cupertino, California
```

② **개체 기반 프롬프트 생성**

```python
def create_prompt_from_entities(parsed_entities: dict) -> str:
    """추출된 개체를 바탕으로 생성 프롬프트 작성"""
    parts = []

    if parsed_entities["PERSON"]:
        parts.append(f"{parsed_entities['PERSON'][0]} works at")
    if parsed_entities["ORG"]:
        parts.append(f"{parsed_entities['ORG'][0]}")
    if parsed_entities["LOC"]:
        parts.append(f"located in {parsed_entities['LOC'][0]}.")

    prompt = " ".join(parts)
    return prompt

prompt = create_prompt_from_entities(parsed)
print(f"생성된 프롬프트: {prompt}")
```

예상 결과:
```
생성된 프롬프트: Sarah Johnson works at Apple Inc. located in Cupertino, California.
```

③ **세 가지 전략으로 생성 결과 비교**

```python
def generate_with_strategy(prompt: str, strategy: str, max_length: int = 50) -> str:
    """지정된 전략으로 텍스트 생성"""
    params = {"max_length": max_length, "num_return_sequences": 1}

    if strategy == "greedy":
        params["do_sample"] = False
    elif strategy == "top_p":
        params["do_sample"] = True
        params["top_p"] = 0.9
        params["temperature"] = 1.0
    elif strategy == "temperature":
        params["do_sample"] = True
        params["top_p"] = 1.0
        params["temperature"] = 0.7

    result = generator(prompt, **params)
    return result[0]["generated_text"]

# 각 전략으로 생성
prompt = create_prompt_from_entities(parsed)
print(f"원본 프롬프트: {prompt}\n")

results = {}
for strategy in ["greedy", "top_p", "temperature"]:
    generated = generate_with_strategy(prompt, strategy)
    results[strategy] = generated
    print(f"[{strategy.upper()}]")
    print(f"  {generated}\n")
```

예상 결과:
```
원본 프롬프트: Sarah Johnson works at Apple Inc. located in Cupertino, California.

[GREEDY]
  Sarah Johnson works at Apple Inc. located in Cupertino, California. She is the
  chief executive officer of the company and has been in the position since

[TOP_P]
  Sarah Johnson works at Apple Inc. located in Cupertino, California. Her
  innovations in consumer electronics have transformed the global market.

[TEMPERATURE]
  Sarah Johnson works at Apple Inc. located in Cupertino, California. The company
  is known for its innovative products and technological advancements.
```

④ **시각화 및 정성적 분석**

```python
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# 전략별 특성 분석
def analyze_text_characteristics(text: str) -> dict:
    """생성된 텍스트의 특성 분석"""
    words = text.split()
    return {
        "length": len(words),
        "avg_word_length": sum(len(w) for w in words) / len(words),
        "unique_words": len(set(w.lower() for w in words)),
    }

# 분석 결과
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
strategies_list = ["greedy", "top_p", "temperature"]

for idx, strategy in enumerate(strategies_list):
    text = results[strategy]
    analysis = analyze_text_characteristics(text)

    ax = axes[idx]
    ax.text(0.5, 0.7, strategy.upper(), ha="center", fontsize=12, fontweight="bold")
    ax.text(0.05, 0.5, f"총 단어: {analysis['length']}", fontsize=10)
    ax.text(0.05, 0.4, f"평균 단어길이: {analysis['avg_word_length']:.2f}", fontsize=10)
    ax.text(0.05, 0.3, f"고유 단어: {analysis['unique_words']}", fontsize=10)
    ax.text(0.05, 0.1, f"생성: {text[len(prompt):50]}...",
           fontsize=8, style="italic", wrap=True)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

plt.suptitle("디코딩 전략별 생성 텍스트 특성 비교", fontsize=13)
plt.tight_layout()
plt.savefig("generation_comparison.png", dpi=150, bbox_inches="tight")
print("저장: generation_comparison.png")
plt.close()
```

⑤ **질적 해석 작성**

```python
interpretation = """
**생성 결과 해석**:

[Greedy Search]
- 특징: 매우 결정적이며 매번 같은 결과 생성
- 장점: 신뢰도 높고 예측 가능
- 단점: 반복적이고 단조로운 텍스트 생성
- 평가: "She is the chief executive officer"는 자연스럽지만 너무 형식적

[Top-p Sampling (p=0.9)]
- 특징: 누적확률 90%의 상위 토큰만 고려하여 샘플링
- 장점: 충분히 다양하면서도 극단적 선택 회피
- 단점: 같은 프롬프트도 매번 다른 결과
- 평가: "innovations in consumer electronics"는 매우 자연스럽고 관련성 높음

[Temperature (T=0.7)]
- 특징: 분포를 완화하여 균등하게 샘플링
- 장점: 창의성과 일관성의 좋은 균형
- 단점: 일부 부자연스러운 조합 가능
- 평가: "known for its innovative products"는 자연스럽고 일반적이지만 다소 뻔함

**결론**: NER 기반 생성에서는 Top-p가 가장 균형잡힌 결과를 제공한다.
개체명이 정확하므로 이를 보존하면서 나머지 문맥은 다양하게 생성할 수 있기 때문이다.
"""

print(interpretation)
```

**검증 체크리스트**:
- [ ] NER로 개체명이 정확히 추출되었는가?
- [ ] 추출된 개체가 프롬프트에 올바르게 포함되었는가?
- [ ] 세 가지 전략으로 생성한 결과가 모두 문법적으로 올바른가?
- [ ] 각 전략의 특성 차이(일관성 vs 다양성)가 명확한가?
- [ ] 생성 결과를 정성적으로 분석했는가?

**Copilot 프롬프트 5**:
```
"BERT NER로 추출한 개체명을 GPT-2 프롬프트에 포함시키는 코드를 만들어줄래?
Person, Organization, Location을 별도로 처리해서 프롬프트를 동적으로 생성해야 해."
```

**Copilot 프롬프트 6**:
```
"위에서 만든 프롬프트를 Greedy, Top-p, Temperature 세 가지 전략으로 생성하고
결과를 나란히 비교하는 코드를 작성해줄래?"
```

**선택 프롬프트**:
```
"생성 결과의 길이, 다양성, 자연스러움을 시각화하는 함수를 만들어줄래?
matplotlib으로 전략별 특성을 비교할 수 있게."
```

---

### 제출 안내 (Google Classroom)

**제출 방법**:
- Google Classroom의 "5주차 B회차" 과제에 조별 1부 제출
- 파일명 형식: `group_{조번호}_ch5B.zip`

**포함할 파일**:
```
group_{조번호}_ch5B/
├── ch5B_bert_gpt.py                    # 전체 구현 코드
├── ner_entities.txt                     # 추출된 개체명 예시
├── generation_comparison.png            # 생성 전략별 비교 시각화
├── generation_samples.txt               # 각 전략의 생성 예시 (최소 3개)
└── analysis_report.md                   # 분석 리포트 (1-2페이지)
```

**리포트 포함 항목** (analysis_report.md):

```markdown
# 5주차 B회차 실습 보고서

## 1. 실습 과정 (3-4문장)
- 각 체크포인트에서 무엇을 배웠는가?
- 특히 어려웠던 부분은 무엇인가?

## 2. BERT NER 결과 분석 (3-4문장)
- 추출된 개체명이 정확한가?
- 어떤 개체는 잘 인식되고 어떤 개체는 못 인식되었는가?
- 왜 그렇게 생각하는가?

## 3. 생성 전략별 비교 (각 2-3문장)

### Greedy Search
- 생성 특성:
- 강점:
- 약점:

### Top-p Sampling
- 생성 특성:
- 강점:
- 약점:

### Temperature Sampling
- 생성 특성:
- 강점:
- 약점:

## 4. 통합 시스템의 실무적 의미 (3-4문장)
- NER + 생성을 조합하는 것의 장점은?
- 어떤 응용 분야에서 사용될 수 있을까?
- 개선할 점은?

## 5. Copilot 사용 경험 (2문장)
- 어떤 프롬프트가 가장 효과적이었는가?
- 어떤 부분에서 모델을 수정해야 했는가?
```

**제출 마감**: 수업 종료 후 24시간 이내

---

### 결과 토론 가이드

> **토론 가이드**: 조별 구현 결과를 공유하며, 다른 조의 생성 결과와 비교하고 NER+생성 파이프라인의 실무적 의미를 함께 해석한다

**토론 주제**:

① **개체명 인식의 정확도와 한계**
- 각 조에서 추출한 개체명이 동일한가? 차이가 있다면 왜인가?
- BERT NER이 누락한 개체가 있는가? (예: 암시적 조직명)
- 실무 응용을 위해 어떤 후처리가 필요한가? (예: 중복 제거, 정규화)

② **디코딩 전략 간 차이의 의미**
- Greedy와 Top-p의 생성 결과에서 가장 큰 차이는?
- Temperature는 Top-p와 어떻게 다른 효과를 내는가?
- 각 전략이 적합한 응용 분야는? (번역 vs 창작 vs 대화)

③ **NER + 생성 통합의 실무적 가치**
- "개체 추출 → 프롬프트 구성 → 텍스트 생성"의 파이프라인이 어떤 문제를 해결하는가?
- 예: 뉴스 자동 작성, 질의응답, 대화 시스템 등
- 개체명을 프롬프트에 포함시키면 생성 품질이 개선되는가?

④ **모델의 한계와 개선 방향**
- BERT NER이 실패한 케이스는? (예: 중첩 개체, 약한 신호)
- GPT-2가 생성한 텍스트에서 부자연스러운 부분은?
- 더 나은 모델이나 전략이 있을까? (예: ELECTRA, T5, GPT-3)

⑤ **Transfer Learning의 실제 효과**
- 처음부터 학습하는 것 vs 사전학습 모델 사용의 차이는?
- 미세조정 없이 사전학습 모델만으로 충분한가?
- 언제 미세조정이 필요한가?

**발표 형식**:
- 각 조 4~6분 발표 (BERT NER + GPT-2 통합 결과, 주요 인사이트)
- 다른 조의 질문에 답변 (2~3개 질문)
- 교수의 보충 설명 및 피드백

---

### 다음 주 예고

다음 주 6장 A회차에서는 **사전학습 모델의 미세조정(Fine-tuning)**을 깊이 있게 다룬다.

**예고 내용**:
- 미세조정의 원리: 왜 소량의 레이블 데이터로도 높은 성능을 달성하는가?
- 미세조정 vs 프롬프트 엔지니어링: 언제 어느 것을 사용할 것인가?
- 실습: BERT를 감정 분류, NER, 질의응답 세 가지 과제에 미세조정하고 성능 비교
- B회차에서는 실제로 미세조정된 모델을 평가하고, 과적합을 방지하는 기법 학습

**사전 준비**:
- 5장 내용(BERT, GPT, 디코딩 전략)을 다시 읽어두기
- Hugging Face Trainer API 문서 미리 확인하기
- 작은 규모의 레이블 데이터셋 준비하기 (감성분석, 스팸 분류 등)

---

## 참고 자료

**실습 코드**:
- _전체 구현 코드는 practice/chapter5/code/5-2-bert-ner-gpt-generation.py 참고_
- _각 체크포인트별 세부 코드: practice/chapter5/code/5-2-checkpoint-*.py_

**권장 읽기**:
- Hugging Face Course. Fine-tuning a pretrained model. https://huggingface.co/course/chapter3
- Bommasani, R., et al. (2021). On the Opportunities and Risks of Foundation Models. arXiv. https://arxiv.org/abs/2108.07258
- Raffel, C., et al. (2020). Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer. JMLR. https://arxiv.org/abs/1910.10683
- Holtzman, A., Buys, J., Du, L., Forbes, M., & Choi, Y. (2020). The Curious Case of Neural Text Degeneration. ICLR. https://arxiv.org/abs/1904.09751

**모델 상세 정보**:
- BERT Large (NER): https://huggingface.co/dbmdz/bert-large-cased-finetuned-conll03-english
- GPT-2: https://huggingface.co/gpt2
- Hugging Face Model Hub: https://huggingface.co/models
