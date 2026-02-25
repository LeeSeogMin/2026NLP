# 제5장 C: BERT와 GPT 활용 — 모범 구현과 해설

> 이 문서는 B회차 과제 제출 후 공개된다. 제출 전에는 열람하지 않는다.

---

## 개요: BERT와 GPT를 사용하는 이유

이번 장에서 핵심은 다음이다:

- **BERT**: Encoder 기반 양방향 모델 — "빈칸 채우기"에 강점, **이해와 분류**에 최적
- **GPT**: Decoder 기반 자기회귀 모델 — "다음 단어 예측"에 강점, **생성**에 최적
- **Transfer Learning의 강력함**: 사전학습된 모델을 소량 데이터로 미세조정하면 높은 성능을 달성할 수 있다

B회차 과제는 이 둘을 조합하는 실무적 파이프라인을 구축한다: 개체명을 BERT로 추출 → GPT로 기반 텍스트 생성.

---

## 체크포인트 1 모범 구현: BERT Tokenizer + NER 모델

### WordPiece 토크나이저의 작동 원리

BERT는 **WordPiece** 토크나이저를 사용한다. 이는 단어를 더 작은 서브워드 단위로 분해하는 방식이다.

**직관적 이해**: "subword"는 "단어보다 작은 단위"다. 예를 들어 "playing"은 "play" + "##ing"으로 분해된다. "##"는 "이전 토큰의 연속"을 의미하는 표식이다.

```python
from transformers import AutoTokenizer

# BERT-Base 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# 샘플 문장
sentence = "John Smith works at Google in California"

# [1단계] 문장을 토큰으로 분해
tokens = tokenizer.tokenize(sentence)
print(f"토큰: {tokens}")
# 출력: ['john', 'smith', 'works', 'at', 'google', 'in', 'california']

# [2단계] 토큰을 어휘 사전의 ID로 변환
token_ids = tokenizer.convert_tokens_to_ids(tokens)
print(f"토큰 ID: {token_ids}")
# 출력: [2062, 4765, 2573, 1037, 1043, 1999, 1029]

# [3단계] ID를 다시 토큰으로 복원 (검증)
recovered = tokenizer.convert_ids_to_tokens(token_ids)
print(f"복원된 토큰: {recovered}")
# 출력: ['john', 'smith', 'works', 'at', 'google', 'in', 'california']

# [4단계] 인코딩: CLS, SEP 토큰을 자동으로 추가
encoded = tokenizer.encode(sentence, return_tensors="pt")
print(f"인코딩 (CLS/SEP 포함): {encoded}")
# 출력: tensor([[  101,  2062,  4765,  2573,  1037,  1043,  1999,  1029,   102]])
#       CLS(101) ──────────────────────────────────────────────────── SEP(102)
```

**핵심 포인트**:
- `tokenize()`: 문장을 토큰으로 분해 (ID 없음)
- `convert_tokens_to_ids()`: 토큰 → ID
- `encode()`: 한 번에 처리 (CLS, SEP 자동 추가)

### 단어를 분해하는 경우

더 긴 또는 복잡한 단어는 어떻게 분해되는가?

```python
# 긴 단어
text = "playing"
tokens = tokenizer.tokenize(text)
print(f"'playing' 토큰화: {tokens}")
# 출력: ['play', '##ing']
#       "##"는 "이전 토큰의 후속"을 의미

# 여러 언어나 부호 포함
text = "don't have $100"
tokens = tokenizer.tokenize(text)
print(f"특수문자 포함 토큰화: {tokens}")
# 출력: ['do', '##n', "'", 't', 'have', '$', '100']

# 개체명 (대문자)
text = "Microsoft Azure"
tokens = tokenizer.tokenize(text)
print(f"개체명 토큰화: {tokens}")
# 출력: ['microsoft', 'azure']  (BERT는 uncased: 대문자 무시)
```

**주의**: BERT-base-uncased는 모든 문자를 소문자로 변환하므로 대소문자 정보가 손실된다. 대소문자를 구분해야 한다면 `bert-base-cased` 또는 다국어 모델을 사용해야 한다.

### BERT NER 파이프라인

개체명 인식은 **태그 시퀀스 라벨링** 과제다. 각 토큰에 대해 B-PER (Person 시작), I-PER (Person 내부), B-ORG (Organization 시작), O (기타) 같은 태그를 예측한다.

```python
from transformers import pipeline

# 사전학습된 BERT NER 모델 로드
ner_pipeline = pipeline(
    "ner",
    model="dbmdz/bert-large-cased-finetuned-conll03-english"
)

# 샘플 텍스트
text = "John Smith works at Google in Mountain View. He studied at Stanford University."

# NER 수행
entities = ner_pipeline(text)

print("추출된 개체:")
for entity in entities:
    print(f"  단어: '{entity['word']}', 태그: {entity['entity']}, 확신도: {entity['score']:.4f}")

# 예상 출력:
# 추출된 개체:
#   단어: 'John', 태그: B-PER, 확신도: 0.9991
#   단어: 'Smith', 태그: I-PER, 확신도: 0.9989
#   단어: 'Google', 태그: B-ORG, 확신도: 0.9987
#   단어: 'Mountain', 태그: B-LOC, 확신도: 0.9925
#   단어: 'View', 태그: I-LOC, 확신도: 0.9897
#   단어: 'Stanford', 태그: B-ORG, 확신도: 0.9967
#   단어: 'University', 태그: I-ORG, 확신도: 0.9873
```

**태그 설명**:
- `B-` (Begin): 개체명의 시작
- `I-` (Inside): 개체명의 내부 또는 연속
- `O`: 개체명이 아닌 기타 (Other)
- `PER`: Person (사람)
- `ORG`: Organization (조직)
- `LOC`: Location (지역)

### 개체 그룹화 및 정제

Pipeline의 결과는 **토큰 단위**로 제공되므로, 이를 **단어 단위 개체명**으로 그룹화해야 한다.

```python
from typing import Dict, List

def extract_entities_grouped(text: str, ner_pipeline) -> Dict[str, List[str]]:
    """
    BERT NER 결과를 태그별로 그룹화

    Args:
        text: 입력 문장
        ner_pipeline: NER Pipeline 객체

    Returns:
        그룹화된 개체명 사전 {"PER": [...], "ORG": [...], "LOC": [...]}
    """
    entities = ner_pipeline(text)

    # 그룹화 딕셔너리 초기화
    grouped = {"PER": [], "ORG": [], "LOC": []}

    # 현재 처리 중인 개체명
    current_entity = ""
    current_type = None

    for token in entities:
        tag = token["entity"]                    # "B-PER", "I-PER", ...
        word = token["word"]                     # 토큰 문자

        # [핵심] ##으로 시작하면 이전 토큰과 연결
        if word.startswith("##"):
            # 예: "Smith" + "##son" → "Smithson"
            current_entity += word[2:]
        else:
            # 이전 개체 저장
            if current_entity and current_type:
                grouped[current_type].append(current_entity)

            # 새 개체 시작
            current_entity = word
            entity_type = tag.split("-")[1]      # "B-PER" → "PER"
            current_type = entity_type

    # 마지막 개체 저장
    if current_entity and current_type:
        grouped[current_type].append(current_entity)

    return grouped

# 사용 예시
text = "John Smith works at Google. Maria Garcia is from Barcelona."
grouped = extract_entities_grouped(text, ner_pipeline)

print("개체별 분류:")
for entity_type, entities in grouped.items():
    if entities:
        print(f"  {entity_type}: {', '.join(entities)}")

# 예상 출력:
# 개체별 분류:
#   PER: John Smith, Maria Garcia
#   ORG: Google
#   LOC: Barcelona
```

### 핵심 포인트: WordPiece vs BPE vs SentencePiece

BERT가 WordPiece를 사용하는 이유:

| 방식 | 장점 | 단점 | 사용 사례 |
|------|------|------|----------|
| **WordPiece** | 의미 있는 단위로 분해 | 구현 복잡 | BERT, Electra |
| **BPE** | 효율적, 유연함 | 부분 최적화 | GPT-2, GPT-3 |
| **SentencePiece** | 다국어 친화적 | 학습 필요 | ALBERT, XLNet |

```python
# WordPiece의 효과를 확인하는 예시
test_words = [
    "playing",           # play + ##ing
    "internationalize",  # inter + ##national + ##ize
    "unsure",           # un + ##sure
]

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

for word in test_words:
    tokens = tokenizer.tokenize(word)
    print(f"'{word}' → {tokens}")

# 출력:
# 'playing' → ['play', '##ing']
# 'internationalize' → ['inter', '##national', '##ize']
# 'unsure' → ['un', '##sure']
```

BERT는 이렇게 분해된 서브워드를 각각 임베딩하여 처리한다. 이로 인해 희귀 단어나 오류 있는 텍스트(오타)도 어느 정도 처리할 수 있다.

### 흔한 실수

1. **NER 결과를 그룹화하지 않기**
   ```python
   # 틀림 - 토큰 단위 결과를 그대로 사용
   entities = ner_pipeline(text)
   for entity in entities:
       print(entity['word'])  # "John", "Smith" 각각 출력

   # 맞음 - 개체명 그룹화
   grouped = extract_entities_grouped(text, ner_pipeline)
   for names in grouped["PER"]:
       print(names)  # "John Smith" 통합 출력
   ```

2. **##로 시작하는 토큰을 무시하기**
   ```python
   # 틀림
   def extract_entities_wrong(text, ner_pipeline):
       entities = ner_pipeline(text)
       names = []
       for token in entities:
           if not token["word"].startswith("##"):  # ##은 버리기
               names.append(token["word"])
       return names
   # 결과: "Smith", "son"이 분리됨 → "Smithson"이 "Smith"로 손실됨

   # 맞음 - ## 토큰을 이전 토큰과 연결
   ```

3. **대소문자 정보 손실**
   ```python
   # BERT-base-uncased 사용 시 주의
   text = "IBM and ibm"
   # 두 개 모두 소문자로 변환되므로 구분 불가능

   # 해결책: BERT-base-cased 사용 (하지만 속도는 느림)
   tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
   ```

---

## 체크포인트 2 모범 구현: GPT-2 텍스트 생성 및 디코딩 전략

### 텍스트 생성의 기본 원리

GPT는 **자기회귀(Autoregressive)** 모델이다. 즉, 현재까지의 텍스트를 기반으로 다음 토큰의 확률 분포를 예측하고, 이로부터 다음 토큰을 선택한다.

**직관적 이해**: "오늘 날씨는 정말 ___"이라는 문장이 주어졌을 때, GPT는 다음 단어의 확률을 계산한다. 예를 들어:
- "좋다" 40%
- "추우니까" 30%
- "더워서" 20%
- 기타 10%

이 확률 분포에서 다음 단어를 어떻게 선택할지가 **디코딩 전략**의 핵심이다.

### 디코딩 전략 1: Greedy Search (탐욕 검색)

매번 가장 높은 확률의 토큰을 선택한다.

```python
from transformers import pipeline

# GPT-2 Pipeline 로드
generator = pipeline("text-generation", model="gpt2", device=0)

prompt = "The future of artificial intelligence is"

# Greedy: 매번 최고 확률 선택
result = generator(
    prompt,
    max_length=30,
    num_return_sequences=1,
    do_sample=False,        # Greedy 옵션
    temperature=1.0         # 무시됨 (do_sample=False일 때)
)

generated_text = result[0]["generated_text"]
print(f"Greedy 결과:\n{generated_text}")

# 예상 출력:
# The future of artificial intelligence is very bright. We will see many new
```

**특성**:
- ✓ **결정적**: 같은 프롬프트 → 항상 같은 결과
- ✓ **신뢰도 높음**: 모델이 가장 확신하는 선택
- ✗ **단조로움**: 반복적이고 지루한 텍스트 생성
- ✗ **기울기 소실**: "가장 확률 높은 토큰"이 너무 극단적이면 기울기가 0에 가까워짐

**사용 시기**: 높은 신뢰도가 필요한 경우 (번역, 요약, 팩트 기반 질의응답)

### 디코딩 전략 2: Top-p (Nucleus) Sampling

누적확률이 p에 도달할 때까지의 토큰만 고려하여 샘플링한다.

```python
def generate_with_top_p(prompt: str, generator, p: float = 0.9,
                        max_length: int = 30) -> str:
    """Top-p Sampling으로 텍스트 생성"""
    result = generator(
        prompt,
        max_length=max_length,
        num_return_sequences=1,
        do_sample=True,         # 샘플링 활성화
        top_p=p,                # 누적확률 p까지만 고려
        temperature=1.0         # 표준 확률 분포 (변경 없음)
    )
    return result[0]["generated_text"]

prompt = "The future of artificial intelligence is"

# p=0.9: 누적확률 90%에 도달하는 상위 토큰들만
text_p90 = generate_with_top_p(prompt, generator, p=0.9)
print(f"Top-p (p=0.9):\n{text_p90}\n")

# p=0.5: 더 제한적, 상위 소수 토큰만
text_p50 = generate_with_top_p(prompt, generator, p=0.5)
print(f"Top-p (p=0.5):\n{text_p50}\n")

# 예상 출력:
# Top-p (p=0.9):
# The future of artificial intelligence is dependent on how we approach ethical
#
# Top-p (p=0.5):
# The future of artificial intelligence is very important and the most critical
```

**핵심 아이디어**:

```python
# 예: 토큰 확률 분포
probs = {
    "very": 0.50,      # 누적: 0.50
    "crucial": 0.25,   # 누적: 0.75
    "critical": 0.15,  # 누적: 0.90 ← p=0.9에서 여기까지 포함
    "important": 0.07, # 누적: 0.97 ← 제외
    "other": 0.03      # 누적: 1.00 ← 제외
}

# p=0.9인 경우: ["very", "crucial", "critical"] 중에서만 샘플링
# 이들의 확률을 정규화: 0.50/(0.50+0.25+0.15) = 0.526...
```

**특성**:
- ✓ **다양성**: 여러 토큰에서 샘플링하므로 다양한 결과
- ✓ **자동 조절**: 토큰 확률에 따라 유효 어휘가 동적으로 변함
- ✓ **극단적 토큰 회피**: 확률 극단값은 자동 제외
- ✗ **비결정적**: 매번 다른 결과
- ✗ **p 설정이 중요**: p가 작으면 제한적, 크면 Greedy와 유사

**사용 시기**: 창의성과 품질의 균형이 필요한 경우 (대화, 창작 글쓰기)

### 디코딩 전략 3: Temperature Sampling

확률 분포의 "날카로움"을 조절한다.

```python
def generate_with_temperature(prompt: str, generator,
                             temperature: float = 0.7,
                             max_length: int = 30) -> str:
    """Temperature로 확률 분포 조절"""
    result = generator(
        prompt,
        max_length=max_length,
        num_return_sequences=1,
        do_sample=True,
        temperature=temperature,
        top_p=1.0               # Top-p 비활성화
    )
    return result[0]["generated_text"]

prompt = "The future of artificial intelligence is"

# 다양한 온도로 생성
temperatures = [0.3, 0.7, 1.5]
print("Temperature별 생성 결과:")
print("=" * 80)

for temp in temperatures:
    text = generate_with_temperature(prompt, generator, temperature=temp)
    print(f"T={temp}: {text[:70]}...")
    print()

# 예상 출력:
# T=0.3: The future of artificial intelligence is very important and the future
# T=0.7: The future of artificial intelligence is going to be a key part of how
# T=1.5: The future of artificial intelligence is perhaps not just the computers
```

**Temperature의 수학적 의미**:

원래 로짓(logit)을 `z`라 하면:

P(word) = exp(z / T) / Σ exp(z_i / T)

- **T → 0** (T=0.1): 최고 확률 토큰의 확률 → 1 (극단적)
- **T = 1** (기본): 확률 분포 변화 없음
- **T → ∞** (T=10): 모든 토큰 확률 → 1/N (균등)

```python
import numpy as np
import matplotlib.pyplot as plt

# 예: 3개 토큰의 로짓
logits = np.array([2.0, 1.0, 0.0])

# 온도별 확률 계산
temperatures = [0.3, 0.7, 1.0, 1.5]
fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(3)
width = 0.2

for i, temp in enumerate(temperatures):
    probs = np.exp(logits / temp) / np.sum(np.exp(logits / temp))
    ax.bar(x + i*width, probs, width, label=f'T={temp}')

ax.set_ylabel('확률')
ax.set_xlabel('토큰')
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(['토큰 0\n(로짓=2.0)', '토큰 1\n(로짓=1.0)', '토큰 2\n(로짓=0.0)'])
ax.legend()
ax.set_title('온도에 따른 확률 분포 변화')
plt.tight_layout()
plt.savefig('temperature_effect.png', dpi=150)

# T가 낮을수록: 높은 확률의 토큰에 집중
# T가 높을수록: 모든 토큰에 균등하게 분산
```

**특성**:
- ✓ **직관적**: T를 낮추면 보수적, 높이면 창의적
- ✓ **세밀한 제어**: 다양한 수준의 창의성 조절
- ✗ **온도 설정이 까다로움**: 도메인마다 최적값이 다름
- ✗ **극단값의 문제**: T가 매우 작으면 오류 확률도 높아짐

**사용 시기**: 특정 창의성 수준이 필요한 경우 (시조·율문 생성, 대화 다양화)

### 세 전략의 비교

```python
# 같은 프롬프트로 세 전략 비교

prompt = "Artificial intelligence"

strategies = {
    "Greedy": {
        "do_sample": False,
        "top_p": 1.0,
        "temperature": 1.0
    },
    "Top-p (0.9)": {
        "do_sample": True,
        "top_p": 0.9,
        "temperature": 1.0
    },
    "Temperature (0.7)": {
        "do_sample": True,
        "top_p": 1.0,
        "temperature": 0.7
    }
}

print("같은 프롬프트, 다른 전략:")
print("=" * 80)

for strategy_name, params in strategies.items():
    result = generator(
        prompt,
        max_length=25,
        num_return_sequences=1,
        **params
    )
    generated = result[0]["generated_text"][len(prompt):]
    print(f"\n{strategy_name}:")
    print(f"  → {generated}")

# 예상 출력:
# Greedy:
#   → will be the most important technology of the century
#
# Top-p (0.9):
#   → has transformed every aspect of how we solve complex problems
#
# Temperature (0.7):
#   → will reshape industries and create new opportunities for innovation
```

| 특성 | Greedy | Top-p | Temperature |
|------|--------|-------|-------------|
| **결정성** | ✓ 결정적 | ✗ 비결정적 | ✗ 비결정적 |
| **다양성** | ✗ 없음 | ✓ 높음 | ✓ 높음 |
| **품질** | ✓ 높음 | ✓ 높음 | △ 중간 |
| **속도** | ✓ 빠름 | ✗ 느림 | ✗ 느림 |
| **구현** | ✓ 간단 | ✗ 복잡 | ✓ 간단 |
| **추천 용도** | 번역, 요약 | 대화, 창작 | 창작, 다양화 |

### 흔한 실수

1. **do_sample=False인데 온도 설정**
   ```python
   # 틀림 - temperature는 무시됨
   result = generator(
       prompt,
       do_sample=False,        # Greedy
       temperature=0.7         # 무시됨!
   )

   # 맞음
   result = generator(
       prompt,
       do_sample=True,         # 샘플링 활성화
       temperature=0.7
   )
   ```

2. **Top-p와 Temperature를 동시에 설정**
   ```python
   # 틀림 - 두 제약이 충돌
   result = generator(
       prompt,
       top_p=0.9,
       temperature=0.5
   )

   # 맞음 - 하나만 사용
   # Top-p 중심
   result = generator(prompt, top_p=0.9, temperature=1.0)
   # 또는 Temperature 중심
   result = generator(prompt, top_p=1.0, temperature=0.5)
   ```

3. **max_length가 너무 짧음**
   ```python
   # 틀림 - 문장이 끝나지 않은 상태로 끝남
   result = generator(prompt, max_length=10)
   # "The future of" 정도만 생성

   # 맞음 - 충분한 길이 설정
   result = generator(prompt, max_length=50)
   ```

---

## 체크포인트 3 모범 구현: BERT NER + GPT-2 통합 시스템

### 통합 파이프라인의 설계

실무에서 가장 일반적인 패턴은:

1. **BERT NER**: 입력 텍스트에서 개체명 추출
2. **프롬프트 구성**: 추출된 개체명을 활용하여 생성 프롬프트 작성
3. **GPT-2 생성**: 프롬프트를 기반으로 텍스트 생성
4. **결과 평가**: 세 디코딩 전략의 결과 비교

### 단계 1: 입력 텍스트 → 개체명 추출

```python
from transformers import pipeline
from typing import Dict, List

# Pipeline 로드
ner_pipeline = pipeline(
    "ner",
    model="dbmdz/bert-large-cased-finetuned-conll03-english"
)

# 입력 텍스트
input_text = "Sarah Johnson is the CEO of Apple Inc., based in Cupertino, California."

# 개체 추출
entities = ner_pipeline(input_text)

# 결과 출력
print(f"입력: {input_text}\n")
print(f"NER 결과 (토큰 단위):")
for entity in entities:
    print(f"  {entity['word']:15s} → {entity['entity']:10s} (score: {entity['score']:.4f})")

# 예상 출력:
# 입력: Sarah Johnson is the CEO of Apple Inc., based in Cupertino, California.
#
# NER 결과 (토큰 단위):
#   Sarah            → B-PER      (score: 0.9994)
#   Johnson          → I-PER      (score: 0.9993)
#   Apple            → B-ORG      (score: 0.9986)
#   Inc.             → I-ORG      (score: 0.9974)
#   Cupertino        → B-LOC      (score: 0.9945)
#   California       → B-LOC      (score: 0.9925)
```

### 단계 2: 개체 그룹화 및 프롬프트 구성

```python
def parse_entities(text: str, ner_result: list) -> Dict[str, List[str]]:
    """NER 결과를 정리하여 개체명 추출"""
    grouped = {"PERSON": [], "ORG": [], "LOC": []}
    current = {"text": "", "tag": None}

    for token in ner_result:
        word = token["word"]
        tag = token["entity"].split("-")[1]  # "B-PER" → "PER"

        # ## 토큰 처리
        if word.startswith("##"):
            current["text"] += word[2:]
        else:
            # 이전 개체 저장
            if current["text"] and current["tag"]:
                tag_map = {"PER": "PERSON", "ORG": "ORG", "LOC": "LOC"}
                grouped[tag_map[current["tag"]]].append(current["text"])

            # 새 개체 시작
            current = {"text": word, "tag": tag}

    # 마지막 개체 저장
    if current["text"] and current["tag"]:
        tag_map = {"PER": "PERSON", "ORG": "ORG", "LOC": "LOC"}
        grouped[tag_map[current["tag"]]].append(current["text"])

    return grouped

# 개체 파싱
parsed = parse_entities(input_text, entities)

print(f"그룹화된 개체명:")
for entity_type, names in parsed.items():
    if names:
        print(f"  {entity_type}: {', '.join(names)}")

# 예상 출력:
# 그룹화된 개체명:
#   PERSON: Sarah Johnson
#   ORG: Apple Inc.
#   LOC: Cupertino, California
```

### 단계 3: 개체 기반 프롬프트 생성

```python
def create_prompt_from_entities(parsed_entities: Dict[str, List[str]]) -> str:
    """추출된 개체를 바탕으로 생성 프롬프트 작성"""
    parts = []

    # 사람이 있으면
    if parsed_entities["PERSON"]:
        person = parsed_entities["PERSON"][0]
        parts.append(f"{person} works at")

    # 조직이 있으면
    if parsed_entities["ORG"]:
        org = parsed_entities["ORG"][0]
        parts.append(f"{org}")

    # 지역이 있으면
    if parsed_entities["LOC"]:
        loc = parsed_entities["LOC"][0]
        parts.append(f"located in {loc}.")

    # 최종 프롬프트 조합
    prompt = " ".join(parts)
    return prompt

# 프롬프트 생성
prompt = create_prompt_from_entities(parsed)
print(f"생성된 프롬프트:\n  {prompt}")

# 예상 출력:
# 생성된 프롬프트:
#   Sarah Johnson works at Apple Inc. located in Cupertino.
```

### 단계 4: 세 전략으로 텍스트 생성

```python
from transformers import pipeline

# GPT-2 Pipeline 로드
generator = pipeline("text-generation", model="gpt2", device=0)

def generate_with_strategy(prompt: str, strategy: str,
                          max_length: int = 50) -> str:
    """지정된 전략으로 텍스트 생성"""
    params = {
        "max_length": max_length,
        "num_return_sequences": 1
    }

    if strategy == "greedy":
        params["do_sample"] = False
        params["temperature"] = 1.0
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
strategies = ["greedy", "top_p", "temperature"]
results = {}

print(f"프롬프트: {prompt}\n")
print("=" * 80)

for strategy in strategies:
    generated = generate_with_strategy(prompt, strategy)
    results[strategy] = generated

    # 프롬프트를 제외한 생성 부분만 출력
    generation_only = generated[len(prompt):]

    print(f"\n[{strategy.upper()}]")
    print(f"  {generated}")

# 예상 출력:
# 프롬프트: Sarah Johnson works at Apple Inc. located in Cupertino.
#
# ================================================================================
#
# [GREEDY]
#   Sarah Johnson works at Apple Inc. located in Cupertino. She is the chief
#   executive officer and has been with the company since 2014.
#
# [TOP_P]
#   Sarah Johnson works at Apple Inc. located in Cupertino. Her leadership in
#   innovation has driven the company's transformation in the last decade.
#
# [TEMPERATURE]
#   Sarah Johnson works at Apple Inc. located in Cupertino. The company is
#   known for its revolutionary products and technological advancements.
```

### 단계 5: 결과 분석 및 시각화

```python
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_generation_quality(results: dict) -> dict:
    """생성 결과의 특성 분석"""
    analysis = {}

    for strategy, text in results.items():
        words = text.split()
        analysis[strategy] = {
            "length": len(words),
            "avg_word_len": sum(len(w) for w in words) / len(words),
            "unique_words": len(set(w.lower() for w in words)),
            "repetition_ratio": 1 - len(set(words)) / len(words),
        }

    return analysis

# 분석 수행
analysis = analyze_generation_quality(results)

# 시각화
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle("디코딩 전략별 생성 텍스트 특성", fontsize=14, fontweight='bold')

metrics = {
    (0, 0): ("length", "총 단어 수"),
    (0, 1): ("avg_word_len", "평균 단어길이"),
    (1, 0): ("unique_words", "고유 단어 수"),
    (1, 1): ("repetition_ratio", "반복률")
}

for (i, j), (metric, label) in metrics.items():
    ax = axes[i, j]
    strategies_list = list(analysis.keys())
    values = [analysis[s][metric] for s in strategies_list]

    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    ax.bar(strategies_list, values, color=colors)
    ax.set_ylabel(label, fontweight='bold')
    ax.set_title(label)

    # 값 표시
    for k, v in enumerate(values):
        ax.text(k, v, f'{v:.2f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('generation_analysis.png', dpi=150, bbox_inches='tight')
print("저장: generation_analysis.png")
plt.close()

# 분석 결과 출력
print("\n생성 텍스트 특성 분석:")
print("=" * 80)
for strategy, metrics in analysis.items():
    print(f"\n{strategy.upper()}:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.2f}")
```

### 단계 6: 정성적 해석

```python
interpretation = """
**생성 결과 해석**

[Greedy Search]
특징: 결정적이며 매번 같은 결과. "CEO"와 "since 2014" 등 구체적인 정보 추가.
강점: 매우 신뢰도 높고, 사실에 가까운 내용 생성.
약점: 단조로운 표현 (형식적 느낌).
평가: 회사 공식 보도자료에 적합.

[Top-p Sampling]
특징: 다양한 표현. "leadership", "innovation", "transformation" 등 추상적 개념 사용.
강점: 자연스럽고 창의적이며 다양한 각도 제시.
약점: 매번 다른 결과라 재현 어려움.
평가: 마케팅 자료나 신문 기사 작성에 적합.

[Temperature Sampling]
특징: 균형잡힌 표현. "revolutionary", "technological advancements" 등 일반적 표현.
강점: 창의성과 신뢰성의 균형.
약점: 일부 뻔한 표현.
평가: 소셜 미디어 또는 일반 콘텐츠 생성에 적합.

**결론**
- NER이 정확하므로 추출된 개체명(Sarah Johnson, Apple Inc.)을 프롬프트에 포함시키면
  생성 결과의 관련성이 높아진다.
- 각 전략의 선택은 **최종 사용 목적**에 따라 결정해야 한다:
  * 신뢰도 우선 → Greedy
  * 창의성 우선 → Top-p
  * 균형 → Temperature
"""

print(interpretation)
```

### 완전한 통합 코드

```python
from transformers import pipeline
from typing import Dict, List

# ============================================================================
# [1단계] NER 파이프라인 설정
# ============================================================================

ner_pipeline = pipeline(
    "ner",
    model="dbmdz/bert-large-cased-finetuned-conll03-english"
)

# ============================================================================
# [2단계] 유틸리티 함수 정의
# ============================================================================

def parse_entities(text: str, ner_result: list) -> Dict[str, List[str]]:
    """NER 결과를 정리하여 개체명 추출"""
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


def create_prompt_from_entities(parsed: Dict[str, List[str]]) -> str:
    """개체명으로부터 프롬프트 생성"""
    parts = []

    if parsed["PERSON"]:
        parts.append(f"{parsed['PERSON'][0]} works at")
    if parsed["ORG"]:
        parts.append(parsed["ORG"][0])
    if parsed["LOC"]:
        parts.append(f"located in {parsed['LOC'][0]}.")

    return " ".join(parts)


def generate_text(prompt: str, generator, strategy: str,
                 max_length: int = 50) -> str:
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


# ============================================================================
# [3단계] GPT-2 파이프라인 설정
# ============================================================================

generator = pipeline("text-generation", model="gpt2", device=0)

# ============================================================================
# [4단계] 실행
# ============================================================================

input_text = "Sarah Johnson is the CEO of Apple Inc., based in Cupertino, California."

# 개체 추출
entities = ner_pipeline(input_text)
parsed = parse_entities(input_text, entities)

# 프롬프트 생성
prompt = create_prompt_from_entities(parsed)

# 세 전략으로 생성
print(f"입력: {input_text}\n")
print(f"추출 개체: {parsed}\n")
print(f"프롬프트: {prompt}\n")
print("=" * 80)

for strategy in ["greedy", "top_p", "temperature"]:
    text = generate_text(prompt, generator, strategy)
    print(f"\n[{strategy.upper()}]\n{text}")
```

### 흔한 실수

1. **개체가 없으면 프롬프트가 비어짐**
   ```python
   # 틀림
   prompt = create_prompt_from_entities(parsed)
   if not prompt:  # 개체가 없으면 에러!
       # ...

   # 맞음
   def create_prompt_from_entities_safe(parsed):
       if not any(parsed.values()):
           return "Write a short text about:"  # 기본값
       # ... 원래 로직
   ```

2. **생성된 텍스트가 너무 짧거나 김**
   ```python
   # max_length를 적절히 설정
   # 토큰 단위이므로 문장으로는 대략 max_length/4~5
   result = generate_text(prompt, generator, "greedy", max_length=50)
   # 약 10~15 단어 생성
   ```

3. **프롬프트에 특수문자나 오류 포함**
   ```python
   # 틀림 - 토큰화 오류 발생 가능
   prompt = "Sarah Johnson,, works at  Apple Inc.."  # 중복 부호

   # 맞음 - 정제
   prompt = prompt.strip().replace(",,", ",").replace("  ", " ")
   ```

---

## 심화 학습: 모델 구조와 한계

### BERT의 양방향 이해와 한계

BERT는 **Masked Language Model(MLM)**으로 학습된다:

```
원문: "The [MASK] is very bright."
목표: [MASK] 위치의 토큰 예측

BERT는 앞뒤 모두를 보고 예측하므로 양방향 이해 가능.
```

**한계**:
- **생성에 부적합**: MLM은 이미 있는 문맥을 채우는 과제이므로, 처음부터 생성하기는 어려움
- **Causal Attention이 없음**: 따라서 순차 생성 불가능

### GPT의 자기회귀 생성과 한계

GPT는 **Causal Language Model(CLM)**로 학습된다:

```
프롬프트: "The future"
다음 토큰 예측: "is" (40%), "will" (30%), ...
선택된 토큰: "is"
새 프롬프트: "The future is"
반복...
```

**한계**:
- **환각(Hallucination)**: 모델이 사실이 아닌 내용을 그럴듯하게 생성할 수 있음
- **지식 한계**: 학습 데이터(2019년)보다 이후의 정보 부족
- **일관성 부족**: 긴 텍스트에서 맥락 일관성 저하

### 개선 방안: RAG와 Fine-tuning

```python
# RAG (Retrieval Augmented Generation): 외부 지식 기반 생성
# Fine-tuning: 도메인 특화 데이터로 모델 재학습

# 다음 장에서 자세히 다룸
```

---

## 참고 자료 및 추가 학습

### 완전한 구현 코드 위치

- **practice/chapter5/code/5-2-bert-ner-gpt-generation.py** — 통합 시스템 전체 코드
- **practice/chapter5/code/5-2-checkpoint-1-ner.py** — NER 전용
- **practice/chapter5/code/5-2-checkpoint-2-generation.py** — 생성 전용
- **practice/chapter5/code/5-2-checkpoint-3-integration.py** — 통합 시스템

### 권장 읽기

- Devlin, J., et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. arXiv. https://arxiv.org/abs/1810.04805
- Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., & Sutskever, I. (2019). Language Models are Unsupervised Multitask Learners. https://openai.com/blog/better-language-models/
- Holtzman, A., Buys, J., Du, L., Forbes, M., & Choi, Y. (2020). The Curious Case of Neural Text Degeneration. ICLR. https://arxiv.org/abs/1904.09751
- Hugging Face Course, Chapter 2: Using Transformers. https://huggingface.co/course/chapter2

### 추가 실습 (자율 학습)

1. **다국어 모델 시험**: `bert-multilingual-cased`로 한국어 NER 수행
2. **다양한 텍스트 생성**: 뉴스, 시, 대화 등 다양한 도메인에서 생성 비교
3. **미세조정**: 특정 도메인(의학, 법률 등)의 작은 데이터셋으로 BERT 미세조정
4. **평가 지표**: 생성 텍스트의 BLEU, ROUGE 점수 계산

### 다음 장으로의 연결

다음 장(6장)에서는:
- **프롬프트 엔지니어링**: GPT처럼 강력한 모델을 프롬프트만으로 제어하는 기술
- **API 활용**: OpenAI GPT-3/4, 혹은 오픈소스 모델의 API 사용
- **고급 디코딩**: Beam Search, Constrained Decoding 등
- **실무 시스템**: 생성된 텍스트의 품질 평가 및 필터링

이 5장의 BERT와 GPT를 완전히 이해한 후 다음 장으로 진행하자.

---

**생성일**: 2026-02-25
**대상 독자**: 컴퓨터공학/AI 전공 학부생 (3~4학년)
**난이도**: 중급 (PyTorch, Transformer 기초 선수)
