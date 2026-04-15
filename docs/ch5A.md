## 1. LLM이란 무엇인가?

### 1.1. 한 줄 정의

대규모 언어모델(LLM)은 수십억~수천억 개의 단어 패턴을 학습한 **"초거대 자동완성 엔진"**이다.

<aside>
💡

**비유**: 카카오톡 자동완성이 다음 글자를 추천하는 것과 원리가 같다. 다만 LLM은 그 규모가 수만 배 크고, 거의 모든 종류의 글쓰기를 할 수 있다.

</aside>

- **범용성**: 하나의 모델로 번역, 요약, 코드 생성, 대화 등 수십 가지 일 수행 — **만능 요리사**처럼 레시피(프롬프트)만 바꾸면 된다
- **제로샷/퓨샷 학습**: 예시 몇 개만 보여주면 새 작업 가능 — 똑똑한 신입사원에게 매뉴얼 한 장 주는 것과 비슷
- **지식 저장소**: 학습 중 백과사전급 지식을 내부에 저장

### 1.2. LLM은 어떻게 발전해왔나? (2026년 기준)

- **크기 경쟁 → 효율 경쟁으로**: GPT-3(1,750억) → GPT-4(2023) → GPT-4o/4.5(2024~25) → 더 작지만 똑똑한 모델 시대로
- **추론(생각하는) 모델 등장**: OpenAI o1/o3, DeepSeek-R1 등 "답하기 전에 생각하는" 모델이 수학/코드에서 압도적 성능
- **학습법 혁신**: RLHF → DPO(직접 선호도 최적화), GRPO 등 더 간단한 정렬 기법으로 진화
- **멀티모달 보편화**: GPT-4o, Claude 4, Gemini 2 등 텍스트+이미지+음성을 하나의 모델로 처리
- **오픈소스 약진**: LLaMA 3(Meta), DeepSeek-V3, Qwen 2.5(알리바바) 등이 상용 모델에 근접
- **한국어 모델**: HyperCLOVA X(네이버), SOLAR(Upstage), EXAONE(LG AI Research) 등 국내 LLM 생태계 형성

### 1.3. NLP 패러다임의 변화

| 구분 | 과거 (2018년 이전) | 현재 (LLM 시대) |
| --- | --- | --- |
| **모델** | 태스크마다 전용 모델 제작 | 하나의 범용 모델 |
| **학습 방식** | 레이블 데이터 필수 (지도학습) | 레이블 없는 대규모 텍스트 (자기지도학습) |
| **핵심 역량** | 모델 아키텍처 설계 | 프롬프트 엔지니어링 + 데이터 품질 |

'GPT-3': 대화와 자연어 처리를 중심으로 하는 모델
'Instruct GPT': 명령을 통해 특정 작업을 수행하는 인터페이스](https://prod-files-secure.s3.us-west-2.amazonaws.com/3835dc70-4e17-421a-b58e-c8cfd1017381/acf1a194-c7ae-412d-ad3a-d92079104e12/Untitled.png)

## 2. 세 가지 LLM 아키텍처

### 2.1. 트랜스포머: 모든 LLM의 뼈대

2017년 구글이 발표한 트랜스포머는 현대 LLM의 공통 골격이다.

<aside>
🏗️

**비유 — 트랜스포머는 "회의실"이다**

- **셀프 어텐션** = 회의 참석자 전원이 서로의 발언을 동시에 참고하는 것
- **멀티헤드 어텐션** = 여러 분과위원회가 각각 다른 관점(문법, 의미, 감정…)에서 동시에 논의
- **포지셔널 인코딩** = 발언 순서표 (누가 먼저 말했는지 기록)
- **피드포워드 네트워크** = 각 참석자가 자기 메모를 정리하는 개인 노트
</aside>

이 뼈대 위에 **세 가지 변형**이 발전했다. 아래 그림을 보자:

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/3835dc70-4e17-421a-b58e-c8cfd1017381/ca892ff3-8cf9-4c24-8e47-2f718059b0d6/Untitled.png)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/3835dc70-4e17-421a-b58e-c8cfd1017381/d71c40ea-06ff-487d-bfad-35c54c64d8e8/Untitled.png)

### 2.2. 인코더 전용 — "독해왕" (BERT 계열)

<aside>
📖

**비유**: 시험지를 받으면 **문제 전체를 훑어본 뒤** 답을 고르는 학생. 앞뒤 맥락을 모두 보기 때문에 "이해" 작업에 강하다.

</aside>

- **학습법**: 문장 중간에 빈칸(마스크)을 뚚고 맞추기 → 양방향 문맥 파악
- **잘하는 것**: 감정 분석, 스팸 탐지, 개체명 인식 등 **분류/이해** 태스크
- **대표 모델**: BERT(110M~340M), RoBERTa, DeBERTa

### 2.3. 디코더 전용 — "이야기꾼" (GPT 계열)

<aside>
✍️

**비유**: 소설가가 앞 문장만 보고 **다음 문장을 이어 쓰는** 방식. 글을 "생성"하는 데 최적화되어 있다.

</aside>

- **학습법**: 앞 단어들로 다음 단어 예측 (자기회귀)
- **잘하는 것**: 대화, 창작, 코드 생성 등 **생성** 태스크
- **대표 모델**: GPT-4o/4.5, Claude 4, Gemini 2, LLaMA 3, DeepSeek-V3, Qwen 2.5

### 2.4. 인코더-디코더 — "통역사" (T5, BART)

<aside>
🔄

**비유**: 동시통역사처럼 **입력을 완전히 이해(인코더)**한 뒤 **다른 형태로 출력(디코더)**한다.

</aside>

- **학습법**: 문장 일부를 훼손 → 원본 복원
- **잘하는 것**: 번역, 요약, 질의응답 등 **입력→출력 변환** 태스크
- **대표 모델**: T5, BART, mT5

### 2.5. 한눈에 비교

| 아키텍처 유형 | 적합한 태스크 | 덜 적합한 태스크 |
| --- | --- | --- |
| **인코더 전용** | 분류, 개체명 인식, 감성 분석, 자연어 추론 | 장문 텍스트 생성, 번역, 요약 |
| **디코더 전용** | 텍스트 생성, 창작 글쓰기, 대화, 코드 생성 | 세밀한 분류, 개체명 인식 |
| **인코더-디코더** | 번역, 요약, 질의응답, 패러프레이징 | 단순 분류, 단일 레이블 예측 |

## 3. 사전 학습과 파인튜닝 (30분)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/3835dc70-4e17-421a-b58e-c8cfd1017381/522fc7d9-f69b-436f-98c3-20e706147d8b/Untitled.png)

### 3.1. 사전 학습: "세상 모든 책을 읽히는 단계"

<aside>
📚

**비유**: 사전 학습은 **의대생이 교과서 전체를 읽는 것**과 같다. 아직 환자를 진료하는 건 아니지만, 기초 지식을 쌓는 단계다.

</aside>

- 레이블 없는 대규모 텍스트(인터넷, 책, 논문)로 언어의 패턴을 스스로 학습
- 비용이 매우 높지만(GPT-4 학습 추정 1억 달러 이상, LLaMA 3 405B도 수천만 달러), 한 번 하면 여러 곳에 재활용

### 3.2. 네 가지 사전 학습 방법

아래 그림들을 보면서 각 방법의 직관적 차이를 이해하자:

**1) 마스크드 언어 모델링 (MLM)** — BERT 방식

문장의 15%를 가리고 맞추기. **빈칸 채우기 시험**과 같다.

![image.png](attachment:03286cfc-4a60-439a-8e12-92bf483ae7c7:image.png)

---

**2) 인과적 언어 모델링 (CLM)** — GPT 방식

앞 단어로 다음 단어 예측. **끝말잇기**와 비슷한 원리.

![image.png](attachment:dac573e2-6ed2-4538-8fb0-3fcfd3ec4c69:image.png)

---

**3) 스팬 손상 및 복원** — T5/BART 방식

문장 일부를 지우고 복원. **훼손된 고문서를 복원하는 고고학자**처럼.

![image.png](attachment:a84c3718-8d43-467d-a237-a7d0538317a3:image.png)

---

**4) 다중 태스크 사전 학습** — FLAN/T0 방식

다양한 NLP 문제를 동시에 학습. **국·영·수를 한꺿번에 공부하는 것**과 같다.

![image.png](attachment:19cac7ad-991a-4464-b338-8ef081be4973:image.png)

---

### 3.3. 전이 학습: "배운 걸 새 일에 써먹기"

<aside>
🚲

**비유**: 자전거를 탈 줄 아는 사람은 오토바이도 금방 배운다. 이미 익힌 **균형 감각**(= 언어 지식)을 새 상황에 전이하는 것이다.

</aside>

전이 학습의 핵심 가치:

- **자원 절약**: 처음부터 학습하면 수억 원 → 파인튜닝은 수만 원
- **데이터 절약**: 레이블 데이터 100개만으로도 경쟁력 있는 성능
- **빠른 개발**: 사전 학습 모델 + 약간의 조정 → 즉시 서비스 가능

### 3.4. 전이 학습 세 가지 방법 비교(이하는 중간고사 이후 다시 학습함)

<aside>
🎓

**비유로 이해하기**

- **특징 추출** = 요리사가 만든 육수를 그대로 받아서 **내 소스만 따로 만드는 것** (육수 레시피는 안 건드림)
- **파인튜닝** = 요리사의 레시피 전체를 **내 입맛에 맞게 수정**하는 것
- **PEFT(LoRA 등)** = 레시피는 그대로 두고 **양념 몇 가지만 바꾸는 것** (효율적!)
</aside>

| 구분 | 특징 추출 (Feature Extraction) | 파인튜닝 (Fine-tuning) | PEFT |
| --- | --- | --- | --- |
| **원리** | 모델 가중치 고정, 특징만 추출하여 새 분류기 학습 | 사전 학습된 모델 전체 가중치 업데이트 | 소수의 추가 매개변수만 학습 |
| **학습 매개변수** | 분류기 매개변수만 (원본 모델 0%) | 모델 전체 매개변수 (100%) | 전체의 약 0.01~4% |
| **계산 효율성** | 매우 높음 | 낮음 (많은 자원 필요) | 중간~높음 |
| **메모리 요구량** | 매우 낮음 | 매우 높음 | 낮음 |
| **학습 속도** | 매우 빠름 | 느림 | 빠름 |
| **성능** | 제한적 | 최고 수준 | 파인튜닝에 근접 |
| **적응성** | 매우 제한적 | 매우 높음 | 중간~높음 |
| **적합 상황** | 간단한 태스크, 매우 제한된 자원 | 복잡한 태스크, 충분한 자원 | 대형 모델, 제한된 자원 |
| **대표 기법** | SVM, 로지스틱 회귀를 특징 위에 학습 | 전체 모델 그래디언트 업데이트 | LoRA, 어댑터, 프롬프트 튜닝 |
| **구현 복잡성** | 매우 낮음 | 중간 | 중간~높음 |
| **모델 크기 확장성** | 모든 크기에 적합 | 작은~중간 크기 모델에 적합 | 큰 모델일수록 효과적 |

| 구분 | 특징 추출 | 파인튜닝 | PEFT (LoRA 등) |
| --- | --- | --- | --- |
| **원리** | 모델 고정, 위에 분류기만 학습 | 모델 전체 가중치 업데이트 | 소수 매개변수만 학습 |
| **학습량** | 0% (원본 모델) | 100% | 0.01~4% |
| **비용** | 매우 낮음 | 매우 높음 | 낮음 |
| **성능** | 제한적 | 최고 | 파인튜닝에 근접 |
| **언제 쓸까?** | 간단한 분류, 자원 부족 | 복잡한 태스크, 자원 충분 | 대형 모델 + 제한된 자원 |

### 3.5. PEFT 심화: LoRA, 어댑터, 프롬프트 튜닝

**a) LoRA (Low-Rank Adaptation)** — 아래 그림 참고

<aside>
🧬

**비유**: 거대한 도서관의 책 내용은 그대로 두고, **각 책에 작은 포스트잇 메모**만 붙여서 새 지식을 추가하는 방법. 포스트잇(저차원 행렬)은 가볍지만 효과는 크다.

</aside>

- 원본 가중치 W는 고정, 작은 행렬 B×A만 추가 학습 (W' = W + BA)
- 전체의 1~4%만 학습 → 메모리·비용 절감
- 학습 후 원본과 합쳐서 추론 시 추가 비용 없음
- **QLoRA**(2023): 4bit 양자화 + LoRA를 결합하여, 단일 GPU로도 650억 파라미터 모델 파인튜닝 가능 — 현재 가장 널리 쓰이는 방법

![image.png](attachment:206728bd-94c8-4d0f-9cee-919e0cd30021:image.png)

**b) 어댑터 (Adapter)** — 아래 그림 참고

<aside>
🔌

**비유**: 해외여행 갈 때 콘센트에 끼우는 **변환 플러그**. 본체(모델)는 그대로 두고, 각 레이어 사이에 작은 어댑터 모듈만 꼽는 것.

</aside>

- 병목 구조: 차원 축소 → 비선형 변환 → 차원 복원
- 태스크별로 다른 어댑터를 꼽았다 뻔다 가능

![image.png](attachment:b7d435da-07a1-4f1c-b771-37ff1b1fecfb:image.png)

**c) 프롬프트 튜닝 (Prompt Tuning)** — 아래 그림 참고

<aside>
🏷️

**비유**: 모델에게 매번 **"너는 의사야"라는 이름표를 붙여주는 것**. 모델 자체는 전혀 바꾸지 않고, 입력 앞에 학습된 가상 토큰(소프트 프롬프트)만 추가.

</aside>

- 모델 파라미터의 0.01% 이하만 학습
- 초대형 모델일수록 효과적

![image.png](attachment:164176cd-d3f4-4382-89ce-d3ed04073866:image.png)

**비교 표**

| 항목 | LoRA | 어댑터(Adapter) | 프롬프트 튜닝(Prompt Tuning) |
| --- | --- | --- | --- |
| **구현 방식** | 가중치 행렬에 저차원 업데이트 | 레이어 사이에 소형 신경망 삽입 | 입력에 소프트 프롬프트 추가 |
| **적용 위치** | Attention/FFN 가중치 | 각 레이어 끝 | 입력 시퀀스 맨 앞 |
| **파라미터** | 1~4% | 1~3% | 0.01% 이하 |
| **추론 오버헤드** | 없음 (가중치 통합) | 약간 있음 (모듈 유지) | 최소 (입력 길이 증가) |
| **모듈성** | 가중치 기반, 간단한 교체 | 독립 모듈, 명시적 교체 가능 | 입력 기반, 매우 쉬운 교체 |
| **성능** | 전체 파인튜닝에 근접 | 전체 파인튜닝에 근접 | 큰 모델에서 근접, 작은 모델은 약함 |
| **적합 모델 크기** | 중소형~대형 | 중소형~대형 | 초대형 모델에 최적 |

## 4. 실전 파인튜닝 실습

앞에서 배운 LoRA를 실제로 적용해보자.

**실습코드:**

https://colab.research.google.com/drive/1cTsj_rZoZuic9CIBf8rfOcsSCjFvCdCD?usp=sharing

### 4.1 20대 한국인 말투 챗봇 개발

### 4.1.1 프로젝트 개요

20대 한국인의 자연스러운 말투와 대화 패턴을 학습한 챗봇을 개발한다. 한국어에 특화된 언어 모델을 파인튜닝하여 신조어, 줄임말, 이모티콘 등 20대의 특징적인 소통 방식을 구현해본다. 학생들은 다양한 페르소나(친구, 선배, 동아리 멤버 등)를 설정하여 개성 있는 챗봇을 만들 수 있다.

### 4.1.2 사용 기술 및 모델

- 기본 모델: skt/kogpt2, klue/bert-base 또는 beomi/KoGPT2-base (한국어 특화 모델)
- 프레임워크: Hugging Face Transformers, PyTorch
- 토크나이저: 한국어 특화 토크나이저 (KoNLPy, Mecab)
- 인터페이스: Gradio (웹 인터페이스)
- 배포: Hugging Face Spaces (무료 호스팅)
- 학습 환경: Google Colab (GPU 지원)

### 4.1.3 데이터 준비 및 수집

20대 한국인 말투 챗봇 개발을 위한 데이터는 다음과 같은 방법으로 수집할 수 있다:

1. **직접 데이터 수집**
    - 동료, 친구들과의 실제 대화 수집 (개인정보 제거)
    - 학생들이 직접 작성한 가상 대화 시나리오
2. **공개 데이터 활용**
    - AIHub 한국어 대화 데이터셋
    - 웹 소설, 드라마 대본, 유튜브 자막 등 공개 자료
3. **합성 데이터 생성**
    - 기존 대화 데이터를 기반으로 변형 생성
    - ChatGPT 등을 활용한 20대 말투 대화 시나리오 생성

데이터 수집 과정에서는 다음과 같은 20대 한국인 말투의 특징을 고려해야 한다:

- 신조어 및 줄임말 사용 (ex: 인싸, 꿀팁, 핵노잼, 친삭, 갑통알 등)
- 이모티콘과 의성어 활용 (ex: ㅋㅋㅋ, ㅠㅠ, 헐, 대박 등)
- 반말/존댓말 전환 패턴
- 영어 단어 혼용 패턴
- 문장 끝 생략 및 비격식체 표현

아래는 데이터 예시이다:

```python
conversation_data = [
    {
        "instruction": "오늘 뭐해?",
        "response": "아 그냥 집에서 넷플 보고 있어 ㅋㅋㅋ 너는?"
    },
    {
        "instruction": "시험 어떻게 봤어?",
        "response": "아 진짜 망했어 ㅠㅠ 교수님이 진짜 이상한 문제 내셨는데... 너는 괜찮았어?"
    },
    {
        "instruction": "내일 술 마실래?",
        "response": "내일? 가능할듯! 어디서 마실건데? 혹시 밥도 같이 먹을거야?"
    },
    {
        "instruction": "이 드라마 봤어?",
        "response": "아직 못 봤어ㅠㅠ 요즘 핫하다며? 1화만 봤는데 재밌더라 빨리 정주행 해야겠다"
    },
    {
        "instruction": "과제 다 했어?",
        "response": "아니 미쳤어 진짜 아직도 절반도 못했어 ㅋㅋㅋㅋㅋ 담주 월요일까지인데 큰일났다 진짜..."
    }
]

```

### 4.4. 핵심 코드: LoRA 파인튜닝

포스트잇 메모(→LoRA)를 모델에 붙이는 과정을 코드로 보자:

```python
import re
from konlpy.tag import Okt  # Mecab 대신 Okt 사용
import pandas as pd

# 데이터 로드
df = pd.DataFrame(conversation_data)

# 한국어 텍스트 전처리 함수
def preprocess_korean_text(text):
    # URL 제거
    text = re.sub(r'http\S+', '', text)
    # 반복되는 이모티콘 정규화 (예: ㅋㅋㅋㅋㅋ -> ㅋㅋㅋ)
    text = re.sub(r'([ㄱ-ㅎㅏ-ㅣ가-힣])\1{3,}', r'\1\1\1', text)
    # 특수문자 처리 - 한글, 영문, 숫자, 일반 문장부호, 공백 유지
    text = re.sub(r'[^\w\s,.!?()\'\":;ㄱ-ㅎㅏ-ㅣ가-힣]', '', text)
    return text
    
# 전처리 적용
df['cleaned_instruction'] = df['instruction'].apply(preprocess_korean_text)
df['cleaned_response'] = df['response'].apply(preprocess_korean_text)

# Okt 형태소 분석기 초기화
okt = Okt()

df['tokenized_normalized_response'] = df['cleaned_response'].apply(
    lambda x: ' '.join(okt.morphs(okt.normalize(x)))
)

# 최종 데이터 확인
print(df.head())
```

한국어 토큰화는 영어와 다른 접근이 필요하다. KoGPT 또는 KoBERT 모델의 토크나이저를 사용하면 한국어에 특화된 토큰화가 가능하다:

```python
# 토크나이저 로드 및 패딩 토큰 설정
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # 종료 토큰을 패딩 토큰으로 설정

# 토큰화 함수 수정
def tokenize_function(examples):
    texts = []
    for instruction, response in zip(examples["cleaned_instruction"], examples["cleaned_response"]):
        text = f"<usr>{instruction}<sys>{response}</s>"
        texts.append(text)
    
    # 패딩 명시적 설정
    encodings = tokenizer(
        texts, 
        truncation=True, 
        padding="max_length",  
        max_length=256,
        return_tensors="pt"
    )
    
    # 자기회귀적 학습을 위한 레이블 설정 (입력을 그대로 예측)
    encodings["labels"] = encodings["input_ids"].clone()
    return encodings
```

### 4.1.5 LoRA를 활용한 파인튜닝

한국어 모델 파인튜닝에도 LoRA(Low-Rank Adaptation) 기법을 적용하여 효율적인 학습을 진행한다:

```python
# 필요한 라이브러리 임포트
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, TaskType, get_peft_model
import torch
from datasets import Dataset

# 1) 모델 로드
model_name = "skt/kogpt2-base-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# 2) 특수 토큰 추가
special_tokens_dict = {
    'pad_token': '<pad>',
    'additional_special_tokens': ['<usr>', '<sys>']
}
tokenizer.add_special_tokens(special_tokens_dict)
model.resize_token_embeddings(len(tokenizer))

# 3) LoRA 설정 — 포스트잇 메모를 어디에 붙일지 정하기
lora_config = LoraConfig(
    r=16,                              # 포스트잇 크기 (랭크)
    lora_alpha=32,                     # 학습 강도
    target_modules=["c_attn", "c_proj"],  # 메모를 붙일 위치
    lora_dropout=0.05,
    task_type=TaskType.CAUSAL_LM,
    fan_in_fan_out=True
)

# 4) LoRA 적용
model = get_peft_model(model, lora_config)
print(f"학습 파라미터: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

# 5) 학습 실행
training_args = TrainingArguments(
    output_dir="./korean-chatbot",
    learning_rate=3e-4,
    per_device_train_batch_size=4,
    num_train_epochs=5,
    fp16=True,
    report_to=["none"],
)
trainer = Trainer(model=model, args=training_args, train_dataset=tokenized_dataset)
trainer.train()
```

### 4.5. 대화 테스트

```python
# 모델 로드
from peft import PeftModel, PeftConfig

# LoRA 적용 모델 로드
peft_model_path = "./korean-chatbot/checkpoint-10"
config = PeftConfig.from_pretrained(peft_model_path)

# 특수 토큰 추가 (모델 로드 전에 수행)
special_tokens_dict = {
    'pad_token': '<pad>',
    'additional_special_tokens': ['<usr>', '<sys>']
}

# 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path) # 토크나이저 먼저 로드
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict) # 특수 토큰 추가
print(f"추가된 특수 토큰 수: {num_added_toks}")

# 모델 로드 후 어휘 크기 조정
model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
model.resize_token_embeddings(len(tokenizer)) # 모델 로드 후 어휘 크기 조정

model = PeftModel.from_pretrained(model, peft_model_path)

# 응답 생성 함수
def generate_response(instruction, max_length=100):
    # 입력 텍스트 준비
    input_text = f"<usr>{instruction}<sys>"
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids

    # 응답 생성
    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            max_length=max_length,
            temperature=0.8,  # 응답 다양성 조절
            top_p=0.92,  # 상위 확률 샘플링
            repetition_penalty=1.2,  # 반복 방지
            eos_token_id=tokenizer.eos_token_id,
        )

    # 결과 디코딩
    generated_text = tokenizer.decode(output[0], skip_special_tokens=False)

    # 응답 부분만 추출
    response = generated_text.split("<sys>")[1].replace("</s>", "").strip()
    return response

# 테스트 대화
test_instructions = [
    "오늘 저녁에 뭐 먹을래?",
    "요즘 너무 피곤해서 힘들다ㅠㅠ",
    "이번 주말에 영화 볼래?",
    "요즘 인기있는 넷플릭스 추천해줘",
    "과제 제출 언제까지야?"
]

for instruction in test_instructions:
    response = generate_response(instruction)
    print(f"질문: {instruction}")
    print(f"응답: {response}")
    print("-" * 50)

```

### 4.6. Gradio 웹 인터페이스 배포

```python
import gradio as gr

def chat_response(message, chat_history):
    bot_response = generate_response(message)
    chat_history.append((message, bot_response))
    return "", chat_history

with gr.Blocks() as demo:
    gr.Markdown("# 20대 한국인 말투 챗봇")
    chatbot = gr.Chatbot(height=400)
    msg = gr.Textbox(placeholder="메시지를 입력하세요...")
    msg.submit(chat_response, [msg, chatbot], [msg, chatbot])

demo.launch()
```

### **참고)**

https://colab.research.google.com/drive/1-B4IqNGyO769IZBFmoCCN-e3bG6GZ4er?usp=sharing

## 5. 정리 및 향후 전망 (10분)

### 5.1. 오늘 배운 것 한 줄 요약

| 주제 | 핵심 내용 |
| --- | --- |
| **LLM이란?** | 수천억 단어를 학습한 만능 자동완성 엔진 |
| **아키텍처 3종** | 독해왕(BERT) / 이야기꾼(GPT) / 통역사(T5) |
| **사전학습→파인튜닝** | 교과서 읽기(사전학습) → 전공 실습(파인튜닝) |
| **PEFT** | 포스트잇(LoRA), 플러그(어댑터), 이름표(프롬프트 튜닝) |
| **실습** | KoGPT2 + LoRA로 20대 말투 챗봇 만들기 |

### 5.2. 2025~26년 핸심 트렌드

- **추론(생각) 모델**: o1/o3, DeepSeek-R1 등 "답하기 전에 생각하는" 모델이 수학/코딩에서 압도적 성능
- **AI 에이전트**: LLM이 도구를 직접 사용(API 호출, 웹 검색, 코드 실행)하는 자율형 시스템
- **오픈소스 약진**: DeepSeek-V3, LLaMA 3, Qwen 2.5 등이 상용 모델 수준에 근접
- **한국어 LLM 생태계**: HyperCLOVA X(네이버), SOLAR(Upstage), EXAONE(LG) 등 국내 모델 급성장
- **DPO/GRPO**: RLHF보다 간단한 정렬 기법이 주류로
- **QLoRA + Unsloth**: 단일 GPU로도 대형 모델 파인튜닝 가능 — 파인튜닝의 민주화

### 5.3. 과제

<aside>
📝

실습 Colab에서 **자신만의 대화 데이터 10개 이상**을 추가하여 챗봇을 파인튜닝하고, 결과를 제출하세요.

</aside>

## 참고 문헌

1. Vaswani, A. et al. (2017). Attention is all you need. *NeurIPS*.
2. Brown, T. et al. (2020). Language models are few-shot learners. *NeurIPS*.
3. Hu, E. J. et al. (2022). LoRA: Low-rank adaptation of large language models. *ICLR*.
4. Dettmers, T. et al. (2023). QLoRA: Efficient finetuning of quantized LLMs. *NeurIPS*.
5. Rafailov, R. et al. (2023). Direct Preference Optimization (DPO). *NeurIPS*.
6. DeepSeek-AI (2025). DeepSeek-R1: Incentivizing reasoning in LLMs via RL.
7. Dubey, A. et al. (2024). The LLaMA 3 herd of models. *Meta AI*.
8. Devlin, J. et al. (2019). BERT: Pre-training of deep bidirectional transformers. *NAACL*.
9. Park, K. et al. (2021). KoBERT: Korean BERT model.