# 제6장 C: LLM API 활용과 프롬프트 엔지니어링 — 모범 구현과 해설

> 이 문서는 B회차 과제 제출 후 공개된다. 제출 전에는 열람하지 않는다.

---

## 체크포인트 1 모범 구현: 프롬프팅 기법 비교

프롬프팅 기법의 핵심은 같은 모델을 다르게 "지시"함으로써 성능을 끌어올리는 것이다. 다음은 Zero-shot, Few-shot, Chain-of-Thought의 완전한 구현이다.

### OpenAI API 기초 설정

```python
import os
from openai import OpenAI
from dotenv import load_dotenv

# 환경 변수 로드
# .env 파일에 OPENAI_API_KEY=sk-xxx... 형태로 저장
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 테스트 데이터: 한국 뉴스 기사 5개
# 감성 분류 기준:
# - 긍정: 경제 호황, 실적 개선, 기업 활동 증대
# - 중립: 정책 발표, 양론 공존, 중장기 전망
# - 부정: 경제 둔화, 부도, 구조적 악화
test_articles = [
    {
        "title": "삼성전자, 올해 영업이익 최고 기록 달성",
        "text": ("삼성전자가 반도체 가격 상승과 수요 회복에 힘입어 사상 최고의 영업이익을 거뒀다. "
                "업계 전문가들은 향후 3개월간 긍정적 모멘텀이 지속될 것으로 예상하고 있다."),
        "true_label": "긍정"
    },
    {
        "title": "경제 성장률 둔화, 전망 하락",
        "text": ("한국은행이 올해 경제 성장률을 종전 예측보다 0.3%p 낮춘 2.8%로 수정했다. "
                "수출 부진과 소비 위축이 주요 원인으로 지목되었다."),
        "true_label": "부정"
    },
    {
        "title": "정부, 신산업 투자 정책 발표",
        "text": ("정부가 반도체, 바이오, 전기차 등 3개 전략산업에 연 100조 원을 투자한다고 발표했다. "
                "업계에서는 긍정적으로 평가하면서도 구체적 시행 방안을 주목하고 있다."),
        "true_label": "중립"
    },
    {
        "title": "기술 중소기업 부도 급증",
        "text": ("금리 인상과 자금 조달 곤란으로 기술 중소기업의 부도가 전년 동기 대비 45% 증가했다. "
                "창업 생태계의 위축 우려가 높아지고 있다."),
        "true_label": "부정"
    },
    {
        "title": "수소 충전소 100개 달성",
        "text": ("정부 주도로 건설된 수소 충전소가 100개에 도달했다. "
                "이는 수소 경제 활성화를 위한 인프라 구축의 이정표로 평가되고 있다."),
        "true_label": "긍정"
    },
]

print("=" * 70)
print("프롬프팅 기법별 감성 분류 성능 비교")
print("=" * 70)
print()
```

### 프롬프팅 기법 1: Zero-shot

Zero-shot은 사전 예시 없이 과제 지시만으로 수행하는 기법이다. 가장 빠르고 저렴하지만, 모델이 패턴을 추론해야 하므로 정확도가 낮을 수 있다.

```python
def zero_shot_classification(article_text):
    """
    Zero-shot 프롬프팅으로 감성 분류

    핵심: 예시 없이 과제 지시만으로 분류
    - 과제의 목표를 명확히: "뉴스 기사의 감성을 분류하세요"
    - 분류 기준을 정의: "긍정/중립/부정 중 하나"
    - 출력 형식을 지정: "한 단어만"

    장점: 빠르고 저렴함
    단점: 모델이 패턴을 스스로 이해해야 함
    """
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": f"""다음 뉴스 기사의 감성을 '긍정', '중립', '부정' 중 하나로 분류하세요.
분류 기준은 기사의 전체 톤과 내용입니다.

기사:
{article_text}

감성(한 단어만):"""
            }
        ],
        temperature=0,      # 결정적 출력 (항상 같은 결과)
        max_tokens=10,      # 출력 토큰 최소화
    )
    return response

print("[1] Zero-shot Prompting\n")
zero_shot_results = []
zero_shot_total_input_tokens = 0
zero_shot_total_output_tokens = 0

for i, article in enumerate(test_articles, 1):
    article_text = f"제목: {article['title']}\n본문: {article['text']}"
    response = zero_shot_classification(article_text)

    # 응답 파싱
    prediction = response.choices[0].message.content.strip()
    zero_shot_results.append(prediction)

    # 토큰 사용량 기록
    zero_shot_total_input_tokens += response.usage.prompt_tokens
    zero_shot_total_output_tokens += response.usage.completion_tokens

    # 정답 여부 확인
    match = "O" if prediction == article['true_label'] else "X"
    print(f"기사 {i}: {prediction:5} (정답: {article['true_label']:5}) [{match}]")
    print(f"  토큰: 입력 {response.usage.prompt_tokens:3d}, 출력 {response.usage.completion_tokens}")

# 정확도 계산
zero_shot_accuracy = sum(1 for result, article in zip(zero_shot_results, test_articles)
                         if result == article['true_label']) / len(test_articles)
print(f"\n정확도: {zero_shot_accuracy:.1%}")
print(f"총 토큰: 입력 {zero_shot_total_input_tokens}, 출력 {zero_shot_total_output_tokens}\n")
```

**예상 실행 결과**:
```
[1] Zero-shot Prompting

기사 1: 긍정   (정답: 긍정  ) [O]
  토큰: 입력  42, 출력   1
기사 2: 부정   (정답: 부정  ) [O]
  토큰: 입력  42, 출력   1
기사 3: 중립   (정답: 중립  ) [X]  ← 중립을 정확히 분류하지 못함
  토큰: 입력  41, 출력   1
기사 4: 부정   (정답: 부정  ) [O]
  토큰: 입력  41, 출력   1
기사 5: 긍정   (정답: 긍정  ) [O]
  토큰: 입력  41, 출력   1

정확도: 80.0%
총 토큰: 입력 207, 출력 5
```

### 프롬프팅 기법 2: Few-shot

Few-shot은 3~5개의 예시를 제공하여 모델이 패턴을 학습하게 하는 기법이다. 입력 토큰이 증가하지만 정확도가 크게 향상된다.

**직관적 이해**: 아이에게 "이게 사과야, 이게 귤이야, 이게 포도야"라고 예시를 보여주고 다음 과일을 분류하게 하는 것. 설명만 들을 때보다 예시를 본 후가 훨씬 정확하다.

```python
def few_shot_classification(article_text):
    """
    Few-shot 프롬프팅으로 감성 분류

    핵심: 실제 예시 3~5개를 제공하여 패턴을 보여줌

    구조:
    1. 예시 입력-출력 쌍 제시
    2. 테스트 입력 제시
    3. 모델이 패턴을 따라 분류

    장점: 정확도 대폭 향상
    단점: 입력 토큰 증가 → 비용 증가
    """

    few_shot_examples = """예시 1:
제목: 국내 수출액 사상 최대 기록
본문: 올해 수출이 역사상 최고 수치를 기록했으며, 기업들의 실적도 호조를 보이고 있다.
감성: 긍정

예시 2:
제목: 실업률 급상승, 경제 악화 신호
본문: 실업률이 10년 만에 최고 수치를 기록했으며, 고용 지표가 악화되고 있다.
감성: 부정

예시 3:
제목: 정부 정책 발표, 시장 평가 엇갈려
본문: 정부가 새로운 규제를 발표했는데, 업계에서는 긍정과 우려의 목소리가 함께 나오고 있다.
감성: 중립"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": f"""다음 예시들을 참고하여 뉴스 기사의 감성을 분류하세요.

{few_shot_examples}

이제 다음 기사를 분류하세요:
제목: {article_text.split(chr(10))[0].replace('제목: ', '')}
본문: {article_text.split(chr(10))[1].replace('본문: ', '')}
감성:"""
            }
        ],
        temperature=0,
        max_tokens=10,
    )
    return response

print("[2] Few-shot Prompting\n")
few_shot_results = []
few_shot_total_input_tokens = 0
few_shot_total_output_tokens = 0

for i, article in enumerate(test_articles, 1):
    article_text = f"제목: {article['title']}\n본문: {article['text']}"
    response = few_shot_classification(article_text)

    prediction = response.choices[0].message.content.strip()
    few_shot_results.append(prediction)
    few_shot_total_input_tokens += response.usage.prompt_tokens
    few_shot_total_output_tokens += response.usage.completion_tokens

    match = "O" if prediction == article['true_label'] else "X"
    print(f"기사 {i}: {prediction:5} (정답: {article['true_label']:5}) [{match}]")
    print(f"  토큰: 입력 {response.usage.prompt_tokens:3d}, 출력 {response.usage.completion_tokens}")

few_shot_accuracy = sum(1 for result, article in zip(few_shot_results, test_articles)
                        if result == article['true_label']) / len(test_articles)
print(f"\n정확도: {few_shot_accuracy:.1%}")
print(f"총 토큰: 입력 {few_shot_total_input_tokens}, 출력 {few_shot_total_output_tokens}\n")
```

**예상 실행 결과**:
```
[2] Few-shot Prompting

기사 1: 긍정   (정답: 긍정  ) [O]
  토큰: 입력 143, 출력   1
기사 2: 부정   (정답: 부정  ) [O]
  토큰: 입력 143, 출력   1
기사 3: 중립   (정답: 중립  ) [O]  ← 예시 덕분에 정확히 분류!
  토큰: 입력 145, 출력   1
기사 4: 부정   (정답: 부정  ) [O]
  토큰: 입력 141, 출력   1
기사 5: 긍정   (정답: 긍정  ) [O]
  토큰: 입력 140, 출력   1

정확도: 100.0%
총 토큰: 입력 712, 출력 5
```

### 프롬프팅 기법 3: Chain-of-Thought (CoT)

Chain-of-Thought는 모델에게 **단계별로 생각하고 그 과정을 설명하도록** 지시하는 기법이다. 출력 토큰이 크게 증가하지만, 복잡한 문제에서 정확도가 극적으로 향상된다.

**직관적 이해**: 학생에게 "답을 말해"라고 하면 뭔가 틀릴 수 있지만, "과정을 보여주면서 답을 구해"라고 하면 실수할 확률이 줄어든다. 왜냐하면 단계별 추론이 더 신중하기 때문이다.

```python
def cot_classification(article_text):
    """
    Chain-of-Thought (CoT) 프롬프팅으로 감성 분류

    핵심: 모델에게 단계별 추론을 명시하도록 지시

    구조:
    1. "다음과 같이 생각하세요: (1) ..., (2) ..., (3) ..."
    2. 모델이 각 단계를 거쳐서 추론
    3. 마지막에 최종 결론 도출

    장점:
    - 복잡한 문제에서 정확도 극적 향상
    - 모델의 추론 과정이 보이므로 "왜" 그렇게 판단했는지 이해 가능
    - 오류 수정이 용이 (어느 단계에서 실수했는지 파악 가능)

    단점:
    - 출력 토큰 대폭 증가 → 비용 증가
    - 응답 시간 증가
    """

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": f"""다음 뉴스 기사의 감성을 분류하되, 단계별로 생각하여 이유를 명시하세요.

기사:
{article_text}

단계별 분석:
1. 기사의 주요 키워드를 파악하세요 (긍정적/부정적/중립적 단어들)
2. 기사의 전체 톤(tone)을 평가하세요
3. 경제적 영향도를 고려하세요
4. 최종 감성을 판단하세요

분석:"""
            }
        ],
        temperature=0,
        max_tokens=150,  # CoT는 더 긴 출력 필요
    )
    return response

print("[3] Chain-of-Thought (CoT) Prompting\n")
cot_results = []
cot_total_input_tokens = 0
cot_total_output_tokens = 0

for i, article in enumerate(test_articles, 1):
    article_text = f"제목: {article['title']}\n본문: {article['text']}"
    response = cot_classification(article_text)

    # CoT 응답에서 감성 추출
    # 모델이 "감성: 긍정" 형식으로 마지막에 답을 쓴다고 가정
    response_text = response.choices[0].message.content.strip()

    # 마지막 문장에서 감성 추출
    if "감성:" in response_text:
        prediction = response_text.split("감성:")[-1].strip().split("\n")[0].strip()
    else:
        # 만약 "감성:"이 없으면 응답에서 "긍정", "중립", "부정" 찾기
        for label in ["긍정", "중립", "부정"]:
            if label in response_text:
                prediction = label
                break
        else:
            prediction = "중립"  # 기본값

    cot_results.append(prediction)
    cot_total_input_tokens += response.usage.prompt_tokens
    cot_total_output_tokens += response.usage.completion_tokens

    match = "O" if prediction == article['true_label'] else "X"
    print(f"기사 {i}: {prediction:5} (정답: {article['true_label']:5}) [{match}]")
    print(f"  토큰: 입력 {response.usage.prompt_tokens:3d}, 출력 {response.usage.completion_tokens:3d}")
    # 추론 과정의 일부 출력
    preview = response_text[:80].replace("\n", " ")
    print(f"  분석: {preview}...\n")

cot_accuracy = sum(1 for result, article in zip(cot_results, test_articles)
                   if result == article['true_label']) / len(test_articles)
print(f"\n정확도: {cot_accuracy:.1%}")
print(f"총 토큰: 입력 {cot_total_input_tokens}, 출력 {cot_total_output_tokens}\n")
```

**예상 실행 결과**:
```
[3] Chain-of-Thought (CoT) Prompting

기사 1: 긍정   (정답: 긍정  ) [O]
  토큰: 입력  76, 출력  89
  분석: 1. 주요 키워드: "최고 기록", "수요 회복", "긍정적 모멘텀"...

기사 2: 부정   (정답: 부정  ) [O]
  토큰: 입력  76, 출력  75
  분석: 1. 주요 키워드: "둔화", "하락", "악화", "부진"...

기사 3: 중립   (정답: 중립  ) [O]
  토큰: 입력  75, 출력  81
  분석: 1. 주요 키워드: "긍정적", "우려", "엇갈려"...

기사 4: 부정   (정답: 부정  ) [O]
  토큰: 입력  73, 출력  72
  분석: 1. 주요 키워드: "부도 급증", "곤란", "위축"...

기사 5: 긍정   (정답: 긍정  ) [O]
  토큰: 입력  74, 출력  78
  분석: 1. 주요 키워드: "달성", "활성화", "이정표"...

정확도: 100.0%
총 토큰: 입력 374, 출력 395
```

### 핵심 포인트: 성능과 비용의 트레이드오프

```python
print("\n" + "=" * 70)
print("성능 및 비용 종합 비교")
print("=" * 70 + "\n")

# 정확도 비교
print("정확도 순위:")
print(f"  1위: Few-shot, CoT (동점) — {few_shot_accuracy:.1%}")
print(f"  3위: Zero-shot — {zero_shot_accuracy:.1%}")
print()

# 토큰 사용량 비교
print("토큰 사용량:")
print(f"  Zero-shot:  입력 {zero_shot_total_input_tokens:3d}, 출력 {zero_shot_total_output_tokens:3d}, 합 {zero_shot_total_input_tokens + zero_shot_total_output_tokens:3d}")
print(f"  Few-shot:   입력 {few_shot_total_input_tokens:3d}, 출력 {few_shot_total_output_tokens:3d}, 합 {few_shot_total_input_tokens + few_shot_total_output_tokens:3d}")
print(f"  CoT:        입력 {cot_total_input_tokens:3d}, 출력 {cot_total_output_tokens:3d}, 합 {cot_total_input_tokens + cot_total_output_tokens:3d}")
print()

# 비용 계산 (GPT-4o 기준: 입력 $2.50/M토큰, 출력 $10.00/M토큰)
gpt4o_input_price = 2.50 / 1_000_000
gpt4o_output_price = 10.00 / 1_000_000
krw_usd = 1300  # 환율

zero_shot_cost = (zero_shot_total_input_tokens * gpt4o_input_price +
                  zero_shot_total_output_tokens * gpt4o_output_price) * krw_usd
few_shot_cost = (few_shot_total_input_tokens * gpt4o_input_price +
                 few_shot_total_output_tokens * gpt4o_output_price) * krw_usd
cot_cost = (cot_total_input_tokens * gpt4o_input_price +
            cot_total_output_tokens * gpt4o_output_price) * krw_usd

print("비용 추정 (GPT-4o 기준, 원화):")
print(f"  Zero-shot:  {zero_shot_cost:8.2f}원 (기준)")
print(f"  Few-shot:   {few_shot_cost:8.2f}원 ({few_shot_cost/zero_shot_cost:.1f}배)")
print(f"  CoT:        {cot_cost:8.2f}원 ({cot_cost/zero_shot_cost:.1f}배)")
print()

# 정확도 향상당 비용
print("정확도 1% 향상당 추가 비용:")
if few_shot_accuracy > zero_shot_accuracy:
    cost_per_percent = (few_shot_cost - zero_shot_cost) / ((few_shot_accuracy - zero_shot_accuracy) * 100)
    print(f"  Zero-shot → Few-shot: {cost_per_percent:.2f}원")

if cot_accuracy > zero_shot_accuracy:
    cost_per_percent = (cot_cost - zero_shot_cost) / ((cot_accuracy - zero_shot_accuracy) * 100)
    print(f"  Zero-shot → CoT:      {cost_per_percent:.2f}원")
print()

# 결론
print("실무 선택 기준:")
print("  ✓ Zero-shot: 빠른 프로토타이핑, 정확도가 80% 이상 필요 없을 때")
print("  ✓ Few-shot: 정확도와 비용의 최적 균형 (권장)")
print("  ✓ CoT: 복잡한 추론이 필요하거나, 정확도가 중요한 업무")
```

### 흔한 실수

1. **응답 파싱에서 "감성:" 뒤의 공백 처리 안 함**
   ```python
   # 틀림
   prediction = response.split("감성:")[-1]
   # 문제: "감성: 긍정" 대신 " 긍정" (앞 공백 포함)으로 추출

   # 맞음
   prediction = response.split("감성:")[-1].strip()  # 앞뒤 공백 제거
   ```

2. **Few-shot 예시 개수를 너무 많게 설정**
   ```python
   # 효율 낮음
   few_shot_examples = "예시 1: ... 예시 2: ... ... 예시 10: ..."
   # 문제: 입력 토큰 폭증, 비용 대폭 증가, 모델이 중요한 예시를 놓칠 수 있음

   # 적당함
   few_shot_examples = "예시 1: ... 예시 2: ... 예시 3: ..."  # 3~5개 충분
   ```

3. **CoT에서 max_tokens를 너무 작게 설정**
   ```python
   # 틀림
   max_tokens=10  # CoT는 추론 과정을 써야 하므로 불충분

   # 맞음
   max_tokens=150  # CoT는 길이가 필요
   ```

4. **Temperature를 조정하지 않음**
   ```python
   # 기본값 (1.0)으로 두면 매번 다른 답 가능
   response = client.chat.completions.create(model="gpt-4o", ...)
   # 문제: 비결정적 출력, 비교하기 어려움

   # 맞음
   response = client.chat.completions.create(
       model="gpt-4o",
       temperature=0,  # 결정적 출력
       ...
   )
   ```

---

## 체크포인트 2 모범 구현: Function Calling

Function Calling은 LLM이 자신이 어떤 외부 도구를 호출해야 하는지를 **결정하고**, 그 도구의 입력 파라미터를 **지정하는** 기술이다. 실제 도구 실행은 애플리케이션이 담당한다.

### Function Calling 4단계 흐름

```python
import json
from typing import Optional

print("\n" + "=" * 70)
print("Function Calling 실습")
print("=" * 70 + "\n")

# 도구 정의 (JSON 스키마)
# 도구를 LLM에게 "이런 도구가 있습니다"라고 설명하는 방식
search_tool = {
    "type": "function",
    "function": {
        "name": "search_news",
        "description": "특정 주제로 최신 뉴스 기사를 검색합니다",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "검색 키워드 (예: '삼성전자 실적', '경제 성장률')"
                },
                "count": {
                    "type": "integer",
                    "description": "반환할 기사 개수 (기본값 3)",
                    "default": 3
                }
            },
            "required": ["query"]
        }
    }
}

weather_tool = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "특정 지역의 현재 날씨 정보를 조회합니다",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "지역명 (예: '서울', '부산', '대구')"
                }
            },
            "required": ["location"]
        }
    }
}

# 실제 도구 함수 구현 (프로덕션에서는 실제 API 호출)
def search_news(query: str, count: int = 3) -> dict:
    """
    뉴스 검색 도구 (시뮬레이션)

    실제로는 Google News API, Naver News API 등을 호출하지만,
    여기서는 시뮬레이션 데이터를 반환한다.
    """
    news_database = {
        "삼성전자 실적": [
            {"title": "삼성전자 영업이익 사상 최고", "sentiment": "긍정"},
            {"title": "삼성전자 반도체 생산 증대", "sentiment": "긍정"},
        ],
        "경제 성장률": [
            {"title": "경제 성장률 둔화", "sentiment": "부정"},
            {"title": "경기 회복 신호", "sentiment": "긍정"},
        ],
    }

    results = news_database.get(query, [{"title": f"{query}에 관한 뉴스", "sentiment": "중립"}])
    return {
        "query": query,
        "results": results[:count],
        "total_count": len(results)
    }

def get_weather(location: str) -> dict:
    """
    날씨 조회 도구 (시뮬레이션)

    실제로는 OpenWeatherMap API, 기상청 API 등을 호출하지만,
    여기서는 시뮬레이션 데이터를 반환한다.
    """
    weather_data = {
        "서울": {"temperature": 3, "condition": "맑음", "humidity": 45},
        "부산": {"temperature": 8, "condition": "구름많음", "humidity": 55},
        "대구": {"temperature": 5, "condition": "흐림", "humidity": 60},
    }

    result = weather_data.get(location, {"temperature": -999, "condition": "정보 없음", "humidity": 0})
    return {
        "location": location,
        "temperature": result["temperature"],
        "condition": result["condition"],
        "humidity": result["humidity"]
    }

# 4단계 Function Calling 데모
def function_calling_demo():
    """
    Function Calling의 4단계 흐름:

    [단계 1] 도구 정의를 LLM에 전달
    ↓
    [단계 2] LLM이 어떤 도구를 호출할지 결정
    ↓
    [단계 3] 애플리케이션이 실제 도구 함수 실행
    ↓
    [단계 4] 실행 결과를 LLM에 다시 전달하여 최종 응답 생성
    """

    user_query = "삼성전자 실적과 현재 서울 날씨를 알려줄 수 있어?"
    print(f"사용자: {user_query}\n")

    # ─────────────────────────────────────────────────────────
    # 단계 1: 도구 정의와 함께 LLM 호출
    # ─────────────────────────────────────────────────────────
    print("─" * 70)
    print("단계 1: 도구 정의를 LLM에 전달")
    print("─" * 70)
    print()

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": user_query
            }
        ],
        tools=[search_tool, weather_tool],
        tool_choice="auto",  # LLM이 자동으로 도구 선택
    )

    print(f"LLM 응답 타입: {response.choices[0].message.content}")
    print()

    # ─────────────────────────────────────────────────────────
    # 단계 2: LLM이 호출할 도구 결정
    # ─────────────────────────────────────────────────────────
    print("─" * 70)
    print("단계 2: LLM이 호출할 도구 결정")
    print("─" * 70)
    print()

    tool_calls = response.choices[0].message.tool_calls
    if not tool_calls:
        print("LLM이 도구를 호출할 필요가 없다고 판단했습니다.")
        return

    print(f"호출할 도구 개수: {len(tool_calls)}\n")
    for i, tool_call in enumerate(tool_calls, 1):
        print(f"도구 호출 {i}:")
        print(f"  함수명: {tool_call.function.name}")
        print(f"  인자: {tool_call.function.arguments}")
        print()

    # ─────────────────────────────────────────────────────────
    # 단계 3: 실제 도구 함수 실행
    # ─────────────────────────────────────────────────────────
    print("─" * 70)
    print("단계 3: 실제 도구 함수 실행")
    print("─" * 70)
    print()

    tool_results = []
    for tool_call in tool_calls:
        tool_name = tool_call.function.name
        tool_input = json.loads(tool_call.function.arguments)

        print(f"실행: {tool_name}({tool_input})")

        # 도구 함수 실행
        if tool_name == "search_news":
            result = search_news(**tool_input)
        elif tool_name == "get_weather":
            result = get_weather(**tool_input)
        else:
            result = {"error": f"Unknown tool: {tool_name}"}

        print(f"결과: {json.dumps(result, ensure_ascii=False, indent=2)}\n")

        # 결과를 LLM 메시지로 변환
        tool_results.append({
            "tool_call_id": tool_call.id,
            "tool_name": tool_name,
            "result": result
        })

    # ─────────────────────────────────────────────────────────
    # 단계 4: 도구 실행 결과를 LLM에 전달 후 최종 응답 생성
    # ─────────────────────────────────────────────────────────
    print("─" * 70)
    print("단계 4: 도구 실행 결과를 LLM에 전달 후 최종 응답 생성")
    print("─" * 70)
    print()

    # 메시지 히스토리 구성
    messages = [
        {"role": "user", "content": user_query},
        response.choices[0].message,  # LLM의 도구 호출 결정
    ]

    # 각 도구 실행 결과를 메시지로 추가
    for tool_result in tool_results:
        messages.append({
            "role": "tool",
            "tool_call_id": tool_result["tool_call_id"],
            "content": json.dumps(tool_result["result"], ensure_ascii=False)
        })

    # LLM에게 최종 응답 생성 요청
    final_response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
    )

    print(f"최종 응답:\n{final_response.choices[0].message.content}")

function_calling_demo()
```

**예상 실행 결과**:
```
──────────────────────────────────────────────────────────────────
단계 1: 도구 정의를 LLM에 전달
──────────────────────────────────────────────────────────────────

──────────────────────────────────────────────────────────────────
단계 2: LLM이 호출할 도구 결정
──────────────────────────────────────────────────────────────────

호출할 도구 개수: 2

도구 호출 1:
  함수명: search_news
  인자: {"query": "삼성전자 실적"}

도구 호출 2:
  함수명: get_weather
  인자: {"location": "서울"}

──────────────────────────────────────────────────────────────────
단계 3: 실제 도구 함수 실행
──────────────────────────────────────────────────────────────────

실행: search_news({'query': '삼성전자 실적'})
결과: {
  "query": "삼성전자 실적",
  "results": [
    {"title": "삼성전자 영업이익 사상 최고", "sentiment": "긍정"},
    {"title": "삼성전자 반도체 생산 증대", "sentiment": "긍정"}
  ],
  "total_count": 2
}

실행: get_weather({'location': '서울'})
결과: {
  "location": "서울",
  "temperature": 3,
  "condition": "맑음",
  "humidity": 45
}

──────────────────────────────────────────────────────────────────
단계 4: 도구 실행 결과를 LLM에 전달 후 최종 응답 생성
──────────────────────────────────────────────────────────────────

최종 응답:
삼성전자의 최근 실적은 긍정적입니다. 영업이익이 사상 최고를 기록했으며,
반도체 생산도 증대되는 추세입니다.

현재 서울의 날씨는 맑고, 기온은 3°C로 쌀쌀합니다. 습도는 45%로 쾌적합니다.
```

### Function Calling의 핵심 포인트

#### LLM이 도구를 "직접 실행하지 않는다"는 설계의 의미

Function Calling의 가장 중요한 원칙:

```python
# ❌ 틀린 이해
# "LLM이 자동으로 get_weather() 함수를 실행한다"
# → 문제: LLM이 외부 함수를 실행할 수 없음 (보안, 격리)

# ✓ 올바른 이해
# 단계 1: LLM은 "get_weather를 호출해야 한다"고 **결정**만 함
# 단계 2: 우리 코드가 실제로 get_weather를 호출하고 결과를 얻음
# 단계 3: 결과를 LLM에게 알려줌
# 단계 4: LLM이 결과를 바탕으로 최종 답변 작성

# 장점:
# 1. 보안: LLM이 임의 코드를 실행하지 않음
# 2. 통제: 우리가 도구 실행을 완전히 제어함
# 3. 유연성: 실제 도구를 바꿀 수 있음 (API, DB 등)
```

#### 여러 도구를 동시에 호출할 때

```python
def multi_tool_calling_example():
    """여러 도구를 동시에 호출하고 처리하는 패턴"""

    queries = [
        "광주와 부산, 대구의 날씨를 비교해줄래?",
        "반도체와 배터리 뉴스를 찾고 서울 날씨도 알려줄래?",
    ]

    for query in queries:
        print(f"\n질문: {query}")
        print("─" * 70)

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": query}],
            tools=[search_tool, weather_tool],
            tool_choice="auto",
        )

        tool_calls = response.choices[0].message.tool_calls
        print(f"호출할 도구 수: {len(tool_calls) if tool_calls else 0}")

        if tool_calls:
            for i, tc in enumerate(tool_calls, 1):
                args = json.loads(tc.function.arguments)
                print(f"  도구 {i}: {tc.function.name}({args})")

        # 실제 실행 및 최종 응답 생성 (위의 4단계 반복)

multi_tool_calling_example()
```

### 흔한 실수

1. **도구 JSON 스키마에서 required 필드 누락**
   ```python
   # 틀림
   "parameters": {
       "type": "object",
       "properties": {
           "location": {"type": "string"}
       }
       # required 필드 없음
   }

   # 맞음
   "parameters": {
       "type": "object",
       "properties": {
           "location": {"type": "string"}
       },
       "required": ["location"]  # 필수 인자 명시
   }
   ```

2. **tool_call.function.arguments가 JSON 문자열임을 간과**
   ```python
   # 틀림
   args = tool_call.function.arguments  # 딕셔너리로 착각
   search_news(**args)  # TypeError

   # 맞음
   args = json.loads(tool_call.function.arguments)  # JSON 파싱
   search_news(**args)
   ```

3. **tool_call_id를 메시지에 포함하지 않음**
   ```python
   # 틀림
   messages.append({
       "role": "tool",
       "content": json.dumps(result)
       # tool_call_id 없음
   })

   # 맞음
   messages.append({
       "role": "tool",
       "tool_call_id": tool_call.id,  # 호출과 결과 연결
       "content": json.dumps(result)
   })
   ```

---

## 체크포인트 3 모범 구현: Structured Output + LLM-as-a-Judge

### Structured Output: Pydantic 모델로 강제 구조화

Structured Output은 LLM의 자유 형식 출력을 **강제로 구조화**하는 기술이다. Pydantic 모델을 정의하면, OpenAI API가 응답을 항상 그 형식으로 반환하도록 보장한다.

```python
from pydantic import BaseModel, Field
from typing import List

print("\n" + "=" * 70)
print("Structured Output + LLM-as-a-Judge")
print("=" * 70 + "\n")

class NewsAnalysis(BaseModel):
    """
    뉴스 기사 분석 결과를 구조화된 형식으로 정의

    장점:
    - 파싱 오류 없음: 항상 유효한 형식으로 응답
    - 타입 안전성: 각 필드의 타입이 보장됨
    - 프로그래밍 용이: 딕셔너리 대신 객체 속성으로 접근 가능
    """
    title: str = Field(description="기사 제목")
    sentiment: str = Field(description="감성 분류 (긍정/중립/부정)")
    key_topics: List[str] = Field(description="주요 주제 (예: 경제, 기술)")
    economic_impact: str = Field(description="경제적 영향 (긍정/중립/부정)")
    confidence_score: float = Field(description="분류 신뢰도 (0.0~1.0)")
    summary: str = Field(description="한 문장 요약 (20단어 이내)")

def analyze_article_structured(article_text: str) -> NewsAnalysis:
    """
    Structured Output으로 뉴스 분석

    response_format=NewsAnalysis를 지정하면,
    OpenAI가 응답을 항상 NewsAnalysis 형식으로 반환함을 보장한다.
    """

    response = client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": f"""다음 뉴스 기사를 분석하고, 지정된 JSON 형식으로 결과를 제공하세요.

기사:
{article_text}

분석 요청:
- 감성은 '긍정', '중립', '부정' 중 하나
- 주요 주제는 리스트로 (예: ['경제', '기술'])
- 경제적 영향은 '긍정', '중립', '부정' 중 하나
- 신뢰도는 0.0~1.0 범위
- 요약은 20단어 이내"""
            }
        ],
        response_format=NewsAnalysis,  # ← Pydantic 모델 지정
    )

    return response.choices[0].message.parsed

# 구조화된 분석 수행
print("구조화된 뉴스 분석 결과:\n")
structured_results = []

for i, article in enumerate(test_articles, 1):
    article_text = f"제목: {article['title']}\n본문: {article['text']}"

    try:
        analysis = analyze_article_structured(article_text)
        structured_results.append(analysis)

        print(f"기사 {i}: {analysis.title}")
        print(f"  감성: {analysis.sentiment}")
        print(f"  주제: {', '.join(analysis.key_topics)}")
        print(f"  경제영향: {analysis.economic_impact}")
        print(f"  신뢰도: {analysis.confidence_score:.2f}")
        print(f"  요약: {analysis.summary}\n")
    except Exception as e:
        print(f"기사 {i} 분석 실패: {e}\n")
```

**예상 실행 결과**:
```
구조화된 뉴스 분석 결과:

기사 1: 삼성전자, 올해 영업이익 최고 기록 달성
  감성: 긍정
  주제: 경제, 기술, 기업실적
  경제영향: 긍정
  신뢰도: 0.95
  요약: 반도체 호황에 힘입어 사상 최고 영업이익 달성.

기사 2: 경제 성장률 둔화, 전망 하락
  감성: 부정
  주제: 경제, 정책, 고용
  경제영향: 부정
  신뢰도: 0.92
  요약: 경제 성장률 하향 조정, 고용 지표 악화.
...
```

### LLM-as-a-Judge: 자동 평가 시스템

```python
class EvaluationResult(BaseModel):
    """평가 결과 구조"""
    accuracy: int = Field(description="정확도 점수 (0~10)")
    reasoning_quality: int = Field(description="추론 품질 (0~10)")
    conciseness: int = Field(description="간결성 (0~10)")
    overall_score: float = Field(description="종합 점수 (0~10)")
    feedback: str = Field(description="개선 의견 (3문장 이내)")

def evaluate_analysis(article: dict, analysis: NewsAnalysis) -> EvaluationResult:
    """
    LLM-as-a-Judge: 다른 LLM에게 분석 결과 평가 위임

    장점:
    - 자동 평가로 사람 평가 비용 절감
    - 일관된 기준으로 대량 평가 가능
    - 피드백으로 모델 개선 방향 파악

    주의:
    - 같은 모델이 자신의 출력을 높게 평가하는 "자기 편향" 주의
    - 여러 모델의 평가를 조합하면 더 신뢰성 높음 (교차 검증)
    """

    evaluation_prompt = f"""당신은 NLP 전문가입니다. 다음 뉴스 기사 분석을 평가하세요.

기사:
  제목: {article['title']}
  본문: {article['text']}

분석 결과:
  감성: {analysis.sentiment}
  주제: {', '.join(analysis.key_topics)}
  경제영향: {analysis.economic_impact}
  신뢰도: {analysis.confidence_score}
  요약: {analysis.summary}

정답 감성: {article['true_label']}

평가 기준:
1. 정확도 (0~10): 분석이 기사 내용을 올바르게 파악했는가?
2. 추론 품질 (0~10): 주제와 경제영향이 논리적으로 타당한가?
3. 간결성 (0~10): 요약이 명확하고 정보 손실이 없는가?
4. 종합 점수: 세 항목의 평균"""

    response = client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[{"role": "user", "content": evaluation_prompt}],
        response_format=EvaluationResult,
    )

    return response.choices[0].message.parsed

print("=" * 70)
print("LLM-as-a-Judge 자동 평가")
print("=" * 70 + "\n")

total_scores = {
    "accuracy": 0,
    "reasoning_quality": 0,
    "conciseness": 0,
    "overall_score": 0
}

for i, (article, analysis) in enumerate(zip(test_articles, structured_results), 1):
    evaluation = evaluate_analysis(article, analysis)

    print(f"기사 {i} 평가:")
    print(f"  정확도:      {evaluation.accuracy}/10")
    print(f"  추론 품질:   {evaluation.reasoning_quality}/10")
    print(f"  간결성:      {evaluation.conciseness}/10")
    print(f"  종합 점수:   {evaluation.overall_score:.1f}/10")
    print(f"  피드백:      {evaluation.feedback}\n")

    total_scores["accuracy"] += evaluation.accuracy
    total_scores["reasoning_quality"] += evaluation.reasoning_quality
    total_scores["conciseness"] += evaluation.conciseness
    total_scores["overall_score"] += evaluation.overall_score

num_articles = len(test_articles)
print("─" * 70)
print("평균 점수:")
print(f"  정확도:      {total_scores['accuracy']/num_articles:.1f}/10")
print(f"  추론 품질:   {total_scores['reasoning_quality']/num_articles:.1f}/10")
print(f"  간결성:      {total_scores['conciseness']/num_articles:.1f}/10")
print(f"  종합 점수:   {total_scores['overall_score']/num_articles:.1f}/10")
```

**예상 실행 결과**:
```
기사 1 평가:
  정확도:      9/10
  추론 품질:   9/10
  간결성:      8/10
  종합 점수:   8.7/10
  피드백:      분석이 정확합니다. 신뢰도 점수도 합리적입니다.
              주제 분류도 타당합니다.
...

────────────────────────────────────────────────────────────────────
평균 점수:
  정확도:      9.0/10
  추론 품질:   8.6/10
  간결성:      8.4/10
  종합 점수:   8.7/10
```

### Structured Output의 핵심 포인트

#### Pydantic 검증의 강력함

```python
# 좋은 예: 자동 검증
class NewsAnalysis(BaseModel):
    confidence_score: float = Field(
        description="신뢰도 (0.0~1.0)",
        ge=0.0,  # greater or equal
        le=1.0   # less or equal
    )

# LLM이 1.5를 반환하려고 하면?
# → OpenAI API가 자동으로 거부하고 다시 생성하도록 함
# → 수동으로 파싱/검증할 필요 없음
```

#### 자유 형식 vs Structured Output

```python
# 자유 형식 출력
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[...],
    # response_format 없음
)
output = response.choices[0].message.content
# 문제: "감성은 긍정입니다" vs "긍정" vs "positive" 등 일관성 없음
# 파싱 필요: if "긍정" in output then ...

# Structured Output
response = client.beta.chat.completions.parse(
    model="gpt-4o",
    messages=[...],
    response_format=NewsAnalysis
)
output = response.choices[0].message.parsed
# 장점: output.sentiment는 항상 문자열, 유효한 값만 가능
# 파싱 불필요: 바로 output.sentiment 접근 가능
```

### 흔한 실수

1. **response_format 파라미터 없이 parse() 호출**
   ```python
   # 틀림
   response = client.beta.chat.completions.parse(
       model="gpt-4o",
       messages=[...]
       # response_format 없음
   )

   # 맞음
   response = client.beta.chat.completions.parse(
       model="gpt-4o",
       messages=[...],
       response_format=NewsAnalysis
   )
   ```

2. **필드 설명(description)을 생략**
   ```python
   # 덜 명확함
   sentiment: str

   # 더 명확함
   sentiment: str = Field(
       description="감성 분류 (긍정/중립/부정)"
   )
   ```

3. **LLM-as-a-Judge에서 모델 편향 무시**
   ```python
   # 문제: GPT-4o가 자신의 (또는 같은 계열의) 출력을 높게 평가할 수 있음
   evaluator = client.beta.chat.completions.parse(
       model="gpt-4o",  # 같은 모델
       ...
   )

   # 개선: 여러 모델로 교차 검증
   # gpt-4o, claude, gemini 등의 평가를 조합하면 더 신뢰성 높음
   ```

---

## 종합 해설: 중간고사 대비

6주차는 **LLM 시대의 엔지니어링 기초**를 완성하는 주차이다. 지금부터는 API를 직접 호출하고, 프롬프팅으로 성능을 제어하며, 외부 도구를 연동하는 **실무 엔지니어**가 되는 단계다.

### 1~6주차 핵심 개념 요약

| 주차 | 핵심 개념 | 실무 역할 |
|---:|------|--------|
| 1 | PyTorch 기초, 텐서, 역전파 | 딥러닝 기본기 습득 |
| 2 | MLP, 활성화, 학습 루프 | 신경망 설계 및 학습 |
| 3 | 임베딩, RNN, Attention, Self-Attention | 시퀀스 모델의 핵심 이해 |
| 4 | Transformer, Encoder/Decoder, 토크나이저 | 모던 아키텍처 구현 |
| 5 | BERT (사전학습), GPT (생성), 파인튜닝 | 대규모 모델 활용 |
| **6** | **API, 프롬프팅, Function Calling, 평가** | **LLM을 도구로 다루기** |

### 중간고사 대비 핵심

**객관식 60% (개념 이해, 계산)**:
- Attention 수식: Attention(Q,K,V) = softmax(QKᵀ/√dₖ)V
- 토큰: 입력 $2.50/M, 출력 $10/M (입력이 4배 비쌈)
- 프롬프팅: Zero-shot < Few-shot ≈ CoT (정확도)
- 비용: Zero-shot < Few-shot ≈ CoT (입력 토큰)

**주관식 40% (코드 분석, 프롬프팅)**:
- "감성 분류 문제에 Few-shot 프롬프팅을 추천하는 이유는?"
  → 답: 정확도 향상(80% → 100%), 입력 토큰 증가(3배) 지만 여전히 비용 효율적
- "Function Calling 4단계 흐름을 설명하시오"
  → 답: (1) 도구 정의 전달, (2) LLM이 호출 결정, (3) 함수 실행, (4) 결과 전달

---

## 참고 코드 파일

다음 파일에서 전체 구현을 확인할 수 있다:

- `practice/chapter6/code/6-1-api기초.py` — OpenAI API 호출 기초
- `practice/chapter6/code/6-3-function-calling.py` — Function Calling 4단계 구현
- `practice/chapter6/code/6-5-프롬프트실습.py` — 프롬프팅 기법 비교 + Structured Output

### 코드 실행 방법

```bash
# 가상환경 활성화
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate    # Windows

# 환경 변수 설정
# .env 파일에 OPENAI_API_KEY=sk-xxx... 저장

# 각 실습 실행
python practice/chapter6/code/6-1-api기초.py
python practice/chapter6/code/6-3-function-calling.py
python practice/chapter6/code/6-5-프롬프트실습.py
```

---

## 최종 학습 정리

### 6주차 핵심 내용

1. **LLM API 생태계**: OpenAI, Anthropic, Google 등 주요 제공자와 오픈소스 선택 기준
2. **프롬프팅 5계층**: Zero-shot(빠름) → Few-shot(균형) → CoT(정확) → Self-Consistency(최고)
3. **토큰 과금**: 입력과 출력의 가격 차이, 월간 예산 관리의 중요성
4. **Structured Output**: Pydantic으로 파싱 오류 제거
5. **Function Calling**: LLM이 도구를 "결정"하고 우리가 "실행"하는 구조
6. **LLM-as-a-Judge**: 자동 평가로 모델 개선 루프 자동화

### 다음 단계로의 연결

- **8주차 RAG**: Function Calling으로 벡터 DB를 조회하는 시스템
- **12주차 AI Agent**: Function Calling을 확장하여 여러 도구를 자율적으로 조합
- **전체 워크플로우**: 데이터 수집 → 모델 학습 → API 배포 → 프롬프팅 최적화 → Agent 개발

6주차를 완벽히 이해한다면, 이후 모든 실무 주차(8-14주차)의 기초가 마련된다.

---

**생성일**: 2026-02-25
**대상 독자**: 컴퓨터공학/AI 전공 학부생 (3~4학년)
**난이도**: 중급-고급 (PyTorch, 딥러닝 기초, API 개념 선수)
**예상 학습 시간**: 90분 (B회차) + 추가 복습 2시간
