## 6주차 B회차: LLM API 활용 및 프롬프팅 실습

> **미션**: 도메인 특화 텍스트 분석 시스템을 구현하고, 프롬프팅 기법별 성능을 비교하며, Function Calling으로 외부 도구를 연동할 수 있다

### 수업 타임라인

| 시간 | 내용 | Copilot 사용 |
|------|------|-------------|
| 00:00~00:05 | 조 편성 + 역할 배분 (조원 A/B) | 사용 안 함 |
| 00:05~00:10 | A회차 핵심 리캡 + 과제 스펙 재확인 | 사용 안 함 |
| 00:10~00:55 | 2인1조 Copilot 구현 (체크포인트 3회) | 적극 사용 |
| 00:55~01:00 | Google Classroom 제출 (조별 1부) | 사용 안 함 |
| 01:00~01:20 | 결과 토론 (기법별 성능 비교·리소스 분석) | 사용 안 함 |
| 01:20~01:28 | 교수 피드백 + 핵심 정리 | 사용 안 함 |
| 01:28~01:30 | 다음 주 예고 | 사용 안 함 |

---

### A회차 핵심 리캡

**LLM API의 개념 및 구조**:
- API는 거대 모델을 원격 서버에서 실행하고 HTTP 요청으로만 상호작용하는 방식이다
- OpenAI(GPT-4o), Anthropic(Claude 4.5) 등 상용 API는 전문성, 신뢰성, 자동 업데이트가 장점이고, 오픈소스(Llama, Mistral)는 비용 통제와 자유도가 장점이다
- 모든 LLM API는 동일한 패턴을 따른다: **메시지 배열을 보내면 생성된 텍스트를 받는다**

**프롬프트 엔지니어링의 5계층**:
- Zero-shot: 예시 없이 과제만 설명 (빠르지만 정확도 낮음)
- Few-shot: 3~5개의 예시로 패턴 학습 (비용 중간, 정확도 향상)
- Chain-of-Thought (CoT): 단계별 추론 유도 (비용 높지만 복잡한 문제에서 극적인 정확도 향상)
- Self-Consistency: CoT를 여러 번 수행한 뒤 다수결 선택 (최고 정확도, 비용 최고)

**Structured Output의 가치**:
- LLM의 자유 형식 출력을 Pydantic 모델로 강제하여 JSON 구조 보장
- 파싱 오류 없이 즉시 프로그래밍에서 활용 가능

**Function Calling 4단계 흐름**:
- 1단계: 도구 목록을 JSON으로 정의하여 전송
- 2단계: LLM이 어떤 도구를 어떤 인자로 호출할지 결정
- 3단계: 애플리케이션이 실제로 도구 함수를 실행
- 4단계: 실행 결과를 LLM에 다시 전달하여 최종 자연어 응답 생성

**LLM 평가와 할루시네이션**:
- 자동 평가: BLEU(정밀도 기반), ROUGE(재현율 기반)
- LLM-as-a-Judge: 다른 LLM에게 평가 위임
- 할루시네이션 완화: 교차 검증, CoT 요청, RAG 연동

---

### 과제 스펙 + 체크포인트

**과제**: 뉴스 기사 감성 분류 및 정보 추출 시스템 구축

**제출 형태**: 조별 1부, Google Classroom 업로드

**파일 구성**:
- 구현 코드 파일 (`*.py`)
- 성능 비교 리포트 (프롬프팅 기법별 정확도, 토큰 사용량)
- 평가 결과 분석 (각 기법의 장단점 및 실무 적용 방안)

**검증 기준**:
- ✓ Zero-shot, Few-shot, CoT 프롬프팅 3가지 기법 구현
- ✓ Pydantic으로 구조화된 정보 추출 성공
- ✓ LLM-as-a-Judge 자동 평가 실행
- ✓ 토큰 사용량 측정 및 비용 계산
- ✓ 3가지 기법의 성능 및 비용 비교 분석 완료

---

### 2인1조 실습

> **Copilot 활용**: Copilot에게 "OpenAI API로 감성 분석을 해줄 수 있어?", "이 코드에 Few-shot 예시를 추가해줄래?", "CoT 프롬프트를 이용한 단계적 분석을 구현해줄 수 있어?"와 같이 단계적으로 요청한다. 생성된 코드의 API 버전(openai 1.0+, anthropic 0.28+)이 최신인지, 응답 파싱 방식이 정확한지 검증하는 과정에서 API의 세부 구조를 깊이 있게 이해할 수 있다.

**역할 분담**:
- **조원 A (드라이버)**: 코드 작성 및 API 호출, 실행 결과 확인
- **조원 B (네비게이터)**: Copilot 프롬프트 설계, API 문서 검증, 응답 파싱 로직 검토
- **체크포인트마다 역할 교대**: 드라이버와 네비게이터를 번갈아가며 진행하여 두 명 모두 전체 구현을 이해한다

---

#### 체크포인트 1: 프롬프팅 기법 비교 (15분)

**목표**: Zero-shot, Few-shot, CoT 세 가지 기법으로 뉴스 기사 감성 분류를 수행하고, 정확도와 토큰 사용량을 측정한다

**핵심 단계**:

① **테스트 데이터 준비**

```python
import os
from openai import OpenAI
from dotenv import load_dotenv

# 환경변수 로드
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 뉴스 기사 샘플 (실제 한국 뉴스 스타일)
test_articles = [
    {
        "title": "삼성전자, 올해 영업이익 최고 기록 달성",
        "text": "삼성전자가 반도체 가격 상승과 수요 회복에 힘입어 사상 최고의 영업이익을 거뒀다. "
                "업계 전문가들은 향후 3개월간 긍정적 모멘텀이 지속될 것으로 예상하고 있다.",
        "true_label": "긍정"
    },
    {
        "title": "경제 성장률 둔화, 전망 하락",
        "text": "한국은행이 올해 경제 성장률을 종전 예측보다 0.3%p 낮춘 2.8%로 수정했다. "
                "수출 부진과 소비 위축이 주요 원인으로 지목되었다.",
        "true_label": "부정"
    },
    {
        "title": "정부, 신산업 투자 정책 발표",
        "text": "정부가 반도체, 바이오, 전기차 등 3개 전략산업에 연 100조 원을 투자한다고 발표했다. "
                "업계에서는 긍정적으로 평가하면서도 구체적 시행 방안을 주목하고 있다.",
        "true_label": "중립"
    },
    {
        "title": "기술 중소기업 부도 급증",
        "text": "금리 인상과 자금 조달 곤란으로 기술 중소기업의 부도가 전년 동기 대비 45% 증가했다. "
                "창업 생태계의 위축 우려가 높아지고 있다.",
        "true_label": "부정"
    },
    {
        "title": "수소 충전소 100개 달성",
        "text": "정부 주도로 건설된 수소 충전소가 100개에 도달했다. "
                "이는 수소 경제 활성화를 위한 인프라 구축의 이정표로 평가되고 있다.",
        "true_label": "긍정"
    },
]

print("=== 프롬프팅 기법별 감성 분류 성능 비교 ===\n")
```

② **Zero-shot 프롬프팅**

```python
def zero_shot_classification(article_text):
    """Zero-shot 프롬프팅으로 감성 분류"""
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
        temperature=0,
        max_tokens=10,
    )
    return response

print("[1] Zero-shot Prompting\n")
zero_shot_results = []
zero_shot_total_input_tokens = 0
zero_shot_total_output_tokens = 0

for i, article in enumerate(test_articles, 1):
    article_text = f"제목: {article['title']}\n본문: {article['text']}"
    response = zero_shot_classification(article_text)

    prediction = response.choices[0].message.content.strip()
    zero_shot_results.append(prediction)
    zero_shot_total_input_tokens += response.usage.prompt_tokens
    zero_shot_total_output_tokens += response.usage.completion_tokens

    match = "✓" if prediction == article['true_label'] else "✗"
    print(f"기사 {i}: {prediction:5} (정답: {article['true_label']:5}) [{match}]")
    print(f"  입력토큰: {response.usage.prompt_tokens}, 출력토큰: {response.usage.completion_tokens}")

zero_shot_accuracy = sum(1 for result, article in zip(zero_shot_results, test_articles)
                         if result == article['true_label']) / len(test_articles)
print(f"\n총 정확도: {zero_shot_accuracy:.1%}")
print(f"총 토큰: 입력 {zero_shot_total_input_tokens}, 출력 {zero_shot_total_output_tokens}\n")
```

예상 결과:
```
[1] Zero-shot Prompting

기사 1: 긍정   (정답: 긍정  ) [✓]
  입력토큰: 42, 출력토큰: 1
기사 2: 부정   (정답: 부정  ) [✓]
  입력토큰: 42, 출력토큰: 1
기사 3: 중립   (정답: 중립  ) [✗] (실제로는 "긍정"이라고 분류할 수 있음)
  입력토큰: 41, 출력토큰: 1
기사 4: 부정   (정답: 부정  ) [✓]
  입력토큰: 41, 출력토큰: 1
기사 5: 긍정   (정답: 긍정  ) [✓]
  입력토큰: 41, 출력토큰: 1

총 정확도: 80.0%
총 토큰: 입력 207, 출력 5
```

③ **Few-shot 프롬프팅**

```python
def few_shot_classification(article_text):
    """Few-shot 프롬프팅으로 감성 분류"""
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

    match = "✓" if prediction == article['true_label'] else "✗"
    print(f"기사 {i}: {prediction:5} (정답: {article['true_label']:5}) [{match}]")
    print(f"  입력토큰: {response.usage.prompt_tokens}, 출력토큰: {response.usage.completion_tokens}")

few_shot_accuracy = sum(1 for result, article in zip(few_shot_results, test_articles)
                        if result == article['true_label']) / len(test_articles)
print(f"\n총 정확도: {few_shot_accuracy:.1%}")
print(f"총 토큰: 입력 {few_shot_total_input_tokens}, 출력 {few_shot_total_output_tokens}\n")
```

예상 결과:
```
[2] Few-shot Prompting

기사 1: 긍정   (정답: 긍정  ) [✓]
  입력토큰: 143, 출력토큰: 1
기사 2: 부정   (정답: 부정  ) [✓]
  입력토큰: 143, 출력토큰: 1
기사 3: 중립   (정답: 중립  ) [✓]
  입력토큰: 145, 출력토큰: 1
기사 4: 부정   (정답: 부정  ) [✓]
  입력토큰: 141, 출력토큰: 1
기사 5: 긍정   (정답: 긍정  ) [✓]
  입력토큰: 140, 출력토큰: 1

총 정확도: 100.0%
총 토큰: 입력 712, 출력 5
```

④ **Chain-of-Thought (CoT) 프롬프팅**

```python
def cot_classification(article_text):
    """CoT 프롬프팅으로 감성 분류 (단계별 추론)"""
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
        max_tokens=150,
    )
    return response

print("[3] Chain-of-Thought (CoT) Prompting\n")
cot_results = []
cot_total_input_tokens = 0
cot_total_output_tokens = 0

for i, article in enumerate(test_articles, 1):
    article_text = f"제목: {article['title']}\n본문: {article['text']}"
    response = cot_classification(article_text)

    # 응답에서 마지막 줄의 감성 추출
    response_text = response.choices[0].message.content.strip()
    # "감성: 긍정" 형식으로 끝나도록 가정
    if "감성:" in response_text:
        prediction = response_text.split("감성:")[-1].strip().split("\n")[0].strip()
    else:
        prediction = "중립"  # 기본값

    cot_results.append(prediction)
    cot_total_input_tokens += response.usage.prompt_tokens
    cot_total_output_tokens += response.usage.completion_tokens

    match = "✓" if prediction == article['true_label'] else "✗"
    print(f"기사 {i}: {prediction:5} (정답: {article['true_label']:5}) [{match}]")
    print(f"  입력토큰: {response.usage.prompt_tokens}, 출력토큰: {response.usage.completion_tokens}")
    print(f"  분석: {response_text[:100]}...\n")

cot_accuracy = sum(1 for result, article in zip(cot_results, test_articles)
                   if result == article['true_label']) / len(test_articles)
print(f"\n총 정확도: {cot_accuracy:.1%}")
print(f"총 토큰: 입력 {cot_total_input_tokens}, 출력 {cot_total_output_tokens}\n")
```

예상 결과:
```
[3] Chain-of-Thought (CoT) Prompting

기사 1: 긍정   (정답: 긍정  ) [✓]
  입력토큰: 76, 출력토큰: 89
  분석: 1. 주요 키워드: "최고 기록", "수요 회복", "긍정적 모멘텀"
기사 2: 부정   (정답: 부정  ) [✓]
  입력토큰: 76, 출력토큰: 75
  분석: 1. 주요 키워드: "둔화", "하락", "악화", "부진"
기사 3: 중립   (정답: 중립  ) [✓]
  입력토큰: 75, 출력토큰: 81
  분석: 1. 주요 키워드: "긍정적", "우려", "엇갈려"...
기사 4: 부정   (정답: 부정  ) [✓]
  입력토큰: 73, 출력토큰: 72
  분석: 1. 주요 키워드: "부도 급증", "곤란", "위축"...
기사 5: 긍정   (정답: 긍정  ) [✓]
  입력토큰: 74, 출력토큰: 78
  분석: 1. 주요 키워드: "달성", "활성화", "이정표"...

총 정확도: 100.0%
총 정확도: 100.0%
총 토큰: 입력 374, 출력 395
```

⑤ **성능 및 비용 비교 분석**

```python
print("\n" + "="*60)
print("성능 및 비용 종합 비교")
print("="*60 + "\n")

# 정확도 비교
print("정확도 비교:")
print(f"  Zero-shot:  {zero_shot_accuracy:.1%}")
print(f"  Few-shot:   {few_shot_accuracy:.1%}")
print(f"  CoT:        {cot_accuracy:.1%}")

# 토큰 사용량 비교
print("\n토큰 사용량 비교:")
print(f"  Zero-shot:  입력 {zero_shot_total_input_tokens:3d}, 출력 {zero_shot_total_output_tokens:3d}, 총 {zero_shot_total_input_tokens + zero_shot_total_output_tokens:3d}")
print(f"  Few-shot:   입력 {few_shot_total_input_tokens:3d}, 출력 {few_shot_total_output_tokens:3d}, 총 {few_shot_total_input_tokens + few_shot_total_output_tokens:3d}")
print(f"  CoT:        입력 {cot_total_input_tokens:3d}, 출력 {cot_total_output_tokens:3d}, 총 {cot_total_input_tokens + cot_total_output_tokens:3d}")

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

print("\n비용 추정 (GPT-4o 기준, 원화):")
print(f"  Zero-shot:  {zero_shot_cost:7.2f}원 (기준)")
print(f"  Few-shot:   {few_shot_cost:7.2f}원 ({few_shot_cost/zero_shot_cost:.1f}배)")
print(f"  CoT:        {cot_cost:7.2f}원 ({cot_cost/zero_shot_cost:.1f}배)")

# 정확도 대비 비용 효율성
print("\n정확도 대비 비용 효율성 (정확도 1% 향상당 추가 비용):")
zero_to_few_efficiency = (few_shot_cost - zero_shot_cost) / (few_shot_accuracy - zero_shot_accuracy) if few_shot_accuracy > zero_shot_accuracy else 0
zero_to_cot_efficiency = (cot_cost - zero_shot_cost) / (cot_accuracy - zero_shot_accuracy) if cot_accuracy > zero_shot_accuracy else 0
print(f"  Zero-shot → Few-shot: {zero_to_few_efficiency:.1f}원/1% 향상")
print(f"  Zero-shot → CoT:      {zero_to_cot_efficiency:.1f}원/1% 향상 (또는 추론 상세도가 높음)")

print("\n결론:")
if few_shot_accuracy >= zero_shot_accuracy and few_shot_cost <= zero_shot_cost * 1.5:
    print("  ✓ Few-shot이 정확도와 비용의 최적 균형")
elif cot_accuracy >= zero_shot_accuracy:
    print("  ✓ CoT는 가장 높은 정확도를 제공 (비용 대비)")
else:
    print("  ✓ Zero-shot도 충분한 정확도 제공")
```

예상 결과:
```
============================================================
성능 및 비용 종합 비교
============================================================

정확도 비교:
  Zero-shot:  80.0%
  Few-shot:   100.0%
  CoT:        100.0%

토큰 사용량 비교:
  Zero-shot:  입력 207, 출력   5, 총 212
  Few-shot:   입력 712, 출력   5, 총 717
  CoT:        입력 374, 출력 395, 총 769

비용 추정 (GPT-4o 기준, 원화):
  Zero-shot:    3.54원 (기준)
  Few-shot:    12.03원 (3.4배)
  CoT:         13.27원 (3.7배)

정확도 대비 비용 효율성 (정확도 1% 향상당 추가 비용):
  Zero-shot → Few-shot: 0.45원/1% 향상
  Zero-shot → CoT:      0.47원/1% 향상 (또는 추론 상세도가 높음)

결론:
  ✓ Few-shot이 정확도와 비용의 최적 균형
```

**검증 체크리스트**:
- [ ] 세 가지 프롬프팅 기법이 모두 구현되었는가?
- [ ] API 응답에서 토큰 수를 정확히 추출했는가?
- [ ] 각 기법의 정확도가 측정되었는가?
- [ ] 비용 계산이 현재 요율(GPT-4o)을 반영했는가?
- [ ] 결과 비교 그래프나 표가 명확한가?

**Copilot 프롬프트 1**:
```
"OpenAI API로 뉴스 기사의 감성을 분류하는 코드를 작성해줄 수 있어?
Zero-shot, Few-shot, CoT 세 가지 프롬프팅 기법을 모두 구현해야 하고,
각 기법별로 정확도와 토큰 사용량을 측정해야 해."
```

**Copilot 프롬프트 2**:
```
"이 코드에서 세 기법의 정확도를 계산하고 비교 표를 만들어줄 수 있어?
토큰 사용량도 비교하고, GPT-4o 기준(입력 $2.50/M, 출력 $10/M)으로 비용을 계산해줘."
```

---

#### 체크포인트 2: Function Calling으로 도구 연동 (15분)

**목표**: Function Calling을 사용하여 검색 도구와 날씨 도구를 구현하고, 뉴스 기사를 분석할 때 필요한 외부 정보를 동적으로 조회한다

**핵심 단계**:

① **도구 정의 및 함수 구현**

```python
import json
from typing import Optional

# 검색 도구 정의
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
                    "description": "지역명 (예: '서울', '부산', '서울시강남구')"
                }
            },
            "required": ["location"]
        }
    }
}

# 실제 도구 함수 구현 (실제로는 API를 호출하지만, 여기서는 시뮬레이션)
def search_news(query: str, count: int = 3) -> dict:
    """검색 도구 구현 (시뮬레이션)"""
    # 실제로는 Google News API, Naver News API 등을 호출
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
    """날씨 조회 도구 구현 (시뮬레이션)"""
    # 실제로는 OpenWeatherMap, 기상청 API 등을 호출
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

print("[Function Calling 실습]\n")
```

② **LLM이 도구를 호출하는 4단계 흐름**

```python
def function_calling_demo():
    """Function Calling 4단계 흐름 데모"""

    # 사용자 질문
    user_query = "삼성전자 실적과 현재 서울 날씨를 알려줄 수 있어?"
    print(f"사용자: {user_query}\n")

    # 단계 1: 도구 정의와 함께 요청 전송
    print("─" * 60)
    print("단계 1: 도구 정의와 함께 LLM 호출")
    print("─" * 60)

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": user_query
            }
        ],
        tools=[search_tool, weather_tool],
        tool_choice="auto",
    )

    print(f"LLM 응답: {response.choices[0].message}")

    # 단계 2: LLM의 도구 호출 결정 확인
    print("\n" + "─" * 60)
    print("단계 2: LLM이 호출할 도구 결정")
    print("─" * 60)

    tool_calls = response.choices[0].message.tool_calls
    if not tool_calls:
        print("LLM이 도구를 호출할 필요가 없다고 판단했습니다.")
        return

    for i, tool_call in enumerate(tool_calls, 1):
        print(f"\n도구 호출 {i}:")
        print(f"  함수명: {tool_call.function.name}")
        print(f"  인자: {tool_call.function.arguments}")

    # 단계 3: 실제 도구 함수 실행
    print("\n" + "─" * 60)
    print("단계 3: 실제 도구 함수 실행")
    print("─" * 60)

    tool_results = []
    for tool_call in tool_calls:
        tool_name = tool_call.function.name
        tool_input = json.loads(tool_call.function.arguments)

        print(f"\n실행: {tool_name}({tool_input})")

        if tool_name == "search_news":
            result = search_news(**tool_input)
        elif tool_name == "get_weather":
            result = get_weather(**tool_input)
        else:
            result = {"error": f"Unknown tool: {tool_name}"}

        print(f"결과: {json.dumps(result, ensure_ascii=False, indent=2)}")

        tool_results.append({
            "tool_call_id": tool_call.id,
            "tool_name": tool_name,
            "result": result
        })

    # 단계 4: 결과를 LLM에 다시 전달하여 최종 응답 생성
    print("\n" + "─" * 60)
    print("단계 4: 도구 실행 결과를 LLM에 전달 후 최종 응답 생성")
    print("─" * 60)

    messages = [
        {"role": "user", "content": user_query},
        response.choices[0].message,
    ]

    for tool_result in tool_results:
        messages.append({
            "role": "tool",
            "tool_call_id": tool_result["tool_call_id"],
            "content": json.dumps(tool_result["result"], ensure_ascii=False)
        })

    final_response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
    )

    print(f"\n최종 응답:\n{final_response.choices[0].message.content}")

function_calling_demo()
```

예상 결과:
```
사용자: 삼성전자 실적과 현재 서울 날씨를 알려줄 수 있어?

────────────────────────────────────────────────────────────
단계 1: 도구 정의와 함께 LLM 호출
────────────────────────────────────────────────────────────

────────────────────────────────────────────────────────────
단계 2: LLM이 호출할 도구 결정
────────────────────────────────────────────────────────────

도구 호출 1:
  함수명: search_news
  인자: {"query": "삼성전자 실적"}

도구 호출 2:
  함수명: get_weather
  인자: {"location": "서울"}

────────────────────────────────────────────────────────────
단계 3: 실제 도구 함수 실행
────────────────────────────────────────────────────────────

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

────────────────────────────────────────────────────────────
단계 4: 도구 실행 결과를 LLM에 전달 후 최종 응답 생성
────────────────────────────────────────────────────────────

최종 응답:
삼성전자의 최근 실적은 긍정적입니다. 삼성전자가 영업이익으로 사상 최고를
기록했으며, 반도체 생산도 증대되고 있는 추세입니다.

현재 서울의 날씨는 맑고, 기온은 3°C로 꽤 쌀쌀합니다. 습도는 45%로 쾌적한 수준입니다.
```

③ **Multiple Tool Calls 처리**

```python
def multi_tool_calling_test():
    """여러 도구를 동시에 호출하는 경우 테스트"""

    test_queries = [
        "광주와 대구의 날씨를 비교해줄래?",
        "반도체 뉴스 검색하고 서울 날씨도 알려줄래?",
        "경제 성장률에 대한 뉴스를 찾고, 부산 날씨도 확인해줘",
    ]

    for query in test_queries:
        print(f"\n질문: {query}")
        print("─" * 60)

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": query}],
            tools=[search_tool, weather_tool],
            tool_choice="auto",
        )

        tool_calls = response.choices[0].message.tool_calls
        print(f"호출된 도구 수: {len(tool_calls) if tool_calls else 0}")

        for i, tc in enumerate(tool_calls or [], 1):
            print(f"  도구 {i}: {tc.function.name} (인자: {tc.function.arguments})")

multi_tool_calling_test()
```

**검증 체크리스트**:
- [ ] 두 도구(search_news, get_weather) 모두 JSON으로 정의되었는가?
- [ ] LLM이 자동으로 어떤 도구를 호출할지 판단하는가?
- [ ] 도구 호출 ID가 올바르게 추출되었는가?
- [ ] 실행 결과가 tool 메시지로 올바르게 전달되었는가?
- [ ] 최종 응답이 도구 결과를 자연어로 잘 통합했는가?

**Copilot 프롬프트 3**:
```
"OpenAI API의 Function Calling으로 검색 도구와 날씨 도구를 구현해줄 수 있어?
LLM이 필요한 도구를 자동으로 선택하도록 tool_choice='auto'를 사용해야 해.
4단계 흐름(도구 정의 → LLM 호출 → 함수 실행 → 결과 전달)을 명확히 보여줄래?"
```

**Copilot 프롬프트 4**:
```
"위의 Function Calling 코드에서 여러 도구를 동시에 호출하는 경우를 처리하는 코드를 추가해줄 수 있어?
tool_calls가 리스트로 반환될 수 있으므로, 각 호출을 반복해서 실행해야 해."
```

---

#### 체크포인트 3: Structured Output + 자동 평가 (10분)

**목표**: Pydantic 모델을 사용하여 뉴스 기사 분석 결과를 구조화하고, LLM-as-a-Judge로 자동 평가한다

**핵심 단계**:

① **Pydantic 모델 정의 및 Structured Output**

```python
from pydantic import BaseModel, Field
from typing import List

class NewsAnalysis(BaseModel):
    """뉴스 기사 분석 결과"""
    title: str = Field(description="기사 제목")
    sentiment: str = Field(description="감성 분류 (긍정/중립/부정)")
    key_topics: List[str] = Field(description="주요 주제 (예: 경제, 기술, 정책)")
    economic_impact: str = Field(description="경제적 영향 (긍정/중립/부정)")
    confidence_score: float = Field(description="분류 신뢰도 (0.0~1.0)")
    summary: str = Field(description="한 문장 요약 (20단어 이내)")

print("[Structured Output 실습]\n")

def analyze_article_structured(article_text: str):
    """Structured Output으로 뉴스 분석"""

    response = client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": f"""다음 뉴스 기사를 분석하고, 지정된 형식으로 결과를 제공하세요.

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
        response_format=NewsAnalysis,
    )

    return response.choices[0].message.parsed

# 5개 기사 구조화된 분석
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

예상 결과:
```
구조화된 뉴스 분석 결과:

기사 1: 삼성전자, 올해 영업이익 최고 기록 달성
  감성: 긍정
  주제: 경제, 기술, 기업실적
  경제영향: 긍정
  신뢰도: 0.95
  요약: 삼성전자가 반도체 호황에 힘입어 사상 최고 영업이익 기록.

기사 2: 경제 성장률 둔화, 전망 하락
  감성: 부정
  주제: 경제, 정책, 고용
  경제영향: 부정
  신뢰도: 0.92
  요약: 경제 성장률이 예상보다 낮아지고 고용 지표 악화.

기사 3: 정부, 신산업 투자 정책 발표
  감성: 중립
  주제: 정책, 기술, 투자
  경제영향: 긍정
  신뢰도: 0.88
  요약: 정부가 반도체·바이오·전기차에 연 100조 원 투자.

기사 4: 기술 중소기업 부도 급증
  감성: 부정
  주제: 경제, 기술, 중소기업
  경제영향: 부정
  신뢰도: 0.90
  요약: 금리 인상으로 기술 중소기업 부도가 45% 증가.

기사 5: 수소 충전소 100개 달성
  감성: 긍정
  주제: 에너지, 기술, 정책
  경제영향: 긍정
  신뢰도: 0.87
  요약: 정부 수소 충전소 인프라가 100개 달성 이정표.
```

② **LLM-as-a-Judge 자동 평가**

```python
class EvaluationResult(BaseModel):
    """평가 결과"""
    accuracy: int = Field(description="정확도 (0~10)")
    reasoning_quality: int = Field(description="추론 품질 (0~10)")
    conciseness: int = Field(description="간결성 (0~10)")
    overall_score: float = Field(description="종합 점수 (0~10)")
    feedback: str = Field(description="개선 의견 (3문장 이내)")

def evaluate_analysis(article: dict, analysis: NewsAnalysis):
    """LLM-as-a-Judge: 분석 결과 자동 평가"""

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

print("\n" + "="*60)
print("LLM-as-a-Judge 자동 평가")
print("="*60 + "\n")

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
print("─" * 60)
print("평균 점수:")
print(f"  정확도:      {total_scores['accuracy']/num_articles:.1f}/10")
print(f"  추론 품질:   {total_scores['reasoning_quality']/num_articles:.1f}/10")
print(f"  간결성:      {total_scores['conciseness']/num_articles:.1f}/10")
print(f"  종합 점수:   {total_scores['overall_score']/num_articles:.1f}/10")
```

예상 결과:
```
============================================================
LLM-as-a-Judge 자동 평가
============================================================

기사 1 평가:
  정확도:      9/10
  추론 품질:   9/10
  간결성:      8/10
  종합 점수:   8.7/10
  피드백:      분석이 정확합니다. 신뢰도 점수도 합리적입니다.
              주제 분류도 타당합니다.

기사 2 평가:
  정확도:      10/10
  추론 품질:   9/10
  간결성:      9/10
  종합 점수:   9.3/10
  피드백:      부정적 감성을 정확히 포착했습니다.
              경제적 영향 분석도 우수합니다.
              요약이 간결하고 명확합니다.

기사 3 평가:
  정확도:      8/10
  추론 품질:   8/10
  간결성:      8/10
  종합 점수:   8.0/10
  피드백:      중립 감성 판단이 타당합니다.
              정부 정책의 긍정적 면을 경제영향에 반영한 점이 좋습니다.
              주제 분류도 포괄적입니다.

기사 4 평가:
  정확도:      9/10
  추론 품질:   9/10
  간결성:      9/10
  종합 점수:   9.0/10
  피드백:      부정적 신호를 명확히 파악했습니다.
              통계 수치도 올바르게 반영되었습니다.
              실용적인 요약입니다.

기사 5 평가:
  정확도:      9/10
  추론 품질:   8/10
  간결성:      8/10
  종합 점수:   8.3/10
  피드백:      긍정적 감성을 정확히 포착했습니다.
              인프라 구축의 정책적 의미도 잘 이해했습니다.
              에너지 주제 분류가 적절합니다.

────────────────────────────────────────────────────────
평균 점수:
  정확도:      9.0/10
  추론 품질:   8.6/10
  간결성:      8.4/10
  종합 점수:   8.7/10
```

③ **최종 종합 리포트 생성**

```python
print("\n" + "="*60)
print("최종 종합 분석 리포트")
print("="*60 + "\n")

print("## 1. 프롬프팅 기법별 성능 비교\n")
print(f"정확도 순위:")
print(f"  1위: Few-shot, CoT (동점) - 100.0%")
print(f"  3위: Zero-shot - 80.0%")
print(f"\n토큰 효율성 순위 (정확도 1% 향상당 비용):")
print(f"  1위: Few-shot - {zero_to_few_efficiency:.1f}원/1%")
print(f"  2위: CoT - {zero_to_cot_efficiency:.1f}원/1%")
print(f"  (Zero-shot은 기준)")

print("\n## 2. Function Calling 활용 결과\n")
print("✓ 검색 도구(search_news)와 날씨 도구(get_weather) 모두 정상 작동")
print("✓ LLM이 필요한 도구를 자동으로 선택")
print("✓ 복합 질의(여러 도구 동시 호출)도 지원")

print("\n## 3. Structured Output + 자동 평가\n")
print(f"분석 품질 평가 (LLM-as-a-Judge):")
print(f"  정확도:      {total_scores['accuracy']/num_articles:.1f}/10 (우수)")
print(f"  추론 품질:   {total_scores['reasoning_quality']/num_articles:.1f}/10 (우수)")
print(f"  간결성:      {total_scores['conciseness']/num_articles:.1f}/10 (양호)")
print(f"  종합 점수:   {total_scores['overall_score']/num_articles:.1f}/10 (우수)")

print("\n## 4. 권장 사항\n")
print("실무 적용 전략:")
print("  - 간단한 분류 과제 → Few-shot (비용 효율적, 100% 정확도)")
print("  - 복잡한 추론 필요 → CoT (약간의 비용 증가, 상세 근거 제공)")
print("  - 실시간 정보 필요 → Function Calling 연동 (검색, 날씨, DB 조회)")
print("  - 구조화된 출력 필수 → Structured Output (파싱 오류 방지)")
print("  - 품질 보증 필요 → LLM-as-a-Judge (자동 평가, 사람 평가 전 필터)")
```

**검증 체크리스트**:
- [ ] Pydantic 모델이 모든 필드를 포함하고 있는가?
- [ ] response_format=NewsAnalysis 파라미터가 올바르게 전달되었는가?
- [ ] Structured Output이 항상 유효한 JSON을 반환하는가?
- [ ] LLM-as-a-Judge 평가가 객관적이고 일관성 있는가?
- [ ] 종합 리포트가 모든 체크포인트의 결과를 정리했는가?

**Copilot 프롬프트 5**:
```
"Pydantic 모델을 이용해서 OpenAI API의 response_format으로 뉴스 분석 결과를
구조화해줄 수 있어? NewsAnalysis 모델이 title, sentiment, key_topics,
economic_impact, confidence_score, summary를 포함해야 해."
```

**Copilot 프롬프트 6**:
```
"위의 분석 결과를 LLM-as-a-Judge로 평가하는 코드를 만들어줄 수 있어?
정확도, 추론 품질, 간결성을 0~10 점수로 평가하고, 피드백을 3문장 이내로 제공해야 해."
```

---

### 제출 안내 (Google Classroom)

**제출 방법**:
- Google Classroom의 "6주차 B회차" 과제에 조별 1부 제출
- 파일명 형식: `group_{조번호}_ch6B.zip`

**포함할 파일**:
```
group_{조번호}_ch6B/
├── ch6B_실습.py              # 전체 구현 코드
│                            (체크포인트 1-3 모두 포함)
├── 성능비교_리포트.md         # 프롬팅 기법별 정확도/토큰 비교
│                            (Zero-shot, Few-shot, CoT)
├── function_calling_log.txt  # Function Calling 실행 결과
│                            (호출된 도구, 인자, 결과)
├── 평가결과분석.md            # LLM-as-a-Judge 평가 결과
│                            (각 기법의 장단점, 실무 적용 방안)
└── requirements.txt          # 필요 패키지 목록
                            (openai, anthropic, pydantic, dotenv)
```

**리포트 포함 항목** (성능비교_리포트.md):
- 세 가지 프롬프팅 기법(Zero-shot, Few-shot, CoT)의 정확도 비교표 (3-4문장)
- 토큰 사용량 및 비용 분석 (2-3문장)
- 정확도 대비 비용 효율성 분석 (2-3문장)
- 각 기법의 장단점 정리 (각 기법별 2-3문장)
- 실무 적용 시 권장 기법과 이유 (3-4문장)

**평가결과분석.md 포함 항목**:
- Function Calling 동작 원리 및 단계별 결과 (3-4문장)
- Structured Output의 장점과 구현 결과 (2-3문장)
- LLM-as-a-Judge 평가 결과 해석 (3-4문장)
- 세 가지 기법의 조합 활용 방안 (3-4문장)
- Copilot 활용 경험담: 어떤 프롬프트가 효과적이었나 (2-3문장)

**제출 마감**: 수업 종료 후 24시간 이내

---

### 결과 토론 가이드

> **토론 가이드**: 조별 구현 결과를 공유하며, 프롬프팅 기법별 성능 차이를 분석하고, Function Calling의 실무 가치를 논의한다

**토론 주제**:

① **프롬프팅 기법의 트레이드오프**
- 왜 Zero-shot은 빠르지만 덜 정확한가?
- Few-shot이 정확도와 비용의 "스위트 스팟"인 이유는?
- CoT는 정확하지만 왜 많은 토큰을 소비하는가?
- 실제 서비스에서는 어떤 기법을 선택할까? (응답 속도 vs 정확도)

② **토큰 과금의 현실성**
- 입력과 출력의 토큰 가격이 다른 이유는? (출력이 4배 비쌈)
- 한국어 서비스의 비용이 영어보다 높은 이유는?
- 월간 API 비용을 예측하고 예산 운영하는 방법은?

③ **Function Calling의 잠재력**
- 검색 엔진, 날씨 API, 데이터베이스를 LLM과 연동할 때의 장점은?
- "LLM이 도구를 직접 실행하지 않는다"는 설계의 의미는?
- AI Agent와 Function Calling의 관계는? (12주차 선행 학습)

④ **Structured Output의 신뢰성**
- 자유 형식 출력 vs 강제된 구조화 출력의 차이점은?
- Pydantic의 자동 검증이 왜 중요한가?
- 파싱 오류를 제거하는 것이 프로덕션에서 얼마나 중요한가?

⑤ **LLM-as-a-Judge의 한계와 활용**
- 같은 모델이 자신의 출력에 높은 점수를 주는 "자기 편향"을 어떻게 완화할까?
- 여러 모델(GPT-4o, Claude, Gemini)을 교차 평가자로 쓰는 이유는?
- 사람의 평가를 완전히 대체할 수 있을까?

**발표 형식**:
- 각 조 3~5분 발표 (프롬프팅 기법 비교 결과 + Function Calling 경험)
- 다른 조의 질문에 답변 (2~3개 질문)
- 교수의 보충 설명 및 피드백

---

### 교수 피드백 포인트

**강화할 점**:
- 프롬프팅은 "기술"이 아니라 **LLM과의 소통 기술**임을 강조한다. 명확한 지시, 구체적인 예시, 원하는 형식을 명시하는 것이 핵심이다.
- Few-shot의 선택지가 결과 품질에 큰 영향을 미친다는 점을 강조한다. 대표적이고 다양한 예시를 선택해야 한다.
- CoT는 단순히 "정확도를 올린다"는 데 그치지 않고, **모델의 추론 과정을 투명하게 한다**는 점을 강조한다. 이것이 AI 신뢰성의 기초다.
- Function Calling은 "LLM에 팔다리를 달아주는 기술"이며, 이것이 AI Agent의 기초임을 명확히 한다.

**주의할 점**:
- "프롬프팅 하나로 모든 문제가 해결된다"는 오해를 방지한다. 결국 **좋은 데이터, 적절한 모델 크기, 충분한 평가**가 필요하다.
- LLM-as-a-Judge의 편향성(자기 편향, 위치 편향, 장문 편향)을 인지하도록 한다. 평가는 항상 불완전하며, 여러 평가자의 합의가 중요하다.
- API 비용의 누적 효과를 강조한다. 작은 요청 하나는 싸지만, 하루 수천 개 요청하면 월간 예산을 초과할 수 있다.

**다음 학습으로의 연결**:
- 7주차는 **중간고사**이므로, 1~6주차 전체 내용(기초 개념, 아키텍처, 임베딩, Attention, Transformer, API/프롬프팅)을 종합적으로 이해해야 한다.
- 8주차부터는 **실무 기술**인 토픽 모델링, RAG, Fine-tuning, 평가 등을 다룬다. 6주차의 API와 프롬프팅은 이 모든 기술의 기초가 된다.
- 12주차의 **AI Agent**는 Function Calling을 바탕으로 여러 도구를 자율적으로 조합하는 고급 주제이므로, 이 주차의 Function Calling 개념을 철저히 이해해야 한다.

---

### 다음 주 예고

다음 주 7주차는 **중간고사**이다.

**시험 범위**: 1~6주차 전체
- 1주차: NLP의 기본 개념과 BERT 구조
- 2주차: Word Embedding과 Word2Vec 원리
- 3주차: Self-Attention과 Multi-Head Attention
- 4주차: Transformer 아키텍처와 구현
- 5주차: BERT 사전 학습과 Fine-tuning
- 6주차: LLM API와 프롬프트 엔지니어링

**시험 형식**: 90분, 객관식 60% + 주관식 40%
- 객관식: 개념 이해, 계산 문제
- 주관식: 프롬프팅 작성, 간단한 코드 분석

**시험 중 참고 자료**:
- API Key 없음 (코드 작성만)
- Copilot, 인터넷 접근 금지
- 계산기 사용 가능

**사전 준비**:
- 지난 6주간 배운 이론 복습
- 실습 코드 다시 읽기 (특히 프롬프팅 기법과 Function Calling)
- A회차 "Exit ticket" 문항들 풀이

---

## 참고 자료

**실습 코드**:
- _전체 구현 코드는 practice/chapter6/code/6-4-실습.py 참고_

**권장 읽기**:
- OpenAI. Chat Completions API Reference. https://platform.openai.com/docs/api-reference/chat/create
- Anthropic. API Documentation. https://docs.anthropic.com/claude/reference
- Lilian Weng. (2023). Prompt Engineering. *Lil'Log*. https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/
- Wang, X., et al. (2023). Self-Consistency Improves Chain of Thought Reasoning in Language Models. *ICLR*. https://arxiv.org/abs/2203.11171
