"""
6장 실습 코드 6-3: Structured Output과 Function Calling
- JSON Mode / Structured Output (Pydantic)
- Function Calling 구현 (날씨 조회)
- 출력 파싱 및 검증

실행 방법:
    pip install openai anthropic python-dotenv pydantic
    python 6-3-function-calling.py

API Key 설정:
    .env 파일에 OPENAI_API_KEY 설정
    키가 없으면 캐시된 응답으로 자동 폴백
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

CACHE_PATH = Path(__file__).parent.parent / "data" / "output" / "cached_responses.json"
CACHED = {}
if CACHE_PATH.exists():
    with open(CACHE_PATH, "r", encoding="utf-8") as f:
        CACHED = json.load(f)


def has_openai_key():
    """OpenAI API 키가 설정되어 있는지 확인한다."""
    return bool(os.getenv("OPENAI_API_KEY"))


# ──────────────────────────────────────────────
# Pydantic 모델 정의
# ──────────────────────────────────────────────
class NewsExtraction(BaseModel):
    """뉴스 기사에서 추출할 구조화된 정보."""
    company: str = Field(description="기업명")
    period: str = Field(description="실적 기간")
    revenue: str = Field(description="매출액")
    growth_rate: str = Field(description="성장률")
    comparison: str = Field(description="비교 기준")


class SentimentResult(BaseModel):
    """감성 분석 결과."""
    sentiment: str = Field(description="긍정/부정/중립")
    confidence: float = Field(description="신뢰도 (0~1)")
    key_phrases: list[str] = Field(description="핵심 표현")
    reasoning: str = Field(description="판단 근거")


# ──────────────────────────────────────────────
# 1. Structured Output (Pydantic)
# ──────────────────────────────────────────────
def structured_output_demo():
    """Pydantic 모델을 사용한 Structured Output 예제."""
    print("=" * 60)
    print("1. Structured Output (Pydantic)")
    print("=" * 60)

    news = (
        "삼성전자가 2024년 3분기 매출 79조원을 기록하며 "
        "전년 동기 대비 17% 성장했다고 발표했다."
    )
    print(f"입력 텍스트: {news}\n")

    if has_openai_key():
        from openai import OpenAI
        client = OpenAI()

        completion = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "뉴스 기사에서 핵심 정보를 추출합니다."},
                {"role": "user", "content": news},
            ],
            response_format=NewsExtraction,
        )

        result = completion.choices[0].message.parsed
        print(f"추출 결과 (Pydantic 객체):")
        print(f"  기업명: {result.company}")
        print(f"  기간: {result.period}")
        print(f"  매출: {result.revenue}")
        print(f"  성장률: {result.growth_rate}")
        print(f"  비교 기준: {result.comparison}")
        print(f"\nJSON 출력:")
        print(f"  {result.model_dump_json(indent=2)}")
    else:
        print("[캐시 모드] OPENAI_API_KEY가 없어 캐시된 응답을 사용합니다.")
        data = CACHED.get("structured_output", {}).get("response", {})
        result = NewsExtraction(**data) if data else None
        if result:
            print(f"추출 결과 (Pydantic 객체):")
            print(f"  기업명: {result.company}")
            print(f"  기간: {result.period}")
            print(f"  매출: {result.revenue}")
            print(f"  성장률: {result.growth_rate}")
            print(f"  비교 기준: {result.comparison}")
            print(f"\nJSON 출력:")
            print(f"  {result.model_dump_json(indent=2)}")


# ──────────────────────────────────────────────
# 2. Function Calling (도구 연동)
# ──────────────────────────────────────────────

# 가상의 날씨 API (실제 서비스를 대체)
def get_weather(location: str, unit: str = "celsius") -> dict:
    """가상 날씨 정보를 반환한다 (실습용)."""
    weather_data = {
        "서울": {"temperature": 3, "condition": "맑음", "humidity": 45},
        "부산": {"temperature": 8, "condition": "구름 많음", "humidity": 62},
        "제주": {"temperature": 10, "condition": "비", "humidity": 80},
    }
    data = weather_data.get(location, {"temperature": 15, "condition": "맑음", "humidity": 50})
    data["location"] = location
    data["unit"] = unit
    return data


# OpenAI Function Calling용 도구 정의
WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "지정된 도시의 현재 날씨 정보를 조회한다",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "도시명 (예: 서울, 부산, 제주)",
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "온도 단위",
                },
            },
            "required": ["location"],
            "additionalProperties": False,
        },
    },
}


def function_calling_demo():
    """Function Calling 4단계 흐름을 시연한다."""
    print("\n" + "=" * 60)
    print("2. Function Calling (4단계 흐름)")
    print("=" * 60)

    user_message = "서울 날씨 알려줘"
    print(f"사용자 입력: {user_message}\n")

    if has_openai_key():
        from openai import OpenAI
        client = OpenAI()

        # 1단계: 사용자 메시지 + 도구 정의 전송
        print("1단계: LLM에게 메시지 + 도구 정보 전송")
        messages = [{"role": "user", "content": user_message}]
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=[WEATHER_TOOL],
        )

        choice = response.choices[0]
        print(f"  finish_reason: {choice.finish_reason}")

        if choice.finish_reason == "tool_calls":
            tool_call = choice.message.tool_calls[0]
            args = json.loads(tool_call.function.arguments)

            # 2단계: LLM이 도구 호출을 요청
            print(f"\n2단계: LLM이 도구 호출 요청")
            print(f"  함수: {tool_call.function.name}")
            print(f"  인자: {args}")

            # 3단계: 로컬에서 도구 실행
            result = get_weather(**args)
            print(f"\n3단계: 도구 실행 결과")
            print(f"  {json.dumps(result, ensure_ascii=False)}")

            # 4단계: 결과를 LLM에게 전달 → 최종 응답
            messages.append(choice.message)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result, ensure_ascii=False),
            })

            final = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                tools=[WEATHER_TOOL],
            )
            print(f"\n4단계: LLM 최종 응답")
            print(f"  {final.choices[0].message.content}")
    else:
        print("[캐시 모드] OPENAI_API_KEY가 없어 캐시된 응답을 사용합니다.\n")
        fc_data = CACHED.get("function_calling", {})

        print("1단계: LLM에게 메시지 + 도구 정보 전송")
        print("  finish_reason: tool_calls")

        tool_call = fc_data.get("tool_call", {})
        print(f"\n2단계: LLM이 도구 호출 요청")
        print(f"  함수: {tool_call.get('name', 'get_weather')}")
        print(f"  인자: {tool_call.get('arguments', {})}")

        args = tool_call.get("arguments", {"location": "서울"})
        result = get_weather(**args)
        print(f"\n3단계: 도구 실행 결과")
        print(f"  {json.dumps(result, ensure_ascii=False)}")

        print(f"\n4단계: LLM 최종 응답")
        print(f"  {fc_data.get('final_response', '(캐시 없음)')}")


# ──────────────────────────────────────────────
# 3. Pydantic 검증 활용
# ──────────────────────────────────────────────
def pydantic_validation_demo():
    """Pydantic으로 LLM 출력을 검증하는 패턴을 시연한다."""
    print("\n" + "=" * 60)
    print("3. Pydantic 검증 패턴")
    print("=" * 60)

    # 정상적인 데이터
    valid_data = {
        "sentiment": "긍정",
        "confidence": 0.92,
        "key_phrases": ["친절하고", "아늑해서", "다시 오고 싶어요"],
        "reasoning": "긍정적 경험을 표현하는 어휘가 다수 포함되어 있다.",
    }

    try:
        result = SentimentResult(**valid_data)
        print(f"검증 성공:")
        print(f"  감성: {result.sentiment}")
        print(f"  신뢰도: {result.confidence}")
        print(f"  핵심 표현: {result.key_phrases}")
        print(f"  판단 근거: {result.reasoning}")
    except Exception as e:
        print(f"검증 실패: {e}")

    # 비정상적인 데이터 (confidence가 문자열)
    print("\n잘못된 데이터 검증 테스트:")
    invalid_data = {
        "sentiment": "긍정",
        "confidence": "높음",  # float가 아닌 문자열
        "key_phrases": "좋아요",  # 리스트가 아닌 문자열
        "reasoning": "좋은 리뷰입니다.",
    }

    try:
        result = SentimentResult(**invalid_data)
        print(f"  검증 성공 (자동 변환됨): confidence={result.confidence}")
    except Exception as e:
        print(f"  검증 실패: {type(e).__name__}")
        print(f"  → Pydantic이 타입 불일치를 자동으로 잡아준다")


# ──────────────────────────────────────────────
# 메인 실행
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("제6장 실습 6-3: Structured Output과 Function Calling\n")

    structured_output_demo()
    function_calling_demo()
    pydantic_validation_demo()

    print("\n" + "=" * 60)
    print("실습 완료!")
    print("=" * 60)
