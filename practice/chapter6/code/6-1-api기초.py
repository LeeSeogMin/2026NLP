"""
6장 실습 코드 6-1: LLM API 기초
- OpenAI / Anthropic API 기본 호출
- temperature, max_tokens 파라미터 실험
- 토큰 수 계산 및 비용 추정

실행 방법:
    pip install openai anthropic python-dotenv tiktoken
    python 6-1-api기초.py

API Key 설정:
    .env 파일에 OPENAI_API_KEY, ANTHROPIC_API_KEY 설정
    키가 없으면 캐시된 응답으로 자동 폴백
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv

# .env 파일에서 API 키 로드
load_dotenv()

# 캐시된 응답 로드 (API 키 없는 학생을 위한 폴백)
CACHE_PATH = Path(__file__).parent.parent / "data" / "output" / "cached_responses.json"
CACHED = {}
if CACHE_PATH.exists():
    with open(CACHE_PATH, "r", encoding="utf-8") as f:
        CACHED = json.load(f)


def has_openai_key():
    """OpenAI API 키가 설정되어 있는지 확인한다."""
    return bool(os.getenv("OPENAI_API_KEY"))


def has_anthropic_key():
    """Anthropic API 키가 설정되어 있는지 확인한다."""
    return bool(os.getenv("ANTHROPIC_API_KEY"))


# ──────────────────────────────────────────────
# 1. OpenAI API 기본 호출
# ──────────────────────────────────────────────
def openai_basic_chat():
    """OpenAI Chat Completions API 기본 호출 예제."""
    print("=" * 60)
    print("1. OpenAI API 기본 호출")
    print("=" * 60)

    if has_openai_key():
        from openai import OpenAI
        client = OpenAI()

        response = client.chat.completions.create(
            model="gpt-4o-mini",  # 실습에서는 비용 절약을 위해 mini 사용
            messages=[
                {"role": "system", "content": "당신은 NLP 전문가입니다."},
                {"role": "user", "content": "자연어처리가 무엇인지 간단히 설명해주세요."},
            ],
            max_tokens=200,
            temperature=0.7,
        )

        print(f"모델: {response.model}")
        print(f"응답: {response.choices[0].message.content}")
        print(f"\n토큰 사용량:")
        print(f"  입력: {response.usage.prompt_tokens} 토큰")
        print(f"  출력: {response.usage.completion_tokens} 토큰")
        print(f"  합계: {response.usage.total_tokens} 토큰")
    else:
        print("[캐시 모드] OPENAI_API_KEY가 없어 캐시된 응답을 사용합니다.")
        data = CACHED.get("basic_chat_openai", {})
        print(f"모델: {data.get('model', 'gpt-4o')}")
        print(f"응답: {data.get('message', '(캐시 없음)')}")
        usage = data.get("usage", {})
        print(f"\n토큰 사용량:")
        print(f"  입력: {usage.get('prompt_tokens', 0)} 토큰")
        print(f"  출력: {usage.get('completion_tokens', 0)} 토큰")
        print(f"  합계: {usage.get('total_tokens', 0)} 토큰")


# ──────────────────────────────────────────────
# 2. Anthropic API 기본 호출
# ──────────────────────────────────────────────
def anthropic_basic_chat():
    """Anthropic Messages API 기본 호출 예제."""
    print("\n" + "=" * 60)
    print("2. Anthropic (Claude) API 기본 호출")
    print("=" * 60)

    if has_anthropic_key():
        import anthropic
        client = anthropic.Anthropic()

        message = client.messages.create(
            model="claude-haiku-4-5-20251001",  # 실습용 경량 모델
            max_tokens=200,
            system="당신은 NLP 전문가입니다.",
            messages=[
                {"role": "user", "content": "자연어처리가 무엇인지 간단히 설명해주세요."},
            ],
        )

        print(f"모델: {message.model}")
        print(f"응답: {message.content[0].text}")
        print(f"\n토큰 사용량:")
        print(f"  입력: {message.usage.input_tokens} 토큰")
        print(f"  출력: {message.usage.output_tokens} 토큰")
    else:
        print("[캐시 모드] ANTHROPIC_API_KEY가 없어 캐시된 응답을 사용합니다.")
        data = CACHED.get("basic_chat_anthropic", {})
        print(f"모델: {data.get('model', 'claude-sonnet-4-5-20250514')}")
        print(f"응답: {data.get('message', '(캐시 없음)')}")
        usage = data.get("usage", {})
        print(f"\n토큰 사용량:")
        print(f"  입력: {usage.get('input_tokens', 0)} 토큰")
        print(f"  출력: {usage.get('output_tokens', 0)} 토큰")


# ──────────────────────────────────────────────
# 3. OpenAI vs Anthropic API 구조 비교
# ──────────────────────────────────────────────
def compare_api_structure():
    """두 API의 호출 구조 차이를 비교한다."""
    print("\n" + "=" * 60)
    print("3. OpenAI vs Anthropic API 구조 비교")
    print("=" * 60)

    comparison = """
    ┌──────────────────┬─────────────────────┬─────────────────────┐
    │ 항목             │ OpenAI              │ Anthropic           │
    ├──────────────────┼─────────────────────┼─────────────────────┤
    │ 클라이언트       │ OpenAI()            │ Anthropic()         │
    │ 메서드           │ chat.completions    │ messages.create()   │
    │                  │ .create()           │                     │
    │ System 메시지    │ messages 배열 내    │ system= 파라미터    │
    │ max_tokens       │ 선택 (기본값 있음)  │ 필수                │
    │ 토큰 사전 계산   │ tiktoken (로컬)     │ count_tokens (서버) │
    │ 응답 텍스트      │ .choices[0]         │ .content[0].text    │
    │                  │ .message.content    │                     │
    └──────────────────┴─────────────────────┴─────────────────────┘
    """
    print(comparison)


# ──────────────────────────────────────────────
# 4. Temperature 파라미터 실험
# ──────────────────────────────────────────────
def temperature_experiment():
    """temperature 파라미터가 출력 다양성에 미치는 영향을 실험한다."""
    print("\n" + "=" * 60)
    print("4. Temperature 파라미터 실험")
    print("=" * 60)

    prompt = "AI의 미래를 한 문장으로 설명해주세요."
    print(f"프롬프트: {prompt}\n")

    if has_openai_key():
        from openai import OpenAI
        client = OpenAI()

        for temp in [0.0, 0.5, 1.0]:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=temp,
            )
            print(f"temperature={temp}: {response.choices[0].message.content}")
    else:
        print("[캐시 모드]")
        data = CACHED.get("temperature_comparison", {})
        for temp in ["0.0", "0.5", "1.0"]:
            key = f"temp_{temp}"
            print(f"temperature={temp}: {data.get(key, '(캐시 없음)')}")

    print("\n해석:")
    print("  temperature=0.0 → 결정적(항상 같은 답), 사실 기반 작업에 적합")
    print("  temperature=0.5 → 적당한 다양성, 일반적 용도에 적합")
    print("  temperature=1.0 → 높은 다양성, 창작/브레인스토밍에 적합")


# ──────────────────────────────────────────────
# 5. 토큰 수 계산 및 비용 추정
# ──────────────────────────────────────────────
def token_counting_and_cost():
    """tiktoken으로 토큰 수를 계산하고 API 비용을 추정한다."""
    print("\n" + "=" * 60)
    print("5. 토큰 수 계산 및 비용 추정")
    print("=" * 60)

    try:
        import tiktoken
    except ImportError:
        print("tiktoken이 설치되지 않았습니다: pip install tiktoken")
        return

    # GPT-4o는 o200k_base 인코딩 사용
    encoding = tiktoken.encoding_for_model("gpt-4o")

    texts = [
        "Hello, world!",
        "자연어처리는 인공지능의 핵심 분야입니다.",
        "Natural language processing (NLP) is a subfield of AI.",
    ]

    print("\n텍스트별 토큰 수:")
    for text in texts:
        tokens = encoding.encode(text)
        print(f"  '{text}'")
        print(f"    → {len(tokens)} 토큰: {tokens[:10]}{'...' if len(tokens) > 10 else ''}")

    # 비용 추정 예시
    print("\n비용 추정 (GPT-4o 기준):")
    input_tokens = 1000
    output_tokens = 500
    input_cost = input_tokens / 1_000_000 * 2.50   # $2.50 per 1M
    output_cost = output_tokens / 1_000_000 * 10.00  # $10.00 per 1M
    total_cost = input_cost + output_cost

    print(f"  입력 {input_tokens} 토큰 × $2.50/1M = ${input_cost:.6f}")
    print(f"  출력 {output_tokens} 토큰 × $10.00/1M = ${output_cost:.6f}")
    print(f"  합계: ${total_cost:.6f} (약 {total_cost * 1400:.2f}원)")

    # 한국어 vs 영어 토큰 효율 비교
    print("\n한국어 vs 영어 토큰 효율 비교:")
    ko_text = "인공지능은 인간의 학습 능력을 모방한 기술이다."
    en_text = "Artificial intelligence is technology that mimics human learning."
    ko_tokens = len(encoding.encode(ko_text))
    en_tokens = len(encoding.encode(en_text))
    print(f"  한국어: '{ko_text}' → {ko_tokens} 토큰")
    print(f"  영어:   '{en_text}' → {en_tokens} 토큰")
    print(f"  비율: 한국어가 영어 대비 약 {ko_tokens / en_tokens:.1f}배 토큰 사용")


# ──────────────────────────────────────────────
# 메인 실행
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("제6장 실습 6-1: LLM API 기초\n")

    openai_basic_chat()
    anthropic_basic_chat()
    compare_api_structure()
    temperature_experiment()
    token_counting_and_cost()

    print("\n" + "=" * 60)
    print("실습 완료!")
    print("=" * 60)
