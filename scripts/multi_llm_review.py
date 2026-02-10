"""
multi_llm_review.py
Multi-LLM 품질 검증 스크립트

외부 LLM(GPT-4o, Grok-4)을 활용하여 교재 챕터의 품질을 검증한다.

사용법:
    python3 scripts/multi_llm_review.py --chapter 1
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# .env 파일에서 환경 변수 로드
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ[key.strip()] = value.strip()


def call_openai(content: str, model: str = "gpt-4o") -> str:
    """OpenAI API를 호출하여 리뷰를 수행한다."""
    import urllib.request

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return "ERROR: OPENAI_API_KEY not set"

    system_prompt = """당신은 대학교 교재 편집자이자 NLP 전문가입니다.
아래 교재 챕터를 다음 기준으로 평가해주세요:

1. **정확성** (10점): 기술적 내용이 정확한가?
2. **가독성** (10점): 학부생이 이해하기 쉽게 작성되었는가?
3. **구조** (10점): 3교시제 형식이 잘 지켜졌는가? 미션/타임라인/교시 구분이 명확한가?
4. **코드 품질** (10점): 코드 예시가 적절하고 실행 결과가 포함되었는가?
5. **직관적 비유** (10점): 핵심 개념에 직관적 비유가 잘 활용되었는가?
6. **완성도** (10점): 학습목표, 핵심정리, 참고문헌 등 필수 요소가 갖춰졌는가?

각 항목에 점수와 짧은 코멘트를 달고, 개선 제안을 3가지 이내로 제시하세요.
JSON 형식으로 답변하세요."""

    data = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"다음 교재 챕터를 리뷰해주세요:\n\n{content[:15000]}"}
        ],
        "temperature": 0.3,
        "max_tokens": 2000
    }).encode()

    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read())
            return result["choices"][0]["message"]["content"]
    except Exception as e:
        return f"ERROR: {e}"


def call_xai(content: str, model: str = "grok-4-1-fast-reasoning") -> str:
    """xAI API를 호출하여 리뷰를 수행한다."""
    import urllib.request

    api_key = os.environ.get("XAI_API_KEY")
    if not api_key:
        return "ERROR: XAI_API_KEY not set"

    system_prompt = """당신은 대학교 교재 편집자이자 NLP 전문가입니다.
아래 교재 챕터를 다음 기준으로 평가해주세요:

1. **정확성** (10점): 기술적 내용이 정확한가?
2. **가독성** (10점): 학부생이 이해하기 쉽게 작성되었는가?
3. **구조** (10점): 3교시제 형식이 잘 지켜졌는가?
4. **코드 품질** (10점): 코드 예시와 실행 결과가 적절한가?
5. **직관적 비유** (10점): 핵심 개념에 직관적 비유가 활용되었는가?
6. **완성도** (10점): 필수 구성 요소가 모두 갖춰졌는가?

각 항목에 점수와 코멘트, 개선 제안 3가지 이내를 JSON 형식으로 답변하세요."""

    data = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"다음 교재 챕터를 리뷰해주세요:\n\n{content[:12000]}"}
        ],
        "temperature": 0.3,
        "max_tokens": 2000
    }, ensure_ascii=False).encode("utf-8")

    req = urllib.request.Request(
        "https://api.x.ai/v1/chat/completions",
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json; charset=utf-8",
            "User-Agent": "Mozilla/5.0 multi-llm-review/1.0"
        }
    )

    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            result = json.loads(resp.read())
            return result["choices"][0]["message"]["content"]
    except urllib.request.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        return f"ERROR: {e.code} {e.reason} - {body[:500]}"
    except Exception as e:
        return f"ERROR: {e}"


def main():
    parser = argparse.ArgumentParser(description="Multi-LLM 교재 품질 검증")
    parser.add_argument("--chapter", type=int, required=True, help="챕터 번호")
    args = parser.parse_args()

    ch = args.chapter
    doc_path = Path(__file__).parent.parent / "docs" / f"ch{ch}.md"

    if not doc_path.exists():
        print(f"ERROR: {doc_path} 파일이 존재하지 않습니다.")
        sys.exit(1)

    content = doc_path.read_text(encoding="utf-8")
    print(f"📖 ch{ch}.md 로드 완료 ({len(content)} 글자, {len(content.splitlines())} 줄)")
    print()

    reviews = {}

    # GPT-4o 리뷰
    print("🔍 [1/2] GPT-4o 리뷰 수행 중...")
    gpt_review = call_openai(content, model="gpt-4o")
    reviews["gpt-4o"] = gpt_review
    print(f"  ✓ GPT-4o 리뷰 완료 ({len(gpt_review)} 글자)")
    print()

    # Grok-4 리뷰
    print("🔍 [2/2] Grok-4 리뷰 수행 중...")
    grok_review = call_xai(content, model="grok-4-1-fast-reasoning")
    reviews["grok-4"] = grok_review
    print(f"  ✓ Grok-4 리뷰 완료 ({len(grok_review)} 글자)")
    print()

    # 결과 저장
    review_dir = Path(__file__).parent.parent / "content" / "reviews"
    review_dir.mkdir(parents=True, exist_ok=True)

    today = datetime.now().strftime("%Y-%m-%d")
    output_path = review_dir / f"ch{ch}_review_{today}.json"

    result = {
        "chapter": ch,
        "date": today,
        "document": str(doc_path),
        "line_count": len(content.splitlines()),
        "char_count": len(content),
        "reviews": reviews
    }

    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"📝 리뷰 결과 저장: {output_path}")
    print()

    # 결과 요약 출력
    print("=" * 60)
    print(f"  Multi-LLM Review 완료 — ch{ch}")
    print("=" * 60)
    print()
    print("--- GPT-4o 리뷰 ---")
    print(gpt_review[:2000])
    print()
    print("--- Grok-4 리뷰 ---")
    print(grok_review[:2000])


if __name__ == "__main__":
    main()
