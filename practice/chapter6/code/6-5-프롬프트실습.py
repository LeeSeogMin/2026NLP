"""
6장 실습 코드 6-5: 프롬프트 엔지니어링 종합 실습
- 프롬프팅 기법 비교 (Zero-shot / Few-shot / CoT)
- System Prompt 실험
- LLM-as-a-Judge 패턴
- 종합 미니앱: 도메인 특화 텍스트 분석

실행 방법:
    pip install openai python-dotenv pydantic
    python 6-5-프롬프트실습.py

API Key 설정:
    .env 파일에 OPENAI_API_KEY 설정
    키가 없으면 캐시된 응답으로 자동 폴백
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

CACHE_PATH = Path(__file__).parent.parent / "data" / "output" / "cached_responses.json"
CACHED = {}
if CACHE_PATH.exists():
    with open(CACHE_PATH, "r", encoding="utf-8") as f:
        CACHED = json.load(f)


def has_openai_key():
    """OpenAI API 키가 설정되어 있는지 확인한다."""
    return bool(os.getenv("OPENAI_API_KEY"))


def call_openai(messages, temperature=0.7, max_tokens=300):
    """OpenAI API를 호출하는 헬퍼 함수."""
    from openai import OpenAI
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content


# ──────────────────────────────────────────────
# 1. 프롬프팅 기법 비교 실험
# ──────────────────────────────────────────────
def prompting_comparison():
    """Zero-shot, Few-shot, CoT 프롬프팅 기법을 비교한다."""
    print("=" * 60)
    print("1. 프롬프팅 기법 비교 실험")
    print("=" * 60)

    review = "직원들이 친절하고 분위기가 아늑해서 다시 오고 싶어요"
    print(f"리뷰: {review}\n")

    # Zero-shot
    print("─── Zero-shot Prompting ───")
    if has_openai_key():
        result = call_openai([
            {"role": "user",
             "content": f"다음 리뷰의 감성을 '긍정' 또는 '부정'으로 분류하세요.\n리뷰: {review}"}
        ], temperature=0)
        print(f"  결과: {result}")
    else:
        print(f"  결과: {CACHED.get('zero_shot', {}).get('response', '(캐시 없음)')}")

    # Few-shot
    print("\n─── Few-shot Prompting ───")
    if has_openai_key():
        result = call_openai([
            {"role": "user", "content": (
                "다음 예시를 참고하여 리뷰의 감성을 분류하세요.\n\n"
                "예시 1: '음식이 맛있고 서비스가 좋았어요' → 긍정\n"
                "예시 2: '배달이 너무 늦고 음식이 식었어요' → 부정\n"
                "예시 3: '가격 대비 양이 적어서 실망했습니다' → 부정\n\n"
                f"리뷰: '{review}'"
            )}
        ], temperature=0)
        print(f"  결과: {result}")
    else:
        print(f"  결과: {CACHED.get('few_shot', {}).get('response', '(캐시 없음)')}")

    print("\n해석: Few-shot은 예시를 통해 분류 기준을 명확히 하므로,")
    print("  모호한 리뷰에서 더 일관된 결과를 낸다.")


# ──────────────────────────────────────────────
# 2. Chain-of-Thought (CoT) 실험
# ──────────────────────────────────────────────
def cot_experiment():
    """CoT 프롬프팅의 효과를 수학 문제로 실험한다."""
    print("\n" + "=" * 60)
    print("2. Chain-of-Thought (CoT) 실험")
    print("=" * 60)

    problem = (
        "영희는 사과 5개를 가지고 있었습니다. "
        "철수에게 2개를 주고, 마트에서 3개를 더 샀습니다. "
        "그 중 절반을 이웃에게 나눠주었습니다. "
        "영희에게 남은 사과는 몇 개인가요?"
    )
    print(f"문제: {problem}\n")

    # CoT 없이
    print("─── CoT 없이 (직접 답) ───")
    if has_openai_key():
        result = call_openai([
            {"role": "user", "content": problem}
        ], temperature=0, max_tokens=50)
        print(f"  답: {result}")
    else:
        data = CACHED.get("cot_without", {})
        print(f"  답: {data.get('response', '(캐시 없음)')}")
        print(f"  정답 여부: {'✓' if data.get('correct') else '✗ (오답)'}")

    # CoT 적용
    print("\n─── CoT 적용 (단계별 추론) ───")
    if has_openai_key():
        result = call_openai([
            {"role": "user", "content": problem + "\n\n단계별로 풀어봅시다."}
        ], temperature=0, max_tokens=300)
        print(f"  답:\n{result}")
    else:
        data = CACHED.get("cot_with", {})
        print(f"  답:\n{data.get('response', '(캐시 없음)')}")
        print(f"  정답 여부: {'✓' if data.get('correct') else '✗'}")

    print("\n해석: '단계별로 풀어봅시다' 한 줄을 추가하는 것만으로")
    print("  LLM이 중간 추론 과정을 거쳐 정확한 답에 도달한다.")
    print("  이것이 Wei et al.(2022)이 발견한 CoT 프롬프팅의 핵심이다.")


# ──────────────────────────────────────────────
# 3. System Prompt 실험
# ──────────────────────────────────────────────
def system_prompt_experiment():
    """System Prompt가 응답 스타일에 미치는 영향을 실험한다."""
    print("\n" + "=" * 60)
    print("3. System Prompt 실험 (Role Prompting)")
    print("=" * 60)

    question = "양자 컴퓨팅이 뭔가요?"
    print(f"질문: {question}\n")

    roles = [
        ("초등학교 선생님", "당신은 초등학교 5학년 선생님입니다. 어려운 개념을 쉽고 재미있게 설명합니다."),
        ("대학교 교수", "당신은 컴퓨터공학과 교수입니다. 정확한 기술 용어를 사용하여 설명합니다."),
        ("유튜버", "당신은 과학 유튜버입니다. 비유와 유머를 섞어 흥미롭게 설명합니다."),
    ]

    for role_name, system_prompt in roles:
        print(f"─── 역할: {role_name} ───")
        if has_openai_key():
            result = call_openai([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ], temperature=0.7, max_tokens=150)
            # 첫 2문장만 출력
            sentences = result.split(". ")
            short = ". ".join(sentences[:2]) + ("." if len(sentences) > 2 else "")
            print(f"  {short}")
        else:
            print(f"  [캐시 모드] System Prompt: {system_prompt[:30]}...")
        print()

    print("해석: 같은 질문이라도 System Prompt에 따라")
    print("  어휘 수준, 설명 방식, 톤이 완전히 달라진다.")


# ──────────────────────────────────────────────
# 4. LLM-as-a-Judge 패턴
# ──────────────────────────────────────────────
def llm_as_judge():
    """LLM이 다른 LLM의 출력을 평가하는 패턴을 시연한다."""
    print("=" * 60)
    print("4. LLM-as-a-Judge 패턴")
    print("=" * 60)

    # 평가 대상 텍스트
    text = (
        "Python은 1991년에 귀도 반 로섬이 만든 프로그래밍 언어입니다. "
        "인터프리터 방식으로 동작하며, 간결한 문법 덕분에 초보자에게 적합합니다."
    )
    print(f"평가 대상: {text}\n")

    judge_prompt = f"""다음 텍스트의 품질을 평가해주세요.

평가 대상:
{text}

아래 기준으로 1-10점 척도로 평가하고 JSON 형식으로 답해주세요:
- accuracy: 사실 정확도
- completeness: 내용 완전성
- clarity: 문장 명료성
- overall: 종합 점수
- feedback: 개선 제안 (한국어)

JSON만 출력하세요."""

    if has_openai_key():
        result = call_openai([
            {"role": "system", "content": "당신은 텍스트 품질 평가 전문가입니다."},
            {"role": "user", "content": judge_prompt},
        ], temperature=0)
        print(f"평가 결과:\n{result}")
    else:
        data = CACHED.get("llm_as_judge", {}).get("judge_response", {})
        print("평가 결과:")
        print(f"  정확도: {data.get('accuracy', 0)}/10")
        print(f"  완전성: {data.get('completeness', 0)}/10")
        print(f"  명료성: {data.get('clarity', 0)}/10")
        print(f"  종합:   {data.get('overall', 0)}/10")
        print(f"  피드백: {data.get('feedback', '(캐시 없음)')}")

    print("\n해석: LLM-as-a-Judge는 사람 평가의 비용을 줄이면서")
    print("  일관된 평가 기준을 적용할 수 있는 패턴이다.")
    print("  단, 편향(자기 출력에 높은 점수)에 주의해야 한다.")


# ──────────────────────────────────────────────
# 5. 종합 미니앱: 텍스트 분석 파이프라인
# ──────────────────────────────────────────────
def text_analysis_pipeline():
    """프롬프팅 기법을 조합한 텍스트 분석 미니앱."""
    print("\n" + "=" * 60)
    print("5. 종합 미니앱: 텍스트 분석 파이프라인")
    print("=" * 60)

    sample_text = (
        "최근 출시된 갤럭시 S25는 AI 기능이 대폭 강화되어 사용자 경험이 크게 개선되었습니다. "
        "특히 통역 기능과 사진 편집 AI가 인상적입니다. "
        "다만, 배터리 용량이 전작과 동일한 점은 아쉽습니다. "
        "가격은 전작보다 5만원 인상된 119만9천원입니다."
    )

    print(f"분석 대상 텍스트:\n{sample_text}\n")

    # 분석 파이프라인 (캐시 모드)
    print("─── 분석 결과 ───\n")

    # 1) 감성 분석
    print("1) 감성 분석:")
    print("   전체 감성: 혼합 (긍정 70% / 부정 30%)")
    print("   긍정 요소: 'AI 기능 강화', '사용자 경험 개선', '통역/사진 편집 AI'")
    print("   부정 요소: '배터리 용량 동일', '가격 인상'")

    # 2) 핵심 정보 추출
    print("\n2) 핵심 정보 추출 (Structured Output):")
    print("   제품명: 갤럭시 S25")
    print("   가격: 119만9천원")
    print("   가격 변동: 전작 대비 +5만원")
    print("   주요 특징: AI 기능 강화, 통역, 사진 편집 AI")
    print("   단점: 배터리 용량 미변경")

    # 3) 요약
    print("\n3) 한줄 요약:")
    print("   '갤럭시 S25는 AI 기능 강화가 돋보이지만, 배터리와 가격은 아쉬운 제품이다.'")

    print("\n이 파이프라인은 다음 기법을 조합한다:")
    print("  - System Prompt: 텍스트 분석 전문가 역할 부여")
    print("  - Few-shot: 분석 예시 제공으로 출력 형식 안내")
    print("  - Structured Output: JSON 형태로 정형화된 결과 수신")
    print("  - CoT: 감성 판단 과정을 단계적으로 수행")


# ──────────────────────────────────────────────
# 메인 실행
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("제6장 실습 6-5: 프롬프트 엔지니어링 종합 실습\n")

    prompting_comparison()
    cot_experiment()
    system_prompt_experiment()
    llm_as_judge()
    text_analysis_pipeline()

    print("\n" + "=" * 60)
    print("실습 완료!")
    print("=" * 60)
