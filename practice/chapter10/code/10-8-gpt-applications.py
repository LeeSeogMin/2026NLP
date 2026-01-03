"""
10-8-gpt-applications.py
GPT 응용 실습

이 스크립트는 GPT-2를 다양한 태스크에 활용하는 방법을 보여준다:
1. 텍스트 완성
2. Zero-shot 프롬프팅
3. Few-shot 프롬프팅
4. 다양한 스타일 생성

실행 방법:
    python 10-8-gpt-applications.py
"""

import warnings
warnings.filterwarnings('ignore')

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, pipeline


def setup_model():
    """모델 설정"""
    print("=" * 60)
    print("GPT-2 모델 로드 중...")
    print("=" * 60)

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    print("모델 로드 완료!")
    return tokenizer, model


def text_completion_demo(tokenizer, model):
    """텍스트 완성 데모"""
    print("\n" + "=" * 60)
    print("[1] 텍스트 완성 (Text Completion)")
    print("=" * 60)

    prompts = [
        "The key to success in life is",
        "Artificial intelligence will change the world by",
        "The best way to learn programming is to",
        "In the year 2050, humans will",
    ]

    print("\n[텍스트 완성 결과]")
    print("-" * 60)

    for prompt in prompts:
        input_ids = tokenizer.encode(prompt, return_tensors='pt')

        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_length=len(input_ids[0]) + 30,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )

        generated = tokenizer.decode(output[0], skip_special_tokens=True)
        print(f"\n프롬프트: \"{prompt}\"")
        print(f"완성: {generated}")


def zero_shot_demo(tokenizer, model):
    """Zero-shot 프롬프팅 데모"""
    print("\n" + "=" * 60)
    print("[2] Zero-shot 프롬프팅")
    print("=" * 60)

    print("\n[개념]")
    print("  - 예시 없이 태스크 지시만으로 수행")
    print("  - 모델의 일반화 능력에 의존")
    print("  - 간단한 태스크에 효과적")

    tasks = [
        {
            "task": "감성 분석",
            "prompt": "Classify the sentiment of the following text as positive or negative.\n\nText: I absolutely loved this movie! It was fantastic.\nSentiment:"
        },
        {
            "task": "번역",
            "prompt": "Translate the following English text to French.\n\nEnglish: Hello, how are you today?\nFrench:"
        },
        {
            "task": "요약",
            "prompt": "Summarize the following text in one sentence.\n\nText: Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It focuses on developing algorithms that can access data and use it to learn for themselves.\n\nSummary:"
        }
    ]

    print("\n[Zero-shot 결과]")
    print("-" * 60)

    for task_info in tasks:
        input_ids = tokenizer.encode(task_info["prompt"], return_tensors='pt')

        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_length=len(input_ids[0]) + 30,
                do_sample=True,
                temperature=0.5,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )

        generated = tokenizer.decode(output[0], skip_special_tokens=True)
        # 프롬프트 이후 부분만 추출
        response = generated[len(task_info["prompt"]):].strip()
        # 첫 줄만 추출
        response = response.split('\n')[0] if response else "(응답 없음)"

        print(f"\n[{task_info['task']}]")
        print(f"  프롬프트: {task_info['prompt'][:60]}...")
        print(f"  응답: {response}")


def few_shot_demo(tokenizer, model):
    """Few-shot 프롬프팅 데모"""
    print("\n" + "=" * 60)
    print("[3] Few-shot 프롬프팅")
    print("=" * 60)

    print("\n[개념]")
    print("  - 몇 개의 예시를 제공하여 태스크 학습")
    print("  - Zero-shot보다 정확도 향상")
    print("  - 모델 크기가 클수록 효과적")

    # 감성 분석 Few-shot
    few_shot_prompt = """Classify the sentiment as positive or negative.

Text: This restaurant has amazing food!
Sentiment: positive

Text: I was very disappointed with the service.
Sentiment: negative

Text: The weather is perfect for a picnic.
Sentiment: positive

Text: I can't believe how terrible this product is.
Sentiment:"""

    print("\n[Few-shot 감성 분석]")
    print("-" * 60)

    input_ids = tokenizer.encode(few_shot_prompt, return_tensors='pt')

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=len(input_ids[0]) + 10,
            do_sample=False,  # 결정적 출력
            pad_token_id=tokenizer.eos_token_id
        )

    generated = tokenizer.decode(output[0], skip_special_tokens=True)
    response = generated[len(few_shot_prompt):].strip().split('\n')[0]

    print(f"  테스트 문장: \"I can't believe how terrible this product is.\"")
    print(f"  예측 감성: {response}")

    # 텍스트 변환 Few-shot
    transform_prompt = """Convert the following sentences to past tense.

Present: I walk to school every day.
Past: I walked to school every day.

Present: She eats breakfast at 7 AM.
Past: She ate breakfast at 7 AM.

Present: They play soccer on weekends.
Past: They played soccer on weekends.

Present: He writes a letter to his friend.
Past:"""

    print("\n[Few-shot 시제 변환]")
    print("-" * 60)

    input_ids = tokenizer.encode(transform_prompt, return_tensors='pt')

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=len(input_ids[0]) + 15,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    generated = tokenizer.decode(output[0], skip_special_tokens=True)
    response = generated[len(transform_prompt):].strip().split('\n')[0]

    print(f"  입력: \"He writes a letter to his friend.\"")
    print(f"  출력: {response}")


def style_generation_demo(tokenizer, model):
    """다양한 스타일 생성 데모"""
    print("\n" + "=" * 60)
    print("[4] 스타일별 텍스트 생성")
    print("=" * 60)

    styles = [
        {
            "style": "뉴스 기사",
            "prompt": "Breaking News: Scientists have discovered a new planet. The research team announced"
        },
        {
            "style": "시적 표현",
            "prompt": "The moon rises over the mountains, casting silver light upon"
        },
        {
            "style": "기술 문서",
            "prompt": "To install this software, first download the package and then"
        },
        {
            "style": "대화",
            "prompt": "A: Hi, how was your day?\nB:"
        }
    ]

    print("\n[스타일별 생성 결과]")
    print("-" * 60)

    for style_info in styles:
        input_ids = tokenizer.encode(style_info["prompt"], return_tensors='pt')

        torch.manual_seed(42)
        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_length=len(input_ids[0]) + 40,
                do_sample=True,
                temperature=0.8,
                top_p=0.92,
                pad_token_id=tokenizer.eos_token_id
            )

        generated = tokenizer.decode(output[0], skip_special_tokens=True)

        print(f"\n[{style_info['style']}]")
        print(f"  {generated}")


def pipeline_demo():
    """Pipeline API 데모"""
    print("\n" + "=" * 60)
    print("[5] Pipeline API 활용")
    print("=" * 60)

    print("\n[개념]")
    print("  - 간단한 인터페이스로 텍스트 생성")
    print("  - 파라미터 조정 용이")

    # 텍스트 생성 파이프라인
    generator = pipeline('text-generation', model='gpt2')

    prompts = [
        "The secret to happiness is",
        "In a world where technology",
    ]

    print("\n[Pipeline 생성 결과]")
    print("-" * 60)

    for prompt in prompts:
        result = generator(
            prompt,
            max_length=50,
            num_return_sequences=2,
            temperature=0.8,
            top_p=0.9,
            do_sample=True
        )

        print(f"\n프롬프트: \"{prompt}\"")
        for i, r in enumerate(result, 1):
            text = r['generated_text']
            print(f"  [{i}] {text}")


def prompt_engineering_tips():
    """프롬프트 엔지니어링 팁"""
    print("\n" + "=" * 60)
    print("[6] 프롬프트 엔지니어링 팁")
    print("=" * 60)

    print("""
┌─────────────────────────────────────────────────────────────┐
│                  효과적인 프롬프트 작성법                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. 명확하고 구체적인 지시                                  │
│     ✗ "글을 써줘"                                          │
│     ✓ "100단어 내외로 AI의 미래에 대한 에세이를 써줘"       │
│                                                             │
│  2. 맥락과 배경 정보 제공                                   │
│     - 역할 부여: "당신은 전문 작가입니다"                   │
│     - 상황 설명: "초등학생을 위한 설명으로..."              │
│                                                             │
│  3. 원하는 출력 형식 명시                                   │
│     - "다음 형식으로 작성해줘: 제목, 본문, 결론"            │
│     - "JSON 형식으로 출력해줘"                              │
│                                                             │
│  4. Few-shot 예시 활용                                      │
│     - 원하는 출력 예시 2-3개 제공                           │
│     - 일관된 형식 유지                                      │
│                                                             │
│  5. Chain-of-Thought 활용                                   │
│     - "단계별로 생각해봅시다" 추가                          │
│     - 복잡한 추론 문제에 효과적                             │
│                                                             │
└─────────────────────────────────────────────────────────────┘

[Temperature 가이드]
  - 0.2~0.4: 사실적, 결정적 (번역, 코드)
  - 0.5~0.7: 균형 (일반 텍스트)
  - 0.8~1.0: 창의적 (창작, 브레인스토밍)
  - 1.0+: 매우 다양함 (실험적)

[Top-p 가이드]
  - 0.9: 일반적인 사용
  - 0.95: 좀 더 다양한 출력
  - 0.5~0.7: 보수적, 안전한 출력
""")


def main():
    """메인 함수"""
    print("=" * 60)
    print("GPT 응용 실습")
    print("=" * 60)

    # 모델 로드
    tokenizer, model = setup_model()

    # 1. 텍스트 완성
    text_completion_demo(tokenizer, model)

    # 2. Zero-shot
    zero_shot_demo(tokenizer, model)

    # 3. Few-shot
    few_shot_demo(tokenizer, model)

    # 4. 스타일 생성
    style_generation_demo(tokenizer, model)

    # 5. Pipeline
    pipeline_demo()

    # 6. 프롬프트 팁
    prompt_engineering_tips()

    print("\n" + "=" * 60)
    print("GPT 응용 실습 완료")
    print("=" * 60)


if __name__ == "__main__":
    main()
