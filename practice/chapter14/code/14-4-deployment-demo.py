"""
14장 실습: Gradio 데모 배포
- 간단한 감성 분석 데모
- Gradio 인터페이스 구축
- 모델 배포 예시
"""

import warnings
warnings.filterwarnings('ignore')


def create_simple_demo():
    """간단한 규칙 기반 감성 분석 데모"""
    print("=" * 60)
    print("1. 간단한 감성 분석 데모 (규칙 기반)")
    print("=" * 60)

    def simple_sentiment(text):
        """간단한 규칙 기반 감성 분석"""
        positive_words = ['좋다', '좋아', '훌륭', '최고', '멋지', '사랑', '행복',
                          '감사', '기쁘', '만족', '추천', '대박', '짱', '굿', '좋은']
        negative_words = ['나쁘', '싫어', '최악', '별로', '실망', '불만', '화나',
                          '슬프', '짜증', '후회', '나쁜', '안좋', '못']

        text_lower = text.lower()

        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)

        if pos_count > neg_count:
            sentiment = "긍정 😊"
            confidence = min(0.5 + pos_count * 0.1, 0.95)
        elif neg_count > pos_count:
            sentiment = "부정 😞"
            confidence = min(0.5 + neg_count * 0.1, 0.95)
        else:
            sentiment = "중립 😐"
            confidence = 0.5

        return f"감성: {sentiment}\n신뢰도: {confidence:.2f}"

    # 테스트
    test_texts = [
        "이 영화 정말 최고예요! 강력 추천합니다.",
        "서비스가 너무 별로였어요. 실망입니다.",
        "오늘 날씨가 흐립니다."
    ]

    print("\n[테스트 결과]")
    for text in test_texts:
        result = simple_sentiment(text)
        print(f"\n입력: {text}")
        print(result)

    return simple_sentiment


def create_transformer_demo():
    """Hugging Face Transformers 기반 감성 분석 데모"""
    print("\n" + "=" * 60)
    print("2. Transformers 기반 감성 분석 데모")
    print("=" * 60)

    try:
        from transformers import pipeline

        # 감성 분석 파이프라인 로드
        print("모델 로드 중...")
        classifier = pipeline(
            "sentiment-analysis",
            model="nlptown/bert-base-multilingual-uncased-sentiment"
        )

        def analyze_sentiment(text):
            """Transformer 기반 감성 분석"""
            if not text.strip():
                return "텍스트를 입력해주세요."

            result = classifier(text)[0]
            label = result['label']
            score = result['score']

            # 별점 형식 (1-5 stars)
            stars = int(label.split()[0])
            star_emoji = "⭐" * stars + "☆" * (5 - stars)

            if stars >= 4:
                sentiment = "긍정"
            elif stars <= 2:
                sentiment = "부정"
            else:
                sentiment = "중립"

            return f"감성: {sentiment}\n별점: {star_emoji} ({stars}/5)\n신뢰도: {score:.4f}"

        # 테스트
        test_texts = [
            "This product is amazing! I love it so much.",
            "Terrible experience. Never buying again.",
            "It's okay, nothing special."
        ]

        print("\n[Transformer 모델 테스트]")
        for text in test_texts:
            result = analyze_sentiment(text)
            print(f"\n입력: {text}")
            print(result)

        return analyze_sentiment

    except Exception as e:
        print(f"Transformers 로드 실패: {e}")
        return None


def create_gradio_interface(analyze_fn):
    """Gradio 인터페이스 생성"""
    print("\n" + "=" * 60)
    print("3. Gradio 인터페이스 생성")
    print("=" * 60)

    try:
        import gradio as gr

        demo = gr.Interface(
            fn=analyze_fn,
            inputs=gr.Textbox(
                label="분석할 텍스트",
                placeholder="텍스트를 입력하세요...",
                lines=3
            ),
            outputs=gr.Textbox(label="분석 결과"),
            title="🎭 감성 분석 데모",
            description="텍스트의 감성(긍정/부정/중립)을 분석합니다.",
            examples=[
                ["이 제품 정말 좋아요! 강력 추천합니다."],
                ["서비스가 너무 별로였어요. 실망입니다."],
                ["그냥 보통이에요. 평범합니다."]
            ],
            theme="soft"
        )

        print("\nGradio 인터페이스가 생성되었습니다.")
        print("\n[실행 방법]")
        print("1. 로컬 실행: demo.launch()")
        print("2. 공유 링크: demo.launch(share=True)")
        print("3. Hugging Face Spaces 배포")

        # 실제 실행은 주석 처리 (자동화 스크립트에서는 실행하지 않음)
        # demo.launch()

        return demo

    except ImportError:
        print("Gradio가 설치되지 않았습니다.")
        print("설치: pip install gradio")
        return None


def show_deployment_code():
    """배포 코드 예시"""
    print("\n" + "=" * 60)
    print("4. FastAPI 배포 코드 예시")
    print("=" * 60)

    fastapi_code = '''
# app.py - FastAPI 배포 예시
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI(title="감성 분석 API")

# 모델 로드 (앱 시작 시 1회)
classifier = pipeline("sentiment-analysis")

class TextInput(BaseModel):
    text: str

class SentimentOutput(BaseModel):
    label: str
    score: float

@app.post("/predict", response_model=SentimentOutput)
async def predict(input: TextInput):
    """감성 분석 예측"""
    result = classifier(input.text)[0]
    return SentimentOutput(
        label=result["label"],
        score=result["score"]
    )

@app.get("/health")
async def health():
    """헬스 체크"""
    return {"status": "healthy"}

# 실행: uvicorn app:app --host 0.0.0.0 --port 8000
'''
    print(fastapi_code)

    print("\n[FastAPI 실행 방법]")
    print("1. pip install fastapi uvicorn")
    print("2. uvicorn app:app --reload")
    print("3. http://localhost:8000/docs 에서 API 테스트")


def show_huggingface_spaces_guide():
    """Hugging Face Spaces 배포 가이드"""
    print("\n" + "=" * 60)
    print("5. Hugging Face Spaces 배포 가이드")
    print("=" * 60)

    print("""
[Hugging Face Spaces 배포 단계]

1. Hugging Face 계정 생성
   - https://huggingface.co/join

2. 새 Space 생성
   - https://huggingface.co/new-space
   - Space SDK: Gradio 선택
   - Hardware: CPU Basic (무료)

3. 필요한 파일 업로드
   - app.py: Gradio 앱 코드
   - requirements.txt: 의존성 목록

4. app.py 예시:
   ```python
   import gradio as gr
   from transformers import pipeline

   classifier = pipeline("sentiment-analysis")

   def predict(text):
       result = classifier(text)[0]
       return f"{result['label']}: {result['score']:.4f}"

   demo = gr.Interface(fn=predict, inputs="text", outputs="text")
   demo.launch()
   ```

5. requirements.txt:
   ```
   transformers
   torch
   gradio
   ```

6. 배포 완료!
   - URL: https://huggingface.co/spaces/{username}/{space-name}
""")


def main():
    """메인 실행 함수"""
    print("=" * 60)
    print("14장 실습: Gradio 데모 배포")
    print("=" * 60)

    # 1. 간단한 규칙 기반 데모
    simple_fn = create_simple_demo()

    # 2. Transformers 기반 데모
    transformer_fn = create_transformer_demo()

    # 3. Gradio 인터페이스 (규칙 기반 사용)
    demo = create_gradio_interface(simple_fn)

    # 4. FastAPI 배포 코드
    show_deployment_code()

    # 5. Hugging Face Spaces 가이드
    show_huggingface_spaces_guide()

    print("\n" + "=" * 60)
    print("데모 배포 실습 완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()
