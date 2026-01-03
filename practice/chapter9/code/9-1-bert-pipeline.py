"""
9-1-bert-pipeline.py
Hugging Face Pipeline API를 활용한 BERT 기본 사용법

이 스크립트는 Hugging Face의 Pipeline API를 사용하여
BERT 기반 모델로 다양한 NLP 태스크를 수행하는 방법을 보여준다.

실행 방법:
    python 9-1-bert-pipeline.py
"""

import warnings
warnings.filterwarnings('ignore')

from transformers import pipeline


def sentiment_analysis_demo():
    """감성 분석 데모"""
    print("=" * 60)
    print("[1] 감성 분석 (Sentiment Analysis)")
    print("=" * 60)

    # 감성 분석 파이프라인 생성
    classifier = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )

    # 테스트 문장
    sentences = [
        "I love this movie! It's absolutely fantastic.",
        "This is the worst product I've ever bought.",
        "The weather is okay today.",
        "I'm so happy to see you again!",
        "This restaurant has terrible service.",
    ]

    print("\n[결과]")
    for sentence in sentences:
        result = classifier(sentence)[0]
        label = result['label']
        score = result['score']
        emoji = "😊" if label == "POSITIVE" else "😞"
        print(f"  {emoji} {label} ({score:.4f}): {sentence[:50]}...")

    return classifier


def ner_demo():
    """개체명 인식 데모"""
    print("\n" + "=" * 60)
    print("[2] 개체명 인식 (Named Entity Recognition)")
    print("=" * 60)

    # NER 파이프라인 생성
    ner = pipeline(
        "ner",
        model="dbmdz/bert-large-cased-finetuned-conll03-english",
        aggregation_strategy="simple"
    )

    # 테스트 문장
    text = "Elon Musk founded SpaceX in Hawthorne, California. " \
           "He also leads Tesla and acquired Twitter in 2022."

    print(f"\n입력 텍스트:")
    print(f"  \"{text}\"")

    # NER 수행
    results = ner(text)

    print("\n[인식된 개체]")
    for entity in results:
        entity_type = entity['entity_group']
        word = entity['word']
        score = entity['score']
        print(f"  [{entity_type}] {word} (신뢰도: {score:.4f})")

    return ner


def question_answering_demo():
    """질의응답 데모"""
    print("\n" + "=" * 60)
    print("[3] 질의응답 (Question Answering)")
    print("=" * 60)

    # QA 파이프라인 생성
    qa = pipeline(
        "question-answering",
        model="distilbert-base-cased-distilled-squad"
    )

    # 지문과 질문
    context = """
    BERT (Bidirectional Encoder Representations from Transformers) is a
    language model developed by Google AI Language team in 2018. It was
    pre-trained on a large corpus of text including BooksCorpus (800 million
    words) and English Wikipedia (2,500 million words). BERT uses a
    transformer architecture with only the encoder component, which allows
    it to understand context from both directions simultaneously.
    """

    questions = [
        "Who developed BERT?",
        "When was BERT created?",
        "What architecture does BERT use?",
        "How many words is BooksCorpus?",
    ]

    print("\n[지문]")
    print(f"  {context.strip()[:200]}...")

    print("\n[질의응답 결과]")
    for question in questions:
        result = qa(question=question, context=context)
        answer = result['answer']
        score = result['score']
        print(f"\n  Q: {question}")
        print(f"  A: {answer} (신뢰도: {score:.4f})")

    return qa


def fill_mask_demo():
    """마스크 토큰 예측 데모 (MLM)"""
    print("\n" + "=" * 60)
    print("[4] 마스크 토큰 예측 (Masked Language Modeling)")
    print("=" * 60)

    # Fill-mask 파이프라인 생성
    unmasker = pipeline(
        "fill-mask",
        model="bert-base-uncased"
    )

    # 마스킹된 문장
    masked_sentences = [
        "The capital of France is [MASK].",
        "I love to [MASK] books in my free time.",
        "The [MASK] is shining brightly today.",
        "She is a brilliant [MASK] at the university.",
    ]

    print("\n[마스크 토큰 예측 결과]")
    for sentence in masked_sentences:
        results = unmasker(sentence)
        print(f"\n  입력: {sentence}")
        print("  예측 (Top 3):")
        for i, result in enumerate(results[:3], 1):
            token = result['token_str']
            score = result['score']
            print(f"    {i}. {token} ({score:.4f})")

    return unmasker


def text_classification_demo():
    """텍스트 분류 데모"""
    print("\n" + "=" * 60)
    print("[5] 제로샷 분류 (Zero-shot Classification)")
    print("=" * 60)

    # 제로샷 분류 파이프라인
    classifier = pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli"
    )

    # 테스트 문장
    text = "The new iPhone 15 Pro has an amazing camera system."
    candidate_labels = ["technology", "sports", "politics", "entertainment"]

    print(f"\n입력 텍스트: \"{text}\"")
    print(f"후보 레이블: {candidate_labels}")

    # 분류 수행
    result = classifier(text, candidate_labels)

    print("\n[분류 결과]")
    for label, score in zip(result['labels'], result['scores']):
        bar = "█" * int(score * 30)
        print(f"  {label:15} {bar} {score:.4f}")

    return classifier


def main():
    """메인 함수: 모든 Pipeline 데모 실행"""
    print("=" * 60)
    print("Hugging Face Pipeline API - BERT 활용 데모")
    print("=" * 60)

    # 1. 감성 분석
    sentiment_analysis_demo()

    # 2. 개체명 인식
    ner_demo()

    # 3. 질의응답
    question_answering_demo()

    # 4. 마스크 토큰 예측
    fill_mask_demo()

    # 5. 제로샷 분류
    text_classification_demo()

    print("\n" + "=" * 60)
    print("Pipeline API 데모 완료")
    print("=" * 60)


if __name__ == "__main__":
    main()
