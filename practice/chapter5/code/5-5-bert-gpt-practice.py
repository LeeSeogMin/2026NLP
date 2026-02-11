"""
5-5-bert-gpt-practice.py
3교시 실습: BERT/GPT 종합 활용

이 스크립트는 BERT와 GPT를 종합적으로 활용하는 3교시 실습이다:
1. BERT 감성 분석 (AutoModel 방식)
2. BERT NER + 문장 유사도
3. GPT-2 텍스트 생성 + 디코딩 전략 실험
4. Hugging Face Pipeline 종합 활용

실행 방법:
    cd practice/chapter5
    pip install -r code/requirements.txt
    python code/5-5-bert-gpt-practice.py
"""

import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn.functional as F
from transformers import (
    BertTokenizer,
    BertModel,
    BertForSequenceClassification,
    GPT2Tokenizer,
    GPT2LMHeadModel,
    AutoTokenizer,
    pipeline,
)


def bert_sentiment_demo():
    """BERT 감성 분석 (AutoModel 방식)"""
    print("=" * 60)
    print("[1] BERT 감성 분석")
    print("=" * 60)

    model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name)
    model.eval()

    texts = [
        "This product is absolutely amazing! Best purchase ever.",
        "Terrible experience. Would not recommend to anyone.",
        "It's okay, nothing special but does the job.",
        "The service was excellent and the staff were very friendly.",
    ]

    print("\n[감성 분석 결과] (1-5 별점)")
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            outputs = model(**inputs)
            probs = F.softmax(outputs.logits, dim=1)[0]
            predicted = torch.argmax(probs).item() + 1  # 1-5 별점

            stars = "*" * predicted
            print(f"\n  텍스트: \"{text[:50]}...\"")
            print(f"  예측: {stars} ({predicted}/5, 신뢰도: {probs[predicted - 1]:.4f})")


def bert_ner_demo():
    """BERT NER 데모"""
    print("\n" + "=" * 60)
    print("[2] BERT 개체명 인식 (NER)")
    print("=" * 60)

    ner = pipeline(
        "ner",
        model="dbmdz/bert-large-cased-finetuned-conll03-english",
        aggregation_strategy="simple",
    )

    texts = [
        "Apple Inc. was founded by Steve Jobs in Cupertino, California.",
        "Microsoft CEO Satya Nadella announced new AI features in Seattle.",
        "The Eiffel Tower in Paris attracts millions of visitors each year.",
    ]

    for text in texts:
        print(f"\n  텍스트: \"{text}\"")
        entities = ner(text)
        if entities:
            print("  발견된 개체:")
            for e in entities:
                print(f"    [{e['entity_group']}] {e['word']} (신뢰도: {e['score']:.4f})")
        else:
            print("  (발견된 개체 없음)")


def bert_similarity_demo():
    """BERT 문장 유사도 계산"""
    print("\n" + "=" * 60)
    print("[3] BERT 문장 유사도")
    print("=" * 60)

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")
    model.eval()

    def get_embedding(text):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        hidden = outputs.last_hidden_state
        mask = inputs["attention_mask"].unsqueeze(-1).expand(hidden.size()).float()
        return (torch.sum(hidden * mask, dim=1) / mask.sum(dim=1))[0]

    pairs = [
        ("The cat sits on the mat.", "A cat is sitting on a mat."),
        ("I love programming in Python.", "Python is my favorite language."),
        ("The weather is sunny today.", "I need to buy groceries."),
        ("Machine learning is fascinating.", "AI is an interesting field."),
    ]

    print("\n[문장 유사도 결과]")
    for s1, s2 in pairs:
        e1 = get_embedding(s1)
        e2 = get_embedding(s2)
        sim = F.cosine_similarity(e1.unsqueeze(0), e2.unsqueeze(0)).item()
        bar_len = int(sim * 20)
        bar = "#" * bar_len + "." * (20 - bar_len)
        print(f"\n  문장 1: \"{s1}\"")
        print(f"  문장 2: \"{s2}\"")
        print(f"  유사도: [{bar}] {sim:.4f}")


def gpt2_generation_experiment():
    """GPT-2 텍스트 생성 실험"""
    print("\n" + "=" * 60)
    print("[4] GPT-2 텍스트 생성 실험")
    print("=" * 60)

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()
    tokenizer.pad_token = tokenizer.eos_token

    prompts = [
        "Artificial intelligence will change",
        "The most important thing in life is",
        "In the year 2030, technology will",
    ]

    strategies = [
        ("Greedy", {"do_sample": False}),
        ("Top-p=0.9, T=0.7", {"do_sample": True, "top_p": 0.9, "top_k": 0, "temperature": 0.7}),
        ("Top-k=50, T=0.9", {"do_sample": True, "top_k": 50, "temperature": 0.9}),
    ]

    for prompt in prompts:
        print(f"\n  프롬프트: \"{prompt}\"")
        print("  " + "-" * 50)
        input_ids = tokenizer.encode(prompt, return_tensors="pt")

        for name, params in strategies:
            torch.manual_seed(42)
            with torch.no_grad():
                output = model.generate(
                    input_ids, max_length=40,
                    pad_token_id=tokenizer.eos_token_id,
                    **params,
                )
            text = tokenizer.decode(output[0], skip_special_tokens=True)
            print(f"  [{name}]")
            print(f"    {text}")


def pipeline_comprehensive_demo():
    """Hugging Face Pipeline 종합 활용"""
    print("\n" + "=" * 60)
    print("[5] Hugging Face Pipeline 종합")
    print("=" * 60)

    # 요약
    print("\n--- 텍스트 요약 ---")
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    article = (
        "Artificial intelligence has transformed many industries over the past decade. "
        "From healthcare to finance, AI systems are being deployed to automate tasks, "
        "improve decision-making, and create new products and services. Machine learning, "
        "a subset of AI, has been particularly impactful, enabling computers to learn "
        "from data without being explicitly programmed. Deep learning, which uses neural "
        "networks with many layers, has achieved remarkable results in areas such as "
        "image recognition, natural language processing, and game playing."
    )
    result = summarizer(article, max_length=50, min_length=20, do_sample=False)
    print(f"  원문: \"{article[:80]}...\"")
    print(f"  요약: \"{result[0]['summary_text']}\"")

    # 제로샷 분류
    print("\n--- 제로샷 분류 ---")
    zs_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    text = "The stock market crashed due to rising interest rates."
    labels = ["finance", "sports", "technology", "politics"]
    result = zs_classifier(text, labels)
    print(f"  텍스트: \"{text}\"")
    print(f"  분류 결과:")
    for label, score in zip(result["labels"], result["scores"]):
        bar = "#" * int(score * 30)
        print(f"    {label:12} {bar} {score:.4f}")


def main():
    """메인 함수"""
    print("=" * 60)
    print("제5장 3교시 실습 — BERT/GPT 종합 활용")
    print("=" * 60)

    # 1. BERT 감성 분석
    bert_sentiment_demo()

    # 2. BERT NER
    bert_ner_demo()

    # 3. BERT 유사도
    bert_similarity_demo()

    # 4. GPT-2 생성 실험
    gpt2_generation_experiment()

    # 5. Pipeline 종합
    pipeline_comprehensive_demo()

    print("\n" + "=" * 60)
    print("3교시 실습 완료")
    print("=" * 60)
    print("\n[과제] BERT 기반 NER 모델 + GPT-2 텍스트 생성기 구현")


if __name__ == "__main__":
    main()
