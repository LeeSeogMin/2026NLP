"""
9-7-bert-applications.py
BERT 활용 응용 실습

이 스크립트는 BERT를 다양한 NLP 태스크에 활용하는 방법을 보여준다:
1. 텍스트 분류 (감성 분석)
2. 개체명 인식 (NER)
3. 텍스트 임베딩 추출
4. 문장 유사도 계산

실행 방법:
    python 9-7-bert-applications.py
"""

import warnings
warnings.filterwarnings('ignore')

import torch
import numpy as np
from transformers import (
    BertTokenizer,
    BertModel,
    BertForSequenceClassification,
    BertForTokenClassification,
    AutoTokenizer,
    AutoModel,
    pipeline
)


def text_classification_demo():
    """텍스트 분류 (감성 분석) 데모"""
    print("=" * 60)
    print("[1] 텍스트 분류 - 감성 분석")
    print("=" * 60)

    # 사전학습된 감성 분석 모델 로드
    model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name)

    # 테스트 문장
    texts = [
        "This product is absolutely amazing! Best purchase ever.",
        "Terrible experience. Would not recommend to anyone.",
        "It's okay, nothing special but does the job.",
    ]

    print("\n[감성 분석 결과] (1-5 별점)")
    print("-" * 60)

    model.eval()
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_class = torch.argmax(logits, dim=1).item() + 1  # 1-5 별점

            # 확률 분포
            probs = torch.softmax(logits, dim=1)[0]

            print(f"\n  텍스트: \"{text[:50]}...\"")
            print(f"  예측 별점: {'⭐' * predicted_class} ({predicted_class}/5)")
            print(f"  신뢰도: {probs[predicted_class-1]:.4f}")


def ner_demo():
    """개체명 인식 (NER) 데모"""
    print("\n" + "=" * 60)
    print("[2] 개체명 인식 (Named Entity Recognition)")
    print("=" * 60)

    # NER 파이프라인
    ner_pipeline = pipeline(
        "ner",
        model="dbmdz/bert-large-cased-finetuned-conll03-english",
        aggregation_strategy="simple"
    )

    # 테스트 텍스트
    texts = [
        "Apple Inc. was founded by Steve Jobs in Cupertino, California.",
        "The Eiffel Tower in Paris attracts millions of visitors each year.",
        "Microsoft CEO Satya Nadella announced new AI features in Seattle.",
    ]

    print("\n[NER 결과]")
    print("-" * 60)

    for text in texts:
        print(f"\n  텍스트: \"{text}\"")
        entities = ner_pipeline(text)

        if entities:
            print("  발견된 개체:")
            for entity in entities:
                entity_type = entity['entity_group']
                word = entity['word']
                score = entity['score']
                type_emoji = {"PER": "👤", "ORG": "🏢", "LOC": "📍", "MISC": "📌"}.get(entity_type, "•")
                print(f"    {type_emoji} [{entity_type}] {word} (신뢰도: {score:.4f})")
        else:
            print("  (발견된 개체 없음)")


def text_embedding_demo():
    """텍스트 임베딩 추출 데모"""
    print("\n" + "=" * 60)
    print("[3] 텍스트 임베딩 추출")
    print("=" * 60)

    # BERT 모델 로드
    model_name = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)

    # 테스트 문장
    text = "BERT produces contextualized word embeddings."

    # 토큰화
    inputs = tokenizer(text, return_tensors="pt")

    print(f"\n입력 텍스트: \"{text}\"")
    print(f"토큰: {tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])}")

    # 모델 추론
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)

    # 출력 분석
    last_hidden_state = outputs.last_hidden_state
    pooler_output = outputs.pooler_output

    print(f"\n[출력 텐서 크기]")
    print(f"  last_hidden_state: {last_hidden_state.shape}")
    print(f"    - (batch_size=1, seq_len={last_hidden_state.shape[1]}, hidden_size={last_hidden_state.shape[2]})")
    print(f"  pooler_output: {pooler_output.shape}")
    print(f"    - [CLS] 토큰의 변환된 표현 (batch_size=1, hidden_size=768)")

    # 임베딩 추출 방법들
    print(f"\n[임베딩 추출 방법]")

    # 방법 1: [CLS] 토큰 임베딩
    cls_embedding = last_hidden_state[0, 0, :]  # 첫 번째 토큰 ([CLS])
    print(f"  1. [CLS] 토큰: shape={cls_embedding.shape}, mean={cls_embedding.mean():.4f}")

    # 방법 2: Pooler Output
    print(f"  2. Pooler Output: shape={pooler_output[0].shape}, mean={pooler_output[0].mean():.4f}")

    # 방법 3: Mean Pooling (전체 토큰 평균)
    # Attention mask를 고려한 평균
    attention_mask = inputs['attention_mask']
    mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    sum_embeddings = torch.sum(last_hidden_state * mask_expanded, dim=1)
    sum_mask = mask_expanded.sum(dim=1)
    mean_pooled = sum_embeddings / sum_mask
    print(f"  3. Mean Pooling: shape={mean_pooled[0].shape}, mean={mean_pooled[0].mean():.4f}")

    return model, tokenizer


def sentence_similarity_demo():
    """문장 유사도 계산 데모"""
    print("\n" + "=" * 60)
    print("[4] 문장 유사도 계산")
    print("=" * 60)

    # BERT 모델 로드
    model_name = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)

    def get_sentence_embedding(text):
        """문장 임베딩 추출 (Mean Pooling)"""
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        # Mean Pooling
        attention_mask = inputs['attention_mask']
        hidden_state = outputs.last_hidden_state
        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_state.size()).float()
        sum_embeddings = torch.sum(hidden_state * mask_expanded, dim=1)
        sum_mask = mask_expanded.sum(dim=1)
        return (sum_embeddings / sum_mask)[0]

    def cosine_similarity(vec1, vec2):
        """코사인 유사도 계산"""
        return torch.nn.functional.cosine_similarity(
            vec1.unsqueeze(0), vec2.unsqueeze(0)
        ).item()

    # 테스트 문장 쌍
    sentence_pairs = [
        ("The cat sits on the mat.", "A cat is sitting on a mat."),  # 유사
        ("I love programming in Python.", "Python is my favorite language."),  # 유사
        ("The weather is sunny today.", "I need to buy groceries."),  # 비유사
        ("Machine learning is fascinating.", "AI is an interesting field."),  # 유사
    ]

    print("\n[문장 유사도 결과]")
    print("-" * 60)

    model.eval()
    for sent1, sent2 in sentence_pairs:
        emb1 = get_sentence_embedding(sent1)
        emb2 = get_sentence_embedding(sent2)
        similarity = cosine_similarity(emb1, emb2)

        # 유사도에 따른 시각화
        bar_length = int(similarity * 20)
        bar = "█" * bar_length + "░" * (20 - bar_length)

        print(f"\n  문장 1: \"{sent1}\"")
        print(f"  문장 2: \"{sent2}\"")
        print(f"  유사도: [{bar}] {similarity:.4f}")


def model_comparison():
    """BERT 변형 모델 비교"""
    print("\n" + "=" * 60)
    print("[5] BERT 변형 모델 정보")
    print("=" * 60)

    models_info = [
        ("bert-base-uncased", "BERT-Base"),
        ("distilbert-base-uncased", "DistilBERT"),
        ("albert-base-v2", "ALBERT-Base"),
    ]

    print("\n[모델별 파라미터 수]")
    print("-" * 60)

    for model_name, display_name in models_info:
        try:
            model = AutoModel.from_pretrained(model_name)
            num_params = sum(p.numel() for p in model.parameters())
            print(f"  {display_name:20} : {num_params:>12,} parameters ({num_params/1e6:.1f}M)")
            del model  # 메모리 해제
        except Exception as e:
            print(f"  {display_name:20} : (로드 실패)")

    print("\n[모델 특징 비교]")
    print("-" * 60)
    print("  BERT-Base    : 12 layers, 768 hidden, 12 heads")
    print("  DistilBERT   : 6 layers, 768 hidden, 12 heads (40% 작음, 60% 빠름)")
    print("  ALBERT-Base  : 12 layers, 768 hidden, 파라미터 공유")


def main():
    """메인 함수"""
    print("=" * 60)
    print("BERT 활용 응용 실습")
    print("=" * 60)

    # 1. 텍스트 분류
    text_classification_demo()

    # 2. 개체명 인식
    ner_demo()

    # 3. 텍스트 임베딩 추출
    text_embedding_demo()

    # 4. 문장 유사도 계산
    sentence_similarity_demo()

    # 5. 모델 비교
    model_comparison()

    print("\n" + "=" * 60)
    print("BERT 활용 실습 완료")
    print("=" * 60)


if __name__ == "__main__":
    main()
