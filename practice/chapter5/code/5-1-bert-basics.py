"""
5-1-bert-basics.py
BERT 기본 사용법: 토크나이저, MLM, 임베딩 추출, Pipeline

이 스크립트는 BERT 모델의 핵심 기능을 실습한다:
1. WordPiece 토크나이저 동작 이해
2. Masked Language Model (MLM) 체험
3. BERT 임베딩 추출 (CLS, Mean Pooling)
4. Hugging Face Pipeline API 활용 (감성분석, NER, QA)
5. BERT 변형 모델 비교

실행 방법:
    cd practice/chapter5
    pip install -r code/requirements.txt
    python code/5-1-bert-basics.py
"""

import warnings
warnings.filterwarnings("ignore")

import torch
from transformers import (
    BertTokenizer,
    BertModel,
    BertForMaskedLM,
    AutoTokenizer,
    AutoModel,
    pipeline,
)


def wordpiece_tokenizer_demo():
    """WordPiece 토크나이저 동작 이해"""
    print("=" * 60)
    print("[1] WordPiece 토크나이저")
    print("=" * 60)

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    print(f"\n어휘 크기: {tokenizer.vocab_size:,}")

    # 다양한 단어 토큰화
    print("\n[WordPiece 분해 결과]")
    words = [
        "playing",
        "tokenization",
        "unbelievable",
        "transformer",
        "antidisestablishmentarianism",
    ]
    for word in words:
        tokens = tokenizer.tokenize(word)
        print(f"  {word:35} -> {tokens}")

    # 문장 토큰화 + Special Tokens
    print("\n[문장 토큰화]")
    sent1 = "BERT uses transformers."
    sent2 = "It was developed by Google."
    encoding = tokenizer(sent1, sent2, return_tensors="pt")

    tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"][0])
    print(f"  문장 1: \"{sent1}\"")
    print(f"  문장 2: \"{sent2}\"")
    print(f"  토큰: {tokens}")
    print(f"  token_type_ids: {encoding['token_type_ids'][0].tolist()}")
    print(f"  → 0=문장A, 1=문장B")

    # Special Tokens 설명
    print("\n[Special Tokens]")
    special = {"[CLS]": 101, "[SEP]": 102, "[MASK]": 103, "[PAD]": 0, "[UNK]": 100}
    for token, tid in special.items():
        print(f"  {token:8} (ID={tid:3d})")

    return tokenizer


def mlm_demo():
    """Masked Language Model (MLM) 체험"""
    print("\n" + "=" * 60)
    print("[2] Masked Language Model (MLM)")
    print("=" * 60)

    unmasker = pipeline("fill-mask", model="bert-base-uncased")

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
            token = result["token_str"]
            score = result["score"]
            print(f"    {i}. {token} ({score:.4f})")


def embedding_extraction_demo():
    """BERT 임베딩 추출"""
    print("\n" + "=" * 60)
    print("[3] BERT 임베딩 추출")
    print("=" * 60)

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")
    model.eval()

    text = "BERT produces contextualized word embeddings."
    inputs = tokenizer(text, return_tensors="pt")

    print(f"\n입력 텍스트: \"{text}\"")
    print(f"토큰: {tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])}")

    with torch.no_grad():
        outputs = model(**inputs)

    last_hidden = outputs.last_hidden_state
    pooler = outputs.pooler_output

    print(f"\n[출력 텐서 크기]")
    print(f"  last_hidden_state: {last_hidden.shape}")
    print(f"    - (batch_size=1, seq_len={last_hidden.shape[1]}, hidden_size={last_hidden.shape[2]})")
    print(f"  pooler_output: {pooler.shape}")
    print(f"    - [CLS] 토큰의 변환된 표현")

    # 3가지 임베딩 추출 방법
    print(f"\n[임베딩 추출 방법]")

    # 방법 1: [CLS] 토큰
    cls_emb = last_hidden[0, 0, :]
    print(f"  1. [CLS] 토큰: shape={cls_emb.shape}, mean={cls_emb.mean():.4f}")

    # 방법 2: Pooler Output
    print(f"  2. Pooler Output: shape={pooler[0].shape}, mean={pooler[0].mean():.4f}")

    # 방법 3: Mean Pooling
    attention_mask = inputs["attention_mask"]
    mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
    sum_emb = torch.sum(last_hidden * mask_expanded, dim=1)
    mean_pooled = sum_emb / mask_expanded.sum(dim=1)
    print(f"  3. Mean Pooling: shape={mean_pooled[0].shape}, mean={mean_pooled[0].mean():.4f}")

    return model, tokenizer


def pipeline_demo():
    """Hugging Face Pipeline API 데모"""
    print("\n" + "=" * 60)
    print("[4] Hugging Face Pipeline API")
    print("=" * 60)

    # 감성 분석
    print("\n--- 감성 분석 ---")
    classifier = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
    )
    sentences = [
        "I love this movie! It's absolutely fantastic.",
        "This is the worst product I've ever bought.",
        "The weather is okay today.",
    ]
    for sentence in sentences:
        result = classifier(sentence)[0]
        label = result["label"]
        score = result["score"]
        print(f"  {label} ({score:.4f}): {sentence}")

    # NER
    print("\n--- 개체명 인식 (NER) ---")
    ner = pipeline(
        "ner",
        model="dbmdz/bert-large-cased-finetuned-conll03-english",
        aggregation_strategy="simple",
    )
    text = "Apple Inc. was founded by Steve Jobs in Cupertino, California."
    print(f"  텍스트: \"{text}\"")
    entities = ner(text)
    print("  발견된 개체:")
    for entity in entities:
        etype = entity["entity_group"]
        word = entity["word"]
        score = entity["score"]
        emoji = {"PER": "PER", "ORG": "ORG", "LOC": "LOC", "MISC": "MISC"}.get(etype, etype)
        print(f"    [{emoji}] {word} (신뢰도: {score:.4f})")

    # QA
    print("\n--- 질의응답 (QA) ---")
    qa = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
    context = (
        "BERT (Bidirectional Encoder Representations from Transformers) is a "
        "language model developed by Google AI Language team in 2018. "
        "It was pre-trained on BooksCorpus and English Wikipedia."
    )
    questions = ["Who developed BERT?", "When was BERT created?"]
    for q in questions:
        result = qa(question=q, context=context)
        print(f"  Q: {q}")
        print(f"  A: {result['answer']} (신뢰도: {result['score']:.4f})")


def bert_variants_demo():
    """BERT 변형 모델 비교"""
    print("\n" + "=" * 60)
    print("[5] BERT 변형 모델 비교")
    print("=" * 60)

    models_info = [
        ("bert-base-uncased", "BERT-Base"),
        ("distilbert-base-uncased", "DistilBERT"),
        ("albert-base-v2", "ALBERT-Base"),
    ]

    print("\n[모델별 파라미터 수]")
    for model_name, display_name in models_info:
        try:
            model = AutoModel.from_pretrained(model_name)
            num_params = sum(p.numel() for p in model.parameters())
            print(f"  {display_name:20}: {num_params:>12,} parameters ({num_params / 1e6:.1f}M)")
            del model
        except Exception as e:
            print(f"  {display_name:20}: (로드 실패: {e})")

    print("\n[모델 특징 비교]")
    print("  BERT-Base  : 12 layers, 768 hidden, 12 heads — 기본 모델")
    print("  DistilBERT : 6 layers, 768 hidden — 지식 증류, 97% 성능, 40% 작음")
    print("  ALBERT-Base: 12 layers, 파라미터 공유 — 파라미터 효율 극대화")


def main():
    """메인 함수"""
    print("=" * 60)
    print("제5장 실습 — BERT 기본 사용법")
    print("=" * 60)

    # 1. WordPiece 토크나이저
    wordpiece_tokenizer_demo()

    # 2. MLM 체험
    mlm_demo()

    # 3. 임베딩 추출
    embedding_extraction_demo()

    # 4. Pipeline API
    pipeline_demo()

    # 5. 변형 모델 비교
    bert_variants_demo()

    print("\n" + "=" * 60)
    print("BERT 기본 실습 완료")
    print("=" * 60)


if __name__ == "__main__":
    main()
