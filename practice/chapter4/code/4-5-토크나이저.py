"""
4장 실습 3: Tokenization 심화
- BPE 알고리즘 밑바닥 구현
- Hugging Face Tokenizer 비교 (BERT WordPiece vs GPT-2 BPE)
- 한국어/영어 토크나이제이션 차이 분석
"""

import os
from collections import Counter

# 재현성을 위한 시드 설정
import random
random.seed(42)


# ============================================================
# 1. BPE (Byte Pair Encoding) 밑바닥 구현
# ============================================================
class SimpleBPE:
    """
    BPE 알고리즘의 간소화 구현

    1. 단어를 글자 단위로 분리
    2. 가장 빈번한 글자 쌍을 찾음
    3. 해당 쌍을 합침
    4. 반복
    """

    def __init__(self, num_merges=10):
        self.num_merges = num_merges
        self.merges = []

    def get_pair_counts(self, vocab):
        """어휘에서 인접 토큰 쌍의 빈도를 계산"""
        pairs = Counter()
        for word, freq in vocab.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += freq
        return pairs

    def merge_pair(self, pair, vocab):
        """어휘에서 특정 쌍을 병합"""
        new_vocab = {}
        bigram = " ".join(pair)
        replacement = "".join(pair)
        for word, freq in vocab.items():
            new_word = word.replace(bigram, replacement)
            new_vocab[new_word] = freq
        return new_vocab

    def fit(self, corpus):
        """학습 데이터에서 BPE 병합 규칙을 학습"""
        # 단어 빈도 계산 (각 단어를 글자 단위로 분리)
        word_freq = Counter()
        for sentence in corpus:
            for word in sentence.split():
                word_freq[word] += 1

        # 각 단어를 글자 단위로 분리 (끝 표시 </w> 추가)
        vocab = {}
        for word, freq in word_freq.items():
            chars = " ".join(list(word)) + " </w>"
            vocab[chars] = freq

        print(f"  초기 어휘:")
        for word, freq in sorted(vocab.items(), key=lambda x: -x[1])[:5]:
            print(f"    '{word}' (빈도: {freq})")

        print(f"\n  BPE 병합 과정 ({self.num_merges}회):")
        for i in range(self.num_merges):
            pairs = self.get_pair_counts(vocab)
            if not pairs:
                break

            best_pair = max(pairs, key=pairs.get)
            best_count = pairs[best_pair]
            self.merges.append(best_pair)

            vocab = self.merge_pair(best_pair, vocab)
            print(f"    병합 {i+1}: '{best_pair[0]}' + '{best_pair[1]}' "
                  f"→ '{''.join(best_pair)}' (빈도: {best_count})")

        print(f"\n  최종 어휘:")
        for word, freq in sorted(vocab.items(), key=lambda x: -x[1])[:5]:
            print(f"    '{word}' (빈도: {freq})")

        return vocab

    def tokenize(self, word):
        """학습된 병합 규칙으로 단어를 토큰화"""
        tokens = list(word) + ["</w>"]
        for pair in self.merges:
            i = 0
            while i < len(tokens) - 1:
                if tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                    tokens[i] = pair[0] + pair[1]
                    del tokens[i + 1]
                else:
                    i += 1
        return tokens


def demo_bpe():
    """BPE 알고리즘 시연"""
    print("-" * 40)
    print("1. BPE 알고리즘 밑바닥 구현")
    print("-" * 40)

    corpus = [
        "low low low low low",
        "lower lower lower",
        "newest newest newest newest",
        "widest widest",
        "new new new new new new",
    ]

    print(f"  학습 말뭉치:")
    for sent in corpus:
        print(f"    {sent}")
    print()

    bpe = SimpleBPE(num_merges=10)
    bpe.fit(corpus)

    # 토큰화 테스트
    print(f"\n  토큰화 테스트:")
    test_words = ["low", "lower", "newest", "new", "lowest"]
    for word in test_words:
        tokens = bpe.tokenize(word)
        print(f"    '{word}' → {tokens}")


# ============================================================
# 2. Hugging Face Tokenizer 비교
# ============================================================
def demo_huggingface_tokenizers():
    """Hugging Face Tokenizer 비교 실험"""
    print("\n" + "-" * 40)
    print("2. Hugging Face Tokenizer 비교")
    print("-" * 40)

    try:
        from transformers import AutoTokenizer

        # BERT (WordPiece) vs GPT-2 (BPE) 토크나이저 로드
        bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")

        test_sentences = [
            "The transformer architecture revolutionized natural language processing.",
            "Unbelievably, the tokenizer handles unknown words gracefully.",
            "PyTorch implementation of self-attention mechanism.",
        ]

        print(f"\n  [영어 토큰화 비교]")
        for sent in test_sentences:
            bert_tokens = bert_tokenizer.tokenize(sent)
            gpt2_tokens = gpt2_tokenizer.tokenize(sent)

            print(f"\n  원문: '{sent}'")
            print(f"    BERT (WordPiece): {bert_tokens}")
            print(f"      → 토큰 수: {len(bert_tokens)}")
            print(f"    GPT-2 (BPE):     {gpt2_tokens}")
            print(f"      → 토큰 수: {len(gpt2_tokens)}")

        # 특수 토큰 비교
        print(f"\n  [특수 토큰 비교]")
        print(f"    BERT 특수 토큰: {bert_tokenizer.special_tokens_map}")
        print(f"    GPT-2 특수 토큰: {gpt2_tokenizer.special_tokens_map}")

        # 어휘 크기 비교
        print(f"\n  [어휘 크기 비교]")
        print(f"    BERT vocab size:  {bert_tokenizer.vocab_size:,}")
        print(f"    GPT-2 vocab size: {gpt2_tokenizer.vocab_size:,}")

        # 서브워드 분해 예시
        print(f"\n  [서브워드 분해 상세 — 'unbelievable']")
        bert_sub = bert_tokenizer.tokenize("unbelievable")
        gpt2_sub = gpt2_tokenizer.tokenize("unbelievable")
        print(f"    BERT: {bert_sub}")
        print(f"    GPT-2: {gpt2_sub}")

        # encode/decode 왕복 테스트
        print(f"\n  [Encode → Decode 왕복 테스트]")
        test = "Attention is all you need."
        bert_ids = bert_tokenizer.encode(test)
        gpt2_ids = gpt2_tokenizer.encode(test)
        print(f"    원문: '{test}'")
        print(f"    BERT IDs:  {bert_ids}")
        print(f"    BERT 복원: '{bert_tokenizer.decode(bert_ids)}'")
        print(f"    GPT-2 IDs: {gpt2_ids}")
        print(f"    GPT-2 복원: '{gpt2_tokenizer.decode(gpt2_ids)}'")

        # 한국어 토크나이제이션 (BERT multilingual)
        print(f"\n  [한국어 토큰화]")
        try:
            mbert_tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
            ko_sentences = [
                "트랜스포머 아키텍처는 자연어처리를 혁신했다.",
                "어텐션 메커니즘이 핵심이다.",
            ]
            for sent in ko_sentences:
                tokens = mbert_tokenizer.tokenize(sent)
                print(f"    원문: '{sent}'")
                print(f"    mBERT: {tokens}")
                print(f"    토큰 수: {len(tokens)}")
        except Exception as e:
            print(f"    mBERT 로드 실패: {e}")

    except ImportError:
        print("  transformers 라이브러리가 설치되지 않았습니다.")
        print("  pip install transformers 로 설치하세요.")


# ============================================================
# 3. 토크나이저 성능 비교 요약
# ============================================================
def tokenizer_comparison_summary():
    """토크나이저 비교 요약"""
    print("\n" + "-" * 40)
    print("3. 토크나이저 비교 요약")
    print("-" * 40)

    print("""
  | 알고리즘      | 핵심 원리                    | 사용 모델         | 어휘 크기   |
  |---------------|------------------------------|-------------------|-------------|
  | BPE           | 빈번한 바이트 쌍 병합        | GPT-2, GPT-3/4    | ~50,000     |
  | WordPiece     | 우도(likelihood) 기반 병합    | BERT, DistilBERT  | ~30,000     |
  | Unigram       | 확률 기반 서브워드 선택       | T5, ALBERT        | 다양        |
  | SentencePiece | 언어 무관 서브워드 분할       | Llama, XLNet      | 다양        |

  [BPE vs WordPiece 핵심 차이]
  - BPE: "가장 자주 함께 등장하는 쌍"을 병합 (빈도 기반)
  - WordPiece: "병합했을 때 전체 우도가 가장 많이 증가하는 쌍"을 병합 (확률 기반)
  - 결과적으로 비슷하지만, WordPiece가 더 의미 있는 서브워드를 만드는 경향""")


def main():
    print("=" * 60)
    print("4장 실습 3: Tokenization 심화")
    print("=" * 60)

    # 1. BPE 밑바닥 구현
    demo_bpe()

    # 2. Hugging Face Tokenizer 비교
    demo_huggingface_tokenizers()

    # 3. 비교 요약
    tokenizer_comparison_summary()

    print("\n" + "=" * 60)
    print("실습 3 완료")
    print("=" * 60)


if __name__ == "__main__":
    main()
