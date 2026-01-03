"""
9-4-bert-tokenizer.py
BERT Tokenizer (WordPiece) 실습

이 스크립트는 BERT의 WordPiece 토크나이저가 어떻게 작동하는지,
그리고 토큰화된 입력이 어떻게 구성되는지를 보여준다.

실행 방법:
    python 9-4-bert-tokenizer.py
"""

import warnings
warnings.filterwarnings('ignore')

from transformers import BertTokenizer, AutoTokenizer


def basic_tokenization():
    """기본 토큰화 데모"""
    print("=" * 60)
    print("[1] 기본 토큰화 (Basic Tokenization)")
    print("=" * 60)

    # BERT 토크나이저 로드
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # 테스트 문장
    text = "Hello, how are you doing today?"

    print(f"\n원본 텍스트: \"{text}\"")

    # 토큰화
    tokens = tokenizer.tokenize(text)
    print(f"\n토큰화 결과: {tokens}")

    # 토큰 → ID 변환
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    print(f"토큰 ID: {token_ids}")

    # encode() 메서드 (Special tokens 포함)
    encoded = tokenizer.encode(text)
    print(f"\nencode() 결과 (Special tokens 포함): {encoded}")

    # 디코딩
    decoded = tokenizer.decode(encoded)
    print(f"decode() 결과: \"{decoded}\"")

    return tokenizer


def wordpiece_demonstration():
    """WordPiece 서브워드 토큰화 데모"""
    print("\n" + "=" * 60)
    print("[2] WordPiece 서브워드 토큰화")
    print("=" * 60)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # 다양한 단어 테스트
    words = [
        "playing",           # play + ##ing
        "unbelievable",      # un + ##bel + ##ie + ##va + ##ble
        "tokenization",      # token + ##ization
        "transformers",      # transform + ##ers
        "antidisestablishmentarianism",  # 매우 긴 단어
        "supercalifragilistic",  # 희귀 단어
    ]

    print("\n[WordPiece 분해 결과]")
    print("-" * 60)

    for word in words:
        tokens = tokenizer.tokenize(word)
        print(f"  {word:35} → {tokens}")

    # ## 접두사 설명
    print("\n[참고] '##' 접두사의 의미:")
    print("  - 단어 내부에서 이어지는 토큰임을 표시")
    print("  - 예: 'playing' → ['play', '##ing']")
    print("  - 'play'는 단어의 시작, '##ing'은 이어지는 부분")

    return tokenizer


def special_tokens_demo():
    """Special Tokens 데모"""
    print("\n" + "=" * 60)
    print("[3] Special Tokens")
    print("=" * 60)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Special tokens 정보
    print("\n[BERT Special Tokens]")
    print(f"  [CLS] token: '{tokenizer.cls_token}' (ID: {tokenizer.cls_token_id})")
    print(f"  [SEP] token: '{tokenizer.sep_token}' (ID: {tokenizer.sep_token_id})")
    print(f"  [MASK] token: '{tokenizer.mask_token}' (ID: {tokenizer.mask_token_id})")
    print(f"  [PAD] token: '{tokenizer.pad_token}' (ID: {tokenizer.pad_token_id})")
    print(f"  [UNK] token: '{tokenizer.unk_token}' (ID: {tokenizer.unk_token_id})")

    # 어휘 크기
    print(f"\n  어휘 크기: {tokenizer.vocab_size:,} tokens")

    # 단일 문장 인코딩
    text = "BERT is amazing!"
    encoded = tokenizer.encode(text)
    tokens = tokenizer.convert_ids_to_tokens(encoded)

    print(f"\n[단일 문장 인코딩]")
    print(f"  입력: \"{text}\"")
    print(f"  토큰: {tokens}")
    print(f"  구조: [CLS] + 문장 + [SEP]")

    return tokenizer


def sentence_pair_encoding():
    """문장 쌍 인코딩 데모"""
    print("\n" + "=" * 60)
    print("[4] 문장 쌍 인코딩 (Sentence Pair)")
    print("=" * 60)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # 두 문장
    sentence1 = "BERT uses transformers."
    sentence2 = "It was developed by Google."

    # 문장 쌍 인코딩
    encoding = tokenizer(
        sentence1,
        sentence2,
        padding=True,
        return_tensors="pt"
    )

    print(f"\n문장 1: \"{sentence1}\"")
    print(f"문장 2: \"{sentence2}\"")

    # 토큰 확인
    tokens = tokenizer.convert_ids_to_tokens(encoding['input_ids'][0])
    print(f"\n[토큰]")
    print(f"  {tokens}")

    print(f"\n[인코딩 결과]")
    print(f"  input_ids shape: {encoding['input_ids'].shape}")
    print(f"  token_type_ids: {encoding['token_type_ids'][0].tolist()}")
    print(f"  attention_mask: {encoding['attention_mask'][0].tolist()}")

    # token_type_ids 설명
    print(f"\n[token_type_ids 설명]")
    print(f"  0: 첫 번째 문장 (sentence1)")
    print(f"  1: 두 번째 문장 (sentence2)")

    # 구조 시각화
    print(f"\n[구조]")
    print(f"  [CLS] 문장1 [SEP] 문장2 [SEP]")
    print(f"    0     0     0     1     1")

    return tokenizer


def batch_encoding():
    """배치 인코딩 데모"""
    print("\n" + "=" * 60)
    print("[5] 배치 인코딩 (Batch Encoding)")
    print("=" * 60)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # 여러 문장
    sentences = [
        "Short sentence.",
        "This is a medium length sentence.",
        "This is a much longer sentence that contains more words.",
    ]

    print("\n[입력 문장들]")
    for i, sent in enumerate(sentences, 1):
        print(f"  {i}. \"{sent}\" (길이: {len(sent.split())} words)")

    # 배치 인코딩 (패딩 포함)
    encoding = tokenizer(
        sentences,
        padding=True,           # 가장 긴 문장에 맞춰 패딩
        truncation=True,        # 최대 길이 초과 시 자르기
        max_length=20,          # 최대 토큰 수
        return_tensors="pt"
    )

    print(f"\n[배치 인코딩 결과]")
    print(f"  input_ids shape: {encoding['input_ids'].shape}")
    print(f"  (배치 크기: {encoding['input_ids'].shape[0]}, "
          f"시퀀스 길이: {encoding['input_ids'].shape[1]})")

    print(f"\n[패딩 적용된 토큰]")
    for i, ids in enumerate(encoding['input_ids']):
        tokens = tokenizer.convert_ids_to_tokens(ids)
        print(f"  {i+1}. {tokens}")

    print(f"\n[Attention Mask] (1=실제 토큰, 0=패딩)")
    for i, mask in enumerate(encoding['attention_mask']):
        print(f"  {i+1}. {mask.tolist()}")

    return tokenizer


def vocabulary_exploration():
    """어휘 탐색 데모"""
    print("\n" + "=" * 60)
    print("[6] 어휘(Vocabulary) 탐색")
    print("=" * 60)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # 어휘 정보
    vocab = tokenizer.get_vocab()

    print(f"\n[어휘 통계]")
    print(f"  총 어휘 수: {len(vocab):,}")

    # Special tokens
    special_tokens = [k for k in vocab.keys() if k.startswith('[') and k.endswith(']')]
    print(f"  Special tokens 수: {len(special_tokens)}")

    # ## 접두사 토큰 (서브워드)
    subword_tokens = [k for k in vocab.keys() if k.startswith('##')]
    print(f"  서브워드(##) 토큰 수: {len(subword_tokens):,}")

    # 일반 토큰
    regular_tokens = len(vocab) - len(special_tokens) - len(subword_tokens)
    print(f"  일반 토큰 수: {regular_tokens:,}")

    # 샘플 토큰 출력
    print(f"\n[서브워드 토큰 샘플]")
    sample_subwords = list(subword_tokens)[:15]
    print(f"  {sample_subwords}")

    # 특정 단어의 ID 확인
    print(f"\n[단어 → ID 예시]")
    sample_words = ['hello', 'world', 'transformer', 'bert', 'ai']
    for word in sample_words:
        if word in vocab:
            print(f"  '{word}' → {vocab[word]}")
        else:
            tokens = tokenizer.tokenize(word)
            print(f"  '{word}' → (분해됨) {tokens}")

    return tokenizer


def main():
    """메인 함수"""
    print("=" * 60)
    print("BERT Tokenizer (WordPiece) 실습")
    print("=" * 60)

    # 1. 기본 토큰화
    basic_tokenization()

    # 2. WordPiece 서브워드
    wordpiece_demonstration()

    # 3. Special Tokens
    special_tokens_demo()

    # 4. 문장 쌍 인코딩
    sentence_pair_encoding()

    # 5. 배치 인코딩
    batch_encoding()

    # 6. 어휘 탐색
    vocabulary_exploration()

    print("\n" + "=" * 60)
    print("BERT Tokenizer 실습 완료")
    print("=" * 60)


if __name__ == "__main__":
    main()
