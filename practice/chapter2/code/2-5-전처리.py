"""
2-5-전처리.py
제2장 실습: 텍스트 데이터 전처리

이 스크립트는 자연어처리의 기본 전처리 단계를 구현한다:
- 토큰화 (Tokenization)
- 정제 (Cleaning)
- 불용어 제거 (Stopword Removal)
- 정규화 (Normalization)

실행 방법:
    python 2-5-전처리.py
"""

import re
from typing import List, Set
from collections import Counter


class TextPreprocessor:
    """텍스트 전처리 클래스"""

    def __init__(self):
        """전처리기 초기화"""
        # 한국어 불용어 리스트 (기본)
        self.korean_stopwords = {
            "이", "그", "저", "것", "수", "등", "들", "및",
            "에", "의", "가", "이", "은", "는", "을", "를",
            "로", "으로", "와", "과", "도", "에서", "까지",
            "부터", "에게", "한테", "께", "더", "만", "뿐",
            "하다", "있다", "되다", "없다", "아니다",
        }

        # 영어 불용어 리스트 (기본)
        self.english_stopwords = {
            "the", "a", "an", "is", "are", "was", "were",
            "be", "been", "being", "have", "has", "had",
            "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "must", "shall",
            "i", "you", "he", "she", "it", "we", "they",
            "this", "that", "these", "those",
            "in", "on", "at", "to", "for", "of", "with",
            "and", "or", "but", "not", "so", "as", "if",
        }

    def clean_text(self, text: str, remove_special: bool = True) -> str:
        """
        텍스트 정제

        Args:
            text: 원본 텍스트
            remove_special: 특수문자 제거 여부
        """
        # 소문자 변환 (영어)
        text = text.lower()

        # HTML 태그 제거
        text = re.sub(r"<[^>]+>", "", text)

        # URL 제거
        text = re.sub(r"http\S+|www\.\S+", "", text)

        # 이메일 제거
        text = re.sub(r"\S+@\S+", "", text)

        if remove_special:
            # 특수문자 제거 (한글, 영어, 숫자, 공백만 유지)
            text = re.sub(r"[^가-힣a-zA-Z0-9\s]", "", text)

        # 연속 공백 제거
        text = re.sub(r"\s+", " ", text)

        return text.strip()

    def tokenize_simple(self, text: str) -> List[str]:
        """간단한 공백 기반 토큰화"""
        return text.split()

    def tokenize_with_regex(self, text: str) -> List[str]:
        """정규식 기반 토큰화 (영어)"""
        # 단어 경계 기준 토큰화
        tokens = re.findall(r"\b\w+\b", text.lower())
        return tokens

    def remove_stopwords(
        self, tokens: List[str], language: str = "korean"
    ) -> List[str]:
        """
        불용어 제거

        Args:
            tokens: 토큰 리스트
            language: 언어 ('korean' 또는 'english')
        """
        if language == "korean":
            stopwords = self.korean_stopwords
        else:
            stopwords = self.english_stopwords

        return [token for token in tokens if token not in stopwords]

    def normalize_text(self, text: str) -> str:
        """
        텍스트 정규화

        - 반복 문자 축소 (ㅋㅋㅋㅋ → ㅋㅋ)
        - 숫자 정규화 (선택)
        """
        # 반복 문자 축소 (3회 이상 → 2회)
        text = re.sub(r"(.)\1{2,}", r"\1\1", text)

        return text

    def preprocess(
        self,
        text: str,
        remove_stopwords: bool = True,
        language: str = "korean",
    ) -> List[str]:
        """
        전체 전처리 파이프라인

        Args:
            text: 원본 텍스트
            remove_stopwords: 불용어 제거 여부
            language: 언어
        """
        # 1. 정규화
        text = self.normalize_text(text)

        # 2. 정제
        text = self.clean_text(text)

        # 3. 토큰화
        tokens = self.tokenize_simple(text)

        # 4. 불용어 제거
        if remove_stopwords:
            tokens = self.remove_stopwords(tokens, language)

        return tokens


def demonstrate_preprocessing():
    """전처리 단계별 데모"""

    preprocessor = TextPreprocessor()

    # 샘플 텍스트
    sample_texts = [
        "안녕하세요!!! 오늘 날씨가 정말 좋네요ㅋㅋㅋㅋㅋ",
        "자연어처리(NLP)는 인공지능의 한 분야입니다.",
        "이메일: test@example.com, 웹사이트: https://example.com",
        "딥러닝은 머신러닝의 하위 분야이며, 신경망을 사용합니다.",
        "The quick brown fox jumps over the lazy dog.",
    ]

    print("=" * 60)
    print("텍스트 전처리 단계별 데모")
    print("=" * 60)

    for i, text in enumerate(sample_texts, 1):
        print(f"\n[텍스트 {i}]")
        print(f"원본: {text}")

        # 정규화
        normalized = preprocessor.normalize_text(text)
        print(f"정규화: {normalized}")

        # 정제
        cleaned = preprocessor.clean_text(normalized)
        print(f"정제: {cleaned}")

        # 토큰화
        tokens = preprocessor.tokenize_simple(cleaned)
        print(f"토큰화: {tokens}")

        # 불용어 제거
        lang = "english" if text[0].isascii() else "korean"
        filtered = preprocessor.remove_stopwords(tokens, lang)
        print(f"불용어 제거: {filtered}")

        print("-" * 40)


def build_vocabulary(corpus: List[str], min_freq: int = 1) -> dict:
    """
    코퍼스에서 어휘 사전 구축

    Args:
        corpus: 문장 리스트
        min_freq: 최소 출현 빈도
    """
    preprocessor = TextPreprocessor()

    # 전체 토큰 수집
    all_tokens = []
    for text in corpus:
        tokens = preprocessor.preprocess(text, remove_stopwords=False)
        all_tokens.extend(tokens)

    # 빈도 계산
    token_counts = Counter(all_tokens)

    # 최소 빈도 이상 토큰만 선택
    vocab = {
        token: count
        for token, count in token_counts.items()
        if count >= min_freq
    }

    # 빈도순 정렬
    vocab = dict(sorted(vocab.items(), key=lambda x: x[1], reverse=True))

    return vocab


def main():
    """텍스트 전처리 실습 메인 함수"""

    print()
    print("╔" + "═" * 58 + "╗")
    print("║         텍스트 데이터 전처리 실습                       ║")
    print("╚" + "═" * 58 + "╝")
    print()

    # === 단계별 데모 ===
    demonstrate_preprocessing()

    # === 어휘 사전 구축 ===
    print("\n" + "=" * 60)
    print("어휘 사전 구축")
    print("=" * 60)

    corpus = [
        "나는 오늘 학교에 갔다",
        "나는 오늘 도서관에서 공부했다",
        "나는 어제 친구를 만났다",
        "오늘 날씨가 정말 좋다",
        "나는 자연어처리를 공부한다",
        "딥러닝은 정말 재미있다",
        "인공지능이 세상을 바꾸고 있다",
    ]

    vocab = build_vocabulary(corpus, min_freq=1)

    print(f"\n총 어휘 수: {len(vocab)}")
    print("\n상위 10개 단어:")
    for i, (token, count) in enumerate(list(vocab.items())[:10], 1):
        print(f"  {i:2}. {token}: {count}회")

    # === 전처리 파이프라인 적용 ===
    print("\n" + "=" * 60)
    print("전처리 파이프라인 적용")
    print("=" * 60)

    preprocessor = TextPreprocessor()

    test_text = "자연어처리(NLP)는 인공지능의 중요한 분야입니다!!!"
    print(f"\n원본: {test_text}")

    processed = preprocessor.preprocess(
        test_text, remove_stopwords=True, language="korean"
    )
    print(f"전처리 결과: {processed}")

    print("\n" + "=" * 60)
    print("실습 완료")
    print("=" * 60)


if __name__ == "__main__":
    main()
