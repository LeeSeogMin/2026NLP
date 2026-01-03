"""
2-2-ngram모델.py
제2장 실습: N-gram 언어 모델 구현 및 텍스트 생성

이 스크립트는 N-gram 언어 모델을 직접 구현하고,
학습된 모델로 텍스트를 생성하는 방법을 보여준다.

실행 방법:
    python 2-2-ngram모델.py
"""

import random
from collections import defaultdict, Counter
from typing import List, Dict, Tuple
import math


class NGramLanguageModel:
    """N-gram 언어 모델 구현 클래스"""

    def __init__(self, n: int = 2, smoothing_k: float = 1.0):
        """
        N-gram 모델 초기화

        Args:
            n: N-gram의 N (2=Bigram, 3=Trigram)
            smoothing_k: Add-k 스무딩의 k 값
        """
        self.n = n
        self.smoothing_k = smoothing_k
        self.ngram_counts = defaultdict(Counter)  # (n-1)-gram → {다음단어: 빈도}
        self.context_counts = Counter()  # (n-1)-gram 빈도
        self.vocabulary = set()
        self.START_TOKEN = "<s>"
        self.END_TOKEN = "</s>"

    def tokenize(self, text: str) -> List[str]:
        """간단한 토큰화 (공백 기준)"""
        # 소문자 변환 및 공백 기준 분리
        tokens = text.lower().split()
        return tokens

    def add_padding(self, tokens: List[str]) -> List[str]:
        """문장 시작/끝 토큰 추가"""
        padding = [self.START_TOKEN] * (self.n - 1)
        return padding + tokens + [self.END_TOKEN]

    def get_ngrams(self, tokens: List[str]) -> List[Tuple]:
        """토큰 리스트에서 N-gram 추출"""
        padded = self.add_padding(tokens)
        ngrams = []
        for i in range(len(padded) - self.n + 1):
            context = tuple(padded[i:i + self.n - 1])
            next_word = padded[i + self.n - 1]
            ngrams.append((context, next_word))
        return ngrams

    def train(self, corpus: List[str]):
        """
        코퍼스로 N-gram 모델 학습

        Args:
            corpus: 문장 리스트
        """
        print(f"=== {self.n}-gram 모델 학습 시작 ===")
        print(f"학습 문장 수: {len(corpus)}")

        for sentence in corpus:
            tokens = self.tokenize(sentence)
            self.vocabulary.update(tokens)

            ngrams = self.get_ngrams(tokens)
            for context, next_word in ngrams:
                self.ngram_counts[context][next_word] += 1
                self.context_counts[context] += 1

        self.vocabulary.add(self.END_TOKEN)
        print(f"어휘 크기: {len(self.vocabulary)}")
        print(f"고유 {self.n-1}-gram 수: {len(self.context_counts)}")
        print()

    def get_probability(self, context: Tuple, word: str) -> float:
        """
        Add-k 스무딩을 적용한 조건부 확률 계산

        P(word|context) = (count(context, word) + k) / (count(context) + k * V)
        """
        count = self.ngram_counts[context][word]
        context_count = self.context_counts[context]
        vocab_size = len(self.vocabulary)

        # Add-k smoothing
        prob = (count + self.smoothing_k) / (
            context_count + self.smoothing_k * vocab_size
        )
        return prob

    def calculate_perplexity(self, test_sentences: List[str]) -> float:
        """
        테스트 문장들의 Perplexity 계산

        PPL = 2^(-1/N * Σlog₂P(wᵢ|context))
        """
        total_log_prob = 0.0
        total_words = 0

        for sentence in test_sentences:
            tokens = self.tokenize(sentence)
            ngrams = self.get_ngrams(tokens)

            for context, word in ngrams:
                prob = self.get_probability(context, word)
                if prob > 0:
                    total_log_prob += math.log2(prob)
                total_words += 1

        avg_log_prob = total_log_prob / total_words if total_words > 0 else 0
        perplexity = 2 ** (-avg_log_prob)
        return perplexity

    def generate_next_word(self, context: Tuple, temperature: float = 1.0) -> str:
        """
        주어진 문맥에서 다음 단어 생성 (확률적 샘플링)

        Args:
            context: 이전 (n-1)개 단어 튜플
            temperature: 샘플링 온도 (높을수록 다양, 낮을수록 결정적)
        """
        # 각 단어의 확률 계산
        word_probs = {}
        for word in self.vocabulary:
            prob = self.get_probability(context, word)
            word_probs[word] = prob ** (1.0 / temperature)

        # 확률 정규화
        total = sum(word_probs.values())
        word_probs = {w: p / total for w, p in word_probs.items()}

        # 확률적 샘플링
        words = list(word_probs.keys())
        probs = list(word_probs.values())
        return random.choices(words, weights=probs, k=1)[0]

    def generate_sentence(self, max_length: int = 20, temperature: float = 1.0) -> str:
        """
        문장 생성

        Args:
            max_length: 최대 단어 수
            temperature: 샘플링 온도
        """
        # 시작 문맥
        context = tuple([self.START_TOKEN] * (self.n - 1))
        generated = []

        for _ in range(max_length):
            next_word = self.generate_next_word(context, temperature)

            if next_word == self.END_TOKEN:
                break

            generated.append(next_word)
            # 문맥 업데이트 (슬라이딩 윈도우)
            context = tuple(list(context)[1:] + [next_word])

        return " ".join(generated)


def main():
    """N-gram 모델 실습 메인 함수"""

    print("╔" + "═" * 50 + "╗")
    print("║       N-gram 언어 모델 실습                      ║")
    print("╚" + "═" * 50 + "╝")
    print()

    # 샘플 코퍼스 (한국어 문장)
    corpus = [
        "나는 오늘 학교에 갔다",
        "나는 오늘 도서관에서 공부했다",
        "나는 어제 친구를 만났다",
        "오늘 날씨가 정말 좋다",
        "오늘 점심은 김치찌개를 먹었다",
        "나는 자연어처리를 공부한다",
        "딥러닝은 정말 재미있다",
        "인공지능이 세상을 바꾸고 있다",
        "나는 파이썬을 좋아한다",
        "오늘 하루도 열심히 살자",
        "나는 매일 운동을 한다",
        "공부는 꾸준히 하는 것이 중요하다",
    ]

    # === Bigram 모델 ===
    print("=" * 50)
    print("1. Bigram 모델 (N=2)")
    print("=" * 50)

    bigram_model = NGramLanguageModel(n=2, smoothing_k=1.0)
    bigram_model.train(corpus)

    # N-gram 빈도 확인
    print("주요 Bigram 빈도:")
    for context, counter in list(bigram_model.ngram_counts.items())[:5]:
        print(f"  {context} → {dict(counter)}")
    print()

    # Perplexity 계산
    test_sentences = ["나는 오늘 공부했다", "날씨가 좋다"]
    ppl = bigram_model.calculate_perplexity(test_sentences)
    print(f"테스트 문장 Perplexity: {ppl:.2f}")
    print()

    # 문장 생성
    print("생성된 문장 (Bigram):")
    for i in range(3):
        sentence = bigram_model.generate_sentence(max_length=10, temperature=0.8)
        print(f"  {i+1}. {sentence}")
    print()

    # === Trigram 모델 ===
    print("=" * 50)
    print("2. Trigram 모델 (N=3)")
    print("=" * 50)

    trigram_model = NGramLanguageModel(n=3, smoothing_k=1.0)
    trigram_model.train(corpus)

    # Perplexity 비교
    ppl_trigram = trigram_model.calculate_perplexity(test_sentences)
    print(f"테스트 문장 Perplexity: {ppl_trigram:.2f}")
    print()

    # 문장 생성
    print("생성된 문장 (Trigram):")
    for i in range(3):
        sentence = trigram_model.generate_sentence(max_length=10, temperature=0.8)
        print(f"  {i+1}. {sentence}")
    print()

    # === 스무딩 효과 비교 ===
    print("=" * 50)
    print("3. 스무딩 효과 비교")
    print("=" * 50)

    for k in [0.01, 0.1, 1.0, 10.0]:
        model = NGramLanguageModel(n=2, smoothing_k=k)
        model.train(corpus)
        ppl = model.calculate_perplexity(test_sentences)
        print(f"  k={k:>5.2f} → Perplexity: {ppl:.2f}")
    print()

    print("=" * 50)
    print("실습 완료")
    print("=" * 50)


if __name__ == "__main__":
    main()
