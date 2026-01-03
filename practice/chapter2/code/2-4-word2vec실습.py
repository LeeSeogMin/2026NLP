"""
2-4-word2vec실습.py
제2장 실습: Word2Vec 모델 학습 및 활용

이 스크립트는 Gensim 라이브러리를 사용하여
Word2Vec 모델을 학습하고 단어 유사도를 측정하는 방법을 보여준다.

실행 방법:
    pip install gensim
    python 2-4-word2vec실습.py
"""

from pathlib import Path


def check_gensim():
    """Gensim 라이브러리 설치 확인"""
    try:
        import gensim
        print(f"Gensim 버전: {gensim.__version__}")
        return True
    except ImportError:
        print("Gensim이 설치되지 않았습니다.")
        print("설치: pip install gensim")
        return False


def train_word2vec_example():
    """간단한 Word2Vec 모델 학습 예제"""
    from gensim.models import Word2Vec
    import numpy as np

    print("=" * 50)
    print("1. Word2Vec 모델 학습")
    print("=" * 50)

    # 샘플 코퍼스 (토큰화된 문장 리스트)
    corpus = [
        ["나는", "오늘", "학교에", "갔다"],
        ["나는", "오늘", "도서관에서", "공부했다"],
        ["나는", "어제", "친구를", "만났다"],
        ["오늘", "날씨가", "정말", "좋다"],
        ["나는", "자연어처리를", "공부한다"],
        ["딥러닝은", "정말", "재미있다"],
        ["인공지능이", "세상을", "바꾸고", "있다"],
        ["나는", "파이썬을", "좋아한다"],
        ["머신러닝은", "인공지능의", "한", "분야이다"],
        ["자연어처리는", "인공지능의", "중요한", "분야이다"],
        ["나는", "매일", "운동을", "한다"],
        ["공부는", "꾸준히", "하는", "것이", "중요하다"],
        ["딥러닝과", "머신러닝은", "관련이", "있다"],
        ["자연어처리와", "컴퓨터비전은", "인공지능의", "분야이다"],
    ]

    # Word2Vec 모델 학습
    # - vector_size: 임베딩 차원 (작은 코퍼스이므로 50으로 설정)
    # - window: 문맥 윈도우 크기
    # - min_count: 최소 출현 빈도
    # - sg: 0=CBOW, 1=Skip-gram
    model = Word2Vec(
        sentences=corpus,
        vector_size=50,  # 임베딩 차원
        window=3,  # 문맥 윈도우
        min_count=1,  # 최소 빈도 (작은 코퍼스용)
        sg=1,  # Skip-gram
        epochs=100,  # 학습 에폭
        seed=42,
    )

    print(f"학습된 어휘 크기: {len(model.wv)}")
    print(f"임베딩 차원: {model.wv.vector_size}")
    print()

    # === 단어 벡터 확인 ===
    print("=" * 50)
    print("2. 단어 벡터 확인")
    print("=" * 50)

    word = "나는"
    if word in model.wv:
        vector = model.wv[word]
        print(f"'{word}'의 벡터 (처음 10차원):")
        print(f"  {vector[:10]}")
        print(f"  벡터 크기: {len(vector)}")
    print()

    # === 단어 유사도 측정 ===
    print("=" * 50)
    print("3. 단어 유사도 측정 (코사인 유사도)")
    print("=" * 50)

    # 유사한 단어 찾기
    if "나는" in model.wv:
        similar_words = model.wv.most_similar("나는", topn=5)
        print("'나는'과 유사한 단어:")
        for word, score in similar_words:
            print(f"  {word}: {score:.4f}")
    print()

    # 두 단어 간 유사도
    word_pairs = [
        ("인공지능의", "분야이다"),
        ("딥러닝은", "머신러닝은"),
        ("나는", "오늘"),
    ]

    print("단어 쌍 유사도:")
    for w1, w2 in word_pairs:
        if w1 in model.wv and w2 in model.wv:
            sim = model.wv.similarity(w1, w2)
            print(f"  '{w1}' - '{w2}': {sim:.4f}")
    print()

    # === CBOW vs Skip-gram 비교 ===
    print("=" * 50)
    print("4. CBOW vs Skip-gram 비교")
    print("=" * 50)

    # CBOW 모델
    cbow_model = Word2Vec(
        sentences=corpus,
        vector_size=50,
        window=3,
        min_count=1,
        sg=0,  # CBOW
        epochs=100,
        seed=42,
    )

    # Skip-gram 모델 (이미 학습됨)
    skipgram_model = model

    print("'나는'과 유사한 단어 비교:")
    print()
    print("CBOW 모델:")
    if "나는" in cbow_model.wv:
        for word, score in cbow_model.wv.most_similar("나는", topn=3):
            print(f"  {word}: {score:.4f}")

    print()
    print("Skip-gram 모델:")
    if "나는" in skipgram_model.wv:
        for word, score in skipgram_model.wv.most_similar("나는", topn=3):
            print(f"  {word}: {score:.4f}")
    print()

    # === 모델 저장 및 로드 ===
    print("=" * 50)
    print("5. 모델 저장 및 로드")
    print("=" * 50)

    output_dir = Path(__file__).parent.parent / "data" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "word2vec_sample.model"
    model.save(str(model_path))
    print(f"모델 저장: {model_path}")

    # 모델 로드
    loaded_model = Word2Vec.load(str(model_path))
    print(f"모델 로드 완료, 어휘 크기: {len(loaded_model.wv)}")
    print()

    return model


def demonstrate_vector_arithmetic():
    """벡터 연산 데모 (개념 설명용)"""
    print("=" * 50)
    print("6. 벡터 연산 개념 (참고)")
    print("=" * 50)

    print("""
벡터 연산 예시 (대규모 코퍼스에서 학습 시):

  '왕' - '남자' + '여자' ≈ '여왕'
  '파리' - '프랑스' + '한국' ≈ '서울'
  '좋다' - '긍정' + '부정' ≈ '나쁘다'

이러한 연산이 가능한 이유:
  - Word2Vec은 단어의 의미적 관계를 벡터 공간에 인코딩
  - 유사한 문맥에서 등장하는 단어는 유사한 벡터를 가짐
  - 의미적 관계가 벡터 방향으로 표현됨

주의: 작은 코퍼스에서는 이러한 관계가 잘 학습되지 않음
      대규모 사전학습 모델(예: Google News Word2Vec)을 사용하면
      더 정확한 벡터 연산이 가능함
""")


def main():
    """Word2Vec 실습 메인 함수"""

    print()
    print("╔" + "═" * 50 + "╗")
    print("║         Word2Vec 실습                           ║")
    print("╚" + "═" * 50 + "╝")
    print()

    if not check_gensim():
        return

    print()
    train_word2vec_example()
    demonstrate_vector_arithmetic()

    print("=" * 50)
    print("실습 완료")
    print("=" * 50)
    print()
    print("참고: 더 정확한 결과를 위해서는")
    print("      대규모 사전학습 Word2Vec 모델을 사용하세요.")
    print("      예: gensim-data의 'word2vec-google-news-300'")


if __name__ == "__main__":
    main()
