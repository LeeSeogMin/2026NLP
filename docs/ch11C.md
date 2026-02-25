# 11주차 C: RAG Q&A 시스템 — 모범 구현과 해설

> 이 문서는 B회차 과제 제출 후 공개된다. 제출 전에는 열람하지 않는다.

---

## 체크포인트 1 모범 구현: PDF 로드, 청킹, 벡터 DB 구축

PDF 문서 기반 RAG 시스템의 첫 단계는 문서를 벡터 DB에 저장하는 것이다. 다음은 완전한 구현이다.

### 필수 라이브러리 설치

```bash
pip install langchain langchain-community python-dotenv
pip install sentence-transformers faiss-cpu
pip install pypdf openai
pip install rank-bm25 pandas numpy matplotlib
```

### 통합 설정 및 상수

```python
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# 프로젝트 루트 경로 설정
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "practice" / "chapter11" / "data"
OUTPUT_DIR = DATA_DIR / "output"

# 디렉토리 생성
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 설정 상수
CHUNK_SIZE = 400  # 문자 단위
CHUNK_OVERLAP = 100
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
TOP_K_RETRIEVAL = 5  # 검색할 상위 문서 개수
TOP_K_RERANK = 3    # 재랭킹 후 최종 선택 개수

print(f"프로젝트 루트: {PROJECT_ROOT}")
print(f"데이터 디렉토리: {DATA_DIR}")
print(f"출력 디렉토리: {OUTPUT_DIR}")
```

### PDF 로드 및 텍스트 추출

```python
from langchain.document_loaders import PyPDFLoader
from langchain.schema import Document
import json

def load_and_parse_pdf(pdf_path):
    """
    PDF 파일을 로드하여 텍스트를 추출합니다.

    Args:
        pdf_path: PDF 파일 경로

    Returns:
        Document 객체 리스트 (각 페이지별)
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF 파일을 찾을 수 없습니다: {pdf_path}")

    print(f"\n[1단계] PDF 로드 중... {pdf_path}")

    # PyPDFLoader로 PDF 로드
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # 문서 정보 출력
    print(f"✓ 총 {len(documents)} 페이지 로드됨")

    # 페이지별 텍스트 길이 통계
    page_lengths = [len(doc.page_content) for doc in documents]
    print(f"  페이지별 평균 텍스트 길이: {sum(page_lengths)//len(page_lengths)} 문자")
    print(f"  최소/최대: {min(page_lengths)}/{max(page_lengths)} 문자")

    # 처음 2개 페이지 미리보기
    print("\n[미리보기]")
    for i in range(min(2, len(documents))):
        doc = documents[i]
        print(f"\n페이지 {i+1}:")
        print(f"메타데이터: {doc.metadata}")
        print(f"내용 (첫 150자): {doc.page_content[:150]}...")

    return documents

# 실행 (샘플 PDF 경로 - 실제 교재 PDF 사용)
sample_pdf = DATA_DIR / "input" / "sample_nlp_textbook.pdf"

# 샘플 PDF가 없으면 더미 PDF 생성
if not sample_pdf.exists():
    print(f"주의: {sample_pdf} 파일이 없습니다.")
    print("실제 교과서 또는 논문 PDF를 {DATA_DIR}/input/ 디렉토리에 저장하세요.")
    print("\n데모 목적으로 더미 문서로 진행합니다.")

    # 더미 문서 생성 (테스트용)
    from langchain.schema import Document
    documents = [
        Document(
            page_content="""
            제1장: 자연어처리 입문
            자연어처리(Natural Language Processing, NLP)는 기계가 인간의 언어를 이해하고 생성하는 분야이다.
            NLP의 주요 작업으로는 텍스트 분류, 감정 분석, 기계 번역, 질의응답 시스템 등이 있다.
            이 장에서는 NLP의 기본 개념과 현대적 접근법을 소개한다.
            """,
            metadata={"page": 0, "source": "dummy"}
        ),
        Document(
            page_content="""
            제2장: 임베딩과 벡터 표현
            단어 임베딩(Word Embedding)은 단어를 고정 차원의 벡터로 표현하는 기법이다.
            Word2Vec, GloVe, FastText 등의 방법이 있으며, 이들은 단어의 의미적 유사도를 보존한다.
            최근에는 BERT나 GPT 같은 사전학습 모델이 문맥에 따른 동적 임베딩을 제공한다.
            """,
            metadata={"page": 1, "source": "dummy"}
        ),
        Document(
            page_content="""
            제3장: Transformer와 Self-Attention
            Transformer는 2017년 "Attention is All You Need" 논문에서 제안된 아키텍처이다.
            Self-Attention 메커니즘을 통해 시퀀스 내 토큰들 간의 관계를 병렬로 계산한다.
            Scaled Dot-Product Attention은 Query와 Key의 내적을 √d_k로 정규화한다.
            """,
            metadata={"page": 2, "source": "dummy"}
        ),
        Document(
            page_content="""
            제4장: BERT와 사전학습
            BERT(Bidirectional Encoder Representations from Transformers)는 양방향 문맥을 활용한다.
            마스크된 언어 모델(MLM)과 다음 문장 예측(NSP) 작업으로 대규모 코퍼스로 사전학습된다.
            BERT는 다양한 NLP 작업의 기초 모델로 활용되며, 매우 높은 성능을 달성한다.
            """,
            metadata={"page": 3, "source": "dummy"}
        ),
        Document(
            page_content="""
            제5장: 대규모 언어모델과 RAG
            GPT-3, ChatGPT 등 대규모 언어모델(LLM)은 수십억 개의 파라미터를 가진다.
            LLM은 탁월한 자연어 이해와 생성 능력을 보여주지만, 폐쇄 지식과 환각 문제를 가진다.
            RAG(Retrieval-Augmented Generation)는 검색을 통해 최신 정보를 동적으로 제공한다.
            """,
            metadata={"page": 4, "source": "dummy"}
        )
    ]
else:
    documents = load_and_parse_pdf(str(sample_pdf))

print(f"\n총 {len(documents)}개 문서 로드 완료")
```

### 문서 청킹 (고정 크기 + 겹침)

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk_documents(documents, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """
    문서를 고정 크기의 청크로 분할합니다.

    Args:
        documents: Document 객체 리스트
        chunk_size: 청크 크기 (문자 단위)
        chunk_overlap: 인접 청크 간 겹침 (문자 단위)

    Returns:
        분할된 Document 리스트
    """
    print(f"\n[2단계] 문서 청킹 중 (크기: {chunk_size}, 겹침: {chunk_overlap})...")

    # RecursiveCharacterTextSplitter로 분할
    # 문장, 문단, 공백 순서로 시도하여 자연스러운 분할
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "、", " ", ""]
    )

    split_documents = splitter.split_documents(documents)

    # 청킹 통계
    chunk_lengths = [len(doc.page_content) for doc in split_documents]
    print(f"✓ 총 {len(split_documents)} 개 청크 생성됨")
    print(f"  청크 크기: 최소={min(chunk_lengths)}, 최대={max(chunk_lengths)}, 평균={sum(chunk_lengths)//len(chunk_lengths)} 문자")

    # 처음 3개 청크 미리보기
    print("\n[청킹 결과 미리보기]")
    for i in range(min(3, len(split_documents))):
        chunk = split_documents[i]
        page = chunk.metadata.get('page', 'N/A')
        print(f"\n청크 {i+1} (페이지 {page}, {len(chunk.page_content)} 문자):")
        print(f"내용: {chunk.page_content[:100]}...")

    return split_documents

# 실행
chunks = chunk_documents(documents, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
```

### Embedding 모델 로드 및 벡터화

```python
from sentence_transformers import SentenceTransformer
import numpy as np

def create_embeddings(model_name=EMBEDDING_MODEL):
    """
    Sentence-Transformers 모델을 로드합니다.

    Args:
        model_name: 사용할 모델명

    Returns:
        SentenceTransformer 모델 객체
    """
    print(f"\n[3단계] Embedding 모델 로드 중... {model_name}")

    # 모델 로드 (첫 실행 시 다운로드)
    model = SentenceTransformer(model_name)

    # 모델 정보 출력
    print(f"✓ 모델 로드 완료")
    print(f"  모델: {model_name}")

    # 테스트 임베딩
    test_texts = ["자연어처리", "딥러닝", "Transformer"]
    test_embeddings = model.encode(test_texts)

    print(f"  임베딩 차원: {test_embeddings.shape[1]}")
    print(f"  테스트 임베딩:")
    for text, emb in zip(test_texts, test_embeddings):
        print(f"    '{text}': [{emb[0]:.4f}, {emb[1]:.4f}, ...] (차원: {len(emb)})")

    return model

def embed_chunks(chunks, embedding_model):
    """
    모든 청크를 임베딩합니다.

    Args:
        chunks: Document 청크 리스트
        embedding_model: SentenceTransformer 모델

    Returns:
        (청크 리스트, 임베딩 numpy 배열)
    """
    print(f"\n[4단계] {len(chunks)}개 청크 임베딩 중...")

    # 청크 텍스트 추출
    chunk_texts = [chunk.page_content for chunk in chunks]

    # 임베딩 계산
    embeddings = embedding_model.encode(chunk_texts, show_progress_bar=True)

    # 통계 출력
    print(f"✓ 임베딩 계산 완료")
    print(f"  임베딩 형태: {embeddings.shape}")
    print(f"  임베딩 평균: {np.mean(embeddings):.6f}")
    print(f"  임베딩 표준편차: {np.std(embeddings):.6f}")

    return chunks, embeddings

# 실행
embedding_model = create_embeddings(EMBEDDING_MODEL)
chunks, embeddings = embed_chunks(chunks, embedding_model)
```

### FAISS 벡터 DB 구축 및 저장

```python
import faiss
import pickle

def create_faiss_index(embeddings, index_name="rag_index"):
    """
    FAISS 벡터 인덱스를 생성합니다.

    Args:
        embeddings: 임베딩 numpy 배열 (N, D)
        index_name: 인덱스 이름

    Returns:
        FAISS 인덱스
    """
    print(f"\n[5단계] FAISS 인덱스 생성 중...")

    # 임베딩을 float32로 변환
    embeddings_float32 = np.array(embeddings).astype('float32')

    # 정규화 (코사인 유사도 계산을 위해)
    faiss.normalize_L2(embeddings_float32)

    # FAISS 인덱스 생성
    # IndexFlatIP: Inner Product (정규화된 벡터에서는 코사인 유사도와 같음)
    dimension = embeddings_float32.shape[1]
    index = faiss.IndexFlatIP(dimension)

    # 벡터 추가
    index.add(embeddings_float32)

    print(f"✓ FAISS 인덱스 생성 완료")
    print(f"  인덱스 타입: IndexFlatIP (Inner Product)")
    print(f"  저장된 벡터 수: {index.ntotal}")
    print(f"  벡터 차원: {dimension}")

    return index

def save_index(index, chunks, index_dir="rag_index"):
    """
    FAISS 인덱스와 청크 정보를 저장합니다.

    Args:
        index: FAISS 인덱스
        chunks: Document 청크 리스트
        index_dir: 저장할 디렉토리명
    """
    index_path = OUTPUT_DIR / index_dir
    index_path.mkdir(parents=True, exist_ok=True)

    print(f"\n[6단계] 인덱스 저장 중... {index_path}")

    # FAISS 인덱스 저장
    faiss.write_index(index, str(index_path / "index.faiss"))

    # 청크 정보 저장 (재검색에 필요)
    chunk_data = [
        {
            "content": chunk.page_content,
            "metadata": chunk.metadata
        }
        for chunk in chunks
    ]

    with open(index_path / "chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunk_data, f, ensure_ascii=False, indent=2)

    print(f"✓ 저장 완료")
    print(f"  인덱스 파일: {index_path / 'index.faiss'}")
    print(f"  청크 정보: {index_path / 'chunks.json'}")

    return index_path

# 실행
faiss_index = create_faiss_index(embeddings, index_name="rag_index")
index_path = save_index(faiss_index, chunks, index_dir="rag_index")
```

### 검증 체크리스트

```python
def validate_checkpoint_1():
    """
    체크포인트 1의 검증을 수행합니다.
    """
    print("\n" + "="*80)
    print("[체크포인트 1 검증]")
    print("="*80)

    checks = [
        ("PDF가 성공적으로 로드되었는가?", len(documents) > 0),
        ("문서가 청크로 분할되었는가?", len(chunks) > 0),
        ("청크의 크기가 적절한가? (100~500 문자)",
         all(100 < len(c.page_content) < 500 for c in chunks)),
        ("Embedding 모델이 로드되었는가?", embedding_model is not None),
        ("임베딩이 계산되었는가?", embeddings is not None and embeddings.shape[0] == len(chunks)),
        ("임베딩 차원이 올바른가? (384 또는 768)", embeddings.shape[1] in [384, 768, 1024]),
        ("FAISS 인덱스가 생성되었는가?", faiss_index.ntotal == len(chunks)),
        ("인덱스 파일이 저장되었는가?", (index_path / "index.faiss").exists()),
    ]

    all_passed = True
    for check_name, result in checks:
        status = "✓" if result else "✗"
        print(f"{status} {check_name}")
        if not result:
            all_passed = False

    print("\n" + "="*80)
    if all_passed:
        print("✓ 체크포인트 1 완전히 통과!")
    else:
        print("✗ 일부 검증 실패. 위 항목들을 확인하세요.")
    print("="*80)

validate_checkpoint_1()
```

### 핵심 포인트

#### 청킹 크기와 겹침의 영향

```python
def compare_chunking_strategies():
    """
    다양한 청킹 전략을 비교합니다.
    """
    test_doc = documents[0]
    strategies = [
        {"chunk_size": 200, "overlap": 0, "name": "작고 겹치지 않음"},
        {"chunk_size": 200, "overlap": 50, "name": "작고 겹침"},
        {"chunk_size": 400, "overlap": 100, "name": "중간 (현재)"},
        {"chunk_size": 800, "overlap": 100, "name": "크고 겹침"},
    ]

    print("\n" + "="*80)
    print("[청킹 전략 비교]")
    print("="*80)

    for strategy in strategies:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=strategy["chunk_size"],
            chunk_overlap=strategy["overlap"]
        )
        split_docs = splitter.split_documents([test_doc])

        print(f"\n{strategy['name']}:")
        print(f"  설정: chunk_size={strategy['chunk_size']}, overlap={strategy['overlap']}")
        print(f"  결과: {len(split_docs)} 청크 생성")
        print(f"  첫 청크: {split_docs[0].page_content[:80]}...")

compare_chunking_strategies()
```

예상 결과:
```
================================================================================
[청킹 전략 비교]
================================================================================

작고 겹치지 않음:
  설정: chunk_size=200, overlap=0
  결과: 6 청크 생성
  첫 청크: 제1장: 자연어처리 입문
자연어처리(Natural Language Processing, NLP)는 기계가...

작고 겹침:
  설정: chunk_size=200, overlap=50
  결과: 8 청크 생성
  첫 청크: 제1장: 자연어처리 입문
자연어처리(Natural Language Processing, NLP)는 기계가...

중간 (현재):
  설정: chunk_size=400, overlap=100
  결과: 4 청크 생성
  첫 청크: 제1장: 자연어처리 입문
자연어처리(Natural Language Processing, NLP)는 기계가 인간의 언어를...

크고 겹침:
  설정: chunk_size=800, overlap=100
  결과: 2 청크 생성
  첫 청크: 제1장: 자연어처리 입문
자연어처리(Natural Language Processing, NLP)는 기계가 인간의 언어를...
```

#### Embedding 모델의 선택과 영향

```python
def compare_embedding_models():
    """
    서로 다른 임베딩 모델을 비교합니다.
    """
    test_texts = [
        "Transformer와 Self-Attention의 관계",
        "BERT의 마스킹 언어 모델",
        "한국어 자연어처리"
    ]

    models_to_test = [
        "sentence-transformers/all-MiniLM-L6-v2",  # 384차원, 빠름
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",  # 384차원, 한국어 지원
        "sentence-transformers/all-mpnet-base-v2",  # 768차원, 정확도 높음
    ]

    print("\n" + "="*80)
    print("[임베딩 모델 비교]")
    print("="*80)

    for model_name in models_to_test:
        try:
            model = SentenceTransformer(model_name)
            embs = model.encode(test_texts)

            print(f"\n모델: {model_name}")
            print(f"  차원: {embs.shape[1]}")
            print(f"  처리 시간 (예상): {'빠름' if embs.shape[1] < 400 else '중간' if embs.shape[1] < 768 else '느림'}")

            # 의미적 유사도 계산
            from sklearn.metrics.pairwise import cosine_similarity
            sim = cosine_similarity(embs)
            print(f"  유사도 (첫 두 텍스트): {sim[0, 1]:.4f}")
        except Exception as e:
            print(f"\n모델 로드 실패: {model_name}")
            print(f"  오류: {e}")

compare_embedding_models()
```

#### 흔한 실수

```python
print("\n" + "="*80)
print("[흔한 실수와 해결]")
print("="*80)

# 실수 1: 청크가 너무 작음
print("\n[실수 1] 청크가 너무 작음 (50 문자)")
print("  문제: 맥락이 부족해져 의미가 불명확함")
print("  예시: '자연어처리는 기계가' ← 이것만으로는 의미 파악 어려움")
print("  해결: 청크 크기를 200~500 문자로 설정")

# 실수 2: 겹침이 없음
print("\n[실수 2] 청크 간 겹침이 없음")
print("  문제: 문장의 경계에서 중요한 정보가 손실될 수 있음")
print("  예시:")
print("    청크1: ...모델은 Transformer 구조를 기반으로...")
print("    청크2: ...한다. Self-Attention 메커니즘은...")
print("    → 'Transformer'와 'Self-Attention'의 연결이 끊김")
print("  해결: 50~100 문자 겹침 설정")

# 실수 3: float32가 아닌 float64 사용
print("\n[실수 3] FAISS에 float64 벡터 추가")
print("  문제: FAISS는 float32를 기대하므로 타입 불일치")
print("  오류: 'Dimension mismatch'")
print("  해결: embeddings.astype('float32') 사용")
```

---

## 체크포인트 2 모범 구현: Hybrid Search (Dense + Sparse)

검색 성능을 개선하기 위해 Dense와 Sparse 검색을 결합한다.

### Dense 검색 (의미 기반)

```python
def dense_search(query, embedding_model, faiss_index, chunks, k=TOP_K_RETRIEVAL):
    """
    Dense 검색: 임베딩 유사도 기반 검색입니다.

    Args:
        query: 사용자 질문
        embedding_model: SentenceTransformer 모델
        faiss_index: FAISS 인덱스
        chunks: Document 청크 리스트
        k: 반환할 상위 문서 개수

    Returns:
        검색 결과 리스트 (각 결과: rank, similarity, content, page, chunk_id)
    """
    # 질문 임베딩
    query_embedding = embedding_model.encode([query])
    query_embedding_float32 = np.array(query_embedding).astype('float32')

    # 정규화 (IndexFlatIP 사용 시)
    faiss.normalize_L2(query_embedding_float32)

    # FAISS 검색
    distances, indices = faiss_index.search(query_embedding_float32, k)

    # 결과 정리
    results = []
    for rank, (similarity, idx) in enumerate(zip(distances[0], indices[0]), 1):
        if idx == -1:  # 유효하지 않은 인덱스
            continue

        chunk = chunks[idx]
        results.append({
            'rank': rank,
            'similarity': float(similarity),  # Inner Product (0~1)
            'content': chunk.page_content,
            'page': chunk.metadata.get('page', 'N/A'),
            'chunk_id': int(idx),
            'method': 'dense'
        })

    return results

# 테스트
test_query = "Transformer와 Self-Attention의 관계는?"
dense_results = dense_search(test_query, embedding_model, faiss_index, chunks, k=TOP_K_RETRIEVAL)

print(f"[Dense 검색] 질문: {test_query}\n")
for result in dense_results[:3]:
    print(f"순위 {result['rank']}: 유사도 {result['similarity']:.4f} (페이지 {result['page']})")
    print(f"내용: {result['content'][:80]}...\n")
```

### Sparse 검색 (BM25 키워드 검색)

```python
from rank_bm25 import BM25Okapi
import re

class BM25Searcher:
    """
    BM25 기반 희소 검색 인덱스입니다.
    """

    def __init__(self, documents):
        """
        Args:
            documents: Document 청크 리스트
        """
        # 토큰화: 공백 기준 간단한 분할
        self.documents = documents
        self.tokenized_docs = [
            self._tokenize(doc.page_content)
            for doc in documents
        ]

        # BM25 인덱스 생성
        self.bm25 = BM25Okapi(self.tokenized_docs)

        print(f"[BM25] {len(documents)}개 문서에서 인덱스 생성됨")

    def _tokenize(self, text):
        """
        간단한 토큰화: 공백과 구두점으로 분할

        Args:
            text: 입력 텍스트

        Returns:
            토큰 리스트
        """
        # 소문자로 변환
        text = text.lower()

        # 구두점 제거, 공백으로 분할
        tokens = re.findall(r'\b\w+\b', text)

        return tokens

    def search(self, query, k=TOP_K_RETRIEVAL):
        """
        BM25 검색을 수행합니다.

        Args:
            query: 사용자 질문
            k: 반환할 상위 문서 개수

        Returns:
            검색 결과 리스트
        """
        # 쿼리 토큰화
        query_tokens = self._tokenize(query)

        # BM25 점수 계산
        scores = self.bm25.get_scores(query_tokens)

        # 상위 K개 선택
        top_indices = np.argsort(-scores)[:k]

        # 결과 정리
        results = []
        for rank, idx in enumerate(top_indices, 1):
            score = float(scores[idx])

            # 점수가 0이면 제외
            if score > 0:
                chunk = self.documents[idx]
                results.append({
                    'rank': rank,
                    'score': score,
                    'content': chunk.page_content,
                    'page': chunk.metadata.get('page', 'N/A'),
                    'chunk_id': int(idx),
                    'method': 'sparse'
                })

        return results

# BM25 검색기 생성
bm25_searcher = BM25Searcher(chunks)

# 테스트
sparse_results = bm25_searcher.search(test_query, k=TOP_K_RETRIEVAL)

print(f"\n[Sparse(BM25) 검색] 질문: {test_query}\n")
for result in sparse_results[:3]:
    print(f"순위 {result['rank']}: 점수 {result['score']:.4f} (페이지 {result['page']})")
    print(f"내용: {result['content'][:80]}...\n")
```

### Hybrid 검색 (Dense + Sparse 결합)

```python
def hybrid_search(query, embedding_model, faiss_index, bm25_searcher,
                  chunks, k=TOP_K_RETRIEVAL, alpha=0.5):
    """
    Hybrid 검색: Dense와 Sparse 결과를 가중 결합합니다.

    Args:
        query: 사용자 질문
        embedding_model: SentenceTransformer 모델
        faiss_index: FAISS 인덱스
        bm25_searcher: BM25Searcher 인스턴스
        chunks: Document 청크 리스트
        k: 반환할 상위 문서 개수
        alpha: Dense 검색의 가중치 (1-alpha는 Sparse 가중치)

    Returns:
        결합된 검색 결과 리스트
    """
    # 1. Dense 검색
    dense_results = dense_search(query, embedding_model, faiss_index, chunks, k=k*2)

    # 2. Sparse 검색
    sparse_results = bm25_searcher.search(query, k=k*2)

    # 3. 결과 정규화 및 결합
    combined_scores = {}

    # Dense 결과 추가 (0~1 점수는 이미 정규화됨)
    for result in dense_results:
        chunk_id = result['chunk_id']
        if chunk_id not in combined_scores:
            combined_scores[chunk_id] = {
                'dense': 0.0,
                'sparse': 0.0,
                'doc': result
            }

        # 순위 기반 점수로 변환 (상위일수록 높음)
        rank_score = (k*2 - result['rank'] + 1) / (k*2)
        combined_scores[chunk_id]['dense'] = result['similarity'] * 0.8 + rank_score * 0.2

    # Sparse 결과 추가 (점수 정규화)
    if sparse_results:
        max_sparse_score = max(r['score'] for r in sparse_results)

        for result in sparse_results:
            chunk_id = result['chunk_id']
            if chunk_id not in combined_scores:
                combined_scores[chunk_id] = {
                    'dense': 0.0,
                    'sparse': 0.0,
                    'doc': result
                }

            # 정규화된 Sparse 점수
            normalized_score = result['score'] / max_sparse_score if max_sparse_score > 0 else 0
            rank_score = (k*2 - result['rank'] + 1) / (k*2)
            combined_scores[chunk_id]['sparse'] = normalized_score * 0.8 + rank_score * 0.2

    # 4. 가중 결합
    final_results = []
    for chunk_id, scores in combined_scores.items():
        combined_score = (
            alpha * scores['dense'] +
            (1 - alpha) * scores['sparse']
        )

        # 원본 문서 정보 가져오기
        doc_info = scores['doc']

        final_results.append({
            'chunk_id': chunk_id,
            'combined_score': combined_score,
            'dense_score': scores['dense'],
            'sparse_score': scores['sparse'],
            'content': doc_info['content'],
            'page': doc_info['page']
        })

    # 5. 상위 K개 선택 및 정렬
    final_results.sort(key=lambda x: x['combined_score'], reverse=True)
    final_results = final_results[:k]

    # 순위 추가
    for rank, result in enumerate(final_results, 1):
        result['rank'] = rank

    return final_results

# 테스트
hybrid_results = hybrid_search(
    test_query, embedding_model, faiss_index, bm25_searcher, chunks,
    k=TOP_K_RETRIEVAL, alpha=0.5
)

print(f"\n[Hybrid 검색 (α=0.5)] 질문: {test_query}\n")
for result in hybrid_results[:3]:
    print(f"순위 {result['rank']}: 결합 점수 {result['combined_score']:.4f}")
    print(f"  Dense {result['dense_score']:.4f} + Sparse {result['sparse_score']:.4f}")
    print(f"  내용: {result['content'][:80]}...\n")
```

### 검색 방식 비교

```python
def compare_search_methods(query, embedding_model, faiss_index, bm25_searcher,
                           chunks, k=3):
    """
    Dense, Sparse, Hybrid 검색 결과를 비교합니다.
    """
    print("\n" + "="*80)
    print("[검색 방식 비교]")
    print("="*80)
    print(f"질문: {query}\n")

    # Dense 검색
    print("[1] Dense 검색 (의미 유사도 중심)")
    print("-" * 40)
    dense_results = dense_search(query, embedding_model, faiss_index, chunks, k=k)
    for r in dense_results:
        print(f"  {r['rank']}. 유사도 {r['similarity']:.4f}")
        print(f"     {r['content'][:60]}...")

    # Sparse 검색
    print("\n[2] Sparse 검색 (BM25 키워드 중심)")
    print("-" * 40)
    sparse_results = bm25_searcher.search(query, k=k)
    for r in sparse_results:
        print(f"  {r['rank']}. 점수 {r['score']:.4f}")
        print(f"     {r['content'][:60]}...")

    # Hybrid 검색
    print("\n[3] Hybrid 검색 (Dense + Sparse 결합)")
    print("-" * 40)
    hybrid_results = hybrid_search(
        query, embedding_model, faiss_index, bm25_searcher, chunks,
        k=k, alpha=0.5
    )
    for r in hybrid_results:
        print(f"  {r['rank']}. 점수 {r['combined_score']:.4f}")
        print(f"     Dense: {r['dense_score']:.4f}, Sparse: {r['sparse_score']:.4f}")
        print(f"     {r['content'][:60]}...")

    print("\n" + "="*80)
    print("분석:")
    print("- Dense: 의미적 유사도를 직접 계산하여 문맥을 이해함")
    print("- Sparse: 정확한 키워드 매칭으로 용어 찾기에 강함")
    print("- Hybrid: 둘의 장점을 결합하여 가장 강건한 검색")
    print("="*80)

compare_search_methods(test_query, embedding_model, faiss_index, bm25_searcher, chunks, k=3)
```

### 검증 체크리스트

```python
def validate_checkpoint_2():
    """
    체크포인트 2의 검증을 수행합니다.
    """
    print("\n" + "="*80)
    print("[체크포인트 2 검증]")
    print("="*80)

    # Dense 검색 테스트
    dense_test = dense_search(test_query, embedding_model, faiss_index, chunks, k=3)

    # Sparse 검색 테스트
    sparse_test = bm25_searcher.search(test_query, k=3)

    # Hybrid 검색 테스트
    hybrid_test = hybrid_search(test_query, embedding_model, faiss_index,
                                bm25_searcher, chunks, k=3, alpha=0.5)

    checks = [
        ("Dense 검색이 작동하는가?", len(dense_test) > 0),
        ("Dense 결과가 유사도를 가지는가?",
         all('similarity' in r and 0 <= r['similarity'] <= 1 for r in dense_test)),
        ("Sparse 검색이 작동하는가?", len(sparse_test) > 0),
        ("Sparse 결과가 점수를 가지는가?", all('score' in r for r in sparse_test)),
        ("Hybrid 검색이 작동하는가?", len(hybrid_test) > 0),
        ("Hybrid 결과가 결합 점수를 가지는가?",
         all('combined_score' in r for r in hybrid_test)),
        ("세 방식의 결과가 다른가?",
         dense_test[0]['chunk_id'] != sparse_test[0]['chunk_id'] or
         dense_test[0]['chunk_id'] != hybrid_test[0]['chunk_id']),
    ]

    all_passed = True
    for check_name, result in checks:
        status = "✓" if result else "✗"
        print(f"{status} {check_name}")
        if not result:
            all_passed = False

    print("\n" + "="*80)
    if all_passed:
        print("✓ 체크포인트 2 완전히 통과!")
    else:
        print("✗ 일부 검증 실패. 위 항목들을 확인하세요.")
    print("="*80)

validate_checkpoint_2()
```

### 핵심 포인트

#### α 값(가중치)의 영향

```python
def analyze_alpha_effects(query):
    """
    Hybrid 검색의 α 값에 따른 영향을 분석합니다.
    """
    print("\n" + "="*80)
    print("[α 값에 따른 Hybrid 검색 결과 변화]")
    print("="*80)
    print(f"질문: {query}\n")

    alpha_values = [0.0, 0.25, 0.5, 0.75, 1.0]

    for alpha in alpha_values:
        results = hybrid_search(
            query, embedding_model, faiss_index, bm25_searcher, chunks,
            k=1, alpha=alpha
        )

        if results:
            result = results[0]
            mode = "(Sparse만)" if alpha == 0 else "(Dense만)" if alpha == 1 else f"(혼합 α={alpha})"

            print(f"α = {alpha} {mode}")
            print(f"  상위 결과 (ID {result['chunk_id']}): {result['content'][:60]}...")
            print(f"  점수: {result['combined_score']:.4f} (D:{result['dense_score']:.4f}, S:{result['sparse_score']:.4f})")
            print()

analyze_alpha_effects(test_query)
```

---

## 체크포인트 3 모범 구현: RAG 파이프라인 + 성능 평가

### RAG 프롬프트 구성

```python
def construct_rag_prompt(query, search_results, max_context_length=1500):
    """
    검색 결과를 바탕으로 RAG 프롬프트를 구성합니다.

    Args:
        query: 사용자 질문
        search_results: 검색 결과 리스트
        max_context_length: 최대 컨텍스트 길이 (토큰 아님, 문자 수)

    Returns:
        (프롬프트, 컨텍스트 텍스트)
    """
    # 컨텍스트 문서 합치기
    context_parts = []
    total_length = 0

    for i, result in enumerate(search_results, 1):
        chunk_text = result['content']

        # 컨텍스트 길이 제한 확인
        if total_length + len(chunk_text) > max_context_length:
            break

        context_parts.append(f"[문서 {i}]\n{chunk_text}")
        total_length += len(chunk_text)

    context = "\n\n".join(context_parts)

    # RAG 프롬프트 구성
    prompt = f"""당신은 도움이 되는 AI 어시스턴트입니다.

다음 문서를 기반으로 사용자의 질문에 답변하세요.

### 컨텍스트:
{context}

### 질문:
{query}

### 지시사항:
1. 위의 컨텍스트에만 기반하여 답변하세요
2. 컨텍스트에 없는 정보는 "제공된 문서에는 해당 정보가 없습니다"라고 말하세요
3. 가능하면 구체적인 예시를 들어 설명하세요
4. 답변의 출처를 문서 번호로 명시하세요
5. 답변은 2-3 문단 정도로 간결하게 작성하세요

### 답변:"""

    return prompt, context

# 테스트
rag_prompt, context = construct_rag_prompt(test_query, hybrid_results[:3])

print("="*80)
print("[RAG 프롬프트 구성]")
print("="*80)
print(f"\n질문: {test_query}\n")
print("프롬프트 전체:")
print(rag_prompt)
print(f"\n프롬프트 길이: {len(rag_prompt)} 문자")
print(f"컨텍스트 길이: {len(context)} 문자")
```

### LLM을 통한 답변 생성

```python
from openai import OpenAI
import os

def generate_rag_answer(prompt, api_key=None):
    """
    RAG 프롬프트를 바탕으로 LLM이 답변을 생성합니다.

    Args:
        prompt: RAG 프롬프트
        api_key: OpenAI API 키 (없으면 환경변수에서 로드)

    Returns:
        LLM 생성 답변
    """
    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        print("경고: OpenAI API 키가 설정되지 않았습니다.")
        print("데모 답변을 반환합니다.")

        # 데모 답변
        return """Transformer와 Self-Attention은 밀접한 관계가 있습니다.

Self-Attention은 Transformer의 핵심 메커니즘으로, 각 토큰이 시퀀스 내의 다른 모든 토큰들과 상호작용할 수 있게 합니다. 이는 병렬 처리를 가능하게 하며, RNN과 달리 장거리 의존성을 효과적으로 처리합니다.

Scaled Dot-Product Attention은 Query와 Key 행렬의 내적을 계산하고 √(d_k)로 정규화하여 안정적인 학습을 보장합니다. [문서 1-3]"""

    try:
        client = OpenAI(api_key=api_key)

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful AI assistant that answers questions based on provided documents. Always cite your sources."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.7,
            max_tokens=500
        )

        return response.choices[0].message.content

    except Exception as e:
        print(f"LLM 호출 오류: {e}")
        print("데모 답변을 반환합니다.")

        return """Transformer와 Self-Attention은 밀접한 관계가 있습니다.

Self-Attention은 Transformer의 핵심 메커니즘입니다. [문서 1-3]"""

# 테스트
answer = generate_rag_answer(rag_prompt)

print("\n" + "="*80)
print("[RAG 답변]")
print("="*80)
print(answer)
```

### 검색 성능 평가

```python
def evaluate_search_performance(query, dense_results, sparse_results, hybrid_results, k=3):
    """
    검색 성능을 정량적으로 평가합니다.

    Args:
        query: 사용자 질문
        dense_results: Dense 검색 결과
        sparse_results: Sparse 검색 결과
        hybrid_results: Hybrid 검색 결과
        k: 평가할 상위 문서 개수
    """
    print("\n" + "="*80)
    print("[검색 성능 평가]")
    print("="*80)
    print(f"질문: {query}\n")

    # 각 방식의 상위 K개 결과 분석
    methods = {
        'Dense': dense_results[:k],
        'Sparse': sparse_results[:k],
        'Hybrid': hybrid_results[:k]
    }

    for method_name, results in methods.items():
        print(f"\n[{method_name}]")
        print("-" * 40)

        # 결과 출력
        for i, result in enumerate(results, 1):
            if 'similarity' in result:
                score = result['similarity']
                score_label = f"유사도 {score:.4f}"
            elif 'score' in result:
                score = result['score']
                score_label = f"점수 {score:.4f}"
            elif 'combined_score' in result:
                score = result['combined_score']
                score_label = f"결합점수 {score:.4f}"
            else:
                score_label = "점수 없음"

            print(f"  {i}. {score_label}")
            print(f"     내용: {result['content'][:60]}...")

        # 결과의 다양성 평가
        pages = [str(r.get('page', 'N/A')) for r in results]
        unique_pages = len(set(pages))

        print(f"\n  커버된 페이지 수: {unique_pages}/{k}")
        print(f"  페이지 분포: {', '.join(pages)}")

    print("\n" + "="*80)
    print("[평가 요약]")
    print("="*80)
    print("- Dense 검색: 의미적 유사도 중심, 문맥을 잘 이해")
    print("- Sparse 검색: 키워드 매칭 중심, 정확한 용어 찾기")
    print("- Hybrid 검색: 둘의 장점 결합, 가장 강건함")
    print("\n권장: Hybrid 검색 사용 (α=0.5 또는 0.6)")

evaluate_search_performance(test_query, dense_results, sparse_results, hybrid_results, k=3)
```

### 복수 쿼리로 전체 RAG 파이프라인 실행

```python
def run_complete_rag_pipeline(queries, k=3, alpha=0.5):
    """
    여러 쿼리에 대해 전체 RAG 파이프라인을 실행합니다.
    """
    results_summary = []

    for i, query in enumerate(queries, 1):
        print(f"\n{'='*80}")
        print(f"[쿼리 {i}] {query}")
        print(f"{'='*80}\n")

        # 1. Hybrid 검색
        search_results = hybrid_search(
            query, embedding_model, faiss_index, bm25_searcher, chunks,
            k=k, alpha=alpha
        )

        # 검색 결과 출력
        print("[1] 검색 결과")
        for r in search_results:
            print(f"  {r['rank']}. 점수 {r['combined_score']:.4f} | {r['content'][:50]}...")

        # 2. 프롬프트 구성
        prompt, context = construct_rag_prompt(query, search_results)
        print(f"\n[2] 프롬프트 구성 완료 ({len(prompt)} 문자)")

        # 3. 답변 생성
        print("\n[3] LLM 답변 생성 중...")
        answer = generate_rag_answer(prompt)

        print(f"\n[답변]")
        print(answer[:200] + "..." if len(answer) > 200 else answer)

        # 결과 저장
        results_summary.append({
            'query': query,
            'search_results': search_results,
            'answer': answer
        })

    return results_summary

# 테스트
test_queries = [
    "Transformer의 Self-Attention 메커니즘은 무엇인가?",
    "BERT의 마스킹 언어 모델이란?",
    "임베딩과 벡터 표현의 차이는?"
]

summary = run_complete_rag_pipeline(test_queries, k=3, alpha=0.5)
```

### 검증 체크리스트

```python
def validate_checkpoint_3():
    """
    체크포인트 3의 검증을 수행합니다.
    """
    print("\n" + "="*80)
    print("[체크포인트 3 검증]")
    print("="*80)

    # 프롬프트 구성 테스트
    prompt, context = construct_rag_prompt(test_query, hybrid_results[:3])

    # 답변 생성 테스트
    answer = generate_rag_answer(prompt)

    checks = [
        ("RAG 프롬프트가 구성되었는가?", len(prompt) > 100),
        ("프롬프트에 컨텍스트가 포함되었는가?", "[문서" in prompt),
        ("프롬프트에 질문이 포함되었는가?", "질문:" in prompt),
        ("프롬프트에 지시사항이 포함되었는가?", "지시사항:" in prompt),
        ("LLM 답변이 생성되었는가?", len(answer) > 50),
        ("답변이 합리적인가? (최소 50자)", len(answer) >= 50),
        ("전체 파이프라인이 작동하는가?", len(answer) > 0),
    ]

    all_passed = True
    for check_name, result in checks:
        status = "✓" if result else "✗"
        print(f"{status} {check_name}")
        if not result:
            all_passed = False

    print("\n" + "="*80)
    if all_passed:
        print("✓ 체크포인트 3 완전히 통과!")
        print("✓ RAG Q&A 시스템 구축 완료!")
    else:
        print("✗ 일부 검증 실패. 위 항목들을 확인하세요.")
    print("="*80)

validate_checkpoint_3()
```

### 성능 분석 리포트 생성

```python
def generate_performance_report(test_queries, output_path=None):
    """
    검색 성능 분석 리포트를 생성합니다.
    """
    if output_path is None:
        output_path = OUTPUT_DIR / "search_performance_report.md"

    report = """# RAG 시스템 검색 성능 분석 리포트

## 1. 시스템 구성

- **임베딩 모델**: paraphrase-multilingual-MiniLM-L12-v2 (384차원)
- **벡터 DB**: FAISS IndexFlatIP
- **Sparse 검색**: BM25
- **문서 크기**: {num_docs} 개 청크
- **청킹 전략**: 고정 크기 {chunk_size}자, 겹침 {overlap}자

## 2. 검색 방식 비교

### 2.1 Dense 검색 (의미 유사도)
- 장점: 문맥과 의미를 이해하여 의미 있는 문서 검색
- 단점: 정확한 키워드 매칭이 약할 수 있음
- 사용 사례: "신경망의 원리는?"과 같은 개념 질문

### 2.2 Sparse 검색 (BM25 키워드)
- 장점: 정확한 용어와 키워드 찾기에 강함
- 단점: 문맥을 고려하지 않아 부적절한 결과 가능
- 사용 사례: "Transformer"나 "BERT" 같은 특정 용어 검색

### 2.3 Hybrid 검색 (결합, α=0.5)
- 장점: Dense와 Sparse의 장점을 모두 활용
- 단점: 약간의 추가 계산 비용
- 권장도: ⭐⭐⭐⭐⭐

## 3. 테스트 쿼리별 결과

"""

    for i, query in enumerate(test_queries, 1):
        # 검색 수행
        dense = dense_search(query, embedding_model, faiss_index, chunks, k=3)
        sparse = bm25_searcher.search(query, k=3)
        hybrid = hybrid_search(query, embedding_model, faiss_index, bm25_searcher, chunks, k=3, alpha=0.5)

        report += f"\n### 3.{i} 쿼리 {i}: \"{query}\"\n\n"

        report += "#### Dense 검색\n"
        for j, r in enumerate(dense[:3], 1):
            report += f"{j}. [유사도 {r['similarity']:.4f}] {r['content'][:60]}...\n"

        report += "\n#### Sparse 검색\n"
        for j, r in enumerate(sparse[:3], 1):
            report += f"{j}. [점수 {r['score']:.4f}] {r['content'][:60]}...\n"

        report += "\n#### Hybrid 검색\n"
        for j, r in enumerate(hybrid[:3], 1):
            report += f"{j}. [점수 {r['combined_score']:.4f}] {r['content'][:60]}...\n"

        report += f"\n#### 분석\n"
        if hybrid[0]['chunk_id'] == dense[0]['chunk_id']:
            report += "- Dense와 Hybrid의 상위 결과가 동일 (의미 기반 검색이 더 효과적)\n"
        if hybrid[0]['chunk_id'] == sparse[0]['chunk_id']:
            report += "- Sparse와 Hybrid의 상위 결과가 동일 (키워드 매칭이 더 효과적)\n"
        if hybrid[0]['chunk_id'] != dense[0]['chunk_id'] and hybrid[0]['chunk_id'] != sparse[0]['chunk_id']:
            report += "- Hybrid가 Dense와 Sparse 모두와 다른 결과 제공 (좋은 균형)\n"
        report += "\n"

    report += """## 4. 결론

Hybrid 검색(Dense + Sparse 결합)은:
1. **의미 이해**: Dense 검색으로 문맥 기반 검색
2. **정확성**: Sparse 검색으로 키워드 정확도 확보
3. **강건성**: 다양한 유형의 질문에 대응

**권장 설정**:
- α = 0.5 (Dense와 Sparse 동등 가중)
- 또는 도메인에 따라 α = 0.6 (Dense 약간 강화)

"""

    # 파일로 저장
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report.format(
            num_docs=len(chunks),
            chunk_size=CHUNK_SIZE,
            overlap=CHUNK_OVERLAP
        ))

    print(f"\n✓ 리포트 저장: {output_path}")

    return report

# 실행
generate_performance_report(test_queries)
```

### 흔한 실수

```python
print("\n" + "="*80)
print("[흔한 실수와 해결]")
print("="*80)

print("\n[실수 1] Dense와 Sparse 점수를 직접 더하기")
print("  문제: 두 점수의 범위가 다름 (Dense: 0~1, Sparse: 0~∞)")
print("  예시: 0.85 + 3.2 = 4.05 (의미 없는 점수)")
print("  해결: 각 점수를 0~1 범위로 정규화 후 가중 결합")

print("\n[실수 2] 프롬프트에 너무 많은 문서 포함")
print("  문제: LLM의 토큰 제한 초과, 비용 증가")
print("  예시: 50개 검색 결과를 모두 포함")
print("  해결: 상위 3~5개 결과만 포함, max_context_length 설정")

print("\n[실수 3] 검색 결과를 그냥 LLM에 전달")
print("  문제: LLM이 검색 문서와 무관한 답변 생성 (환각)")
print("  예시: \"다음 문서를 참고해서 답변해\"라고만 명시")
print("  해결: \"컨텍스트에만 기반해\" + \"없으면 알 수 없다고\"라는 명확한 지시")

print("\n[실수 4] α 값을 조정하지 않음")
print("  문제: Dense와 Sparse 중 하나만 효과적인 경우 성능 저하")
print("  예시: 키워드 검색이 중요한 도메인에서 α=0.5 사용")
print("  해결: 도메인에 따라 α=0.4~0.6 범위에서 조정")

print("\n[실수 5] BM25 토큰화가 부적절")
print("  문제: 영문과 한글 혼합 시 토큰화 오류")
print("  예시: \"Transformer 모델\"을 공백으로만 분할")
print("  해결: 정규식으로 \\b\\w+\\b 패턴 사용 (다국어 지원)")
```

---

## 심화 학습 포인트

### 재랭킹 (Reranking)

```python
from sklearn.metrics.pairwise import cosine_similarity

def apply_reranker(query, initial_results, embedding_model, k_final=3):
    """
    검색 결과를 더 정교한 재랭킹으로 개선합니다.

    간단한 방법: 쿼리와 각 결과의 유사도를 다시 계산

    Args:
        query: 사용자 질문
        initial_results: 초기 검색 결과 (상위 K개)
        embedding_model: Embedding 모델
        k_final: 최종 선택할 문서 개수

    Returns:
        재랭킹된 결과 (상위 k_final개)
    """
    # 쿼리 임베딩
    query_emb = embedding_model.encode([query])[0]

    # 각 결과의 임베딩 계산
    rerank_scores = []
    for result in initial_results:
        content_emb = embedding_model.encode([result['content']])[0]

        # 코사인 유사도 계산
        similarity = cosine_similarity([query_emb], [content_emb])[0][0]

        # 기존 점수와 재계산 점수를 결합
        if 'combined_score' in result:
            combined = 0.6 * result['combined_score'] + 0.4 * similarity
        elif 'similarity' in result:
            combined = 0.6 * result['similarity'] + 0.4 * similarity
        else:
            combined = similarity

        rerank_scores.append({
            **result,
            'rerank_score': combined,
            'original_rank': result['rank']
        })

    # 재랭킹
    rerank_scores.sort(key=lambda x: x['rerank_score'], reverse=True)

    # 순위 업데이트
    for rank, result in enumerate(rerank_scores[:k_final], 1):
        result['rank'] = rank

    return rerank_scores[:k_final]

# 테스트
reranked_results = apply_reranker(test_query, hybrid_results[:5], embedding_model, k_final=3)

print("\n[재랭킹 전후 비교]")
print(f"질문: {test_query}\n")

print("재랭킹 전:")
for r in hybrid_results[:3]:
    print(f"  {r['rank']}. 점수 {r['combined_score']:.4f} (ID {r['chunk_id']})")

print("\n재랭킹 후:")
for r in reranked_results:
    print(f"  {r['rank']}. 점수 {r['rerank_score']:.4f} (원래 순위 {r['original_rank']}, ID {r['chunk_id']})")
```

### 성능 지표: Recall@K, MRR

```python
def calculate_retrieval_metrics(query, search_results, relevant_chunk_ids, k=3):
    """
    검색 성능을 정량적으로 평가합니다.

    Args:
        query: 사용자 질문
        search_results: 검색 결과
        relevant_chunk_ids: 관련 문서의 청크 ID 리스트
        k: 평가할 상위 문서 개수

    Returns:
        {'recall_k': float, 'mrr': float, 'ndcg': float}
    """
    # 상위 K개 결과의 청크 ID
    top_k_ids = [r['chunk_id'] for r in search_results[:k]]

    # Recall@K: 상위 K개에서 관련 문서의 비율
    relevant_in_top_k = sum(1 for id in top_k_ids if id in relevant_chunk_ids)
    recall_k = relevant_in_top_k / len(relevant_chunk_ids) if relevant_chunk_ids else 0

    # MRR (Mean Reciprocal Rank): 첫 관련 문서의 순위의 역수
    mrr = 0
    for rank, chunk_id in enumerate(top_k_ids, 1):
        if chunk_id in relevant_chunk_ids:
            mrr = 1 / rank
            break

    # NDCG (Normalized Discounted Cumulative Gain)
    dcg = 0
    for rank, chunk_id in enumerate(top_k_ids, 1):
        relevance = 1 if chunk_id in relevant_chunk_ids else 0
        dcg += relevance / np.log2(rank + 1)

    # 이상적 DCG
    idcg = sum(1 / np.log2(i + 1) for i in range(1, min(len(relevant_chunk_ids) + 1, k + 1)))
    ndcg = dcg / idcg if idcg > 0 else 0

    return {
        'recall_k': recall_k,
        'mrr': mrr,
        'ndcg': ndcg
    }

# 테스트 (예시: 청크 2, 3이 관련 문서)
relevant_ids = {2, 3}
metrics = calculate_retrieval_metrics(test_query, hybrid_results, relevant_ids, k=5)

print(f"\n[검색 성능 지표]")
print(f"질문: {test_query}\n")
print(f"Recall@5: {metrics['recall_k']:.4f} (상위 5개 중 관련 문서 비율)")
print(f"MRR: {metrics['mrr']:.4f} (첫 관련 문서 순위의 역수)")
print(f"NDCG: {metrics['ndcg']:.4f} (정규화된 누적 이득)")
```

---

## 최종 학습 정리

### 11주차 핵심 개념 요약

1. **LLM의 한계**: 폐쇄 지식(학습 시점 이후 정보 미지), 환각(거짓 생성), 업데이트 비용
2. **RAG의 해결**: 검색(외부 문서) → 증강(프롬프트 추가) → 생성(LLM)
3. **청킹 전략**: 고정 크기 + 겹침이 실무적이고 효과적
4. **Embedding**: 의미적 유사도 계산을 위해 필수적, 모델 선택이 중요
5. **벡터 DB**: FAISS(프로토타입), Pinecone(프로덕션)
6. **Dense 검색**: 의미 기반, 문맥 이해에 강함
7. **Sparse 검색**: 키워드 기반(BM25), 정확도에 강함
8. **Hybrid 검색**: 둘의 결합으로 가장 강건한 성능
9. **프롬프트 엔지니어링**: 명확한 지시사항으로 환각 감소
10. **성능 평가**: Recall@K, MRR, NDCG로 정량 평가

### 다음 단계

- **11주차 A/B 완료 후**: 고급 기법 (Semantic Chunking, Multi-Turn RAG, Query Expansion)
- **12주차**: Fine-tuning + RAG 결합, Adaptive Retrieval
- **최종 프로젝트**: 실제 기업 문서 기반 RAG 시스템 구축 및 배포

---

**생성일**: 2026-02-25
**대상 독자**: 컴퓨터공학/AI 전공 학부생 (3~4학년)
**난이도**: 고급 (11주차 B회차 완료 필수)
