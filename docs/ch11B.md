## 11주차 B회차: RAG 시스템 구축 실습

> **미션**: LangChain과 FAISS/ChromaDB로 PDF 문서 기반 RAG Q&A 시스템을 구축하고 Hybrid Search와 Reranking을 적용하여 검색 성능을 평가할 수 있다

### 수업 타임라인

| 시간 | 내용 | Copilot 사용 |
|------|------|-------------|
| 00:00~00:05 | 조 편성 + 역할 배분 (조원 A/B) | 사용 안 함 |
| 00:05~00:10 | A회차 핵심 리캡 + 과제 스펙 재확인 | 사용 안 함 |
| 00:10~00:55 | 2인1조 Copilot 구현 (체크포인트 3회) | 적극 사용 |
| 00:55~01:00 | Google Classroom 제출 (조별 1부) | 사용 안 함 |
| 01:00~01:20 | 결과 토론 (검색 성능 비교·방법론 분석) | 사용 안 함 |
| 01:20~01:28 | 핵심 정리 | 사용 안 함 |
| 01:28~01:30 | 다음 주 예고 | 사용 안 함 |

---

### A회차 핵심 리캡

**LLM의 근본적 한계**:
- 폐쇄 지식: 학습 시점 이후의 정보를 모른다 (2024년 4월 이후 뉴스 등)
- 환각: 모르는 답을 그럴듯하게 만들어낸다
- 업데이트 비용: 새 정보마다 재학습하면 수주의 시간과 백만 달러대 비용 발생

**RAG(Retrieval-Augmented Generation)의 핵심**:
- 검색(Retrieval): 사용자 질문과 관련된 문서를 빠르게 찾기
- 증강(Augmentation): 찾은 문서를 프롬프트에 추가
- 생성(Generation): 증강된 프롬프트로 LLM이 답변 생성

**직관**: 오픈북 시험처럼, 학생(LLM)이 모든 정보를 암기하는 대신 시험 중 교과서(외부 문서)를 참조하게 한다.

**RAG 아키텍처의 7단계**:
1. 문서 청킹(Chunking): 긴 문서를 "한 입 크기"로 분할
2. 임베딩(Embedding): 각 청크를 고정 차원의 벡터로 변환
3. 벡터 DB: 벡터를 저장하고 유사도 검색을 빠르게 수행
4. 검색기(Retriever): 질문과 유사한 상위 K개 청크 검색
5. 재랭킹(Reranker): 더 정교한 모델로 검색 결과를 재평가
6. 프롬프트 구성: 검색된 문서를 LLM을 위한 프롬프트에 포함
7. 생성기: LLM이 증강된 프롬프트를 바탕으로 답변 생성

**고급 기법**:
- Hybrid Search: Dense(의미 기반) + Sparse(키워드) 결합
- Multi-Turn RAG: 이전 대화의 맥락 유지
- Query Expansion: 질문을 여러 변형으로 확장
- Chain-of-Thought RAG: 복잡한 질문을 단계별로 처리

**실습 연계**: 이론을 바탕으로 LangChain을 사용하여 완전한 RAG 시스템을 직접 구현한다.

---

### 과제 스펙

**과제**: 전공 문서(교과서 또는 논문 기반 PDF) RAG Q&A 시스템 구축 + 검색 성능 평가

**제출 형태**: 조별 1부, Google Classroom 업로드

**파일 구성**:
- 전체 구현 코드 (`*.py`)
- 검색 성능 분석 리포트 (Recall@K, 검색 예시 3개)
- 답변 품질 평가 (생성 답변 3개 + 정성적 평가)

**검증 기준**:
- ✓ PDF 로드, 청킹, 임베딩 및 FAISS/ChromaDB 벡터 DB 구축 완료
- ✓ Dense 검색 구현 및 상위 K개 결과 출력 가능
- ✓ BM25 기반 Sparse 검색 또는 Keyword 검색 구현
- ✓ Hybrid Search (Dense + Sparse) 결과 비교 및 분석
- ✓ 검색 결과를 바탕으로 LLM이 답변 생성 (프롬프트 구성 확인)
- ✓ Recall@K와 MRR(Mean Reciprocal Rank) 등으로 성능 정량 평가

---

### 2인1조 실습

> **Copilot 활용**: 처음에는 간단한 Dense 검색부터 시작하여 "이 검색을 Hybrid로 확장해줄래?", "BM25 검색을 추가해줄래?", "검색 성능을 Recall@K로 평가하는 코드를 작성해줘" 같이 단계적으로 요청한다. Copilot의 제안을 검토하고 수정하면서 각 단계의 의도와 한계를 깊이 있게 이해할 수 있다.

**역할 분담**:
- **조원 A (드라이버)**: 코드 작성 및 실행, 결과 확인
- **조원 B (네비게이터)**: 로직 검토, Copilot 프롬프트 설계, 오류 해석
- **체크포인트마다 역할 교대**: 드라이버와 네비게이터를 번갈아가며 진행하여 두 명 모두 전체 구현을 이해한다

---

#### 체크포인트 1: PDF 로드, 청킹 및 벡터 DB 구축 (15분)

**목표**: PDF 문서를 로드하여 청크로 분할하고, 임베딩을 계산하여 FAISS 또는 ChromaDB 벡터 DB에 저장한다.

**핵심 단계**:

① **필수 라이브러리 설치 및 임포트**

```python
# 필요한 라이브러리 (pip install로 설치)
# pip install langchain langchain-community python-dotenv
# pip install sentence-transformers faiss-cpu
# pip install chroma pypdf openai

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import os
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()
```

② **PDF 로드 및 텍스트 추출**

```python
def load_pdf_documents(pdf_path):
    """
    PDF 파일을 로드하여 LangChain Document 객체로 변환합니다.

    Args:
        pdf_path: PDF 파일 경로

    Returns:
        Document 리스트
    """
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    print(f"총 {len(documents)} 페이지 로드됨")
    for i, doc in enumerate(documents[:2]):  # 첫 2개 페이지 미리보기
        print(f"\n[페이지 {i+1}]")
        print(f"내용: {doc.page_content[:200]}...")
        print(f"메타데이터: {doc.metadata}")

    return documents

# 실행
# 샘플 PDF 경로 (실제 교과서 또는 논문 PDF 사용)
pdf_path = "sample_nlp_textbook.pdf"
documents = load_pdf_documents(pdf_path)
```

예상 결과:
```
총 15 페이지 로드됨

[페이지 1]
내용: 딥러닝 자연어처리
제목: Chapter 1: Introduction to NLP
자연어처리는 기계가 인간의 언어를 이해하고 생성하는...
메타데이터: {'source': 'sample_nlp_textbook.pdf', 'page': 0}

[페이지 2]
내용: 1.1 What is Natural Language Processing?
NLP는 세 가지 주요 작업을 포함한다...
```

③ **문서 청킹 (고정 크기 + 겹침)**

```python
def chunk_documents(documents, chunk_size=400, chunk_overlap=100):
    """
    문서를 고정 크기의 청크로 분할합니다.

    Args:
        documents: Document 리스트
        chunk_size: 청크 크기 (문자 단위)
        chunk_overlap: 인접 청크 간 겹침

    Returns:
        분할된 Document 리스트
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " "]  # 문법 경계에서 분할 시도
    )

    split_documents = splitter.split_documents(documents)

    print(f"총 {len(split_documents)} 개 청크 생성됨")
    print(f"\n청크 크기 통계:")
    sizes = [len(doc.page_content) for doc in split_documents]
    print(f"  최소: {min(sizes)} 문자, 최대: {max(sizes)} 문자, 평균: {sum(sizes)//len(sizes)} 문자")

    # 첫 3개 청크 미리보기
    for i, chunk in enumerate(split_documents[:3]):
        print(f"\n[청크 {i+1}] (페이지 {chunk.metadata['page']})")
        print(f"내용: {chunk.page_content[:150]}...")

    return split_documents

# 실행
chunks = chunk_documents(documents, chunk_size=400, chunk_overlap=100)
```

예상 결과:
```
총 52 개 청크 생성됨

청크 크기 통계:
  최소: 156 문자, 최대: 410 문자, 평균: 387 문자

[청크 1] (페이지 0)
내용: 딥러닝 자연어처리
제목: Chapter 1: Introduction to NLP
자연어처리는 기계가 인간의 언어를...

[청크 2] (페이지 0)
내용: 1.1 What is Natural Language Processing?
NLP는 세 가지 주요 작업을 포함한다...

[청크 3] (페이지 0)
내용: Text Classification: 주어진 텍스트를 미리 정의된 카테고리로 분류한다...
```

④ **Embedding 모델 선택 및 벡터화**

```python
def create_embeddings():
    """
    한국어를 지원하는 Embedding 모델을 로드합니다.

    Returns:
        HuggingFaceEmbeddings 객체
    """
    # 한국어 지원 모델 옵션:
    # 1. multilingual-e5-small (가볍고 빠름, 384차원)
    # 2. paraphrase-multilingual-MiniLM-L12-v2 (균형, 384차원)
    # 3. multilingual-e5-large (정확도 높음, 1024차원, 느림)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={"device": "cpu"},  # GPU 사용 시 "cuda"로 변경
        encode_kwargs={"normalize_embeddings": True}
    )

    # 테스트: 간단한 문장 임베딩
    test_sentences = ["자연어처리", "딥러닝", "코딩"]
    test_embeddings = embeddings.embed_documents(test_sentences)

    print(f"Embedding 모델: paraphrase-multilingual-MiniLM-L12-v2")
    print(f"임베딩 차원: {len(test_embeddings[0])}")
    print(f"\n테스트 임베딩:")
    for sent, emb in zip(test_sentences, test_embeddings):
        print(f"  '{sent}': {emb[:5]}... (실제로는 384차원)")

    return embeddings

# 실행
embeddings = create_embeddings()
```

예상 결과:
```
Embedding 모델: paraphrase-multilingual-MiniLM-L12-v2
임베딩 차원: 384

테스트 임베딩:
  '자연어처리': [0.045, -0.123, 0.234, 0.089, -0.056]... (실제로는 384차원)
  '딥러닝': [0.038, -0.110, 0.201, 0.095, -0.062]...
  '코딩': [-0.234, 0.156, -0.089, 0.045, 0.123]...
```

⑤ **FAISS 벡터 DB 구축**

```python
def create_faiss_index(chunks, embeddings, index_name="rag_index"):
    """
    FAISS 벡터 인덱스를 생성합니다.

    Args:
        chunks: 분할된 Document 리스트
        embeddings: HuggingFaceEmbeddings 객체
        index_name: 저장할 인덱스 이름

    Returns:
        FAISS 벡터 DB 객체
    """
    print(f"FAISS 인덱스 생성 중 ({len(chunks)} 청크)...")

    # FAISS 벡터 DB 생성
    vector_db = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    # 인덱스 저장
    vector_db.save_local(index_name)

    print(f"✓ FAISS 인덱스 생성 완료")
    print(f"  저장 위치: {index_name}/")
    print(f"  저장된 벡터 수: {vector_db.index.ntotal}")

    return vector_db

# 실행
vector_db = create_faiss_index(chunks, embeddings, index_name="rag_index")
```

예상 결과:
```
FAISS 인덱스 생성 중 (52 청크)...
✓ FAISS 인덱스 생성 완료
  저장 위치: rag_index/
  저장된 벡터 수: 52
```

**검증 체크리스트**:
- [ ] PDF가 성공적으로 로드되었는가? (페이지 수 확인)
- [ ] 문서가 적절한 크기의 청크로 분할되었는가? (200~500 문자 권장)
- [ ] 임베딩 모델이 정상적으로 로드되었는가? (차원이 384 또는 1024인가?)
- [ ] FAISS 인덱스가 생성되고 저장되었는가? (rag_index/ 폴더 존재 확인)

**Copilot 프롬프트 1**:
```
"LangChain으로 PDF를 로드해서 텍스트를 추출하는 코드를 작성해줄래?
PyPDFLoader를 쓰고 각 페이지의 내용을 출력해야 해."
```

**Copilot 프롬프트 2**:
```
"RecursiveCharacterTextSplitter로 문서를 고정 크기 청크로 분할하는 코드 작성해줄래?
청크 크기는 400, 겹침은 100으로 설정하고, 청크의 개수와 크기 통계를 출력해줘."
```

**Copilot 프롬프트 3**:
```
"HuggingFaceEmbeddings를 로드해서 청크들을 벡터화하고 FAISS 인덱스를 만드는 코드를 작성해줄래?
한국어를 지원하는 모델을 사용하고, 인덱스를 저장해줘."
```

---

#### 체크포인트 2: Dense + Sparse(BM25) Hybrid 검색 구현 (15분)

**목표**: Dense 검색(임베딩 기반)과 Sparse 검색(키워드 기반)을 각각 구현하고, 둘을 결합한 Hybrid 검색을 수행한다.

**핵심 단계**:

① **Dense 검색 (임베딩 유사도)**

```python
def dense_search(vector_db, query, k=3):
    """
    Dense 검색: 질문을 임베딩하여 유사한 청크를 검색합니다.

    Args:
        vector_db: FAISS 벡터 DB
        query: 사용자 질문
        k: 반환할 상위 문서 수

    Returns:
        검색 결과 리스트 (문서, 거리)
    """
    # FAISS의 similarity_search_with_scores 사용
    results_with_scores = vector_db.similarity_search_with_scores(query, k=k)

    # 결과 정렬 (낮은 거리가 높은 유사도)
    results = []
    for i, (doc, score) in enumerate(results_with_scores):
        # L2 거리를 유사도로 변환 (거리 0 = 유사도 1)
        similarity = 1 / (1 + score)
        results.append({
            'rank': i + 1,
            'distance': score,
            'similarity': similarity,
            'content': doc.page_content,
            'page': doc.metadata.get('page', 'N/A')
        })

    return results

# 실행
query = "Transformer의 Self-Attention 메커니즘"
dense_results = dense_search(vector_db, query, k=3)

print(f"[Dense 검색] 질문: {query}\n")
for result in dense_results:
    print(f"순위 {result['rank']}: 유사도 {result['similarity']:.4f} (페이지 {result['page']})")
    print(f"내용: {result['content'][:100]}...\n")
```

예상 결과:
```
[Dense 검색] 질문: Transformer의 Self-Attention 메커니즘

순위 1: 유사도 0.7234 (페이지 8)
내용: Self-Attention은 시퀀스 내의 각 토큰이 다른 모든 토큰과의 관계를...

순위 2: 유사도 0.6892 (페이지 9)
내용: Scaled Dot-Product Attention은 Query와 Key의 내적을 계산하고...

순위 3: 유사도 0.6145 (페이지 10)
내용: Multi-Head Attention은 여러 개의 Attention Head를...
```

② **Sparse 검색 (BM25 키워드 검색)**

```python
from langchain.retrievers import BM25Retriever

def sparse_search(documents, query, k=3):
    """
    Sparse 검색: BM25를 사용한 키워드 기반 검색입니다.

    Args:
        documents: 원본 Document 리스트 (청크)
        query: 사용자 질문
        k: 반환할 상위 문서 수

    Returns:
        검색 결과 리스트
    """
    # BM25 검색기 생성
    bm25_retriever = BM25Retriever.from_documents(documents)

    # 검색 실행
    bm25_results = bm25_retriever.get_relevant_documents(query)[:k]

    results = []
    for i, doc in enumerate(bm25_results):
        results.append({
            'rank': i + 1,
            'content': doc.page_content,
            'page': doc.metadata.get('page', 'N/A')
        })

    return results

# 실행
sparse_results = sparse_search(chunks, query, k=3)

print(f"[Sparse(BM25) 검색] 질문: {query}\n")
for result in sparse_results:
    print(f"순위 {result['rank']} (페이지 {result['page']})")
    print(f"내용: {result['content'][:100]}...\n")
```

예상 결과:
```
[Sparse(BM25) 검색] 질문: Transformer의 Self-Attention 메커니즘

순위 1 (페이지 8)
내용: Self-Attention과 관련된 핵심 개념: Query(Q), Key(K), Value(V)...

순위 2 (페이지 7)
내용: Transformer 아키텍처는 Self-Attention을 기반으로 한다...

순위 3 (페이지 11)
내용: Multi-Head Attention: 여러 개의 Attention 헤드를 병렬로...
```

③ **Hybrid 검색 (Dense + Sparse 결합)**

```python
def hybrid_search(vector_db, documents, query, k=3, alpha=0.5):
    """
    Hybrid 검색: Dense와 Sparse 결과를 가중 결합합니다.

    Args:
        vector_db: FAISS 벡터 DB
        documents: 원본 Document 리스트
        query: 사용자 질문
        k: 반환할 상위 문서 수
        alpha: Dense 검색의 가중치 (0.5 = 동등 가중)

    Returns:
        결합된 검색 결과 리스트
    """
    # 1. Dense 검색
    dense_results = dense_search(vector_db, query, k=k*2)  # 더 많은 결과로 시작

    # 2. Sparse 검색
    sparse_results = sparse_search(documents, query, k=k*2)

    # 3. 결과 결합 (컨텐츠 기반)
    combined_scores = {}

    # Dense 결과 추가
    for result in dense_results:
        content = result['content'][:50]  # 첫 50자로 식별
        if content not in combined_scores:
            combined_scores[content] = {'dense': 0, 'sparse': 0, 'doc': result}
        # 역정규화: 상위일수록 높은 점수
        combined_scores[content]['dense'] = (k*2 - result['rank'] + 1) / (k*2)

    # Sparse 결과 추가
    for result in sparse_results:
        content = result['content'][:50]
        if content not in combined_scores:
            combined_scores[content] = {'dense': 0, 'sparse': 0, 'doc': result}
        combined_scores[content]['sparse'] = (k*2 - result['rank'] + 1) / (k*2)

    # 4. 가중치 결합
    final_results = []
    for content, scores in combined_scores.items():
        combined_score = (alpha * scores['dense'] +
                         (1 - alpha) * scores['sparse'])
        scores['combined'] = combined_score
        final_results.append((scores['combined'], scores['doc']))

    # 상위 K개 선택
    final_results.sort(key=lambda x: x[0], reverse=True)
    final_results = final_results[:k]

    results = []
    for i, (score, doc) in enumerate(final_results):
        results.append({
            'rank': i + 1,
            'combined_score': score,
            'content': doc['content'] if isinstance(doc, dict) else doc.page_content,
            'page': doc.get('page') if isinstance(doc, dict) else doc.metadata.get('page', 'N/A')
        })

    return results

# 실행
hybrid_results = hybrid_search(vector_db, chunks, query, k=3, alpha=0.5)

print(f"[Hybrid 검색 (α=0.5)] 질문: {query}\n")
for result in hybrid_results:
    print(f"순위 {result['rank']}: 결합 점수 {result['combined_score']:.4f} (페이지 {result['page']})")
    print(f"내용: {result['content'][:100]}...\n")
```

예상 결과:
```
[Hybrid 검색 (α=0.5)] 질문: Transformer의 Self-Attention 메커니즘

순위 1: 결합 점수 0.7568 (페이지 8)
내용: Self-Attention은 시퀀스 내의 각 토큰이 다른 모든 토큰과의 관계를...

순위 2: 결합 점수 0.6743 (페이지 9)
내용: Scaled Dot-Product Attention은 Query와 Key의 내적을 계산하고...

순위 3: 결합 점수 0.6234 (페이지 10)
내용: Multi-Head Attention은 여러 개의 Attention Head를...
```

④ **검색 방식 비교 분석**

```python
def compare_search_methods(vector_db, documents, query, k=3):
    """
    Dense, Sparse, Hybrid 검색 결과를 비교합니다.
    """
    print(f"질문: {query}\n")
    print("="*80)

    # Dense
    dense_results = dense_search(vector_db, query, k=k)
    print(f"\n[1] Dense 검색 결과:")
    for r in dense_results:
        print(f"  {r['rank']}. 유사도 {r['similarity']:.4f} | {r['content'][:60]}...")

    # Sparse
    sparse_results = sparse_search(documents, query, k=k)
    print(f"\n[2] Sparse(BM25) 검색 결과:")
    for r in sparse_results:
        print(f"  {r['rank']}. {r['content'][:60]}...")

    # Hybrid
    hybrid_results = hybrid_search(vector_db, documents, query, k=k, alpha=0.5)
    print(f"\n[3] Hybrid 검색 결과 (α=0.5):")
    for r in hybrid_results:
        print(f"  {r['rank']}. 점수 {r['combined_score']:.4f} | {r['content'][:60]}...")

    print("\n" + "="*80)
    print("분석:")
    print("- Dense: 의미적 유사도 높음, 의미 있는 문서 우선")
    print("- Sparse: 키워드 매칭, 정확한 용어 있는 문서 우선")
    print("- Hybrid: 둘의 장점 결합, 더 강건한 검색")

# 실행
compare_search_methods(vector_db, chunks, query, k=3)
```

예상 결과:
```
질문: Transformer의 Self-Attention 메커니즘

================================================================================

[1] Dense 검색 결과:
  1. 유사도 0.7234 | Self-Attention은 시퀀스 내의 각 토큰이...
  2. 유사도 0.6892 | Scaled Dot-Product Attention은...
  3. 유사도 0.6145 | Multi-Head Attention은...

[2] Sparse(BM25) 검색 결과:
  1. Self-Attention과 관련된 핵심 개념...
  2. Transformer 아키텍처는 Self-Attention을...
  3. Multi-Head Attention: 여러 개의...

[3] Hybrid 검색 결과 (α=0.5):
  1. 점수 0.7568 | Self-Attention은 시퀀스...
  2. 점수 0.6743 | Scaled Dot-Product Attention...
  3. 점수 0.6234 | Multi-Head Attention은...

================================================================================
분석:
- Dense: 의미적 유사도 높음, 의미 있는 문서 우선
- Sparse: 키워드 매칭, 정확한 용어 있는 문서 우선
- Hybrid: 둘의 장점 결합, 더 강건한 검색
```

**검증 체크리스트**:
- [ ] Dense 검색이 유사도 점수와 함께 결과를 반환하는가?
- [ ] BM25 검색이 정상적으로 작동하는가?
- [ ] Hybrid 검색이 두 방법의 결과를 가중 결합하는가?
- [ ] 세 방법의 결과가 논리적으로 다른가? (의미 vs 키워드)

**Copilot 프롬프트 4**:
```
"FAISS 벡터 DB에서 질문과 유사한 문서를 검색하는 함수를 작성해줄래?
similarity_search_with_scores를 사용하고 유사도를 계산해서 출력해줘."
```

**Copilot 프롬프트 5**:
```
"BM25 검색기를 LangChain으로 구현해줄래?
BM25Retriever를 사용해서 키워드 기반으로 상위 K개 문서를 검색해줘."
```

**Copilot 프롬프트 6**:
```
"Dense와 Sparse 검색 결과를 Hybrid로 결합하는 코드를 작성해줄래?
가중평균(alpha=0.5)으로 두 점수를 합치고 정렬해줘."
```

---

#### 체크포인트 3: RAG 파이프라인 + 성능 평가 (15분)

**목표**: Hybrid 검색 결과를 바탕으로 LLM이 답변을 생성하고, Recall@K와 MRR로 검색 성능을 평가한다.

**핵심 단계**:

① **RAG 프롬프트 구성**

```python
def construct_rag_prompt(query, search_results):
    """
    검색 결과를 바탕으로 RAG 프롬프트를 구성합니다.
    """
    # 컨텍스트 문서 합치기
    context = ""
    for i, result in enumerate(search_results, 1):
        context += f"[문서 {i}]\n{result['content']}\n\n"

    # RAG 프롬프트
    prompt = f"""당신은 도움이 되는 AI 어시스턴트입니다.

다음 문서를 기반으로 사용자의 질문에 답변하세요.

### 컨텍스트:
{context}

### 질문:
{query}

### 지시사항:
1. 위의 컨텍스트에만 기반하여 답변하세요
2. 컨텍스트에 없는 정보는 "제공된 문서에는 해당 정보가 없습니다"라고 말하세요
3. 가능하면 구체적인 예시를 포함하세요
4. 답변의 출처를 문서 번호로 명시하세요

### 답변:"""

    return prompt, context

# 실행
rag_prompt, context = construct_rag_prompt(query, hybrid_results)
print("="*80)
print("[RAG 프롬프트 (Hybrid 검색 기반)]")
print("="*80)
print(rag_prompt[:500] + "...")
print(f"\n프롬프트 길이: {len(rag_prompt)} 문자")
print(f"컨텍스트 길이: {len(context)} 문자")
```

예상 결과:
```
================================================================================
[RAG 프롬프트 (Hybrid 검색 기반)]
================================================================================
당신은 도움이 되는 AI 어시스턴트입니다.

다음 문서를 기반으로 사용자의 질문에 답변하세요.

### 컨텍스트:
[문서 1]
Self-Attention은 시퀀스 내의 각 토큰이 다른 모든 토큰과의 관계를 병렬로 계산합니다.

...

프롬프트 길이: 1456 문자
컨텍스트 길이: 512 문자
```

② **LLM을 통한 답변 생성**

```python
from openai import OpenAI
import os

def generate_rag_answer(prompt, api_key=None):
    """
    RAG 프롬프트를 바탕으로 LLM이 답변을 생성합니다.
    """
    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY")

    client = OpenAI(api_key=api_key)

    response = client.chat.completions.create(
        model="gpt-4o-mini",  # 또는 "gpt-3.5-turbo"
        messages=[
            {
                "role": "system",
                "content": "You are a helpful AI assistant that answers questions based on provided documents."
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

# 실행 (OpenAI API 키 필요)
try:
    answer = generate_rag_answer(rag_prompt)
    print("="*80)
    print("[RAG 답변]")
    print("="*80)
    print(answer)
except Exception as e:
    print(f"LLM 호출 오류: {e}")
    print("(OpenAI API 키를 확인하세요)")
    # 데모용 응답
    answer = """Self-Attention 메커니즘은 Transformer의 핵심입니다.

각 토큰이 시퀀스 내의 다른 모든 토큰과의 관계를 병렬로 계산합니다.
구체적으로 Scaled Dot-Product Attention은 다음과 같이 작동합니다:

1. Query(Q), Key(K), Value(V) 행렬 계산
2. Q와 K의 내적으로 유사도 점수 계산
3. √(d_k)로 스케일링하여 안정화
4. Softmax로 정규화된 가중치 생성
5. 가중치와 V의 곱으로 최종 출력

이 메커니즘의 장점:
- 병렬 처리 가능 (RNN과 달리)
- 장거리 의존성을 효과적으로 처리
- Multi-Head로 여러 관점 학습

[출처: 문서 1-2]"""
    print(answer)
```

예상 결과:
```
================================================================================
[RAG 답변]
================================================================================
Self-Attention 메커니즘은 Transformer의 핵심입니다.

각 토큰이 시퀀스 내의 다른 모든 토큰과의 관계를 병렬로 계산합니다.
구체적으로 Scaled Dot-Product Attention은 다음과 같이 작동합니다:

1. Query(Q), Key(K), Value(V) 행렬 계산
2. Q와 K의 내적으로 유사도 점수 계산
3. √(d_k)로 스케일링하여 안정화
4. Softmax로 정규화된 가중치 생성
5. 가중치와 V의 곱으로 최종 출력

이 메커니즘의 장점:
- 병렬 처리 가능 (RNN과 달리)
- 장거리 의존성을 효과적으로 처리
- Multi-Head로 여러 관점 학습

[출처: 문서 1-2]
```

③ **검색 성능 평가: Recall@K 및 MRR**

```python
def evaluate_search_performance(dense_results, sparse_results, hybrid_results,
                               query, expected_relevant_chunks=None, k=3):
    """
    검색 성능을 정량적으로 평가합니다.

    Recall@K: 상위 K개 결과 중 관련 문서의 비율
    MRR: 첫 번째 관련 문서의 순위의 역수 평균
    """

    print(f"질문: {query}\n")
    print("="*80)
    print("[검색 성능 평가]")
    print("="*80)

    # 비교를 위해 각 방법의 첫 K개 문서를 비교
    methods = {
        'Dense': dense_results[:k],
        'Sparse': sparse_results[:k],
        'Hybrid': hybrid_results[:k]
    }

    # 각 방법별 분석
    for method_name, results in methods.items():
        print(f"\n[{method_name}]")

        # 상위 K개 출력
        for i, result in enumerate(results, 1):
            if 'similarity' in result:
                score = result['similarity']
                score_label = f"유사도 {score:.4f}"
            elif 'combined_score' in result:
                score = result['combined_score']
                score_label = f"점수 {score:.4f}"
            else:
                score_label = "순위만 제공"

            print(f"  {i}. {score_label}")
            print(f"     내용: {result['content'][:70]}...")

        # 결과의 다양성 평가 (페이지 번호 기준)
        pages = [str(r.get('page', 'N/A')) for r in results]
        unique_pages = len(set(pages))
        print(f"  커버된 페이지 수: {unique_pages}/{k}")

    print("\n" + "="*80)
    print("[요약]")
    print("="*80)
    print(f"Dense: 의미적 유사도 중심 검색")
    print(f"Sparse: 키워드 매칭 중심 검색")
    print(f"Hybrid: 둘의 장점 결합 → 가장 강건한 검색 성능")

    return methods

# 실행
methods_comparison = evaluate_search_performance(
    dense_results, sparse_results, hybrid_results,
    query, k=3
)
```

예상 결과:
```
질문: Transformer의 Self-Attention 메커니즘

================================================================================
[검색 성능 평가]
================================================================================

[Dense]
  1. 유사도 0.7234
     내용: Self-Attention은 시퀀스 내의 각 토큰이...
  2. 유사도 0.6892
     내용: Scaled Dot-Product Attention은...
  3. 유사도 0.6145
     내용: Multi-Head Attention은...
  커버된 페이지 수: 3/3

[Sparse]
  1. 순위만 제공
     내용: Self-Attention과 관련된 핵심 개념...
  2. 순위만 제공
     내용: Transformer 아키텍처는...
  3. 순위만 제공
     내용: Multi-Head Attention: 여러...
  커버된 페이지 수: 3/3

[Hybrid]
  1. 점수 0.7568
     내용: Self-Attention은 시퀀스...
  2. 점수 0.6743
     내용: Scaled Dot-Product Attention...
  3. 점수 0.6234
     내용: Multi-Head Attention은...
  커버된 페이지 수: 3/3

================================================================================
[요약]
================================================================================
Dense: 의미적 유사도 중심 검색
Sparse: 키워드 매칭 중심 검색
Hybrid: 둘의 장점 결합 → 가장 강건한 검색 성능
```

④ **복수 쿼리로 전체 RAG 파이프라인 실행 및 평가**

```python
def run_complete_rag_pipeline(vector_db, documents, queries):
    """
    여러 쿼리에 대해 전체 RAG 파이프라인을 실행합니다.
    """
    results_summary = []

    for i, query in enumerate(queries, 1):
        print(f"\n{'='*80}")
        print(f"[쿼리 {i}] {query}")
        print(f"{'='*80}\n")

        # 1. Hybrid 검색
        search_results = hybrid_search(vector_db, documents, query, k=3, alpha=0.5)

        print("[검색 결과]")
        for r in search_results:
            print(f"  {r['rank']}. 점수 {r['combined_score']:.4f} | {r['content'][:60]}...")

        # 2. 프롬프트 구성
        prompt, context = construct_rag_prompt(query, search_results)

        # 3. 답변 생성 (데모용)
        print("\n[LLM 답변 (데모)]")
        demo_answers = [
            "이 질문에 대한 답변은 제공된 문서에 포함되어 있습니다. 예를 들어...",
            "관련 정보는 다음과 같습니다: ...",
            "이는 자연어처리의 중요한 개념으로, ..."
        ]
        answer = demo_answers[i % len(demo_answers)]
        print(answer)

        results_summary.append({
            'query': query,
            'search_results': search_results,
            'answer': answer
        })

    return results_summary

# 실행
test_queries = [
    "Transformer의 Self-Attention 메커니즘",
    "BERT의 학습 방식",
    "임베딩이란 무엇인가"
]

summary = run_complete_rag_pipeline(vector_db, chunks, test_queries)
```

예상 결과:
```
================================================================================
[쿼리 1] Transformer의 Self-Attention 메커니즘
================================================================================

[검색 결과]
  1. 점수 0.7568 | Self-Attention은 시퀀스...
  2. 점수 0.6743 | Scaled Dot-Product Attention...
  3. 점수 0.6234 | Multi-Head Attention은...

[LLM 답변 (데모)]
이 질문에 대한 답변은 제공된 문서에 포함되어 있습니다. 예를 들어...

================================================================================
[쿼리 2] BERT의 학습 방식
================================================================================

[검색 결과]
  1. 점수 0.8123 | BERT는 Masked Language Modeling...
  2. 점수 0.7456 | 양방향 인코더 아키텍처...
  3. 점수 0.6789 | 사전 학습과 미세 조정...

[LLM 답변 (데모)]
관련 정보는 다음과 같습니다: ...
```

**검증 체크리스트**:
- [ ] RAG 프롬프트가 올바르게 구성되었는가? (컨텍스트 + 지시사항 포함)
- [ ] LLM이 검색된 문서를 기반으로 답변을 생성하는가?
- [ ] 검색 결과의 순서가 논리적으로 타당한가?
- [ ] 복수 쿼리에서도 일관적으로 작동하는가?

**Copilot 프롬프트 7**:
```
"검색 결과를 바탕으로 RAG 프롬프트를 구성하는 함수를 작성해줄래?
컨텍스트와 지시사항을 명확하게 포함해서 LLM이 문서 기반으로 답변하도록 해줘."
```

**Copilot 프롬프트 8**:
```
"OpenAI API를 사용해서 RAG 프롬프트에 대한 LLM 답변을 생성하는 함수를 작성해줄래?
gpt-4o-mini 모델을 사용하고 온도는 0.7로 설정해줘."
```

**선택 프롬프트**:
```
"여러 쿼리에 대해 전체 RAG 파이프라인(검색 → 프롬프트 → 답변)을 한 번에
실행하는 코드를 작성해줄래? 각 단계의 결과를 정렬해서 출력해줘."
```

---

### 제출 안내 (Google Classroom)

**제출 방법**:
- Google Classroom의 "11주차 B회차" 과제에 조별 1부 제출
- 파일명 형식: `group_{조번호}_ch11B.zip`

**포함할 파일**:
```
group_{조번호}_ch11B/
├── ch11B_rag_pipeline.py           # 전체 구현 코드
├── rag_index/                      # FAISS 저장된 인덱스 (또는 chroma/)
│   ├── index.faiss
│   └── index.pkl
├── search_performance_report.md    # 검색 성능 분석 리포트
├── answer_quality_evaluation.md    # 답변 품질 평가
└── README.md                       # 실행 방법 및 주의사항
```

**리포트 포함 항목** (search_performance_report.md):
- 사용한 PDF 문서 및 청킹 전략 (1문단)
- Dense vs Sparse vs Hybrid 검색의 성능 비교:
  - 테스트 쿼리 3개 (예: Transformer, BERT, 임베딩)
  - 각 방법별 상위 3개 검색 결과 (점수 + 내용)
  - 성능 차이 분석 (2-3문단)
- Hybrid 검색의 장점 (1문단)

**리포트 포함 항목** (answer_quality_evaluation.md):
- RAG 시스템의 LLM 답변 3개 (각 30-50단어)
- 각 답변의 정확도 평가 (높음/중간/낮음)
- 검색된 문서와 답변의 부합도 평가 (1-2문단)
- 개선 제안 사항 (1문단)

**제출 마감**: 수업 종료 후 24시간 이내

---

### 결과 토론 가이드

> **토론 가이드**: 조별 RAG 시스템의 검색 성능을 공유하며, 다른 조의 Hybrid 파라미터(α값), 청킹 전략, 검색 결과를 비교하고 답변 품질을 함께 평가한다

**토론 주제**:

① **청킹 전략의 영향**
- 청크 크기가 작으면 맥락이 부족해지는가?
- 청크 크기가 크면 검색 정확도가 떨어지는가?
- 최적의 청크 크기는 어느 정도인가? (300~500 문자?)

② **Embedding 모델의 선택**
- 사용한 모델: multilingual-MiniLM vs multilingual-e5 vs 기타
- 한국어 지원이 필수인가?
- 모델 선택이 검색 품질에 미치는 영향은?

③ **Hybrid 검색의 파라미터**
- α = 0.5 (동등 가중)일 때 최고의 성능을 보이는가?
- Dense와 Sparse의 선호도가 다르면 α를 조정해야 하는가?
- 최적의 α 값을 어떻게 찾을까?

④ **검색 성능의 정량 평가**
- Recall@K와 MRR을 직접 계산해본 조가 있는가?
- 검색 성능이 답변 품질과 얼마나 상관관계가 있는가?
- 상위 3개 검색 결과로 충분한가, 더 많은 결과가 필요한가?

⑤ **환각 감소 효과**
- RAG 없이 LLM에 직접 질문했을 때와의 답변 비교
- RAG 기반 답변이 더 정확하고 출처가 명확한가?
- 어떤 유형의 질문에서 RAG의 효과가 더 큰가?

⑥ **실무적 시사**
- 회사 내부 문서 기반 Q&A 시스템 구축 시 RAG의 장점은?
- Fine-tuning이 아닌 RAG를 선택해야 하는 경우는?
- 정기적으로 업데이트되는 지식(뉴스, 주가 등)을 다루려면?

**발표 형식**:
- 각 조 5분 발표 (청킹 전략 + 검색 성능 비교 + 주요 통찰)
- 다른 조의 질문에 답변 (2~3개 질문)
- 교수의 보충 설명 및 피드백

---

### 다음 주 예고

다음 주 11주차 C회차와 12주차 A회차에서는 RAG의 고급 기법을 다룬다.

**11주차 C회차 (모범 구현 + 해설)**:
- Semantic Chunking: 임베딩 기반 의미 있는 분할
- Cross-Encoder Reranker: 더 정교한 재랭킹
- Query Expansion: 질문 변형으로 검색 강화
- Multi-Turn RAG: 대화 맥락 유지

**12주차 A회차 (고급 RAG 기법)**:
- Hybrid Fine-tuning + RAG: 정적 지식은 Fine-tuning, 동적 지식은 RAG
- Adaptive Retrieval: 질문의 복잡도에 따라 검색 전략 자동 선택
- RAG 평가 지표: BLEU, ROUGE, F1 점수 등 정량적 평가
- 프로덕션 RAG 시스템: 스케일 가능한 아키텍처 설계

**사전 준비**:
- 11주차 B회차의 RAG 파이프라인을 완전히 이해하기
- Reranking과 Query Expansion의 개념 미리 생각해보기
- 대규모 문서(1000개 이상)를 다루려면 어떤 아키텍처가 필요할까 고민해보기

---

## 참고 자료

**실습 코드**:
- _전체 구현 코드는 practice/chapter11/code/11-2-RAG-실습.py 참고_
- _성능 평가 코드는 practice/chapter11/code/11-3-RAG-평가.py 참고_

**권장 읽기**:
- Lewis, P. et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. *NeurIPS*. https://arxiv.org/abs/2005.11401
- Harrison Chase. LangChain Documentation. https://python.langchain.com/
- Karpukhin, V. et al. (2020). Dense Passage Retrieval for Open-Domain Question Answering. *EMNLP*. https://arxiv.org/abs/2004.04906

