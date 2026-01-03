"""
13-6-langchain-basics.py
LangChain 기본 사용법

이 코드는 LangChain의 핵심 컴포넌트와 기본 사용법을 다룬다.
"""

import os
import tempfile
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

print("=" * 60)
print("LangChain 기본 사용법")
print("=" * 60)

# ============================================================
# 1. Document Loaders
# ============================================================
print("\n[1] Document Loaders")
print("-" * 50)

# 예시 문서 생성
sample_text = """
# 딥러닝 자연어처리 개요

딥러닝은 인공 신경망을 사용하여 데이터에서 패턴을 학습하는 머신러닝의 한 분야이다.
자연어처리(NLP)는 컴퓨터가 인간의 언어를 이해하고 처리하는 기술이다.

## Transformer 아키텍처

2017년 "Attention is All You Need" 논문에서 소개된 Transformer는
Self-Attention 메커니즘을 기반으로 한다. 이 아키텍처는 RNN의 순차적 처리 한계를
극복하고 병렬 처리를 가능하게 했다.

## BERT와 GPT

BERT는 양방향 인코더로 문맥 이해에 강점을 가진다.
GPT는 자기회귀 디코더로 텍스트 생성에 특화되어 있다.
두 모델 모두 Transformer 아키텍처를 기반으로 한다.

## 파인튜닝

사전 학습된 모델을 특정 태스크에 적응시키는 과정을 파인튜닝이라 한다.
Full Fine-tuning은 모든 파라미터를 학습하고,
LoRA와 같은 PEFT 기법은 일부 파라미터만 효율적으로 학습한다.

## RAG (Retrieval-Augmented Generation)

RAG는 검색과 생성을 결합하여 LLM의 한계를 보완한다.
외부 문서에서 관련 정보를 검색하여 답변 생성에 활용한다.
이를 통해 Hallucination을 줄이고 최신 정보를 반영할 수 있다.
"""

# 임시 파일에 저장
temp_dir = tempfile.mkdtemp()
temp_file = os.path.join(temp_dir, "sample.txt")
with open(temp_file, "w", encoding="utf-8") as f:
    f.write(sample_text)

# TextLoader로 문서 로드
loader = TextLoader(temp_file, encoding="utf-8")
documents = loader.load()

print(f"로드된 문서 수: {len(documents)}")
print(f"문서 길이: {len(documents[0].page_content)} 문자")
print(f"메타데이터: {documents[0].metadata}")

# ============================================================
# 2. Text Splitters
# ============================================================
print("\n[2] Text Splitters")
print("-" * 50)

# RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,        # 청크 크기
    chunk_overlap=50,      # 청크 간 중첩
    length_function=len,
    separators=["\n\n", "\n", ".", " ", ""]
)

# 문서 분할
chunks = text_splitter.split_documents(documents)

print(f"원본 문서 수: {len(documents)}")
print(f"분할된 청크 수: {len(chunks)}")
print(f"청크 크기 설정: {text_splitter._chunk_size}")
print(f"청크 중첩: {text_splitter._chunk_overlap}")

print("\n분할된 청크 예시:")
for i, chunk in enumerate(chunks[:3]):
    print(f"\n[청크 {i+1}] ({len(chunk.page_content)} 문자)")
    print(f"  {chunk.page_content[:100]}...")

# ============================================================
# 3. Embeddings
# ============================================================
print("\n[3] Embeddings")
print("-" * 50)

# HuggingFace 임베딩 모델
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 단일 텍스트 임베딩
sample_embedding = embeddings.embed_query("딥러닝이란 무엇인가?")
print(f"임베딩 모델: all-MiniLM-L6-v2")
print(f"임베딩 차원: {len(sample_embedding)}")
print(f"임베딩 예시 (처음 5개 값): {sample_embedding[:5]}")

# 문서 배치 임베딩
doc_texts = [chunk.page_content for chunk in chunks[:3]]
doc_embeddings = embeddings.embed_documents(doc_texts)
print(f"\n문서 배치 임베딩: {len(doc_embeddings)}개 문서")

# ============================================================
# 4. Vector Stores (FAISS)
# ============================================================
print("\n[4] Vector Stores (FAISS)")
print("-" * 50)

# FAISS Vector Store 생성
vectorstore = FAISS.from_documents(chunks, embeddings)

print(f"Vector Store 생성 완료")
print(f"저장된 문서 수: {len(chunks)}")

# 유사도 검색
query = "BERT와 GPT의 차이점은?"
similar_docs = vectorstore.similarity_search(query, k=3)

print(f"\n쿼리: '{query}'")
print(f"검색 결과 (Top-3):")
for i, doc in enumerate(similar_docs):
    print(f"\n[결과 {i+1}]")
    print(f"  {doc.page_content[:150]}...")

# 유사도 점수와 함께 검색
docs_with_scores = vectorstore.similarity_search_with_score(query, k=3)
print(f"\n유사도 점수와 함께 검색:")
for i, (doc, score) in enumerate(docs_with_scores):
    print(f"  {i+1}. (점수: {score:.4f}) {doc.page_content[:80]}...")

# ============================================================
# 5. Retriever
# ============================================================
print("\n[5] Retriever")
print("-" * 50)

# Vector Store를 Retriever로 변환
retriever = vectorstore.as_retriever(
    search_type="similarity",  # 또는 "mmr"
    search_kwargs={"k": 3}
)

# Retriever로 검색
retrieved_docs = retriever.invoke("RAG 시스템이란?")

print(f"Retriever 설정:")
print(f"  - 검색 타입: similarity")
print(f"  - Top-k: 3")
print(f"\n검색 결과:")
for i, doc in enumerate(retrieved_docs):
    print(f"  {i+1}. {doc.page_content[:100]}...")

# ============================================================
# 6. Vector Store 저장 및 로드
# ============================================================
print("\n[6] Vector Store 저장 및 로드")
print("-" * 50)

# 저장
save_path = os.path.join(temp_dir, "faiss_index")
vectorstore.save_local(save_path)
print(f"Vector Store 저장 완료: {save_path}")

# 저장된 파일 확인
import os
saved_files = os.listdir(save_path)
print(f"저장된 파일: {saved_files}")

# 로드
loaded_vectorstore = FAISS.load_local(
    save_path,
    embeddings,
    allow_dangerous_deserialization=True
)
print(f"Vector Store 로드 완료")

# 로드된 Vector Store로 검색
loaded_results = loaded_vectorstore.similarity_search("파인튜닝이란?", k=2)
print(f"\n로드된 Vector Store 검색 결과:")
for i, doc in enumerate(loaded_results):
    print(f"  {i+1}. {doc.page_content[:100]}...")

# 정리
import shutil
shutil.rmtree(temp_dir)

# ============================================================
# 7. 핵심 요약
# ============================================================
print("\n" + "=" * 60)
print("핵심 요약")
print("=" * 60)
print("""
1. Document Loaders
   - TextLoader, PDFLoader, WebBaseLoader 등
   - 다양한 형식의 문서를 로드

2. Text Splitters
   - RecursiveCharacterTextSplitter 권장
   - chunk_size, chunk_overlap 설정

3. Embeddings
   - HuggingFaceEmbeddings 활용
   - all-MiniLM-L6-v2: 빠르고 효율적

4. Vector Stores
   - FAISS, Chroma, Pinecone 등
   - similarity_search로 유사 문서 검색

5. Retriever
   - Vector Store를 as_retriever()로 변환
   - RAG 파이프라인의 핵심 컴포넌트
""")

print("=" * 60)
print("LangChain 기본 사용법 완료")
print("=" * 60)
