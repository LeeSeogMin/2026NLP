"""
13-4-vector-database.py
Vector Database와 FAISS 기초

이 코드는 Vector Database의 개념과 FAISS를 활용한
유사도 검색의 기본 원리를 다룬다.
"""

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import time

print("=" * 60)
print("Vector Database와 FAISS 기초")
print("=" * 60)

# ============================================================
# 1. 벡터 임베딩의 개념
# ============================================================
print("\n[1] 벡터 임베딩의 개념")
print("-" * 50)

# Sentence Transformer 모델 로드
print("임베딩 모델 로드 중...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# 예시 문장
sentences = [
    "인공지능이 세상을 바꾸고 있다.",
    "머신러닝은 데이터에서 패턴을 학습한다.",
    "딥러닝은 신경망을 사용한다.",
    "자연어처리는 텍스트를 이해한다.",
    "오늘 날씨가 좋다.",
    "맛있는 음식을 먹었다.",
]

# 임베딩 생성
embeddings = model.encode(sentences)

print(f"문장 수: {len(sentences)}")
print(f"임베딩 차원: {embeddings.shape[1]}")
print(f"임베딩 타입: {embeddings.dtype}")

# 첫 번째 문장의 임베딩 일부 출력
print(f"\n첫 번째 문장 임베딩 (처음 10개 값):")
print(f"  {embeddings[0][:10]}")

# ============================================================
# 2. 유사도 측정
# ============================================================
print("\n[2] 유사도 측정")
print("-" * 50)


def cosine_similarity(a, b):
    """코사인 유사도 계산"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def euclidean_distance(a, b):
    """유클리드 거리 계산"""
    return np.linalg.norm(a - b)


# 쿼리 문장
query = "AI와 기계학습에 대해 알려줘"
query_embedding = model.encode([query])[0]

print(f"쿼리: '{query}'")
print(f"\n{'문장':<40} | {'코사인 유사도':>12} | {'L2 거리':>10}")
print("-" * 70)

similarities = []
for i, (sent, emb) in enumerate(zip(sentences, embeddings)):
    cos_sim = cosine_similarity(query_embedding, emb)
    l2_dist = euclidean_distance(query_embedding, emb)
    similarities.append((sent, cos_sim, l2_dist))
    print(f"{sent:<40} | {cos_sim:>12.4f} | {l2_dist:>10.4f}")

# 가장 유사한 문장
most_similar = max(similarities, key=lambda x: x[1])
print(f"\n가장 유사한 문장: '{most_similar[0]}' (유사도: {most_similar[1]:.4f})")

# ============================================================
# 3. FAISS 기본 사용법
# ============================================================
print("\n[3] FAISS 기본 사용법")
print("-" * 50)

# 임베딩 차원
d = embeddings.shape[1]

# IndexFlatL2: 정확한 L2 거리 기반 검색 (brute-force)
index = faiss.IndexFlatL2(d)

# 임베딩 추가 (float32로 변환 필요)
embeddings_f32 = embeddings.astype('float32')
index.add(embeddings_f32)

print(f"FAISS 인덱스 생성 완료")
print(f"  - 인덱스 타입: IndexFlatL2")
print(f"  - 벡터 차원: {d}")
print(f"  - 저장된 벡터 수: {index.ntotal}")

# 검색
k = 3  # Top-k 결과
query_f32 = query_embedding.astype('float32').reshape(1, -1)

start_time = time.time()
distances, indices = index.search(query_f32, k)
search_time = time.time() - start_time

print(f"\n쿼리: '{query}'")
print(f"검색 시간: {search_time*1000:.3f}ms")
print(f"\nTop-{k} 검색 결과:")
for i, (idx, dist) in enumerate(zip(indices[0], distances[0])):
    print(f"  {i+1}. '{sentences[idx]}' (L2 거리: {dist:.4f})")

# ============================================================
# 4. 다양한 FAISS 인덱스 비교
# ============================================================
print("\n[4] FAISS 인덱스 비교")
print("-" * 50)

# 대규모 데이터 시뮬레이션
np.random.seed(42)
n_vectors = 10000
d = 384  # all-MiniLM-L6-v2의 차원

# 랜덤 벡터 생성 (실제로는 임베딩 사용)
large_embeddings = np.random.randn(n_vectors, d).astype('float32')
query_vec = np.random.randn(1, d).astype('float32')

print(f"테스트 데이터: {n_vectors}개 벡터, {d}차원")

# 1. IndexFlatL2 (정확, 느림)
index_flat = faiss.IndexFlatL2(d)
index_flat.add(large_embeddings)

start = time.time()
_, _ = index_flat.search(query_vec, 5)
flat_time = time.time() - start

# 2. IndexIVFFlat (근사, 빠름)
nlist = 100  # 클러스터 수
quantizer = faiss.IndexFlatL2(d)
index_ivf = faiss.IndexIVFFlat(quantizer, d, nlist)
index_ivf.train(large_embeddings)
index_ivf.add(large_embeddings)
index_ivf.nprobe = 10  # 검색할 클러스터 수

start = time.time()
_, _ = index_ivf.search(query_vec, 5)
ivf_time = time.time() - start

# 3. IndexHNSWFlat (근사, 빠름, 메모리 효율)
index_hnsw = faiss.IndexHNSWFlat(d, 32)  # 32 neighbors
index_hnsw.add(large_embeddings)

start = time.time()
_, _ = index_hnsw.search(query_vec, 5)
hnsw_time = time.time() - start

print(f"\n{'인덱스 타입':<20} | {'검색 시간':>15} | {'특징':>30}")
print("-" * 70)
print(f"{'IndexFlatL2':<20} | {flat_time*1000:>12.3f}ms | {'정확, 대규모에서 느림':>30}")
print(f"{'IndexIVFFlat':<20} | {ivf_time*1000:>12.3f}ms | {'빠름, 학습 필요':>30}")
print(f"{'IndexHNSWFlat':<20} | {hnsw_time*1000:>12.3f}ms | {'빠름, 메모리 효율적':>30}")

# ============================================================
# 5. 문서 기반 검색 예시
# ============================================================
print("\n[5] 문서 기반 검색 시스템")
print("-" * 50)

# 예시 문서들
documents = [
    "BERT는 양방향 트랜스포머를 사용하여 문맥을 이해한다.",
    "GPT는 자기회귀 방식으로 텍스트를 생성한다.",
    "LoRA는 저랭크 행렬을 사용하여 효율적으로 파인튜닝한다.",
    "RAG는 검색과 생성을 결합하여 정확한 답변을 제공한다.",
    "Attention 메커니즘은 입력의 중요한 부분에 집중한다.",
    "토큰화는 텍스트를 모델이 이해할 수 있는 단위로 분할한다.",
    "사전 학습된 모델은 대규모 코퍼스에서 언어를 학습한다.",
    "파인튜닝은 특정 태스크에 모델을 적응시킨다.",
]

# 임베딩 생성 및 인덱스 구축
doc_embeddings = model.encode(documents).astype('float32')
doc_index = faiss.IndexFlatL2(doc_embeddings.shape[1])
doc_index.add(doc_embeddings)

# 질문에 대한 관련 문서 검색
questions = [
    "효율적인 파인튜닝 방법은?",
    "텍스트 생성 모델은?",
    "검색 기반 생성이란?",
]

print("문서 검색 시스템 테스트:")
for q in questions:
    q_emb = model.encode([q]).astype('float32')
    D, I = doc_index.search(q_emb, 2)
    print(f"\n질문: '{q}'")
    print("관련 문서:")
    for i, (idx, dist) in enumerate(zip(I[0], D[0])):
        print(f"  {i+1}. {documents[idx]} (거리: {dist:.4f})")

# ============================================================
# 6. 핵심 요약
# ============================================================
print("\n" + "=" * 60)
print("핵심 요약")
print("=" * 60)
print("""
1. Vector Embedding
   - 텍스트를 고차원 벡터로 변환
   - Sentence Transformers 활용

2. 유사도 측정
   - 코사인 유사도: 방향 기반
   - L2 거리: 유클리드 거리

3. FAISS 인덱스 타입
   - IndexFlatL2: 정확하지만 느림 (소규모)
   - IndexIVFFlat: 빠른 근사 검색 (대규모)
   - IndexHNSWFlat: 균형 잡힌 성능

4. 검색 시스템 구축
   - 문서 임베딩 → 인덱스 저장 → 쿼리 검색
""")

print("=" * 60)
print("Vector Database 기초 완료")
print("=" * 60)
