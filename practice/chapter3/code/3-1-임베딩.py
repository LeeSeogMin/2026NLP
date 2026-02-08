"""
3장 실습 1: 단어 임베딩 (Word Embedding)

이 코드는 다음을 실습한다:
1. 간단한 Word2Vec 모델 학습
2. 임베딩 공간에서 단어 유사도 측정
3. 단어 벡터의 산술 연산 (왕 - 남자 + 여자 = ?)
4. 임베딩 공간 2D 시각화
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path


# ============================================================
# 1. 간단한 Word2Vec (Skip-gram) 직접 구현
# ============================================================
print("=" * 60)
print("1. Skip-gram Word2Vec 직접 구현")
print("=" * 60)

# 학습용 문장 (간단한 예시)
sentences = [
    "나는 오늘 학교에 갔다",
    "나는 어제 도서관에 갔다",
    "그는 오늘 회사에 갔다",
    "그녀는 어제 학교에 갔다",
    "나는 오늘 커피를 마셨다",
    "그는 어제 차를 마셨다",
    "학교에서 공부를 했다",
    "도서관에서 공부를 했다",
    "회사에서 일을 했다",
    "나는 학생이다",
    "그는 직장인이다",
    "그녀는 학생이다",
]

# 토큰화 및 어휘 사전 구축
words = []
for s in sentences:
    words.extend(s.split())

vocab = sorted(set(words))
word2idx = {w: i for i, w in enumerate(vocab)}
idx2word = {i: w for w, i in word2idx.items()}
vocab_size = len(vocab)

print(f"어휘 크기: {vocab_size}")
print(f"어휘 목록: {vocab[:10]}...")

# Skip-gram 학습 데이터 생성 (window_size=2)
window_size = 2
pairs = []

for sentence in sentences:
    tokens = sentence.split()
    for i, center in enumerate(tokens):
        for j in range(max(0, i - window_size), min(len(tokens), i + window_size + 1)):
            if i != j:
                pairs.append((word2idx[center], word2idx[tokens[j]]))

print(f"학습 쌍 수: {len(pairs)}")


# Skip-gram 모델 정의
class SkipGram(nn.Module):
    """Skip-gram Word2Vec 모델"""

    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        # 중심 단어 임베딩
        self.center_embed = nn.Embedding(vocab_size, embed_dim)
        # 문맥 단어 임베딩
        self.context_embed = nn.Embedding(vocab_size, embed_dim)

    def forward(self, center, context):
        center_vec = self.center_embed(center)    # (batch, embed_dim)
        context_vec = self.context_embed(context)  # (batch, embed_dim)
        # 내적으로 유사도 계산
        score = torch.sum(center_vec * context_vec, dim=1)
        return score


# 모델 학습
embed_dim = 16
model = SkipGram(vocab_size, embed_dim)
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.BCEWithLogitsLoss()

# 학습 데이터 준비 (Negative Sampling 간략 구현)
np.random.seed(42)
torch.manual_seed(42)

for epoch in range(200):
    total_loss = 0
    np.random.shuffle(pairs)

    # 긍정 샘플
    centers = torch.tensor([p[0] for p in pairs])
    contexts = torch.tensor([p[1] for p in pairs])
    labels = torch.ones(len(pairs))

    # 부정 샘플 (랜덤 단어)
    neg_contexts = torch.randint(0, vocab_size, (len(pairs),))
    neg_labels = torch.zeros(len(pairs))

    # 합치기
    all_centers = torch.cat([centers, centers])
    all_contexts = torch.cat([contexts, neg_contexts])
    all_labels = torch.cat([labels, neg_labels])

    scores = model(all_centers, all_contexts)
    loss = criterion(scores, all_labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    total_loss = loss.item()

    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch + 1:3d}: Loss = {total_loss:.4f}")

# 학습된 임베딩 추출
embeddings = model.center_embed.weight.detach().numpy()
print(f"\n임베딩 행렬 크기: {embeddings.shape}")

# ============================================================
# 2. 단어 유사도 측정
# ============================================================
print("\n" + "=" * 60)
print("2. 단어 유사도 측정 (코사인 유사도)")
print("=" * 60)


def cosine_similarity(v1, v2):
    """두 벡터의 코사인 유사도 계산"""
    dot = np.dot(v1, v2)
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    return dot / norm if norm > 0 else 0


# 유사 단어 쌍 테스트
test_pairs = [
    ("나는", "그는"),
    ("학교에", "도서관에서"),
    ("갔다", "마셨다"),
    ("학생이다", "직장인이다"),
    ("오늘", "어제"),
]

print(f"\n{'단어 쌍':<25} {'코사인 유사도':>10}")
print("-" * 40)
for w1, w2 in test_pairs:
    if w1 in word2idx and w2 in word2idx:
        sim = cosine_similarity(
            embeddings[word2idx[w1]], embeddings[word2idx[w2]]
        )
        print(f"{w1} ↔ {w2:<12} {sim:>10.4f}")

# 특정 단어와 가장 유사한 단어 찾기
print("\n'나는'과 가장 유사한 단어 (상위 5개):")
target_vec = embeddings[word2idx["나는"]]
sims = []
for w, idx in word2idx.items():
    if w != "나는":
        sim = cosine_similarity(target_vec, embeddings[idx])
        sims.append((w, sim))

sims.sort(key=lambda x: x[1], reverse=True)
for w, sim in sims[:5]:
    print(f"  {w}: {sim:.4f}")

# ============================================================
# 3. 임베딩 공간 2D 시각화 (PCA)
# ============================================================
print("\n" + "=" * 60)
print("3. 임베딩 공간 2D 시각화")
print("=" * 60)

# PCA로 2D 축소 (NumPy 직접 구현)
mean = embeddings.mean(axis=0)
centered = embeddings - mean
cov = np.cov(centered.T)
eigenvalues, eigenvectors = np.linalg.eigh(cov)

# 가장 큰 고유값 2개에 해당하는 고유벡터
top2_idx = np.argsort(eigenvalues)[-2:][::-1]
pca_components = eigenvectors[:, top2_idx]
embeddings_2d = centered @ pca_components

explained_var = eigenvalues[top2_idx] / eigenvalues.sum() * 100
print(f"PCA 설명 분산: {explained_var[0]:.1f}% + {explained_var[1]:.1f}% = {sum(explained_var):.1f}%")

import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams["font.family"] = "AppleGothic"
matplotlib.rcParams["axes.unicode_minus"] = False

fig, ax = plt.subplots(figsize=(10, 8))
ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c="steelblue", s=100, alpha=0.7)

for i, word in idx2word.items():
    ax.annotate(
        word,
        (embeddings_2d[i, 0], embeddings_2d[i, 1]),
        fontsize=9,
        ha="center",
        va="bottom",
        textcoords="offset points",
        xytext=(0, 5),
    )

ax.set_title("Skip-gram 임베딩 공간 (PCA 2D 투영)", fontsize=14)
ax.set_xlabel(f"PC1 ({explained_var[0]:.1f}%)")
ax.set_ylabel(f"PC2 ({explained_var[1]:.1f}%)")
ax.grid(True, alpha=0.3)

output_dir = Path(__file__).parent.parent / "data" / "output"
output_dir.mkdir(parents=True, exist_ok=True)
fig.savefig(output_dir / "embedding_space.png", dpi=150, bbox_inches="tight")
print(f"시각화 저장: {output_dir / 'embedding_space.png'}")
plt.close()

print("\n실습 1 완료!")
