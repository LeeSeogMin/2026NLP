"""
3장 실습 3: Self-Attention 통합 실습 + Attention 시각화

이 코드는 다음을 실습한다:
1. 문장에 대한 Self-Attention 적용
2. Attention Weight 히트맵 시각화
3. Multi-Head Attention의 Head별 패턴 비교
4. 간단한 Attention 기반 텍스트 분류 모델
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from pathlib import Path

matplotlib.rcParams["font.family"] = "AppleGothic"
matplotlib.rcParams["axes.unicode_minus"] = False

torch.manual_seed(42)
np.random.seed(42)

output_dir = Path(__file__).parent.parent / "data" / "output"
output_dir.mkdir(parents=True, exist_ok=True)


# ============================================================
# 공통 모듈 정의
# ============================================================
def scaled_dot_product_attention(Q, K, V, mask=None):
    """Scaled Dot-Product Attention"""
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    weights = F.softmax(scores, dim=-1)
    output = torch.matmul(weights, V)
    return output, weights


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention"""

    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(weights, V)

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.W_o(attn_output), weights


# ============================================================
# 1. Self-Attention 히트맵 시각화
# ============================================================
print("=" * 60)
print("1. Self-Attention Weight 히트맵")
print("=" * 60)

# 예시 문장
sentence_ko = ["나는", "은행에서", "돈을", "찾았다"]
sentence_en = ["The", "cat", "sat", "on", "the", "mat"]

d_model = 32
num_heads = 4


def visualize_attention(words, d_model, num_heads, title, filename):
    """Attention Weight를 히트맵으로 시각화"""
    seq_len = len(words)
    mha = MultiHeadAttention(d_model, num_heads)

    X = torch.randn(1, seq_len, d_model)
    _, weights = mha(X)  # (1, heads, seq, seq)

    fig, axes = plt.subplots(1, num_heads, figsize=(4 * num_heads, 4))
    if num_heads == 1:
        axes = [axes]

    for h in range(num_heads):
        w = weights[0, h].detach().numpy()
        sns.heatmap(
            w,
            xticklabels=words,
            yticklabels=words,
            annot=True,
            fmt=".2f",
            cmap="YlOrRd",
            vmin=0,
            vmax=1,
            ax=axes[h],
            cbar=False,
        )
        axes[h].set_title(f"Head {h + 1}", fontsize=12)
        axes[h].set_ylabel("Query" if h == 0 else "")

    fig.suptitle(title, fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / filename, dpi=150, bbox_inches="tight")
    print(f"저장: {output_dir / filename}")
    plt.close()


visualize_attention(
    sentence_ko, d_model, num_heads,
    "Self-Attention: '나는 은행에서 돈을 찾았다'",
    "attention_heatmap_ko.png",
)

visualize_attention(
    sentence_en, d_model, num_heads,
    "Self-Attention: 'The cat sat on the mat'",
    "attention_heatmap_en.png",
)

# ============================================================
# 2. Single Head vs Multi-Head 비교
# ============================================================
print("\n" + "=" * 60)
print("2. Single-Head vs Multi-Head Attention 비교")
print("=" * 60)

words = ["I", "love", "natural", "language", "processing"]
seq_len = len(words)
d_model = 32

# Single Head
X = torch.randn(1, seq_len, d_model)
single_head = MultiHeadAttention(d_model, num_heads=1)
_, single_weights = single_head(X)

# Multi Head (4)
multi_head = MultiHeadAttention(d_model, num_heads=4)
_, multi_weights = multi_head(X)

fig, axes = plt.subplots(1, 5, figsize=(22, 4))

# Single Head
sns.heatmap(
    single_weights[0, 0].detach().numpy(),
    xticklabels=words, yticklabels=words,
    annot=True, fmt=".2f", cmap="Blues",
    vmin=0, vmax=1, ax=axes[0], cbar=False,
)
axes[0].set_title("Single Head", fontsize=11)

# Multi Head (4개)
for h in range(4):
    sns.heatmap(
        multi_weights[0, h].detach().numpy(),
        xticklabels=words, yticklabels=words,
        annot=True, fmt=".2f", cmap="Reds",
        vmin=0, vmax=1, ax=axes[h + 1], cbar=False,
    )
    axes[h + 1].set_title(f"Multi-Head {h + 1}", fontsize=11)

fig.suptitle("Single-Head vs Multi-Head Attention 비교", fontsize=14, y=1.02)
fig.tight_layout()
fig.savefig(output_dir / "single_vs_multi_head.png", dpi=150, bbox_inches="tight")
print(f"저장: {output_dir / 'single_vs_multi_head.png'}")
plt.close()

# 다양성 분석
print("\nHead별 Attention 분포 엔트로피 (높을수록 고르게 분포):")
for h in range(4):
    w = multi_weights[0, h].detach().numpy()
    # 각 행의 엔트로피 평균
    entropies = []
    for row in w:
        row = row + 1e-10
        entropy = -np.sum(row * np.log2(row))
        entropies.append(entropy)
    avg_entropy = np.mean(entropies)
    print(f"  Head {h + 1}: 평균 엔트로피 = {avg_entropy:.3f}")


# ============================================================
# 3. Attention 기반 간단한 텍스트 분류
# ============================================================
print("\n" + "=" * 60)
print("3. Attention 기반 텍스트 분류 모델")
print("=" * 60)


class AttentionClassifier(nn.Module):
    """Self-Attention을 활용한 텍스트 분류 모델"""

    def __init__(self, vocab_size, d_model, num_heads, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # 임베딩
        embedded = self.embedding(x)  # (batch, seq, d_model)

        # Self-Attention
        attn_out, weights = self.attention(embedded)

        # 평균 풀링 (모든 토큰의 출력을 평균)
        pooled = attn_out.mean(dim=1)  # (batch, d_model)

        # 분류
        logits = self.classifier(pooled)  # (batch, num_classes)

        return logits, weights


# 간단한 감성 분류 데이터
# 0: 부정, 1: 긍정
train_data = [
    ([1, 2, 3, 4], 1),    # 이 영화 정말 좋다
    ([1, 2, 5, 6], 0),    # 이 영화 매우 싫다
    ([7, 8, 3, 4], 1),    # 그 책은 정말 좋다
    ([7, 8, 5, 6], 0),    # 그 책은 매우 싫다
    ([9, 10, 3, 11], 1),  # 오늘 기분 정말 최고
    ([9, 10, 5, 12], 0),  # 오늘 기분 매우 최악
    ([1, 13, 3, 4], 1),   # 이 음식 정말 좋다
    ([1, 13, 5, 6], 0),   # 이 음식 매우 싫다
]

vocab_map = {
    0: "<pad>", 1: "이", 2: "영화", 3: "정말", 4: "좋다",
    5: "매우", 6: "싫다", 7: "그", 8: "책은", 9: "오늘",
    10: "기분", 11: "최고", 12: "최악", 13: "음식",
}

vocab_size = 14
d_model = 16
num_heads = 2
num_classes = 2

model = AttentionClassifier(vocab_size, d_model, num_heads, num_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

# 학습
X_train = torch.tensor([d[0] for d in train_data])
y_train = torch.tensor([d[1] for d in train_data])

print(f"모델 구조:")
print(f"  Embedding: {vocab_size} → {d_model}")
print(f"  Multi-Head Attention: {num_heads} heads, d_k = {d_model // num_heads}")
print(f"  Classifier: {d_model} → {num_classes}")
print(f"  총 파라미터: {sum(p.numel() for p in model.parameters()):,}")

losses = []
for epoch in range(200):
    logits, _ = model(X_train)
    loss = criterion(logits, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.append(loss.item())
    if (epoch + 1) % 50 == 0:
        preds = logits.argmax(dim=1)
        acc = (preds == y_train).float().mean()
        print(f"  Epoch {epoch + 1:3d}: Loss = {loss.item():.4f}, Accuracy = {acc:.1%}")

# 테스트 및 Attention 시각화
print(f"\n예측 결과:")
model.eval()
with torch.no_grad():
    logits, weights = model(X_train)
    preds = logits.argmax(dim=1)

    labels = ["부정", "긍정"]
    for i, (x, y) in enumerate(train_data):
        words = [vocab_map[idx] for idx in x]
        sentence = " ".join(words)
        pred_label = labels[preds[i]]
        true_label = labels[y]
        correct = "O" if preds[i] == y else "X"
        print(f"  '{sentence}' → {pred_label} (정답: {true_label}) [{correct}]")

# 마지막 문장의 Attention 시각화
fig, axes = plt.subplots(1, num_heads, figsize=(5 * num_heads, 4))
if num_heads == 1:
    axes = [axes]

sample_idx = 0  # "이 영화 정말 좋다"
sample_words = [vocab_map[idx] for idx in train_data[sample_idx][0]]

for h in range(num_heads):
    w = weights[sample_idx, h].numpy()
    sns.heatmap(
        w,
        xticklabels=sample_words,
        yticklabels=sample_words,
        annot=True, fmt=".2f",
        cmap="YlOrRd", vmin=0, vmax=1,
        ax=axes[h], cbar=h == num_heads - 1,
    )
    axes[h].set_title(f"Head {h + 1}", fontsize=12)

fig.suptitle(f"감성 분류: '{' '.join(sample_words)}' (긍정)", fontsize=13, y=1.02)
fig.tight_layout()
fig.savefig(output_dir / "sentiment_attention.png", dpi=150, bbox_inches="tight")
print(f"\n저장: {output_dir / 'sentiment_attention.png'}")
plt.close()

# 학습 곡선
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(losses, color="steelblue", linewidth=1.5)
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.set_title("Attention 분류 모델 학습 곡선")
ax.grid(True, alpha=0.3)
fig.savefig(output_dir / "training_curve.png", dpi=150, bbox_inches="tight")
print(f"저장: {output_dir / 'training_curve.png'}")
plt.close()

print("\n실습 3 완료!")
print(f"\n생성된 시각화 파일:")
for f in sorted(output_dir.glob("*.png")):
    print(f"  {f.name}")
