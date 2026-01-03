"""
4장 실습: 텍스트 분류
- Bag-of-Words 벡터화
- MLP 기반 감성 분석
- 모델 평가 및 시각화
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from collections import Counter
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt


class TextClassificationDataset(Dataset):
    """텍스트 분류용 Dataset"""

    def __init__(self, texts, labels, vocab=None, max_vocab_size=1000):
        self.texts = texts
        self.labels = labels

        # 어휘 사전 구축
        if vocab is None:
            self.vocab = self._build_vocab(texts, max_vocab_size)
        else:
            self.vocab = vocab

        self.vocab_size = len(self.vocab)

    def _build_vocab(self, texts, max_vocab_size):
        """단어 빈도 기반 어휘 사전 구축"""
        word_counts = Counter()
        for text in texts:
            words = text.split()
            word_counts.update(words)

        # 빈도순 정렬 후 상위 단어 선택
        most_common = word_counts.most_common(max_vocab_size - 2)

        vocab = {"<PAD>": 0, "<UNK>": 1}
        for word, _ in most_common:
            vocab[word] = len(vocab)

        return vocab

    def _text_to_bow(self, text):
        """텍스트를 Bag-of-Words 벡터로 변환"""
        bow = np.zeros(self.vocab_size, dtype=np.float32)
        words = text.split()
        for word in words:
            idx = self.vocab.get(word, self.vocab["<UNK>"])
            bow[idx] += 1

        # 정규화 (L2 norm)
        norm = np.linalg.norm(bow)
        if norm > 0:
            bow = bow / norm

        return bow

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        bow = self._text_to_bow(text)
        return torch.FloatTensor(bow), torch.tensor(label, dtype=torch.long)


class TextClassifier(nn.Module):
    """MLP 기반 텍스트 분류 모델"""

    def __init__(self, vocab_size, hidden_size, num_classes, dropout=0.3):
        super(TextClassifier, self).__init__()

        self.classifier = nn.Sequential(
            nn.Linear(vocab_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes),
        )

    def forward(self, x):
        return self.classifier(x)


def create_sample_data():
    """샘플 영화 리뷰 데이터 생성"""
    # 긍정 리뷰
    positive = [
        "이 영화 정말 재미있다 추천한다",
        "감동적인 스토리 배우 연기 최고",
        "완벽한 영화 다시 보고 싶다",
        "기대 이상이다 정말 좋았다",
        "훌륭한 작품 명작이다",
        "재미있고 감동적인 영화",
        "최고의 영화 강력 추천",
        "멋진 영화 배우들 연기 훌륭",
        "좋은 영화 재미있게 봤다",
        "감동 받았다 눈물이 났다",
        "연기 좋고 스토리 좋고 최고",
        "정말 좋은 영화 추천한다",
        "기대했던 대로 정말 좋았다",
        "완벽한 스토리 감동적이다",
        "배우들 연기가 훌륭하다",
        "최고의 감동 영화이다",
        "다시 보고 싶은 영화",
        "정말 재미있는 영화였다",
        "감동적인 명작 영화이다",
        "좋은 영화 감동 받았다",
    ]

    # 부정 리뷰
    negative = [
        "지루하고 재미없다 최악",
        "시간 낭비였다 별로다",
        "기대 이하 실망했다",
        "스토리가 너무 지루하다",
        "연기도 별로 내용도 별로",
        "최악의 영화 돈 아깝다",
        "재미없다 추천 안한다",
        "보다가 잠들었다 지루했다",
        "실망스러운 영화였다",
        "기대했는데 별로였다",
        "스토리가 엉망이다",
        "연기가 너무 어색하다",
        "지루해서 끝까지 못봤다",
        "최악이다 보지마라",
        "시간이 아깝다 별로다",
        "실망이다 기대 이하",
        "재미없는 영화다",
        "별로다 추천 안한다",
        "스토리가 별로다",
        "지루하다 재미없다",
    ]

    texts = positive + negative
    labels = [1] * len(positive) + [0] * len(negative)

    # 데이터 셔플
    indices = np.random.permutation(len(texts))
    texts = [texts[i] for i in indices]
    labels = [labels[i] for i in indices]

    return texts, labels


def train_and_evaluate():
    """모델 학습 및 평가"""
    print("=" * 50)
    print("텍스트 분류 모델 학습")
    print("=" * 50)

    # 데이터 준비
    np.random.seed(42)
    torch.manual_seed(42)

    texts, labels = create_sample_data()
    print(f"\n전체 데이터: {len(texts)} 샘플")
    print(f"긍정: {sum(labels)}, 부정: {len(labels) - sum(labels)}")

    # Dataset 생성
    dataset = TextClassificationDataset(texts, labels, max_vocab_size=200)
    print(f"어휘 크기: {dataset.vocab_size}")

    # Train/Val 분할
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    print(f"학습 데이터: {len(train_dataset)}, 검증 데이터: {len(val_dataset)}")

    # 모델 초기화
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TextClassifier(
        vocab_size=dataset.vocab_size, hidden_size=64, num_classes=2, dropout=0.3
    )
    model = model.to(device)

    print(f"\n모델 구조:\n{model}")

    # 학습 설정
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

    # 학습
    print("\n" + "-" * 50)
    print("학습 시작")
    print("-" * 50)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    num_epochs = 30

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch_X.size(0)
            _, predicted = outputs.max(1)
            train_total += batch_y.size(0)
            train_correct += predicted.eq(batch_y).sum().item()

        train_loss /= train_total
        train_acc = 100.0 * train_correct / train_total

        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)

                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)

                val_loss += loss.item() * batch_X.size(0)
                _, predicted = outputs.max(1)
                val_total += batch_y.size(0)
                val_correct += predicted.eq(batch_y).sum().item()

        val_loss /= val_total
        val_acc = 100.0 * val_correct / val_total

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch [{epoch+1:2d}/{num_epochs}] "
                f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.1f}% | "
                f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.1f}%"
            )

    print(f"\n최종 검증 정확도: {history['val_acc'][-1]:.1f}%")

    # 평가
    print("\n" + "=" * 50)
    print("모델 평가")
    print("=" * 50)

    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.numpy())

    # 평가 지표
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="binary")

    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    print(f"\nConfusion Matrix:")
    print(f"           Predicted")
    print(f"           Neg   Pos")
    print(f"Actual Neg  {cm[0][0]:3d}   {cm[0][1]:3d}")
    print(f"       Pos  {cm[1][0]:3d}   {cm[1][1]:3d}")

    # 시각화
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Loss Curve
    axes[0].plot(history["train_loss"], label="Train")
    axes[0].plot(history["val_loss"], label="Validation")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss Curve")
    axes[0].legend()
    axes[0].grid(True)

    # Accuracy Curve
    axes[1].plot(history["train_acc"], label="Train")
    axes[1].plot(history["val_acc"], label="Validation")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("Accuracy Curve")
    axes[1].legend()
    axes[1].grid(True)

    # Confusion Matrix
    im = axes[2].imshow(cm, cmap="Blues")
    axes[2].set_xticks([0, 1])
    axes[2].set_yticks([0, 1])
    axes[2].set_xticklabels(["Negative", "Positive"])
    axes[2].set_yticklabels(["Negative", "Positive"])
    axes[2].set_xlabel("Predicted")
    axes[2].set_ylabel("Actual")
    axes[2].set_title("Confusion Matrix")

    for i in range(2):
        for j in range(2):
            axes[2].text(j, i, cm[i, j], ha="center", va="center", fontsize=14)

    plt.colorbar(im, ax=axes[2])
    plt.tight_layout()
    plt.savefig("text_classification_results.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("\n그래프 저장: text_classification_results.png")

    # 예측 테스트
    print("\n" + "=" * 50)
    print("새로운 텍스트 예측")
    print("=" * 50)

    test_texts = ["정말 재미있는 영화다", "지루하고 별로다", "최고의 명작 강력 추천"]

    model.eval()
    for text in test_texts:
        bow = dataset._text_to_bow(text)
        input_tensor = torch.FloatTensor(bow).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor)
            prob = torch.softmax(output, dim=1)
            pred = output.argmax(1).item()

        sentiment = "긍정" if pred == 1 else "부정"
        confidence = prob[0][pred].item() * 100
        print(f"'{text}' → {sentiment} (신뢰도: {confidence:.1f}%)")

    print("\n" + "=" * 50)
    print("핵심 정리")
    print("=" * 50)
    print("""
    텍스트 분류 파이프라인:
    1. 텍스트 전처리 (토큰화)
    2. 어휘 사전 구축
    3. 텍스트 벡터화 (Bag-of-Words)
    4. MLP 모델 학습
    5. 평가 지표 계산

    Bag-of-Words:
    - 단어 빈도 기반 벡터
    - 단순하지만 효과적
    - 어순 정보 손실

    평가 지표:
    - Accuracy: 전체 정확도
    - Precision: 양성 예측 정확도
    - Recall: 실제 양성 탐지율
    - F1-Score: Precision과 Recall의 조화 평균
    """)


if __name__ == "__main__":
    train_and_evaluate()
