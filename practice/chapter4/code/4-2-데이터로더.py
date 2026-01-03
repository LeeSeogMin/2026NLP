"""
4장 실습: Dataset과 DataLoader
- 커스텀 Dataset 작성
- DataLoader 활용
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class SimpleDataset(Dataset):
    """간단한 커스텀 Dataset"""

    def __init__(self, X, y):
        """
        Args:
            X: 특성 데이터 (numpy array 또는 tensor)
            y: 레이블 데이터
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self):
        """데이터셋 크기 반환"""
        return len(self.y)

    def __getitem__(self, idx):
        """인덱스로 샘플 접근"""
        return self.X[idx], self.y[idx]


class TextDataset(Dataset):
    """텍스트 데이터용 Dataset"""

    def __init__(self, texts, labels, vocab=None, max_len=50):
        self.texts = texts
        self.labels = labels
        self.max_len = max_len

        # 어휘 사전 구축 또는 사용
        if vocab is None:
            self.vocab = self._build_vocab(texts)
        else:
            self.vocab = vocab

    def _build_vocab(self, texts):
        """간단한 어휘 사전 구축"""
        vocab = {"<PAD>": 0, "<UNK>": 1}
        for text in texts:
            for word in text.split():
                if word not in vocab:
                    vocab[word] = len(vocab)
        return vocab

    def _text_to_indices(self, text):
        """텍스트를 인덱스 시퀀스로 변환"""
        indices = []
        for word in text.split():
            idx = self.vocab.get(word, self.vocab["<UNK>"])
            indices.append(idx)

        # 패딩 또는 자르기
        if len(indices) < self.max_len:
            indices += [self.vocab["<PAD>"]] * (self.max_len - len(indices))
        else:
            indices = indices[:self.max_len]

        return indices

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        indices = self._text_to_indices(text)
        return torch.LongTensor(indices), torch.tensor(label)


def main():
    print("=" * 50)
    print("1. 기본 Dataset과 DataLoader")
    print("=" * 50)

    # 샘플 데이터 생성
    np.random.seed(42)
    X = np.random.randn(100, 10).astype(np.float32)
    y = np.random.randint(0, 2, 100)

    # Dataset 생성
    dataset = SimpleDataset(X, y)
    print(f"데이터셋 크기: {len(dataset)}")
    print(f"첫 번째 샘플: X shape={dataset[0][0].shape}, y={dataset[0][1]}")

    # DataLoader 생성
    dataloader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        num_workers=0  # 단일 프로세스
    )

    print(f"\n배치 수: {len(dataloader)}")

    # 배치 순회
    print("\n[첫 3개 배치]")
    for i, (batch_X, batch_y) in enumerate(dataloader):
        print(f"  배치 {i + 1}: X shape={batch_X.shape}, y shape={batch_y.shape}")
        if i >= 2:
            break

    print("\n" + "=" * 50)
    print("2. 텍스트 Dataset")
    print("=" * 50)

    # 샘플 텍스트 데이터
    texts = [
        "이 영화 정말 재미있다",
        "최악의 영화였다",
        "감동적인 스토리",
        "지루하고 재미없다",
        "배우들의 연기가 훌륭하다",
        "시간 낭비였다",
    ]
    labels = [1, 0, 1, 0, 1, 0]  # 1: 긍정, 0: 부정

    text_dataset = TextDataset(texts, labels, max_len=10)

    print(f"어휘 크기: {len(text_dataset.vocab)}")
    print(f"어휘 사전: {text_dataset.vocab}")

    print("\n[샘플 확인]")
    for i in range(3):
        indices, label = text_dataset[i]
        print(f"  텍스트: '{texts[i]}'")
        print(f"  인덱스: {indices.tolist()}")
        print(f"  레이블: {label.item()}")
        print()

    print("=" * 50)
    print("3. 학습/검증 데이터 분할")
    print("=" * 50)

    from torch.utils.data import random_split

    # 전체 데이터셋
    full_dataset = SimpleDataset(X, y)

    # 80% 학습, 20% 검증
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size]
    )

    print(f"학습 데이터: {len(train_dataset)}")
    print(f"검증 데이터: {len(val_dataset)}")

    # 각각의 DataLoader 생성
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    print(f"학습 배치 수: {len(train_loader)}")
    print(f"검증 배치 수: {len(val_loader)}")

    print("\n" + "=" * 50)
    print("핵심 정리")
    print("=" * 50)
    print("""
    Dataset 작성:
    - __init__: 데이터 로드
    - __len__: 데이터 개수 반환
    - __getitem__: 인덱스로 샘플 반환

    DataLoader 주요 파라미터:
    - batch_size: 배치 크기
    - shuffle: 에폭마다 셔플
    - num_workers: 병렬 로딩 워커 수
    - drop_last: 마지막 불완전 배치 버림

    데이터 분할:
    - random_split(): 무작위 분할
    - 일반적으로 80:20 또는 80:10:10
    """)


if __name__ == "__main__":
    main()
