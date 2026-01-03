"""
5장 실습: Character-level 언어 모델
- 문자 단위 텍스트 생성
- LSTM 기반 언어 모델
- Temperature Sampling
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np


class CharDataset(Dataset):
    """문자 단위 데이터셋"""

    def __init__(self, text, seq_length=50):
        self.seq_length = seq_length

        # 문자-인덱스 매핑
        self.chars = sorted(list(set(text)))
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}
        self.vocab_size = len(self.chars)

        # 텍스트를 인덱스로 변환
        self.data = [self.char_to_idx[ch] for ch in text]

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx : idx + self.seq_length], dtype=torch.long)
        y = torch.tensor(self.data[idx + 1 : idx + self.seq_length + 1], dtype=torch.long)
        return x, y


class CharLSTM(nn.Module):
    """문자 단위 LSTM 언어 모델"""

    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=1):
        super(CharLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 임베딩 레이어
        self.embedding = nn.Embedding(vocab_size, embed_size)

        # LSTM
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)

        # 출력 레이어
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        # x: (batch, seq_len)
        embed = self.embedding(x)  # (batch, seq_len, embed_size)
        output, hidden = self.lstm(embed, hidden)  # (batch, seq_len, hidden_size)
        logits = self.fc(output)  # (batch, seq_len, vocab_size)
        return logits, hidden

    def init_hidden(self, batch_size, device):
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        return (h0, c0)


def generate_text(model, dataset, seed_text, length=100, temperature=1.0, device="cpu"):
    """텍스트 생성"""
    model.eval()

    # 시드 텍스트를 인덱스로 변환
    chars = [dataset.char_to_idx.get(ch, 0) for ch in seed_text]
    input_seq = torch.tensor(chars, dtype=torch.long).unsqueeze(0).to(device)

    generated = seed_text
    hidden = model.init_hidden(1, device)

    with torch.no_grad():
        # 시드 텍스트로 hidden state 초기화
        for i in range(len(seed_text) - 1):
            _, hidden = model(input_seq[:, i : i + 1], hidden)

        # 텍스트 생성
        current_char = input_seq[:, -1:]
        for _ in range(length):
            logits, hidden = model(current_char, hidden)

            # Temperature Sampling
            logits = logits[:, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)

            # 확률적 샘플링
            next_idx = torch.multinomial(probs, 1)
            next_char = dataset.idx_to_char[next_idx.item()]
            generated += next_char

            current_char = next_idx

    return generated


def train_model(model, dataloader, num_epochs, device):
    """모델 학습"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.002)

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            # Hidden state 초기화
            hidden = model.init_hidden(batch_x.size(0), device)

            optimizer.zero_grad()
            logits, _ = model(batch_x, hidden)

            # Loss 계산: (batch * seq_len, vocab_size) vs (batch * seq_len)
            loss = criterion(logits.view(-1, logits.size(-1)), batch_y.view(-1))

            loss.backward()
            # Gradient Clipping (기울기 폭주 방지)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

    return avg_loss


def main():
    print("=" * 50)
    print("Character-level 언어 모델")
    print("=" * 50)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 샘플 텍스트 (한국어)
    sample_text = """
    인공지능은 인간의 학습 능력과 추론 능력을 모방하여 만든 컴퓨터 시스템이다.
    딥러닝은 인공지능의 한 분야로 신경망을 깊게 쌓아 복잡한 패턴을 학습한다.
    자연어처리는 컴퓨터가 인간의 언어를 이해하고 생성하는 기술이다.
    언어 모델은 주어진 문맥에서 다음 단어를 예측하는 모델이다.
    순환 신경망은 순차 데이터를 처리하는 신경망 구조이다.
    장단기 기억 신경망은 장기 의존성 문제를 해결한다.
    게이트 순환 유닛은 장단기 기억 신경망의 간소화된 버전이다.
    트랜스포머는 어텐션 메커니즘을 사용하는 새로운 아키텍처이다.
    대규모 언어 모델은 수십억 개의 파라미터를 가진 언어 모델이다.
    자연어처리 기술은 번역, 요약, 질의응답 등에 활용된다.
    """

    # 반복하여 데이터 늘리기
    sample_text = sample_text * 10

    print(f"\n텍스트 길이: {len(sample_text)} 문자")

    # 데이터셋 생성
    seq_length = 30
    dataset = CharDataset(sample_text, seq_length=seq_length)

    print(f"어휘 크기: {dataset.vocab_size}")
    print(f"시퀀스 길이: {seq_length}")
    print(f"샘플 수: {len(dataset)}")
    print(f"문자 집합: {dataset.chars[:20]}...")

    # DataLoader
    batch_size = 32
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print("\n" + "=" * 50)
    print("1. 모델 생성")
    print("=" * 50)

    # 모델 생성
    embed_size = 64
    hidden_size = 128
    num_layers = 2

    model = CharLSTM(dataset.vocab_size, embed_size, hidden_size, num_layers)
    model = model.to(device)

    print(f"모델 구조:\n{model}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n총 파라미터 수: {total_params:,}")

    print("\n" + "=" * 50)
    print("2. 모델 학습")
    print("=" * 50)

    num_epochs = 30
    final_loss = train_model(model, dataloader, num_epochs, device)
    print(f"\n최종 Loss: {final_loss:.4f}")

    print("\n" + "=" * 50)
    print("3. 텍스트 생성 - Temperature 비교")
    print("=" * 50)

    seed = "인공지능"

    temperatures = [0.5, 0.8, 1.0, 1.5]
    for temp in temperatures:
        generated = generate_text(model, dataset, seed, length=50, temperature=temp, device=device)
        print(f"\n[Temperature = {temp}]")
        print(f"  {generated}")

    print("\n" + "=" * 50)
    print("4. Temperature의 효과")
    print("=" * 50)

    print("""
    Temperature Sampling:
    - logits = logits / temperature
    - probs = softmax(logits)

    Temperature 효과:
    - T < 1 (예: 0.5): 확률 분포가 날카로워짐
      → 높은 확률 문자 선택 확률 증가
      → 더 결정적, 반복적, 안전한 텍스트

    - T = 1: 원래 확률 분포 유지
      → 학습된 분포대로 샘플링

    - T > 1 (예: 1.5): 확률 분포가 평평해짐
      → 낮은 확률 문자도 선택될 가능성 증가
      → 더 창의적, 다양하지만 오류 가능성 증가
    """)

    print("\n" + "=" * 50)
    print("5. 다양한 시드로 생성")
    print("=" * 50)

    seeds = ["딥러닝", "자연어", "신경망"]
    for seed in seeds:
        generated = generate_text(model, dataset, seed, length=40, temperature=0.8, device=device)
        print(f"'{seed}' → {generated}")

    print("\n" + "=" * 50)
    print("핵심 정리")
    print("=" * 50)
    print("""
    Character-level 언어 모델:
    - 문자 단위로 다음 문자 예측
    - 어휘 크기 작음 (OOV 문제 없음)
    - 철자, 형태론적 패턴 학습 가능

    모델 구조:
    - Embedding → LSTM → Linear
    - 각 시간 단계에서 다음 문자 확률 예측

    텍스트 생성:
    - 시드 텍스트로 시작
    - 다음 문자 샘플링 후 입력에 추가
    - 원하는 길이까지 반복

    Gradient Clipping:
    - 기울기 폭주 방지
    - clip_grad_norm_(parameters, max_norm)
    """)


if __name__ == "__main__":
    main()
