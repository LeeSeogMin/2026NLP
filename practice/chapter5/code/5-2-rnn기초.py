"""
5장 실습: RNN 기초
- RNN의 기본 구조 이해
- Hidden State의 역할
- PyTorch nn.RNN 사용법
"""

import torch
import torch.nn as nn
import numpy as np


class SimpleRNN(nn.Module):
    """간단한 RNN 모델"""

    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size

        # RNN 레이어
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)

        # 출력 레이어
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden=None):
        # x: (batch, seq_len, input_size)
        # hidden: (1, batch, hidden_size)

        # RNN 순전파
        out, hidden = self.rnn(x, hidden)
        # out: (batch, seq_len, hidden_size)
        # hidden: (1, batch, hidden_size)

        # 마지막 시간 단계의 출력만 사용
        out = self.fc(out[:, -1, :])
        return out, hidden


class ManualRNNCell(nn.Module):
    """RNN 셀을 수동으로 구현"""

    def __init__(self, input_size, hidden_size):
        super(ManualRNNCell, self).__init__()
        self.hidden_size = hidden_size

        # 가중치 정의
        self.W_xh = nn.Linear(input_size, hidden_size, bias=False)
        self.W_hh = nn.Linear(hidden_size, hidden_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x_t, h_prev):
        # h_t = tanh(W_xh * x_t + W_hh * h_prev + b)
        h_t = torch.tanh(self.W_xh(x_t) + self.W_hh(h_prev) + self.bias)
        return h_t


def main():
    print("=" * 50)
    print("1. RNN 기본 개념")
    print("=" * 50)

    # 설정
    batch_size = 2
    seq_len = 5
    input_size = 3
    hidden_size = 4
    output_size = 2

    # 샘플 데이터
    torch.manual_seed(42)
    x = torch.randn(batch_size, seq_len, input_size)
    print(f"입력 shape: {x.shape}")
    print(f"  - 배치 크기: {batch_size}")
    print(f"  - 시퀀스 길이: {seq_len}")
    print(f"  - 입력 차원: {input_size}")

    print("\n" + "=" * 50)
    print("2. PyTorch nn.RNN 사용")
    print("=" * 50)

    # nn.RNN 직접 사용
    rnn = nn.RNN(input_size, hidden_size, batch_first=True)
    print(f"\nRNN 구조: {rnn}")

    # 초기 hidden state
    h0 = torch.zeros(1, batch_size, hidden_size)
    print(f"초기 hidden state shape: {h0.shape}")

    # 순전파
    output, h_n = rnn(x, h0)
    print(f"\n출력 shape: {output.shape}")
    print(f"  - (batch={batch_size}, seq_len={seq_len}, hidden={hidden_size})")
    print(f"최종 hidden state shape: {h_n.shape}")

    # 시간 단계별 출력 확인
    print("\n[시간 단계별 출력 (첫 번째 배치)]")
    for t in range(seq_len):
        print(f"  t={t}: {output[0, t, :2].detach().numpy()}")  # 처음 2개 값만

    print("\n" + "=" * 50)
    print("3. 수동 RNN Cell 구현")
    print("=" * 50)

    # 수동 구현
    manual_cell = ManualRNNCell(input_size, hidden_size)

    h = torch.zeros(batch_size, hidden_size)
    outputs = []

    print("\n[수동 RNN 순전파]")
    for t in range(seq_len):
        x_t = x[:, t, :]  # 시간 t의 입력
        h = manual_cell(x_t, h)
        outputs.append(h)
        print(f"  t={t}: h shape={h.shape}, h mean={h.mean().item():.4f}")

    # 출력 스택
    manual_output = torch.stack(outputs, dim=1)
    print(f"\n수동 구현 출력 shape: {manual_output.shape}")

    print("\n" + "=" * 50)
    print("4. 다층 RNN")
    print("=" * 50)

    # 2층 RNN
    num_layers = 2
    multi_rnn = nn.RNN(input_size, hidden_size, num_layers=num_layers, batch_first=True)

    h0_multi = torch.zeros(num_layers, batch_size, hidden_size)
    output_multi, h_n_multi = multi_rnn(x, h0_multi)

    print(f"2층 RNN:")
    print(f"  - 출력 shape: {output_multi.shape}")
    print(f"  - hidden shape: {h_n_multi.shape}")
    print(f"    (num_layers={num_layers}, batch={batch_size}, hidden={hidden_size})")

    print("\n" + "=" * 50)
    print("5. 양방향 RNN")
    print("=" * 50)

    # 양방향 RNN
    bi_rnn = nn.RNN(input_size, hidden_size, batch_first=True, bidirectional=True)

    h0_bi = torch.zeros(2, batch_size, hidden_size)  # 2 = 양방향
    output_bi, h_n_bi = bi_rnn(x, h0_bi)

    print(f"양방향 RNN:")
    print(f"  - 출력 shape: {output_bi.shape}")
    print(f"    (batch={batch_size}, seq_len={seq_len}, hidden*2={hidden_size * 2})")
    print(f"  - hidden shape: {h_n_bi.shape}")
    print(f"    (2=양방향, batch={batch_size}, hidden={hidden_size})")

    print("\n" + "=" * 50)
    print("6. 간단한 분류 모델")
    print("=" * 50)

    model = SimpleRNN(input_size, hidden_size, output_size)
    print(f"모델 구조:\n{model}")

    # 파라미터 수
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n총 파라미터 수: {total_params}")

    # 순전파 테스트
    output, final_hidden = model(x)
    print(f"\n분류 출력 shape: {output.shape}")
    print(f"  (batch={batch_size}, output_size={output_size})")

    print("\n" + "=" * 50)
    print("핵심 정리")
    print("=" * 50)
    print("""
    RNN 핵심 개념:
    - 순차 데이터를 처리하는 신경망
    - Hidden State로 과거 정보 유지
    - 시간 단계마다 가중치 공유

    PyTorch nn.RNN:
    - input: (batch, seq_len, input_size) - batch_first=True
    - output: (batch, seq_len, hidden_size)
    - h_n: (num_layers, batch, hidden_size)

    주요 옵션:
    - num_layers: 층 수
    - bidirectional: 양방향 처리
    - batch_first: 배치 차원 순서
    """)


if __name__ == "__main__":
    main()
