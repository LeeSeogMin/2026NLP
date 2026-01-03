"""
5장 실습: LSTM (Long Short-Term Memory)
- LSTM의 구조와 게이트 이해
- Cell State와 Hidden State
- 장기 의존성 문제 해결
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


class LSTMClassifier(nn.Module):
    """LSTM 기반 시퀀스 분류 모델"""

    def __init__(self, input_size, hidden_size, num_classes, num_layers=1, dropout=0.0):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: (batch, seq_len, input_size)

        # LSTM 순전파
        # output: (batch, seq_len, hidden_size)
        # (h_n, c_n): 각각 (num_layers, batch, hidden_size)
        output, (h_n, c_n) = self.lstm(x)

        # 마지막 hidden state 사용
        out = self.fc(h_n[-1])  # (batch, num_classes)
        return out


class ManualLSTMCell(nn.Module):
    """LSTM 셀 수동 구현 (게이트 이해용)"""

    def __init__(self, input_size, hidden_size):
        super(ManualLSTMCell, self).__init__()
        self.hidden_size = hidden_size

        # 4개의 게이트를 위한 가중치 (i, f, g, o)
        self.W_ii = nn.Linear(input_size, hidden_size, bias=True)
        self.W_hi = nn.Linear(hidden_size, hidden_size, bias=False)

        self.W_if = nn.Linear(input_size, hidden_size, bias=True)
        self.W_hf = nn.Linear(hidden_size, hidden_size, bias=False)

        self.W_ig = nn.Linear(input_size, hidden_size, bias=True)
        self.W_hg = nn.Linear(hidden_size, hidden_size, bias=False)

        self.W_io = nn.Linear(input_size, hidden_size, bias=True)
        self.W_ho = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x_t, h_prev, c_prev):
        """
        x_t: (batch, input_size) - 현재 입력
        h_prev: (batch, hidden_size) - 이전 hidden state
        c_prev: (batch, hidden_size) - 이전 cell state
        """
        # Input Gate: 새로운 정보 중 얼마나 저장할지
        i_t = torch.sigmoid(self.W_ii(x_t) + self.W_hi(h_prev))

        # Forget Gate: 이전 cell state에서 얼마나 버릴지
        f_t = torch.sigmoid(self.W_if(x_t) + self.W_hf(h_prev))

        # Cell Gate (Candidate): 새로운 후보 값
        g_t = torch.tanh(self.W_ig(x_t) + self.W_hg(h_prev))

        # Output Gate: 출력할 정보
        o_t = torch.sigmoid(self.W_io(x_t) + self.W_ho(h_prev))

        # Cell State 업데이트
        c_t = f_t * c_prev + i_t * g_t

        # Hidden State 계산
        h_t = o_t * torch.tanh(c_t)

        return h_t, c_t, (i_t, f_t, g_t, o_t)


def visualize_gates(gates_history, seq_len):
    """게이트 활성화 시각화"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    gate_names = ["Input Gate", "Forget Gate", "Cell Gate", "Output Gate"]

    for idx, (ax, name) in enumerate(zip(axes.flat, gate_names)):
        gate_values = [g[idx][0, :4].detach().numpy() for g in gates_history]
        gate_array = np.array(gate_values)

        im = ax.imshow(gate_array.T, aspect="auto", cmap="RdYlGn")
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Hidden Unit")
        ax.set_title(name)
        plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig("lstm_gates.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("게이트 시각화 저장: lstm_gates.png")


def main():
    print("=" * 50)
    print("1. LSTM 기본 구조")
    print("=" * 50)

    # 설정
    batch_size = 2
    seq_len = 10
    input_size = 5
    hidden_size = 8

    torch.manual_seed(42)
    x = torch.randn(batch_size, seq_len, input_size)

    # PyTorch LSTM
    lstm = nn.LSTM(input_size, hidden_size, batch_first=True)

    # 초기 상태
    h0 = torch.zeros(1, batch_size, hidden_size)
    c0 = torch.zeros(1, batch_size, hidden_size)

    print(f"입력 shape: {x.shape}")
    print(f"초기 hidden state shape: {h0.shape}")
    print(f"초기 cell state shape: {c0.shape}")

    # 순전파
    output, (h_n, c_n) = lstm(x, (h0, c0))

    print(f"\n출력 shape: {output.shape}")
    print(f"최종 hidden state shape: {h_n.shape}")
    print(f"최종 cell state shape: {c_n.shape}")

    print("\n" + "=" * 50)
    print("2. 수동 LSTM Cell 구현")
    print("=" * 50)

    manual_lstm = ManualLSTMCell(input_size, hidden_size)

    h = torch.zeros(batch_size, hidden_size)
    c = torch.zeros(batch_size, hidden_size)
    gates_history = []

    print("\n[시간 단계별 LSTM 처리]")
    for t in range(seq_len):
        x_t = x[:, t, :]
        h, c, gates = manual_lstm(x_t, h, c)
        gates_history.append(gates)

        if t < 3 or t >= seq_len - 2:
            i, f, g, o = gates
            print(f"  t={t}: h mean={h.mean().item():.4f}, "
                  f"forget={f.mean().item():.4f}, input={i.mean().item():.4f}")

    # 게이트 시각화
    visualize_gates(gates_history, seq_len)

    print("\n" + "=" * 50)
    print("3. LSTM의 게이트 역할")
    print("=" * 50)

    print("""
    Forget Gate (f_t):
    - 이전 Cell State에서 버릴 정보 결정
    - σ(W_f·[h_{t-1}, x_t] + b_f)
    - 0에 가까우면: 과거 정보 삭제
    - 1에 가까우면: 과거 정보 유지

    Input Gate (i_t):
    - 새로운 정보 중 저장할 부분 결정
    - σ(W_i·[h_{t-1}, x_t] + b_i)

    Cell Gate (g_t):
    - 새로운 후보 값 생성
    - tanh(W_g·[h_{t-1}, x_t] + b_g)

    Output Gate (o_t):
    - Cell State에서 출력할 정보 결정
    - σ(W_o·[h_{t-1}, x_t] + b_o)

    Cell State 업데이트:
    - C_t = f_t ⊙ C_{t-1} + i_t ⊙ g_t
    - 덧셈으로 기울기가 직접 흐름 (기울기 소실 방지)

    Hidden State:
    - h_t = o_t ⊙ tanh(C_t)
    """)

    print("\n" + "=" * 50)
    print("4. 다층 LSTM")
    print("=" * 50)

    num_layers = 2
    multi_lstm = nn.LSTM(
        input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=0.1
    )

    h0_multi = torch.zeros(num_layers, batch_size, hidden_size)
    c0_multi = torch.zeros(num_layers, batch_size, hidden_size)

    output_multi, (h_n_multi, c_n_multi) = multi_lstm(x, (h0_multi, c0_multi))

    print(f"2층 LSTM:")
    print(f"  - 출력 shape: {output_multi.shape}")
    print(f"  - hidden shape: {h_n_multi.shape}")
    print(f"  - cell shape: {c_n_multi.shape}")

    print("\n" + "=" * 50)
    print("5. LSTM 분류 모델")
    print("=" * 50)

    num_classes = 3
    model = LSTMClassifier(input_size, hidden_size, num_classes, num_layers=2)
    print(f"모델 구조:\n{model}")

    # 파라미터 수
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n총 파라미터 수: {total_params}")

    # 순전파
    logits = model(x)
    print(f"출력 shape: {logits.shape}")
    print(f"예측 클래스: {logits.argmax(dim=1).tolist()}")

    print("\n" + "=" * 50)
    print("6. RNN vs LSTM 파라미터 비교")
    print("=" * 50)

    rnn = nn.RNN(input_size, hidden_size, batch_first=True)
    lstm = nn.LSTM(input_size, hidden_size, batch_first=True)

    rnn_params = sum(p.numel() for p in rnn.parameters())
    lstm_params = sum(p.numel() for p in lstm.parameters())

    print(f"RNN 파라미터: {rnn_params}")
    print(f"LSTM 파라미터: {lstm_params}")
    print(f"LSTM/RNN 비율: {lstm_params / rnn_params:.2f}x")
    print(f"(LSTM은 4개 게이트로 인해 약 4배 파라미터)")

    print("\n" + "=" * 50)
    print("핵심 정리")
    print("=" * 50)
    print("""
    LSTM 핵심:
    - Cell State: 장기 기억 저장소
    - 3개 게이트로 정보 흐름 제어
    - 덧셈 연산으로 기울기 소실 방지

    PyTorch nn.LSTM:
    - 출력: output, (h_n, c_n)
    - h_n: hidden state (단기 기억)
    - c_n: cell state (장기 기억)

    장기 의존성 해결:
    - Forget Gate: 선택적 망각
    - Input Gate: 선택적 기억
    - Cell State의 덧셈 업데이트
    """)


if __name__ == "__main__":
    main()
