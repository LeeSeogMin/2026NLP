"""
5장 실습: GRU (Gated Recurrent Unit)
- GRU의 구조와 LSTM과의 차이
- Reset Gate와 Update Gate
- LSTM vs GRU 비교
"""

import torch
import torch.nn as nn
import numpy as np
import time


class GRUClassifier(nn.Module):
    """GRU 기반 시퀀스 분류 모델"""

    def __init__(self, input_size, hidden_size, num_classes, num_layers=1):
        super(GRUClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        output, h_n = self.gru(x)
        # h_n: (num_layers, batch, hidden_size)
        out = self.fc(h_n[-1])
        return out


class ManualGRUCell(nn.Module):
    """GRU 셀 수동 구현"""

    def __init__(self, input_size, hidden_size):
        super(ManualGRUCell, self).__init__()
        self.hidden_size = hidden_size

        # Reset Gate
        self.W_ir = nn.Linear(input_size, hidden_size, bias=True)
        self.W_hr = nn.Linear(hidden_size, hidden_size, bias=False)

        # Update Gate
        self.W_iz = nn.Linear(input_size, hidden_size, bias=True)
        self.W_hz = nn.Linear(hidden_size, hidden_size, bias=False)

        # New Gate (Candidate)
        self.W_in = nn.Linear(input_size, hidden_size, bias=True)
        self.W_hn = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x_t, h_prev):
        """
        x_t: (batch, input_size)
        h_prev: (batch, hidden_size)
        """
        # Reset Gate: 과거 정보를 얼마나 리셋할지
        r_t = torch.sigmoid(self.W_ir(x_t) + self.W_hr(h_prev))

        # Update Gate: 새로운 정보와 과거 정보의 비율
        z_t = torch.sigmoid(self.W_iz(x_t) + self.W_hz(h_prev))

        # New Gate: 새로운 후보 hidden state
        # r_t로 과거 정보를 선택적으로 리셋
        n_t = torch.tanh(self.W_in(x_t) + r_t * self.W_hn(h_prev))

        # Hidden State 업데이트
        # z_t가 1이면: 과거 유지, z_t가 0이면: 새로운 값 사용
        h_t = (1 - z_t) * n_t + z_t * h_prev

        return h_t, (r_t, z_t, n_t)


def compare_lstm_gru():
    """LSTM과 GRU 비교"""
    print("=" * 50)
    print("LSTM vs GRU 비교")
    print("=" * 50)

    input_size = 64
    hidden_size = 128
    seq_len = 100
    batch_size = 32
    num_iterations = 100

    # 모델 생성
    lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
    gru = nn.GRU(input_size, hidden_size, batch_first=True)

    # 파라미터 수 비교
    lstm_params = sum(p.numel() for p in lstm.parameters())
    gru_params = sum(p.numel() for p in gru.parameters())

    print(f"\n[파라미터 수]")
    print(f"  LSTM: {lstm_params:,}")
    print(f"  GRU:  {gru_params:,}")
    print(f"  GRU/LSTM: {gru_params / lstm_params * 100:.1f}%")

    # 속도 비교
    x = torch.randn(batch_size, seq_len, input_size)

    # LSTM 속도
    start = time.time()
    for _ in range(num_iterations):
        _ = lstm(x)
    lstm_time = time.time() - start

    # GRU 속도
    start = time.time()
    for _ in range(num_iterations):
        _ = gru(x)
    gru_time = time.time() - start

    print(f"\n[추론 속도 ({num_iterations} iterations)]")
    print(f"  LSTM: {lstm_time:.4f}s")
    print(f"  GRU:  {gru_time:.4f}s")
    print(f"  GRU 속도 향상: {(lstm_time - gru_time) / lstm_time * 100:.1f}%")

    # 메모리 비교 (대략적)
    print(f"\n[메모리 사용량 (파라미터 기준)]")
    lstm_memory = lstm_params * 4 / 1024  # float32 = 4 bytes
    gru_memory = gru_params * 4 / 1024
    print(f"  LSTM: {lstm_memory:.1f} KB")
    print(f"  GRU:  {gru_memory:.1f} KB")

    return lstm_params, gru_params, lstm_time, gru_time


def main():
    print("=" * 50)
    print("1. GRU 기본 구조")
    print("=" * 50)

    # 설정
    batch_size = 2
    seq_len = 10
    input_size = 5
    hidden_size = 8

    torch.manual_seed(42)
    x = torch.randn(batch_size, seq_len, input_size)

    # PyTorch GRU
    gru = nn.GRU(input_size, hidden_size, batch_first=True)

    h0 = torch.zeros(1, batch_size, hidden_size)

    print(f"입력 shape: {x.shape}")
    print(f"초기 hidden state shape: {h0.shape}")

    # 순전파
    output, h_n = gru(x, h0)

    print(f"\n출력 shape: {output.shape}")
    print(f"최종 hidden state shape: {h_n.shape}")
    print("(GRU는 Cell State가 없음 - LSTM과의 차이)")

    print("\n" + "=" * 50)
    print("2. 수동 GRU Cell 구현")
    print("=" * 50)

    manual_gru = ManualGRUCell(input_size, hidden_size)

    h = torch.zeros(batch_size, hidden_size)

    print("\n[시간 단계별 GRU 처리]")
    for t in range(min(5, seq_len)):
        x_t = x[:, t, :]
        h, (r_t, z_t, n_t) = manual_gru(x_t, h)
        print(
            f"  t={t}: h mean={h.mean().item():.4f}, "
            f"reset={r_t.mean().item():.4f}, update={z_t.mean().item():.4f}"
        )

    print("\n" + "=" * 50)
    print("3. GRU의 게이트 역할")
    print("=" * 50)

    print("""
    Reset Gate (r_t):
    - 과거 hidden state를 얼마나 무시할지 결정
    - r_t = σ(W_r·[h_{t-1}, x_t])
    - 0에 가까우면: 과거 정보 무시 (새로 시작)
    - 1에 가까우면: 과거 정보 유지

    Update Gate (z_t):
    - 과거와 현재의 혼합 비율 결정
    - z_t = σ(W_z·[h_{t-1}, x_t])
    - LSTM의 Forget + Input Gate 역할 통합

    Hidden State 업데이트:
    - h_t = (1 - z_t) ⊙ n_t + z_t ⊙ h_{t-1}
    - z_t가 크면: 과거 유지 (장기 기억)
    - z_t가 작으면: 새로운 값 반영 (단기 기억)
    """)

    print("\n" + "=" * 50)
    print("4. LSTM vs GRU 구조 비교")
    print("=" * 50)

    print("""
    |  항목       |     LSTM          |     GRU           |
    |-------------|-------------------|-------------------|
    | 게이트 수   | 3개               | 2개               |
    |             | (input, forget,   | (reset, update)   |
    |             |  output)          |                   |
    | 상태        | hidden + cell     | hidden만          |
    | 파라미터    | 4 × (in×h + h×h)  | 3 × (in×h + h×h)  |
    | 복잡도      | 높음              | 낮음              |
    """)

    # 실제 비교
    compare_lstm_gru()

    print("\n" + "=" * 50)
    print("5. GRU 분류 모델")
    print("=" * 50)

    num_classes = 3
    model = GRUClassifier(input_size, hidden_size, num_classes)
    print(f"모델 구조:\n{model}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n총 파라미터 수: {total_params}")

    logits = model(x)
    print(f"출력 shape: {logits.shape}")

    print("\n" + "=" * 50)
    print("6. 선택 가이드")
    print("=" * 50)

    print("""
    GRU 선택:
    - 데이터셋이 작을 때 (<10K 샘플)
    - 빠른 학습이 필요할 때
    - 메모리/계산 자원이 제한될 때
    - 시퀀스가 비교적 짧을 때 (< 100 토큰)

    LSTM 선택:
    - 대용량 데이터셋 (>100K 샘플)
    - 매우 긴 시퀀스
    - 복잡한 장기 의존성이 중요할 때
    - 충분한 계산 자원이 있을 때

    실무 팁:
    - 먼저 GRU로 빠르게 실험
    - 성능이 부족하면 LSTM 시도
    - 하이퍼파라미터 튜닝이 더 중요할 수 있음
    """)

    print("\n" + "=" * 50)
    print("핵심 정리")
    print("=" * 50)
    print("""
    GRU 핵심:
    - LSTM의 간소화 버전
    - 2개 게이트 (Reset, Update)
    - Cell State 없음 (Hidden State만)
    - 파라미터 25% 적음, 학습 20-40% 빠름

    LSTM과의 관계:
    - Update Gate = Forget + Input Gate 통합
    - Reset Gate로 과거 정보 선택적 리셋
    - 많은 태스크에서 비슷한 성능

    PyTorch nn.GRU:
    - 출력: output, h_n (c_n 없음!)
    - 사용법은 LSTM과 거의 동일
    """)


if __name__ == "__main__":
    main()
