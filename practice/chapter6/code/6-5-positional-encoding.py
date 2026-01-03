"""
6장 실습: Positional Encoding 구현
- Sinusoidal Positional Encoding
- Learned Positional Encoding
- 시각화
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import math

# 재현성을 위한 시드 설정
torch.manual_seed(42)


class SinusoidalPositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding
    "Attention is All You Need" 논문의 원본 구현

    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """

    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Positional Encoding 행렬 생성
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # div_term = 10000^(2i/d_model)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        # 짝수 인덱스: sin, 홀수 인덱스: cos
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # (1, max_len, d_model)로 변환하여 buffer로 등록
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            x + positional_encoding
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


class LearnedPositionalEncoding(nn.Module):
    """
    Learned Positional Encoding
    GPT-2 등에서 사용하는 학습 가능한 위치 임베딩
    """

    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = nn.Embedding(max_len, d_model)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            x + positional_encoding
        """
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        x = x + self.pe(positions)
        return self.dropout(x)


def visualize_positional_encoding(pe_matrix, title="Positional Encoding"):
    """Positional Encoding 시각화"""
    import os
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'output')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'positional_encoding.png')

    plt.figure(figsize=(12, 5))

    # 히트맵
    plt.subplot(1, 2, 1)
    plt.imshow(pe_matrix[:50, :64].numpy(), cmap='RdBu', aspect='auto')
    plt.colorbar()
    plt.xlabel('Dimension')
    plt.ylabel('Position')
    plt.title(f'{title} (First 50 pos, 64 dims)')

    # 특정 차원의 패턴
    plt.subplot(1, 2, 2)
    positions = np.arange(100)
    for dim in [0, 1, 10, 11, 50, 51]:
        label = f"dim {dim} ({'sin' if dim % 2 == 0 else 'cos'})"
        plt.plot(pe_matrix[:100, dim].numpy(), label=label, alpha=0.7)
    plt.xlabel('Position')
    plt.ylabel('Encoding Value')
    plt.title('PE Pattern by Dimension')
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Positional Encoding visualization saved: {output_path}")


def visualize_pe_comparison():
    """Sin/Cos 조합의 상대적 위치 표현 능력 시각화"""
    import os
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'output')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'pe_similarity.png')

    d_model = 64
    max_len = 100

    # Sinusoidal PE 생성
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    # 위치 간 유사도 (코사인 유사도)
    pe_norm = pe / pe.norm(dim=1, keepdim=True)
    similarity = torch.mm(pe_norm, pe_norm.t())

    plt.figure(figsize=(8, 6))
    plt.imshow(similarity.numpy(), cmap='viridis')
    plt.colorbar(label='Cosine Similarity')
    plt.xlabel('Position j')
    plt.ylabel('Position i')
    plt.title('Positional Encoding Similarity Between Positions')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"PE similarity visualization saved: {output_path}")


def main():
    print("=" * 60)
    print("Positional Encoding 구현 및 분석")
    print("=" * 60)

    # 파라미터
    d_model = 128
    max_len = 100
    seq_len = 20
    batch_size = 2

    print(f"\n[파라미터]")
    print(f"  모델 차원 (d_model): {d_model}")
    print(f"  최대 길이: {max_len}")
    print(f"  시퀀스 길이: {seq_len}")

    # 1. Sinusoidal Positional Encoding
    print("\n" + "-" * 40)
    print("1. Sinusoidal Positional Encoding")
    print("-" * 40)

    sinusoidal_pe = SinusoidalPositionalEncoding(d_model, max_len, dropout=0.0)

    # 파라미터 수 확인
    num_params = sum(p.numel() for p in sinusoidal_pe.parameters())
    print(f"  학습 가능한 파라미터 수: {num_params} (파라미터 없음)")

    # 예시 입력
    x = torch.zeros(batch_size, seq_len, d_model)
    output = sinusoidal_pe(x)

    print(f"  입력 shape: {x.shape}")
    print(f"  출력 shape: {output.shape}")

    # PE 값 분석
    pe_matrix = sinusoidal_pe.pe.squeeze(0)
    print(f"\n  [첫 3개 위치의 PE 값 (처음 8차원)]")
    for pos in range(3):
        values = pe_matrix[pos, :8].tolist()
        print(f"    위치 {pos}: {[f'{v:.3f}' for v in values]}")

    # 시각화
    visualize_positional_encoding(pe_matrix, "Sinusoidal Positional Encoding")

    # 2. Learned Positional Encoding
    print("\n" + "-" * 40)
    print("2. Learned Positional Encoding")
    print("-" * 40)

    learned_pe = LearnedPositionalEncoding(d_model, max_len, dropout=0.0)

    num_params = sum(p.numel() for p in learned_pe.parameters())
    print(f"  학습 가능한 파라미터 수: {num_params:,}")
    print(f"  = max_len × d_model = {max_len} × {d_model}")

    output_learned = learned_pe(x)
    print(f"  출력 shape: {output_learned.shape}")

    # 3. 비교
    print("\n" + "-" * 40)
    print("3. Sinusoidal vs Learned 비교")
    print("-" * 40)

    print("""
    | 특성              | Sinusoidal          | Learned             |
    |-------------------|---------------------|---------------------|
    | 파라미터          | 없음                | max_len × d_model   |
    | 긴 시퀀스 일반화  | 가능                | 불가능 (max_len 제한)|
    | 표현력            | 고정                | 유연 (학습됨)       |
    | 사용 예           | 원본 Transformer    | GPT-2, BERT        |
    """)

    # 4. 위치 간 유사도 시각화
    print("-" * 40)
    print("4. 위치 간 유사도 분석")
    print("-" * 40)

    visualize_pe_comparison()

    # 특정 위치 간 거리 분석
    pe_norm = pe_matrix / pe_matrix.norm(dim=1, keepdim=True)

    print(f"\n  [위치 간 코사인 유사도]")
    pairs = [(0, 1), (0, 5), (0, 10), (0, 50), (10, 11), (10, 20)]
    for i, j in pairs:
        sim = torch.dot(pe_norm[i], pe_norm[j]).item()
        print(f"    위치 {i} ↔ 위치 {j}: {sim:.4f}")

    print("\n  → 가까운 위치일수록 유사도가 높음")
    print("  → 상대적 위치 정보를 인코딩")

    # 5. 긴 시퀀스 테스트
    print("\n" + "-" * 40)
    print("5. 긴 시퀀스 테스트")
    print("-" * 40)

    # Sinusoidal: 학습 시 보지 못한 길이도 처리 가능
    long_seq = torch.zeros(1, 200, d_model)  # max_len=100보다 긴 시퀀스

    # max_len 늘려서 새로 생성
    sinusoidal_pe_long = SinusoidalPositionalEncoding(d_model, max_len=500, dropout=0.0)
    output_long = sinusoidal_pe_long(long_seq)
    print(f"  Sinusoidal PE - 200 길이 시퀀스 처리: {output_long.shape}")

    # Learned: max_len 이상은 처리 불가
    print(f"  Learned PE - max_len={max_len} 초과 시 에러 발생")

    print("\n" + "=" * 60)
    print("Positional Encoding 실습 완료")
    print("=" * 60)


if __name__ == "__main__":
    main()
