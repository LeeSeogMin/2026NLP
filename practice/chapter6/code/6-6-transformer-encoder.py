"""
6장 실습: Transformer Encoder 블록 구현
- Multi-Head Attention
- Position-wise Feed-Forward Network
- Add & Norm (Residual Connection + Layer Normalization)
- 텍스트 분류 적용
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# 재현성을 위한 시드 설정
torch.manual_seed(42)


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention 구현

    MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
    head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
    """

    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Q, K, V, Output을 위한 선형 변환
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        """
        Args:
            query, key, value: (batch, seq_len, d_model)
            mask: (batch, 1, 1, seq_len) or (batch, 1, seq_len, seq_len)
        Returns:
            output: (batch, seq_len, d_model)
            attention_weights: (batch, num_heads, seq_len, seq_len)
        """
        batch_size = query.size(0)

        # 1단계: Q, K, V 선형 변환
        Q = self.W_q(query)  # (batch, seq_len, d_model)
        K = self.W_k(key)
        V = self.W_v(value)

        # 2단계: Multi-Head를 위한 reshape
        # (batch, seq_len, d_model) → (batch, num_heads, seq_len, d_k)
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # 3단계: Scaled Dot-Product Attention
        # scores: (batch, num_heads, seq_len, seq_len)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # 4단계: Value와 가중합
        context = torch.matmul(attention_weights, V)

        # 5단계: Concatenate and project
        # (batch, num_heads, seq_len, d_k) → (batch, seq_len, d_model)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_o(context)

        return output, attention_weights


class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network

    FFN(x) = ReLU(xW_1 + b_1)W_2 + b_2
    차원: d_model → d_ff → d_model (일반적으로 d_ff = 4 * d_model)
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # ReLU 활성화 (최신 모델에서는 GELU 사용)
        x = self.dropout(F.relu(self.linear1(x)))
        x = self.linear2(x)
        return x


class TransformerEncoderBlock(nn.Module):
    """
    Transformer Encoder 블록

    구조:
    Input → Multi-Head Self-Attention → Add & Norm →
          → Feed-Forward Network → Add & Norm → Output
    """

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()

        # Multi-Head Self-Attention
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)

        # Feed-Forward Network
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)

        # Layer Normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        Args:
            x: (batch, seq_len, d_model)
            mask: optional attention mask
        Returns:
            output: (batch, seq_len, d_model)
        """
        # Self-Attention with residual connection
        attn_output, attn_weights = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))

        # Feed-Forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x, attn_weights


class TransformerEncoder(nn.Module):
    """
    N개의 Encoder 블록으로 구성된 Transformer Encoder
    """

    def __init__(self, d_model, num_heads, d_ff, num_layers, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x, mask=None):
        attention_weights_all = []
        for layer in self.layers:
            x, attn_weights = layer(x, mask)
            attention_weights_all.append(attn_weights)
        return x, attention_weights_all


class TransformerClassifier(nn.Module):
    """
    Transformer Encoder 기반 텍스트 분류 모델
    """

    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers,
                 num_classes, max_len=512, dropout=0.1):
        super().__init__()

        # Token Embedding
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Positional Encoding
        self.pos_encoding = self._create_positional_encoding(max_len, d_model)

        # Transformer Encoder
        self.encoder = TransformerEncoder(d_model, num_heads, d_ff, num_layers, dropout)

        # Classification Head
        self.classifier = nn.Linear(d_model, num_classes)

        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model

    def _create_positional_encoding(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return nn.Parameter(pe.unsqueeze(0), requires_grad=False)

    def forward(self, x, mask=None):
        """
        Args:
            x: (batch, seq_len) - token indices
        Returns:
            logits: (batch, num_classes)
        """
        seq_len = x.size(1)

        # Embedding + Positional Encoding
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = x + self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x)

        # Transformer Encoder
        x, attention_weights = self.encoder(x, mask)

        # Classification: [CLS] 토큰 (첫 번째 위치) 사용
        # 또는 평균 풀링 사용
        x = x.mean(dim=1)  # (batch, d_model)

        # Classification Head
        logits = self.classifier(x)

        return logits


def count_parameters(model):
    """모델의 파라미터 수 계산"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def main():
    print("=" * 60)
    print("Transformer Encoder 구현")
    print("=" * 60)

    # 하이퍼파라미터 (BERT-base와 유사하게 설정)
    d_model = 256
    num_heads = 8
    d_ff = 1024  # 4 * d_model
    num_layers = 4
    dropout = 0.1

    batch_size = 2
    seq_len = 10

    print(f"\n[모델 설정]")
    print(f"  d_model: {d_model}")
    print(f"  num_heads: {num_heads}")
    print(f"  d_k (= d_model / num_heads): {d_model // num_heads}")
    print(f"  d_ff: {d_ff}")
    print(f"  num_layers: {num_layers}")

    # 1. Multi-Head Attention 테스트
    print("\n" + "-" * 40)
    print("1. Multi-Head Attention")
    print("-" * 40)

    mha = MultiHeadAttention(d_model, num_heads, dropout)
    x = torch.randn(batch_size, seq_len, d_model)

    output, attn_weights = mha(x, x, x)

    print(f"  입력 shape: {x.shape}")
    print(f"  출력 shape: {output.shape}")
    print(f"  Attention weights shape: {attn_weights.shape}")
    print(f"    → (batch, num_heads, seq_len, seq_len)")

    # 2. Encoder Block 테스트
    print("\n" + "-" * 40)
    print("2. Transformer Encoder Block")
    print("-" * 40)

    encoder_block = TransformerEncoderBlock(d_model, num_heads, d_ff, dropout)
    output_block, attn_weights_block = encoder_block(x)

    print(f"  입력 shape: {x.shape}")
    print(f"  출력 shape: {output_block.shape}")

    total_params, trainable_params = count_parameters(encoder_block)
    print(f"  파라미터 수: {total_params:,}")

    # 3. Full Encoder 테스트
    print("\n" + "-" * 40)
    print("3. Transformer Encoder (N layers)")
    print("-" * 40)

    encoder = TransformerEncoder(d_model, num_heads, d_ff, num_layers, dropout)
    output_encoder, attn_weights_all = encoder(x)

    print(f"  입력 shape: {x.shape}")
    print(f"  출력 shape: {output_encoder.shape}")
    print(f"  레이어 수: {num_layers}")
    print(f"  각 레이어 attention weights: {len(attn_weights_all)}")

    total_params, trainable_params = count_parameters(encoder)
    print(f"  총 파라미터 수: {total_params:,}")

    # 4. 분류 모델 테스트
    print("\n" + "-" * 40)
    print("4. Transformer Classifier")
    print("-" * 40)

    vocab_size = 10000
    num_classes = 2  # 이진 분류

    classifier = TransformerClassifier(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        num_layers=num_layers,
        num_classes=num_classes,
        max_len=512,
        dropout=dropout
    )

    # 더미 입력 (토큰 인덱스)
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    logits = classifier(input_ids)

    print(f"  Vocab size: {vocab_size}")
    print(f"  Num classes: {num_classes}")
    print(f"  입력 shape: {input_ids.shape}")
    print(f"  출력 logits shape: {logits.shape}")

    total_params, trainable_params = count_parameters(classifier)
    print(f"  총 파라미터 수: {total_params:,}")

    # 5. BERT-base와 비교
    print("\n" + "-" * 40)
    print("5. 모델 크기 비교")
    print("-" * 40)

    print("""
    | 모델           | d_model | heads | layers | d_ff  | 파라미터      |
    |----------------|---------|-------|--------|-------|---------------|
    | 현재 모델      | 256     | 8     | 4      | 1024  | {0:,}   |
    | BERT-base      | 768     | 12    | 12     | 3072  | ~110M         |
    | BERT-large     | 1024    | 16    | 24     | 4096  | ~340M         |
    | GPT-2 small    | 768     | 12    | 12     | 3072  | ~117M         |
    """.format(total_params))

    # 6. PyTorch 내장 모듈과 비교
    print("-" * 40)
    print("6. PyTorch nn.TransformerEncoder 비교")
    print("-" * 40)

    # PyTorch 내장 Transformer Encoder
    pytorch_encoder_layer = nn.TransformerEncoderLayer(
        d_model=d_model,
        nhead=num_heads,
        dim_feedforward=d_ff,
        dropout=dropout,
        batch_first=True
    )
    pytorch_encoder = nn.TransformerEncoder(pytorch_encoder_layer, num_layers=num_layers)

    output_pytorch = pytorch_encoder(x)
    print(f"  PyTorch 내장 Encoder 출력 shape: {output_pytorch.shape}")

    pytorch_params = sum(p.numel() for p in pytorch_encoder.parameters())
    print(f"  PyTorch 내장 Encoder 파라미터: {pytorch_params:,}")
    print(f"  직접 구현 Encoder 파라미터: {sum(p.numel() for p in encoder.parameters()):,}")

    print("\n" + "=" * 60)
    print("Transformer Encoder 실습 완료")
    print("=" * 60)


if __name__ == "__main__":
    main()
