"""
4장 실습 2: Transformer Encoder/Decoder 밑바닥 구현
- Multi-Head Attention (3장 복습 + 확장)
- Transformer Encoder Block (MHA + FFN + Add&Norm)
- Transformer Decoder Block (Masked MHA + Cross-Attention + FFN)
- Transformer Encoder 기반 텍스트 분류 모델
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# 재현성을 위한 시드 설정
torch.manual_seed(42)


# ============================================================
# 1. Multi-Head Attention (3장 코드 기반 개선)
# ============================================================
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

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # Linear projection
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)

        # Reshape: (batch, seq, d_model) → (batch, heads, seq, d_k)
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Scaled Dot-Product Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        context = torch.matmul(attention_weights, V)

        # Concatenate and project
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_o(context)

        return output, attention_weights


# ============================================================
# 2. Position-wise Feed-Forward Network
# ============================================================
class PositionwiseFeedForward(nn.Module):
    """
    FFN(x) = GELU(xW₁ + b₁)W₂ + b₂
    차원: d_model → d_ff → d_model
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # GELU 활성화 (Transformer 최신 표준)
        x = self.dropout(F.gelu(self.linear1(x)))
        x = self.linear2(x)
        return x


# ============================================================
# 3. Transformer Encoder Block (Post-LN 방식)
# ============================================================
class TransformerEncoderBlock(nn.Module):
    """
    Transformer Encoder Block (Post-LN)

    Input → Multi-Head Self-Attention → Add & Norm →
          → Feed-Forward Network → Add & Norm → Output
    """

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-Attention + Residual + Norm
        attn_output, attn_weights = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))

        # Feed-Forward + Residual + Norm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x, attn_weights


# ============================================================
# 4. Transformer Decoder Block
# ============================================================
class TransformerDecoderBlock(nn.Module):
    """
    Transformer Decoder Block

    Input → Masked Self-Attention → Add & Norm →
          → Cross-Attention (Encoder 참조) → Add & Norm →
          → Feed-Forward Network → Add & Norm → Output
    """

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.masked_self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        # 1. Masked Self-Attention (미래 토큰 마스킹)
        attn_output, self_attn_weights = self.masked_self_attention(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))

        # 2. Cross-Attention (Encoder 출력 참조)
        # Query: Decoder, Key/Value: Encoder
        cross_output, cross_attn_weights = self.cross_attention(
            x, encoder_output, encoder_output, src_mask
        )
        x = self.norm2(x + self.dropout(cross_output))

        # 3. Feed-Forward
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))

        return x, self_attn_weights, cross_attn_weights


# ============================================================
# 5. Sinusoidal Positional Encoding
# ============================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1), :]


# ============================================================
# 6. Transformer Encoder (N layers)
# ============================================================
class TransformerEncoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList(
            [TransformerEncoderBlock(d_model, num_heads, d_ff, dropout)
             for _ in range(num_layers)]
        )

    def forward(self, x, mask=None):
        attn_weights_all = []
        for layer in self.layers:
            x, attn_weights = layer(x, mask)
            attn_weights_all.append(attn_weights)
        return x, attn_weights_all


# ============================================================
# 7. Transformer Encoder 기반 텍스트 분류 모델
# ============================================================
class TransformerClassifier(nn.Module):
    """
    Transformer Encoder 기반 텍스트 분류 모델

    Token Embedding + Positional Encoding → Encoder Stack → Mean Pooling → Classifier
    """

    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers,
                 num_classes, max_len=512, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.encoder = TransformerEncoder(d_model, num_heads, d_ff, num_layers, dropout)
        self.classifier = nn.Linear(d_model, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model

    def forward(self, x, mask=None):
        # Embedding + Positional Encoding
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)

        # Encoder
        x, attn_weights = self.encoder(x, mask)

        # Mean Pooling → Classification
        x = x.mean(dim=1)
        logits = self.classifier(x)
        return logits


def count_parameters(model):
    """모델의 총 파라미터 수와 학습 가능한 파라미터 수를 반환"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def create_causal_mask(seq_len):
    """Causal (Look-ahead) Mask 생성"""
    mask = torch.tril(torch.ones(seq_len, seq_len))
    return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)


def main():
    print("=" * 60)
    print("4장 실습 2: Transformer Encoder/Decoder 구현")
    print("=" * 60)

    # 하이퍼파라미터
    d_model = 256
    num_heads = 8
    d_ff = 1024  # 4 × d_model
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

    # --------------------------------------------------------
    # 1. Multi-Head Attention 테스트
    # --------------------------------------------------------
    print("\n" + "-" * 40)
    print("1. Multi-Head Attention")
    print("-" * 40)

    mha = MultiHeadAttention(d_model, num_heads, dropout)
    x = torch.randn(batch_size, seq_len, d_model)
    output, attn_weights = mha(x, x, x)

    print(f"  입력 shape: {x.shape}")
    print(f"  출력 shape: {output.shape}")
    print(f"  Attention weights shape: {attn_weights.shape}")
    print(f"    → (batch={batch_size}, heads={num_heads}, "
          f"seq={seq_len}, seq={seq_len})")

    # --------------------------------------------------------
    # 2. Encoder Block 테스트
    # --------------------------------------------------------
    print("\n" + "-" * 40)
    print("2. Transformer Encoder Block")
    print("-" * 40)

    encoder_block = TransformerEncoderBlock(d_model, num_heads, d_ff, dropout)
    output_block, _ = encoder_block(x)

    total, trainable = count_parameters(encoder_block)
    print(f"  입력 shape: {x.shape}")
    print(f"  출력 shape: {output_block.shape}")
    print(f"  파라미터 수: {total:,}")

    # --------------------------------------------------------
    # 3. Full Encoder (N layers)
    # --------------------------------------------------------
    print("\n" + "-" * 40)
    print("3. Transformer Encoder ({} layers)".format(num_layers))
    print("-" * 40)

    encoder = TransformerEncoder(d_model, num_heads, d_ff, num_layers, dropout)
    output_enc, attn_all = encoder(x)

    total, _ = count_parameters(encoder)
    print(f"  입력 shape: {x.shape}")
    print(f"  출력 shape: {output_enc.shape}")
    print(f"  총 파라미터 수: {total:,}")

    # --------------------------------------------------------
    # 4. Decoder Block + Causal Mask 테스트
    # --------------------------------------------------------
    print("\n" + "-" * 40)
    print("4. Transformer Decoder Block + Causal Mask")
    print("-" * 40)

    decoder_block = TransformerDecoderBlock(d_model, num_heads, d_ff, dropout)

    # Causal Mask 생성
    causal_mask = create_causal_mask(seq_len)
    print(f"  Causal Mask shape: {causal_mask.shape}")
    print(f"  Causal Mask (5×5 부분):")

    mask_display = causal_mask[0, 0, :5, :5]
    for i in range(5):
        row = [f"{mask_display[i, j]:.0f}" for j in range(5)]
        print(f"    [{', '.join(row)}]")

    # Decoder 실행 (Encoder 출력을 참조)
    decoder_input = torch.randn(batch_size, seq_len, d_model)
    decoder_output, self_attn, cross_attn = decoder_block(
        decoder_input, output_enc, tgt_mask=causal_mask
    )

    total_dec, _ = count_parameters(decoder_block)
    print(f"\n  Decoder 입력 shape: {decoder_input.shape}")
    print(f"  Encoder 출력 shape (참조): {output_enc.shape}")
    print(f"  Decoder 출력 shape: {decoder_output.shape}")
    print(f"  Self-Attention weights: {self_attn.shape}")
    print(f"  Cross-Attention weights: {cross_attn.shape}")
    print(f"  Decoder Block 파라미터 수: {total_dec:,}")

    # Causal Mask 적용 확인
    print(f"\n  [Causal Mask 적용 확인 — Self-Attention Weights]")
    first_row = self_attn[0, 0, 0].detach()
    second_row = self_attn[0, 0, 1].detach()
    print(f"    첫 번째 토큰이 참조하는 가중치: "
          f"[{first_row[0]:.3f}, {first_row[1]:.3f}, ..., {first_row[-1]:.3f}]")
    print(f"    → 첫 토큰은 자기 자신만 참조 (나머지 ≈ 0)")
    print(f"    두 번째 토큰이 참조하는 가중치: "
          f"[{second_row[0]:.3f}, {second_row[1]:.3f}, {second_row[2]:.3f}, ...]")
    print(f"    → 두 번째 토큰은 위치 0, 1만 참조 (위치 2부터 ≈ 0)")

    # --------------------------------------------------------
    # 5. 분류 모델 테스트
    # --------------------------------------------------------
    print("\n" + "-" * 40)
    print("5. Transformer Classifier")
    print("-" * 40)

    vocab_size = 10000
    num_classes = 2

    classifier = TransformerClassifier(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        num_layers=num_layers,
        num_classes=num_classes,
        max_len=512,
        dropout=dropout,
    )

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    logits = classifier(input_ids)

    total_cls, _ = count_parameters(classifier)
    print(f"  Vocab size: {vocab_size}")
    print(f"  Num classes: {num_classes}")
    print(f"  입력 shape: {input_ids.shape}")
    print(f"  출력 logits shape: {logits.shape}")
    print(f"  총 파라미터 수: {total_cls:,}")

    # --------------------------------------------------------
    # 6. 모델 크기 비교
    # --------------------------------------------------------
    print("\n" + "-" * 40)
    print("6. 모델 크기 비교")
    print("-" * 40)

    print(f"""
  | 모델            | d_model | heads | layers | d_ff  | 파라미터        |
  |-----------------|---------|-------|--------|-------|-----------------|
  | 현재 모델       | {d_model:>5}   | {num_heads:>3}   | {num_layers:>4}   | {d_ff:>4}  | {total_cls:>12,}  |
  | BERT-base       |   768   |  12   |   12   | 3072  |        ~110M    |
  | BERT-large      |  1024   |  16   |   24   | 4096  |        ~340M    |
  | GPT-2 small     |   768   |  12   |   12   | 3072  |        ~117M    |
  | GPT-3 175B      | 12288   |  96   |   96   | 49152 |       ~175B     |""")

    # --------------------------------------------------------
    # 7. PyTorch 내장 모듈과 비교
    # --------------------------------------------------------
    print("\n" + "-" * 40)
    print("7. PyTorch nn.TransformerEncoder 비교")
    print("-" * 40)

    pytorch_layer = nn.TransformerEncoderLayer(
        d_model=d_model, nhead=num_heads, dim_feedforward=d_ff,
        dropout=dropout, batch_first=True
    )
    pytorch_encoder = nn.TransformerEncoder(pytorch_layer, num_layers=num_layers)

    output_pytorch = pytorch_encoder(x)
    pytorch_params = sum(p.numel() for p in pytorch_encoder.parameters())
    custom_params = sum(p.numel() for p in encoder.parameters())

    print(f"  PyTorch 내장 Encoder 출력 shape: {output_pytorch.shape}")
    print(f"  PyTorch 내장 파라미터: {pytorch_params:,}")
    print(f"  직접 구현 파라미터:    {custom_params:,}")

    print("\n" + "=" * 60)
    print("실습 2 완료")
    print("=" * 60)


if __name__ == "__main__":
    main()
