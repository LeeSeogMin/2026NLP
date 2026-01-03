"""
12-3-lora-basics.py
LoRA(Low-Rank Adaptation)의 기본 개념을 이해하기 위한 코드

이 코드는 LoRA의 핵심 아이디어인 저랭크 행렬 분해를 직접 구현하여
수학적 원리를 시각적으로 이해할 수 있도록 한다.
"""

import torch
import torch.nn as nn
import numpy as np

print("=" * 60)
print("LoRA (Low-Rank Adaptation) 기본 개념")
print("=" * 60)

# ============================================================
# 1. 저랭크 행렬 분해 개념
# ============================================================
print("\n[1] 저랭크 행렬 분해 (Low-Rank Matrix Decomposition)")
print("-" * 50)

# 원본 가중치 행렬 (d x k)
d, k = 768, 768  # BERT-base의 hidden size
original_weight = torch.randn(d, k)

# Full fine-tuning: 모든 파라미터 학습
full_params = d * k
print(f"원본 가중치 크기: {d} x {k} = {full_params:,} 파라미터")

# LoRA: 저랭크 분해 (ΔW = B × A)
rank = 8  # 저랭크 차원
B = torch.randn(d, rank)  # d x r
A = torch.randn(rank, k)  # r x k

lora_params = d * rank + rank * k
print(f"LoRA 파라미터 (r={rank}): {d}×{rank} + {rank}×{k} = {lora_params:,} 파라미터")
print(f"파라미터 감소율: {(1 - lora_params/full_params)*100:.2f}%")

# ============================================================
# 2. Rank 값에 따른 파라미터 수 비교
# ============================================================
print("\n[2] Rank 값에 따른 파라미터 수 비교")
print("-" * 50)

print(f"{'Rank':>6} | {'LoRA 파라미터':>15} | {'감소율':>10} | {'원본 대비':>10}")
print("-" * 50)

for r in [1, 4, 8, 16, 32, 64]:
    lora_p = d * r + r * k
    reduction = (1 - lora_p / full_params) * 100
    ratio = lora_p / full_params * 100
    print(f"{r:>6} | {lora_p:>15,} | {reduction:>9.2f}% | {ratio:>9.2f}%")

# ============================================================
# 3. LoRA 레이어 구현
# ============================================================
print("\n[3] LoRA 레이어 직접 구현")
print("-" * 50)


class LoRALayer(nn.Module):
    """
    LoRA 레이어 구현

    W' = W + (α/r) × B × A

    - W: 원본 가중치 (동결)
    - B: d × r 행렬 (학습)
    - A: r × k 행렬 (학습)
    - α: 스케일링 팩터
    - r: 저랭크 차원
    """

    def __init__(self, original_layer, rank=8, alpha=16):
        super().__init__()
        self.original_layer = original_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # 원본 레이어 동결
        for param in self.original_layer.parameters():
            param.requires_grad = False

        # LoRA 행렬 초기화
        in_features = original_layer.in_features
        out_features = original_layer.out_features

        # A는 Kaiming 초기화, B는 0으로 초기화 (학습 시작 시 ΔW = 0)
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

    def forward(self, x):
        # 원본 출력 + LoRA 출력
        original_output = self.original_layer(x)
        lora_output = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return original_output + lora_output

    def get_merged_weight(self):
        """LoRA 가중치를 원본에 병합"""
        delta_W = (self.lora_B @ self.lora_A) * self.scaling
        return self.original_layer.weight + delta_W


# 테스트
print("LoRA 레이어 생성:")
original_linear = nn.Linear(768, 768)
lora_layer = LoRALayer(original_linear, rank=8, alpha=16)

# 파라미터 수 비교
original_params = sum(p.numel() for p in original_linear.parameters())
lora_trainable = sum(p.numel() for p in lora_layer.parameters() if p.requires_grad)
print(f"  - 원본 레이어 파라미터: {original_params:,}")
print(f"  - LoRA 학습 파라미터: {lora_trainable:,}")
print(f"  - 학습 비율: {lora_trainable/original_params*100:.2f}%")

# Forward 테스트
x = torch.randn(2, 10, 768)  # (batch, seq_len, hidden)
output = lora_layer(x)
print(f"  - 입력 shape: {x.shape}")
print(f"  - 출력 shape: {output.shape}")

# ============================================================
# 4. Alpha/Rank 스케일링 효과
# ============================================================
print("\n[4] Alpha/Rank 스케일링 효과")
print("-" * 50)

print("스케일링 팩터 = α / r")
print(f"{'Alpha':>6} | {'Rank':>6} | {'Scaling':>10}")
print("-" * 30)

for alpha in [8, 16, 32]:
    for r in [4, 8, 16]:
        scaling = alpha / r
        print(f"{alpha:>6} | {r:>6} | {scaling:>10.2f}")

print("\n권장 설정: α = 2 × r (scaling factor = 2)")

# ============================================================
# 5. 모델에 LoRA 적용 시뮬레이션
# ============================================================
print("\n[5] BERT-base에 LoRA 적용 시뮬레이션")
print("-" * 50)

# BERT-base 구조 시뮬레이션
class SimpleBERTLayer(nn.Module):
    def __init__(self, hidden_size=768):
        super().__init__()
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, hidden_size)
        self.ffn_up = nn.Linear(hidden_size, hidden_size * 4)
        self.ffn_down = nn.Linear(hidden_size * 4, hidden_size)


class SimpleBERT(nn.Module):
    def __init__(self, num_layers=12, hidden_size=768):
        super().__init__()
        self.layers = nn.ModuleList([
            SimpleBERTLayer(hidden_size) for _ in range(num_layers)
        ])


model = SimpleBERT(num_layers=12, hidden_size=768)

# 전체 파라미터 계산
total_params = sum(p.numel() for p in model.parameters())
print(f"BERT-base 시뮬레이션 모델 파라미터: {total_params:,}")

# LoRA 적용 대상별 파라미터 계산
hidden = 768
rank = 8

# Query, Value만 적용
qv_lora_per_layer = 2 * (hidden * rank + rank * hidden)
qv_lora_total = qv_lora_per_layer * 12

# Query, Key, Value, Output 모두 적용
qkvo_lora_per_layer = 4 * (hidden * rank + rank * hidden)
qkvo_lora_total = qkvo_lora_per_layer * 12

# 모든 Linear 레이어 적용 (Q,K,V,O + FFN)
all_lora_per_layer = qkvo_lora_per_layer + (hidden * rank + rank * hidden * 4) + (hidden * 4 * rank + rank * hidden)
all_lora_total = all_lora_per_layer * 12

print(f"\nLoRA 적용 범위별 파라미터 수 (r={rank}):")
print(f"  - Query, Value만: {qv_lora_total:,} ({qv_lora_total/total_params*100:.2f}%)")
print(f"  - Q, K, V, Output: {qkvo_lora_total:,} ({qkvo_lora_total/total_params*100:.2f}%)")
print(f"  - 모든 Linear 레이어: {all_lora_total:,} ({all_lora_total/total_params*100:.4f}%)")

# ============================================================
# 6. LoRA의 장점 요약
# ============================================================
print("\n[6] LoRA의 주요 장점")
print("-" * 50)
print("""
1. 메모리 효율성
   - 학습 파라미터 99%+ 감소
   - 그래디언트 저장 공간 대폭 감소
   - 옵티마이저 상태 최소화

2. 저장 효율성
   - 태스크별 어댑터만 저장 (수 MB)
   - 베이스 모델 공유 가능

3. 추론 효율성
   - 병합 후 추가 지연 없음
   - W' = W + BA로 단일 행렬 연산

4. 카타스트로픽 포겟팅 방지
   - 원본 가중치 보존
   - 사전 학습 지식 유지
""")

print("=" * 60)
print("LoRA 기본 개념 학습 완료")
print("=" * 60)
