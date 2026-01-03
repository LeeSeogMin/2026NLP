# 7장 리서치 결과: 중간 점검 - 핵심 개념 복습

**조사일**: 2026-01-03
**조사 주제**: 1-6장 핵심 내용 요약 및 중간고사 대비

---

## 1. 1장 핵심 요약: AI 시대의 개막

### AI/ML/DL 관계
- **AI (인공지능)**: 인간의 지능을 모방하는 시스템의 총칭
- **ML (머신러닝)**: 데이터로부터 패턴을 학습하는 AI의 하위 분야
- **DL (딥러닝)**: 심층 신경망을 사용하는 ML의 하위 분야
- 관계: AI ⊃ ML ⊃ DL (포함 관계)

### 학습 패러다임
| 패러다임 | 설명 | 예시 |
|----------|------|------|
| 지도 학습 | 입력-출력 쌍으로 학습 | 분류, 회귀 |
| 비지도 학습 | 레이블 없이 패턴 발견 | 클러스터링, 토픽 모델링 |
| 강화 학습 | 보상 최대화 행동 학습 | 게임 AI, 로봇 |

---

## 2. 2장 핵심 요약: 언어 모델

### N-gram 모델
- 이전 n-1개 단어로 다음 단어 예측
- P(wₙ | w₁, ..., wₙ₋₁) ≈ P(wₙ | wₙ₋ₖ₊₁, ..., wₙ₋₁)
- 한계: 희소성 문제, 장거리 의존성 무시

### Perplexity
- 언어 모델 평가 지표
- PP(W) = P(w₁, w₂, ..., wₙ)^(-1/n)
- 낮을수록 좋은 모델

### Word Embedding
- **Word2Vec**: 단어를 밀집 벡터로 표현
  - CBOW: 주변 단어로 중심 단어 예측
  - Skip-gram: 중심 단어로 주변 단어 예측
- 의미적 관계: king - man + woman ≈ queen

---

## 3. 3장 핵심 요약: 딥러닝 기초

### 신경망 구조
- **퍼셉트론**: y = σ(Wx + b)
- **MLP**: 여러 층의 퍼셉트론 연결
- **은닉층**: 입력과 출력 사이의 중간 레이어

### 활성화 함수
| 함수 | 수식 | 특징 |
|------|------|------|
| Sigmoid | σ(x) = 1/(1+e^(-x)) | 출력 0-1, 기울기 소실 문제 |
| ReLU | max(0, x) | 계산 효율적, dying ReLU 문제 |
| GELU | x·Φ(x) | BERT/GPT에서 사용 |
| Softmax | eᶻⁱ/Σeᶻʲ | 다중 클래스 분류 출력 |

### 손실 함수
- **MSE (회귀)**: L = (1/n)Σ(yᵢ - ŷᵢ)²
- **Cross-Entropy (분류)**: L = -Σyᵢlog(ŷᵢ)

### 최적화
- **경사 하강법**: θ = θ - α∇L(θ)
- **역전파**: Chain Rule로 기울기 계산
- **Adam**: 모멘텀 + RMSprop 결합

---

## 4. 4장 핵심 요약: PyTorch 기반 개발

### PyTorch 핵심 요소
```python
# 모델 정의
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.layer(x)

# 학습 루프
for epoch in range(epochs):
    output = model(x)           # Forward
    loss = criterion(output, y) # Loss 계산
    loss.backward()             # Backward
    optimizer.step()            # 파라미터 업데이트
    optimizer.zero_grad()       # 기울기 초기화
```

### 데이터 처리
- `Dataset`: 데이터 로딩 인터페이스
- `DataLoader`: 배치 처리, 셔플링

### 평가 지표
- **Accuracy**: 정확도 = 맞춘 수 / 전체 수
- **Precision**: 정밀도 = TP / (TP + FP)
- **Recall**: 재현율 = TP / (TP + FN)
- **F1-Score**: 2 × (P × R) / (P + R)

---

## 5. 5장 핵심 요약: RNN/LSTM/GRU

### RNN (Recurrent Neural Network)
- 순환 연결로 시퀀스 처리
- Hidden State: hₜ = tanh(Wₓₕxₜ + Wₕₕhₜ₋₁ + b)
- 문제점: 장기 의존성, 기울기 소실/폭주

### LSTM (Long Short-Term Memory)
- Cell State (cₜ): 장기 기억
- Hidden State (hₜ): 단기 기억/출력
- **3개의 게이트**:
  - Forget Gate: fₜ = σ(Wf·[hₜ₋₁, xₜ] + bf) → 과거 정보 삭제
  - Input Gate: iₜ = σ(Wi·[hₜ₋₁, xₜ] + bi) → 새 정보 저장
  - Output Gate: oₜ = σ(Wo·[hₜ₋₁, xₜ] + bo) → 출력 결정

### GRU (Gated Recurrent Unit)
- LSTM의 간소화 버전
- **2개의 게이트**:
  - Reset Gate: rₜ = σ(Wr·[hₜ₋₁, xₜ])
  - Update Gate: zₜ = σ(Wz·[hₜ₋₁, xₜ])

### RNN vs LSTM vs GRU 비교
| 특성 | RNN | LSTM | GRU |
|------|-----|------|-----|
| 게이트 수 | 0 | 3 | 2 |
| 파라미터 | 적음 | 많음 | 중간 |
| 장기 의존성 | 약함 | 강함 | 강함 |
| 학습 속도 | 빠름 | 느림 | 중간 |

---

## 6. 6장 핵심 요약: Transformer

### RNN의 한계 → Transformer
- 순차 처리 → 병렬 처리 불가
- O(n) 경로 길이 → 장거리 의존성 문제
- Transformer: Attention만으로 시퀀스 처리

### Attention 메커니즘
- **Query (Q)**: 현재 찾고 있는 정보
- **Key (K)**: 각 위치의 색인
- **Value (V)**: 실제 정보 내용

### Scaled Dot-Product Attention
```
Attention(Q, K, V) = softmax(QKᵀ/√dₖ)V
```
1. QKᵀ: Query-Key 유사도 계산
2. /√dₖ: 스케일링 (기울기 안정화)
3. softmax: 확률 분포로 변환 (Attention Weight)
4. × V: 가중합 계산

### Self-Attention
- 같은 시퀀스 내에서 Q, K, V 모두 생성
- 모든 위치 간 직접 연결 → O(1) 경로 길이

### Multi-Head Attention
- 여러 관점에서 동시에 Attention
- head 수 h, 차원 dₖ = d_model / h
- Concat 후 Linear 투영

### Positional Encoding
- Transformer는 순서 정보가 없음 → PE 추가 필요
- **Sinusoidal PE**:
  - PE(pos, 2i) = sin(pos/10000^(2i/d))
  - PE(pos, 2i+1) = cos(pos/10000^(2i/d))
- **Learned PE**: 학습 가능한 임베딩

### Transformer 구조
- **Encoder**: Self-Attention + FFN (반복)
- **Decoder**: Masked Self-Attention + Cross-Attention + FFN
- **Residual Connection + LayerNorm**: 학습 안정화

---

## 7. 핵심 수식 정리

| 개념 | 수식 |
|------|------|
| Perplexity | PP = P(w₁...wₙ)^(-1/n) |
| Cross-Entropy | L = -Σyᵢlog(ŷᵢ) |
| Gradient Descent | θ = θ - α∇L(θ) |
| RNN Hidden State | hₜ = tanh(Wₓₕxₜ + Wₕₕhₜ₋₁) |
| LSTM Forget Gate | fₜ = σ(Wf·[hₜ₋₁, xₜ]) |
| Attention | softmax(QKᵀ/√dₖ)V |

---

## 8. 중간고사 출제 유형

### 객관식
- 개념 이해 확인 (AI/ML/DL 관계, 활성화 함수 특성)
- 수식 해석 (Attention, LSTM 게이트)

### 단답형
- 용어 정의 (Hidden State, Self-Attention)
- 수식 작성 (Scaled Dot-Product Attention)

### 서술형
- RNN vs Transformer 비교
- LSTM 게이트 역할 설명
- Self-Attention 계산 과정

### 코딩 문제
- PyTorch 기본 구현
- Attention Score 계산

---

## 참고자료

- 1-6장 교재 내용
- Vaswani, A., et al. (2017). Attention is All You Need.
- Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory.
