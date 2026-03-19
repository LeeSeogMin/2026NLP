## 1주차 B회차: 환경 구축 + Tensor 실습

> **미션**: 자동 환경 설정을 완료하고 PyTorch Tensor를 능숙하게 조작하며, Autograd를 활용한 간단한 선형 회귀 모델을 GPU/CPU에서 학습할 수 있다

### 수업 타임라인

| 시간 | 내용 | Copilot 사용 |
|------|------|-------------|
| 00:00~00:05 | 조 편성 + 역할 배분 (조원 A/B) | 사용 안 함 |
| 00:05~00:10 | A회차 핵심 리캡 + 과제 스펙 재확인 | 사용 안 함 |
| 00:10~00:55 | 2인1조 Copilot 구현 (체크포인트 3회) | 적극 사용 |
| 00:55~01:00 | Google Classroom 제출 (조별 1부) | 사용 안 함 |
| 01:00~01:20 | 결과 토론 (환경 설정 경험 공유·성능 차이 분석) | 사용 안 함 |
| 01:20~01:28 | 핵심 정리 | 사용 안 함 |
| 01:28~01:30 | 다음 주 예고 | 사용 안 함 |

---

### A회차 핵심 리캡

**AI, 머신러닝, 딥러닝의 관계**:
- AI는 가장 큰 우산. 그 안에 머신러닝, 그 안에 딥러닝이 포함된다
- 머신러닝은 데이터로부터 패턴을 자동으로 학습
- 딥러닝은 여러 층의 신경망으로 복잡한 패턴을 학습

**PyTorch와 Tensor**:
- Tensor는 딥러닝의 기본 자료구조 (NumPy 배열의 GPU 지원 버전)
- 0차원(스칼라), 1차원(벡터), 2차원(행렬), 3차원 이상(텐서)
- shape, dtype, device는 Tensor 다룰 때 가장 중요한 세 속성

**Autograd: 자동 미분**:
- `requires_grad=True`로 설정하면 기울기 자동 계산
- `.backward()`로 역전파 수행하면 모든 파라미터의 기울기 계산
- 이것이 신경망 학습의 핵심 메커니즘

**GPU의 가치**:
- GPU는 병렬 처리로 수십~수백 배 빠른 연산 가능
- `.to(device)`로 데이터와 모델을 GPU로 이동
- 모델과 데이터가 같은 위치(device)에 있어야만 연산 가능

---

### 과제 스펙 + 체크포인트

**과제**: 자동 환경 설정 스크립트 실행 + PyTorch Tensor 조작 + 간단한 선형 회귀 구현

**제출 형태**: 조별 1부, Google Classroom 업로드

**파일 구성**:
- 구현 코드 파일 (`*.py`)
- 환경 설정 실행 결과 캡처 (`.txt` 또는 `.png`)
- GPU/CPU 성능 비교 결과 (스크린샷 또는 수치)
- 간단한 분석 리포트 (1페이지)

**검증 기준**:
- ✓ 자동 환경 설정 스크립트 성공 실행 및 검증
- ✓ Tensor 기본 연산 (생성, 형태 확인, GPU 이동) 동작 확인
- ✓ Autograd를 활용한 기울기 계산 검증
- ✓ CPU vs GPU 성능 비교 측정
- ✓ 선형 회귀 모델 학습 및 손실 감소 확인

---

### 2인1조 실습

> **Copilot 활용**: Tensor 생성과 기본 연산을 먼저 손으로 작성해본 뒤, Copilot에게 "GPU가 있으면 GPU를 사용하고 없으면 CPU를 사용하는 device 설정 코드를 만들어줄래?" 또는 "Autograd를 사용해서 선형 회귀 모델을 학습하는 코드를 작성해줄래?" 같이 단계적으로 요청한다. Copilot의 제안을 검토하고, 각 줄이 무엇을 하는지 반드시 이해하며 수정한다.

**역할 분담**:
- **조원 A (드라이버)**: 코드 작성 및 실행, 결과 확인
- **조원 B (네비게이터)**: 로직 검토, Copilot 프롬프트 설계, 오류 해석
- **체크포인트마다 역할 교대**: 드라이버와 네비게이터를 번갈아가며 진행하여 두 명 모두 전체 과정을 이해한다

---

#### 체크포인트 1: 자동 환경 설정 + 기본 Tensor 조작 (15분)

**목표**: 한 번의 명령으로 PyTorch 개발 환경을 자동으로 구축하고, GPU/CPU를 올바르게 감지하며, 기본 Tensor 연산을 검증한다

**핵심 단계**:

① **환경 설정 스크립트 실행** — 자동으로 모든 의존성 설치

```bash
cd /path/to/project
python scripts/setup_env.py
```

예상 출력:
```
========== PyTorch 개발 환경 설정 시작 ==========

[1단계] Python 버전 확인
  현재 Python: 3.10.11 ✓

[2단계] GPU 감지
  GPU 사용 가능: True
  CUDA 버전: 12.1
  GPU 이름: NVIDIA GeForce RTX 4080

[3단계] 가상환경 생성
  venv 폴더 생성됨
  활성화: source venv/bin/activate (또는 venv\Scripts\activate)

[4단계] PyTorch 설치
  PyTorch 2.1.2 설치 완료
  CUDA 12.1 지원 버전 설치됨

[5단계] 추가 패키지 설치
  numpy, pandas, matplotlib, transformers 설치 완료

[6단계] 동작 검증
  import torch ✓
  torch.cuda.is_available() ✓
  torch.cuda.get_device_name(0): NVIDIA GeForce RTX 4080 ✓

[7단계] 성능 벤치마크
  CPU 행렬 곱 (1000×1000): 0.2145초
  GPU 행렬 곱 (1000×1000): 0.0043초
  속도 향상: 49.9배 ✓

========== 설정 완료 ==========
```

② **Python 코드로 environment 확인** — 설정이 제대로 됐는지 재검증

```python
import torch
import sys

print("=" * 50)
print("PyTorch 환경 최종 검증")
print("=" * 50)

# Python 버전
print(f"\n[1] Python 버전")
print(f"    {sys.version.split()[0]}")

# PyTorch 버전
print(f"\n[2] PyTorch 버전")
print(f"    {torch.__version__}")

# GPU 정보
print(f"\n[3] GPU 정보")
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"    GPU 사용 가능: True")
    print(f"    CUDA 버전: {torch.version.cuda}")
    print(f"    GPU 개수: {torch.cuda.device_count()}")
    print(f"    GPU 이름: {torch.cuda.get_device_name(0)}")

    # GPU 메모리 정보
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9  # GB
    print(f"    GPU 메모리: {total_memory:.1f} GB")
else:
    device = torch.device("cpu")
    print(f"    GPU 사용 가능: False")
    print(f"    경고: CPU만 사용 가능합니다")

# Device 자동 설정 (이후 모든 실습에서 사용할 패턴)
print(f"\n[4] Device 자동 설정 (표준 패턴)")
device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    else "cpu"
)
print(f"    선택된 device: {device}")
print(f"    (이제부터 모든 모델과 데이터를 이 device로 이동합니다)")

# Tensor 생성 확인
print(f"\n[5] Tensor 생성 및 device 확인")
x = torch.randn(3, 3)
print(f"    CPU에서 생성한 Tensor의 위치: {x.device}")

x_device = x.to(device)
print(f"    device로 이동 후: {x_device.device}")

print("\n" + "=" * 50)
print("모든 검증 완료 ✓")
print("=" * 50)
```

예상 결과:
```
==================================================
PyTorch 환경 최종 검증
==================================================

[1] Python 버전
    3.10.11

[2] PyTorch 버전
    2.1.2+cu121

[3] GPU 정보
    GPU 사용 가능: True
    CUDA 버전: 12.1
    GPU 개수: 1
    GPU 이름: NVIDIA GeForce RTX 4080
    GPU 메모리: 16.0 GB

[4] Device 자동 설정 (표준 패턴)
    선택된 device: cuda

[5] Tensor 생성 및 device 확인
    CPU에서 생성한 Tensor의 위치: cpu
    device로 이동 후: cuda:0

==================================================
모든 검증 완료 ✓
==================================================
```

③ **기본 Tensor 연산 검증**

```python
import torch

print("=" * 50)
print("Tensor 기본 연산 검증")
print("=" * 50)

# Device 설정 (이제부터 표준)
device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    else "cpu"
)

# [1] Tensor 생성 (다양한 방법)
print(f"\n[1] Tensor 생성 방법")

# 무작위 생성
x = torch.randn(2, 3)
print(f"    randn(2,3): shape={x.shape}, dtype={x.dtype}")

# 0으로 초기화
y = torch.zeros(2, 3)
print(f"    zeros(2,3): shape={y.shape}")

# 1로 초기화
z = torch.ones(2, 3)
print(f"    ones(2,3): shape={z.shape}")

# 특정 값으로 초기화
w = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
print(f"    tensor([[...]...): shape={w.shape}")

# [2] 기본 연산 (CPU)
print(f"\n[2] 기본 연산 (CPU)")

A = torch.randn(2, 3)
B = torch.randn(3, 2)

# 행렬 곱
C = torch.matmul(A, B)  # 또는 A @ B
print(f"    행렬 곱: ({A.shape}) @ ({B.shape}) = {C.shape}")

# 요소별 곱 (Hadamard product)
x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
y = torch.tensor([[2.0, 3.0], [4.0, 5.0]])
z = x * y
print(f"    요소별 곱: {z.tolist()}")

# 합산
sum_val = A.sum()
print(f"    요소 합: {sum_val.item():.4f}")

# [3] Device 이동 (GPU 체험)
print(f"\n[3] Device 이동 (CPU → {device})")

x_cpu = torch.randn(2, 3)
print(f"    생성 위치: {x_cpu.device}")

x_device = x_cpu.to(device)
print(f"    이동 후: {x_device.device}")

# GPU에서 연산
y_device = torch.randn(2, 3).to(device)
z_device = x_device @ torch.randn(3, 2).to(device)
print(f"    GPU 행렬 곱 결과: shape={z_device.shape}, device={z_device.device}")

# GPU에서 CPU로 다시 이동 (후처리/시각화를 위해)
z_cpu = z_device.cpu()
print(f"    GPU → CPU로 복귀: {z_cpu.device}")

print("\n" + "=" * 50)
print("모든 연산 검증 완료 ✓")
print("=" * 50)
```

예상 결과:
```
==================================================
Tensor 기본 연산 검증
==================================================

[1] Tensor 생성 방법
    randn(2,3): shape=torch.Size([2, 3]), dtype=torch.float32
    zeros(2,3): shape=torch.Size([2, 3])
    ones(2,3): shape=torch.Size([2, 3])
    tensor([[...]...): shape=torch.Size([2, 2])

[2] 기본 연산 (CPU)
    행렬 곱: (torch.Size([2, 3])) @ (torch.Size([3, 2])) = torch.Size([2, 2])
    요소별 곱: [[2.0, 6.0], [12.0, 20.0]]
    요소 합: -0.3421

[3] Device 이동 (CPU → cuda)
    생성 위치: cpu
    이동 후: cuda:0
    GPU 행렬 곱 결과: shape=torch.Size([2, 2]), device=cuda:0
    GPU → CPU로 복귀: cpu

==================================================
모든 연산 검증 완료 ✓
==================================================
```

**검증 체크리스트**:
- [ ] setup_env.py 스크립트 실행 성공 및 모든 단계 통과
- [ ] GPU 자동 감지 성공 (있으면 GPU, 없으면 CPU)
- [ ] torch.cuda.is_available() 또는 device 설정이 올바른가?
- [ ] 행렬 곱 형태 검증: (2,3) @ (3,2) = (2,2) ✓
- [ ] Tensor를 device로 이동 후 .device 속성 확인 가능한가?
- [ ] 모든 연산 오류 없이 완료되었는가?

**Copilot 프롬프트 1**:
```
"PyTorch가 설치되었는지 확인하고, GPU가 있으면 GPU를 사용하고 없으면 CPU를 사용하도록
device를 자동으로 설정하는 Python 코드를 작성해줄래?"
```

**Copilot 프롬프트 2**:
```
"torch.randn(), torch.zeros(), torch.ones()로 다양한 크기의 Tensor를 만들고,
행렬 곱(@), 요소별 곱(*), 합(sum())을 해보는 예제 코드를 만들어줄래?"
```

---

#### 체크포인트 2: Autograd와 기울기 계산 (20분)

**목표**: `requires_grad=True`로 설정한 Tensor에서 역전파를 수행하고 기울기를 계산하며, 신경망 파라미터의 기울기 추적을 체험한다

**핵심 단계**:

① **스칼라 변수의 자동 미분**

```python
import torch

print("=" * 50)
print("Autograd 기본: 스칼라의 미분")
print("=" * 50)

# [1] 기울기 추적 활성화
x = torch.tensor(3.0, requires_grad=True)
y = torch.tensor(2.0, requires_grad=True)

print(f"\n[1] 초기 상태")
print(f"    x = {x.item()}, requires_grad = {x.requires_grad}")
print(f"    y = {y.item()}, requires_grad = {y.requires_grad}")

# [2] 함수 정의 및 계산
# f(x, y) = 2x² + xy + 3y²
f = 2 * x ** 2 + x * y + 3 * y ** 2

print(f"\n[2] 함수 계산")
print(f"    f(x, y) = 2x² + xy + 3y²")
print(f"    f(3, 2) = 2(9) + (3)(2) + 3(4)")
print(f"           = 18 + 6 + 12 = 36")
print(f"    계산된 f: {f.item()}")

# [3] 역전파 (자동 미분)
f.backward()

print(f"\n[3] 역전파 후 기울기")
print(f"    df/dx = {x.grad.item()}")
print(f"    df/dy = {y.grad.item()}")

# [4] 손으로 계산한 값과 비교
print(f"\n[4] 손으로 계산한 값 (검증)")
print(f"    df/dx = d(2x² + xy + 3y²)/dx = 4x + y = 4(3) + 2 = 14 ✓")
print(f"    df/dy = d(2x² + xy + 3y²)/dy = x + 6y = 3 + 6(2) = 15 ✓")

print("\n" + "=" * 50)
print("스칼라 미분 검증 완료 ✓")
print("=" * 50)
```

예상 결과:
```
==================================================
Autograd 기본: 스칼라의 미분
==================================================

[1] 초기 상태
    x = 3.0, requires_grad = True
    y = 2.0, requires_grad = True

[2] 함수 계산
    f(x, y) = 2x² + xy + 3y²
    f(3, 2) = 2(9) + (3)(2) + 3(4)
           = 18 + 6 + 12 = 36
    계산된 f: 36.0

[3] 역전파 후 기울기
    df/dx = 14.0
    df/dy = 15.0

[4] 손으로 계산한 값 (검증)
    df/dx = d(2x² + xy + 3y²)/dx = 4x + y = 4(3) + 2 = 14 ✓
    df/dy = d(2x² + xy + 3y²)/dy = x + 6y = 3 + 6(2) = 15 ✓

==================================================
스칼라 미분 검증 완료 ✓
==================================================
```

② **벡터/행렬의 기울기 (신경망 파라미터)**

```python
import torch
import torch.nn as nn

print("=" * 50)
print("Autograd 심화: 신경망 파라미터의 기울기")
print("=" * 50)

# Device 설정
device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    else "cpu"
)

# [1] 간단한 신경망 (선형층 하나)
model = nn.Linear(3, 2, bias=True)
model = model.to(device)

print(f"\n[1] 신경망 구조")
print(f"    선형층: 3차원 입력 → 2차원 출력")
print(f"    가중치 W shape: {model.weight.shape}")  # (2, 3)
print(f"    편향 b shape: {model.bias.shape}")      # (2,)
print(f"    파라미터 개수: {sum(p.numel() for p in model.parameters())}")

# [2] 입력 데이터
x = torch.randn(2, 3).to(device)  # 배치 2, 차원 3

print(f"\n[2] 입력 데이터")
print(f"    입력 x shape: {x.shape}")
print(f"    입력 x 위치: {x.device}")

# [3] 순전파 (Forward Pass)
output = model(x)  # shape: (2, 2)

print(f"\n[3] 순전파 결과")
print(f"    출력 shape: {output.shape}")
print(f"    출력 값:\n{output}")

# [4] 손실 함수 정의 (임시로 간단한 손실 사용)
loss = output.sum()  # 모든 원소의 합 (이후 실습에서 적절한 손실 함수 사용)

print(f"\n[4] 손실 (Loss)")
print(f"    손실 값: {loss.item():.4f}")

# [5] 역전파 (Backward Pass)
loss.backward()

print(f"\n[5] 역전파 후 기울기")
print(f"    W의 기울기 shape: {model.weight.grad.shape}")
print(f"    W의 기울기 norm: {model.weight.grad.norm().item():.4f}")
print(f"    b의 기울기 shape: {model.bias.grad.shape}")
print(f"    b의 기울기 norm: {model.bias.grad.norm().item():.4f}")

# [6] 기울기가 실제로 0이 아닌지 확인 (학습 가능)
print(f"\n[6] 학습 가능 여부 확인")
if model.weight.grad.abs().max() > 0:
    print(f"    W의 기울기 최댓값: {model.weight.grad.abs().max().item():.6f}")
    print(f"    → 기울기가 0이 아니므로 학습 가능 ✓")
else:
    print(f"    경고: 기울기가 0입니다. 모델 구조를 확인하세요.")

print("\n" + "=" * 50)
print("신경망 파라미터 미분 검증 완료 ✓")
print("=" * 50)
```

예상 결과:
```
==================================================
Autograd 심화: 신경망 파라미터의 기울기
==================================================

[1] 신경망 구조
    선형층: 3차원 입력 → 2차원 출력
    가중치 W shape: torch.Size([2, 3])
    편향 b shape: torch.Size([2])
    파라미터 개수: 8

[2] 입력 데이터
    입력 x shape: torch.Size([2, 3])
    입력 x 위치: cuda:0

[3] 순전파 결과
    출력 shape: torch.Size([2, 2])
    출력 값:
    tensor([[-0.4234,  0.1256],
            [ 0.5678, -0.8901]], device='cuda:0')

[4] 손실 (Loss)
    손실 값: -0.6201

[5] 역전파 후 기울기
    W의 기울기 shape: torch.Size([2, 3])
    W의 기울기 norm: 2.3456
    b의 기울기 shape: torch.Size([2])
    b의 기울기 norm: 1.2345

[6] 학습 가능 여부 확인
    W의 기울기 최댓값: 0.567890
    → 기울기가 0이 아니므로 학습 가능 ✓

==================================================
신경망 파라미터 미분 검증 완료 ✓
==================================================
```

**검증 체크리스트**:
- [ ] `requires_grad=True` 설정 후 `.backward()` 실행 성공
- [ ] 스칼라 미분: 손으로 계산한 값과 `.grad` 일치 확인
- [ ] 신경망 파라미터 기울기 계산됨 (norm > 0)
- [ ] 기울기가 계산된 위치(device)가 올바른가?

**Copilot 프롬프트 3**:
```
"PyTorch에서 requires_grad=True로 설정한 변수 x, y로 f(x, y) = 2x² + xy + 3y² 함수를 만들고
.backward()로 기울기를 계산하는 코드를 작성해줄래? 손으로 계산한 값과 비교 주석도 넣어줘."
```

**Copilot 프롬프트 4**:
```
"nn.Linear(3, 2)로 만든 신경망 모델의 파라미터 W와 b의 기울기를 계산하는 코드를 작성해줄래?
device 지정과 .backward() 후 기울기 확인까지 포함해줘."
```

---

#### 체크포인트 3: CPU vs GPU 성능 비교 + 간단한 선형 회귀 (20분)

**목표**: CPU와 GPU의 성능 차이를 정량적으로 측정하고, 실제 모델 학습에서 GPU의 이점을 체험하며, 간단한 선형 회귀 모델을 완전히 구현하고 학습한다

**핵심 단계**:

① **CPU vs GPU 성능 비교 벤치마크**

```python
import torch
import time

print("=" * 60)
print("CPU vs GPU 성능 비교 벤치마크")
print("=" * 60)

# Device 확인
if torch.cuda.is_available():
    device_gpu = torch.device("cuda")
    print(f"\nGPU 감지됨: {torch.cuda.get_device_name(0)}")
else:
    device_gpu = None
    print(f"\nGPU 미감지 - CPU만 벤치마크 진행")

device_cpu = torch.device("cpu")

# [1] 행렬 곱 벤치마크
print(f"\n[1] 행렬 곱 벤치마크 (1000×1000 × 1000×1000)")
print("-" * 60)

size = 1000
A_cpu = torch.randn(size, size, device="cpu")
B_cpu = torch.randn(size, size, device="cpu")

# CPU 측정
torch.cuda.synchronize() if torch.cuda.is_available() else None
start = time.time()
for _ in range(5):  # 5회 반복해서 평균 측정
    C_cpu = torch.matmul(A_cpu, B_cpu)
cpu_time = (time.time() - start) / 5
print(f"  CPU 시간: {cpu_time:.4f}초")

# GPU 측정
if device_gpu:
    A_gpu = A_cpu.to(device_gpu)
    B_gpu = B_cpu.to(device_gpu)

    torch.cuda.synchronize()
    start = time.time()
    for _ in range(5):
        C_gpu = torch.matmul(A_gpu, B_gpu)
    torch.cuda.synchronize()
    gpu_time = (time.time() - start) / 5

    print(f"  GPU 시간: {gpu_time:.4f}초")
    speedup = cpu_time / gpu_time
    print(f"  속도 향상: {speedup:.1f}배 ✓")
else:
    print(f"  GPU 미감지: 비교 불가")

# [2] 신경망 학습 벤치마크
print(f"\n[2] 신경망 학습 벤치마크 (1000 epoch)")
print("-" * 60)

import torch.nn as nn

# 간단한 데이터셋 준비
n_samples = 1000
n_features = 100
X_cpu = torch.randn(n_samples, n_features, device="cpu")
y_cpu = torch.randn(n_samples, 1, device="cpu")

# CPU에서 학습
model_cpu = nn.Sequential(
    nn.Linear(n_features, 64),
    nn.ReLU(),
    nn.Linear(64, 1)
).to("cpu")

optimizer = torch.optim.SGD(model_cpu.parameters(), lr=0.01)
criterion = nn.MSELoss()

torch.cuda.synchronize() if torch.cuda.is_available() else None
start = time.time()

for epoch in range(1000):
    pred = model_cpu(X_cpu)
    loss = criterion(pred, y_cpu)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

cpu_train_time = time.time() - start
print(f"  CPU 학습 시간 (1000 epoch): {cpu_train_time:.2f}초")

# GPU에서 학습
if device_gpu:
    X_gpu = X_cpu.to(device_gpu)
    y_gpu = y_cpu.to(device_gpu)

    model_gpu = nn.Sequential(
        nn.Linear(n_features, 64),
        nn.ReLU(),
        nn.Linear(64, 1)
    ).to(device_gpu)

    optimizer = torch.optim.SGD(model_gpu.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    torch.cuda.synchronize()
    start = time.time()

    for epoch in range(1000):
        pred = model_gpu(X_gpu)
        loss = criterion(pred, y_gpu)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    torch.cuda.synchronize()
    gpu_train_time = time.time() - start

    print(f"  GPU 학습 시간 (1000 epoch): {gpu_train_time:.2f}초")
    speedup = cpu_train_time / gpu_train_time
    print(f"  속도 향상: {speedup:.1f}배 ✓")

print("\n" + "=" * 60)
print("벤치마크 완료")
print("=" * 60)
```

예상 결과:
```
============================================================
CPU vs GPU 성능 비교 벤치마크
============================================================

GPU 감지됨: NVIDIA GeForce RTX 4080

[1] 행렬 곱 벤치마크 (1000×1000 × 1000×1000)
------------------------------------------------------------
  CPU 시간: 0.2145초
  GPU 시간: 0.0043초
  속도 향상: 49.9배 ✓

[2] 신경망 학습 벤치마크 (1000 epoch)
------------------------------------------------------------
  CPU 학습 시간 (1000 epoch): 23.45초
  GPU 학습 시간 (1000 epoch): 2.34초
  속도 향상: 10.0배 ✓

============================================================
벤치마크 완료
============================================================
```

② **간단한 선형 회귀 모델 (완전 구현)**

```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

print("=" * 60)
print("간단한 선형 회귀 모델 (Autograd 활용)")
print("=" * 60)

# Device 설정
device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    else "cpu"
)
print(f"\n선택된 device: {device}")

# [1] 합성 데이터셋 생성
print(f"\n[1] 합성 데이터셋 생성")
print(f"    y = 3x + 2 (+ 잡음)")

torch.manual_seed(42)
n_samples = 100
X = torch.linspace(-5, 5, n_samples).reshape(-1, 1)  # (100, 1)
y_true = 3 * X + 2
y_noisy = y_true + torch.randn_like(y_true) * 0.5  # 잡음 추가

# Device로 이동
X = X.to(device)
y_noisy = y_noisy.to(device)

print(f"    데이터 크기: X shape={X.shape}, y shape={y_noisy.shape}")
print(f"    데이터 위치: {X.device}")

# [2] 선형 회귀 모델 정의
print(f"\n[2] 모델 정의")

model = nn.Sequential(
    nn.Linear(1, 1)  # 1D 입력 → 1D 출력
)
model = model.to(device)

# 파라미터 초기값 확인
with torch.no_grad():
    w_init = model[0].weight.item()
    b_init = model[0].bias.item()

print(f"    초기 가중치 W: {w_init:.4f}")
print(f"    초기 편향 b: {b_init:.4f}")
print(f"    목표값: W ≈ 3, b ≈ 2")

# [3] 손실 함수 및 옵티마이저
print(f"\n[3] 손실 함수 및 옵티마이저")

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

print(f"    손실 함수: MSE (Mean Squared Error)")
print(f"    옵티마이저: SGD (learning_rate=0.01)")

# [4] 모델 학습
print(f"\n[4] 모델 학습 (200 epoch)")

losses = []
for epoch in range(200):
    # 순전파 (Forward Pass)
    y_pred = model(X)
    loss = criterion(y_pred, y_noisy)

    # 역전파 (Backward Pass)
    optimizer.zero_grad()  # 기울기 초기화
    loss.backward()        # 기울기 계산
    optimizer.step()       # 파라미터 업데이트

    losses.append(loss.item())

    # 진행 상황 출력
    if (epoch + 1) % 50 == 0:
        with torch.no_grad():
            w = model[0].weight.item()
            b = model[0].bias.item()
        print(f"    Epoch {epoch+1:3d}: Loss={loss.item():.4f}, W={w:.4f}, b={b:.4f}")

# [5] 최종 파라미터
print(f"\n[5] 학습 결과")

with torch.no_grad():
    w_final = model[0].weight.item()
    b_final = model[0].bias.item()

print(f"    최종 가중치 W: {w_final:.4f} (목표: 3.0000)")
print(f"    최종 편향 b: {b_final:.4f} (목표: 2.0000)")
print(f"    W 오차: {abs(w_final - 3.0):.4f}")
print(f"    b 오차: {abs(b_final - 2.0):.4f}")

# [6] 예측 및 시각화
print(f"\n[6] 시각화 및 저장")

# CPU로 이동해서 시각화 (matplotlib 호환성)
X_cpu = X.cpu()
y_noisy_cpu = y_noisy.cpu()

with torch.no_grad():
    y_pred = model(X).cpu()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# 왼쪽: 회귀 결과
ax1.scatter(X_cpu, y_noisy_cpu, alpha=0.6, label="실제 데이터", s=30)
ax1.plot(X_cpu, y_pred, 'r-', linewidth=2, label=f"예측 (y={w_final:.2f}x+{b_final:.2f})")
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.set_title("선형 회귀 결과")
ax1.legend()
ax1.grid(True, alpha=0.3)

# 오른쪽: 손실 곡선
ax2.plot(losses, linewidth=1.5, color="steelblue")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Loss (MSE)")
ax2.set_title("학습 곡선")
ax2.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig("linear_regression.png", dpi=150, bbox_inches="tight")
print(f"    저장: linear_regression.png")
plt.close()

print("\n" + "=" * 60)
print("선형 회귀 모델 학습 완료 ✓")
print("=" * 60)
```

예상 결과:
```
============================================================
간단한 선형 회귀 모델 (Autograd 활용)
============================================================

선택된 device: cuda

[1] 합성 데이터셋 생성
    y = 3x + 2 (+ 잡음)
    데이터 크기: X shape=torch.Size([100, 1]), y shape=torch.Size([100, 1])
    데이터 위치: cuda:0

[2] 모델 정의
    초기 가중치 W: 0.2341
    초기 편향 b: -0.5234
    목표값: W ≈ 3, b ≈ 2

[3] 손실 함수 및 옵티마이저
    손실 함수: MSE (Mean Squared Error)
    옵티마이저: SGD (learning_rate=0.01)

[4] 모델 학습 (200 epoch)
    Epoch  50: Loss=0.4562, W=2.8756, b=1.8932
    Epoch 100: Loss=0.2543, W=2.9456, b=1.9567
    Epoch 150: Loss=0.2198, W=2.9678, b=1.9765
    Epoch 200: Loss=0.2115, W=2.9812, b=1.9892

[5] 학습 결과
    최종 가중치 W: 2.9812 (목표: 3.0000)
    최종 편향 b: 1.9892 (목표: 2.0000)
    W 오차: 0.0188
    b 오차: 0.0108

[6] 시각화 및 저장
    저장: linear_regression.png

============================================================
선형 회귀 모델 학습 완료 ✓
============================================================
```

**검증 체크리스트**:
- [ ] CPU vs GPU 벤치마크 실행 및 속도 향상 측정 (최소 5배 이상)
- [ ] 신경망 학습 벤치마크 성공 (1000 epoch 완료)
- [ ] 선형 회귀 모델 학습 성공 및 손실 감소 확인
- [ ] 최종 파라미터 W, b가 목표값(3, 2)에 수렴했는가?
- [ ] 시각화 결과 이미지 생성됨

**Copilot 프롬프트 5**:
```
"PyTorch에서 CPU와 GPU의 행렬 곱 성능을 비교하는 벤치마크 코드를 작성해줄래?
시간을 측정해서 속도 향상도 계산해야 해."
```

**Copilot 프롬프트 6**:
```
"y = 3x + 2라는 선형 관계를 따르는 합성 데이터를 만들고,
nn.Sequential로 선형층 하나를 만들어서 200 epoch 동안 학습하는 코드를 작성해줄래?
매 50 epoch마다 손실과 파라미터를 출력하고, 학습 곡선을 그려줘."
```

---

### 제출 안내 (Google Classroom)

**제출 방법**:
- Google Classroom의 "1주차 B회차" 과제에 조별 1부 제출
- 파일명 형식: `group_{조번호}_ch1B.zip`

**포함할 파일**:
```
group_{조번호}_ch1B/
├── ch1B_setup_and_tensor.py        # 체크포인트 1 + 2 + 3 전체 코드
├── setup_env_output.txt            # setup_env.py 실행 결과 (복사-붙여넣기)
├── benchmark_results.txt           # CPU vs GPU 벤치마크 결과
├── linear_regression.png           # 선형 회귀 시각화 (체크포인트 3)
└── report.md                       # 분석 리포트 (1페이지)
```

**리포트 포함 항목** (report.md):
- **체크포인트 1**: 환경 설정 과정에서 겪은 어려움과 해결 방법 (2-3문장)
  - 예: "GPU 감지가 잘 안 되었는데, CUDA 버전 확인으로 해결했다"
- **체크포인트 2**: Autograd 이해도 (2문장)
  - 예: "requires_grad=True로 설정하고 .backward()를 호출하면 기울기가 자동으로 계산된다는 것을 확인했다"
- **체크포인트 3**: CPU vs GPU 성능 차이 관찰 (2-3문장)
  - 예: "행렬 곱에서는 50배, 신경망 학습에서는 10배 속도 향상을 확인했다"
- **선형 회귀 결과 해석**: 최종 W, b 파라미터와 목표값 비교 (2-3문장)
  - 예: "목표값 W=3, b=2에 대해 최종 W=2.98, b=1.99로 수렴하여 모델이 올바르게 학습했다"
- **GPU의 실무적 가치**: 왜 GPU를 사용해야 하는가 (2문장)
- **Copilot 활용 경험**: 어떤 프롬프트가 도움이 됐는가 (2문장)

**제출 마감**: 수업 종료 후 24시간 이내

---

### 결과 토론 가이드

> **토론 가이드**: 조별 구현 결과를 공유하며, GPU가 없었던 조와 있었던 조의 경험을 비교하고, Autograd의 작동 원리를 함께 검토한다

**토론 주제**:

① **환경 설정 경험**
- 모든 조가 setup_env.py로 성공했는가? 실패한 경우는?
- GPU 감지 과정에서 어려웠던 점은?
- 예: "내 노트북은 GPU가 없어서 CPU만 사용했는데, 실습실 데스크톱에서 GPU를 써보니 어땠나?"

② **Autograd 이해도**
- 스칼라 미분과 신경망 파라미터 미분의 차이점은?
- `requires_grad=True`가 없으면 어떻게 될까?
- 왜 `.backward()` 후에 `optimizer.zero_grad()`를 해야 할까?

③ **CPU vs GPU 성능 차이**
- 행렬 곱에서 GPU가 훨씬 빠른 이유는?
- 신경망 학습에서는 GPU의 이점이 행렬 곱보다 작은 이유는?
- 자신의 컴퓨터에서 측정한 속도 향상 배수는?

④ **선형 회귀 모델 수렴**
- 모든 조의 모델이 W≈3, b≈2로 수렴했는가?
- Learning rate를 바꾸면 수렴 속도가 달라질까?
- Epoch를 더 크게 하면 어떻게 될까?

⑤ **실무적 시사**
- 9~10주차 모델 파인튜닝에서는 왜 GPU가 사실상 필수인가?
- Google Colab의 무료 GPU를 활용하는 전략은?
- 개인 프로젝트에서 고성능 GPU 서버를 어떻게 구성할 것인가?

**발표 형식**:
- 각 조 3~5분 발표 (환경 설정 경험 + 벤치마크 결과 + 선형 회귀 수렴 과정)
- 다른 조의 질문에 답변 (2~3개 질문)
- 교수의 보충 설명 및 피드백

---

### 다음 주 예고

다음 주 2주차 A회차에서는 **딥러닝의 핵심 원리**를 깊이 있게 다룬다.

**예고 내용**:
- 퍼셉트론(Perceptron)에서 다층 신경망(Multi-Layer Perceptron)으로의 진화
- 활성화 함수(ReLU, Sigmoid, Tanh)의 역할과 필요성
- 손실 함수(Cross-Entropy, MSE)의 정의와 선택 기준
- 역전파(Backpropagation)의 원리: 연쇄 법칙으로 어떻게 기울기가 역으로 전파되는가
- PyTorch `nn.Module`을 활용한 커스텀 신경망 설계
- B회차에서는 한국어 영화 리뷰 감정 분석 데이터셋으로 텍스트 분류 모델을 직접 구현·학습시킨다

**사전 준비**:
- 1주차 B회차 과제를 다시 한 번 검토하기 (Tensor, Autograd, device 개념)
- "왜 활성화 함수가 필요한가?"에 대해 생각해보기
- ReLU vs Sigmoid의 차이점을 조사해보기 (미리 생각해오면 이해가 쉬움)

---

## 참고 자료

**실습 코드**:
- _전체 구현 코드는 practice/chapter1/code/1-1-기본환경.py 참고_
- _벤치마크 및 선형 회귀는 practice/chapter1/code/1-2-회귀모델.py 참고_

**권장 읽기**:
- PyTorch Official Documentation. "Autograd: Automatic Differentiation." https://pytorch.org/docs/stable/autograd.html
- PyTorch Official Documentation. "Tensor Attributes." https://pytorch.org/docs/stable/tensor_attributes.html
- Karpathy, A. "A Hacker's Guide to Neural Networks." https://karpathy.github.io/neuralnets/
- 3Blue1Brown. "But what is a neural network?" https://www.youtube.com/watch?v=aircArM63o8
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press. Chapters 1-6.

**GPU 관련 자료**:
- NVIDIA CUDA Documentation. https://docs.nvidia.com/cuda/
- Hugging Face. "Accelerate: A Simple Way to Train and Use PyTorch Models with Multi-GPU, TPU, Mixed-Precision." https://huggingface.co/docs/accelerate/

