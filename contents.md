# 딥러닝 자연어처리 (2026) - 상세 강의계획서

## LLM 시대의 NLP 엔지니어링: 원리부터 배포까지

> **대상**: 컴퓨터공학/AI 전공 학부생 (3~4학년)
> **목표**: LLM의 원리를 깊이 이해하고, 실무에서 모델을 설계·튜닝·배포할 수 있는 AI 엔지니어 양성
> **키워드**: PyTorch 구현, 모델 아키텍처, 파인튜닝, AI Agent, 배포

---

## 강의 구성 철학

```
기초 압축 (4주) → LLM 핵심 (2주) → 시험 → 실무 기술 (5주) → 프로젝트 (2주) → 시험
```

- **기초는 빠르게, 실무는 깊게**: RNN은 개념만, Transformer부터 본격 구현
- **밑바닥 구현 → 프레임워크 활용**: 원리를 안 뒤에 도구를 쓴다
- **취업 직결 기술 우선**: Agent, 배포, 평가 등 현업 필수 스킬 포함
- **직관 먼저, 수식 다음**: 모든 개념은 "왜 필요한가"부터 시작하고, 비유와 시각화로 직관을 잡은 뒤 수학적 정의로 넘어간다

---

## 실습 도구

### VS Code + GitHub Copilot

모든 실습은 **VS Code**에서 **GitHub Copilot**을 활용하여 진행한다.

| 항목 | 내용 |
|------|------|
| **IDE** | VS Code 1.102+ |
| **AI 도구** | GitHub Copilot Pro (학생 무료) |
| **Copilot 모드** | Copilot Chat (Agent 모드) |
| **언어** | Python 3.10+ / PyTorch |

**Copilot 역할 (NLP 과목 특성)**:
- 이 과목에서 Copilot은 코드의 **주 작성자**가 아니라 구현의 **가속 도구**이다
- 학생은 Attention, LoRA 등의 원리를 **직접 이해**해야 하며, Copilot은 보일러플레이트 코드와 반복 작업을 돕는다
- **A회차**: Copilot 없이 이론과 원리를 먼저 학습 (교수 시연 시 선택 사용)
- **B회차**: Copilot을 활용하여 2인1조 실습 + 디버깅

**NLP 프로젝트 copilot-instructions.md 예시**:
```markdown
# NLP 프로젝트 지침
- Python 3.10+, PyTorch 2.x, Hugging Face Transformers 4.x
- 모든 모델은 nn.Module을 상속하여 정의
- 한국어 주석 사용, PEP 8 준수
- GPU 사용 시 device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

**추천 MCP 서버** (선택사항):
- **Context7**: PyTorch/Hugging Face 최신 문서 자동 참조
- **GitHub MCP**: 과제 리포지토리 관리

**추천 Agent Skills** (`.github/skills/`):
- `pytorch-model-pattern`: nn.Module 템플릿, training loop 패턴
- `huggingface-pipeline`: Trainer API, 모델 로드 패턴

### GPU/CUDA 환경 설정 정책

모든 실습 코드는 **GPU 가속을 기본으로 활용**하되, CUDA가 없는 환경에서도 **CPU로 자동 폴백**한다.

**표준 디바이스 설정 패턴** (모든 실습 코드에 적용):
```python
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = model.to(device)
```

**주차별 GPU 필요도**:

| 구분 | GPU 필수 | GPU 강력 권장 | GPU 있으면 좋음 | CPU 충분 |
|------|---------|-------------|---------------|---------|
| 주차 | 9, 10 | 5, 8, 13 | 1, 2, 4, 11 | 3, 6, 12 |
| 사유 | 모델 파인튜닝 | 대형 모델 추론/임베딩 | 학습 가속 | API/소규모 연산 |

**1주차 자동 환경 설정**: `scripts/setup_env.py` 스크립트를 실행하면 가상환경 생성, 전체 실습 패키지 설치, GPU 사양 감지 및 적합한 CUDA 버전의 PyTorch를 자동으로 설치한다. 이후 주차에서는 별도 환경 설정 없이 바로 실습에 집중할 수 있다.

---

## 수업 운영 구조 (주2회 90분 A/B 체계)

매주 2회 수업을 진행하며, A회차(이론+시연)와 B회차(실습+토론)로 구분한다:

**A회차 (90분) — 이론 + 시연**

| 시간 | 내용 | Copilot 사용 |
|------|------|-------------|
| 00:00~00:05 | 오늘의 질문 + 빠른 진단(퀴즈 1문항) | 사용 안 함 |
| 00:05~00:55 | 이론 강의 (직관적 비유 → 개념 → 수식 → 원리) | 사용 안 함 |
| 00:55~01:25 | 라이브 코딩 시연 (교수 주도, 핵심 코드 시연) | 교수 시연 시 선택 사용 |
| 01:25~01:28 | 핵심 정리 + B회차 과제 스펙 공개 | |
| 01:28~01:30 | Exit ticket (1문항) | |

**B회차 (90분) — 실습 + 토론**

| 시간 | 내용 | Copilot 사용 |
|------|------|-------------|
| 00:00~00:05 | 2인1조 편성 + 과제 스펙 확인 | |
| 00:05~00:10 | A회차 핵심 코드 빠른 리캡 + Copilot 사용 가이드 | |
| 00:10~00:55 | Copilot 활용 조별 구현 (체크포인트 3개) | 적극 사용 |
| 00:55~01:00 | Google Classroom 제출 (코드 + 결과 + 해석) | |
| 01:00~01:20 | 조별 결과 공유 + 전체 토론 (2-3조 발표) | |
| 01:20~01:28 | 교수 종합 피드백 + 모범 구현 비교 | |
| 01:28~01:30 | 다음 주 예고 | |

**B회차 운영 원칙**:
- **2인1조**: 매 B회차마다 조를 편성한다. 분석/구현은 함께 협의하되, 보고서 작성 책임은 교대한다.
- **체크포인트 3개**: 45분 실습을 3단계로 나누어 중간중간 조원 교차 검증을 수행한다.
- **Google Classroom 제출**: 코드 + 실행 결과 + 해석을 제출한다.
- **토론**: 조별 구현 전략, 하이퍼파라미터 선택, Copilot 활용 경험을 공유한다.

**C파일 (모범 구현 + 해설)**: 과제 제출 후 `docs/ch{N}C.md`로 공개한다. NLP 과제는 "정답"이 하나가 아닐 수 있으므로, 모범 구현 코드 + 핵심 포인트 해설 + 대안적 접근 형태로 제공한다.

---

## 15주 운영 계획

| 주차 | A회차 (이론 + 시연) | B회차 (실습 + 토론) | B회차 과제 |
|:---:|------|------|------|
| 1 | AI 시대의 개막 + PyTorch 기초 시연 | 환경 구축 + Tensor 실습 | GPU 환경 구축 + Tensor 과제 |
| 2 | 신경망 핵심 원리 + PyTorch 패턴 시연 | MLP 텍스트 분류 구현 + 성능 비교 토론 | IMDb 감성 분류 MLP + 분석 |
| 3 | 임베딩 + RNN → Attention 원리 시연 | Self-Attention 구현 + 시각화 | Attention 모듈 구현 + 시각화 리포트 |
| 4 | Transformer 아키텍처 심층 분석 시연 | Transformer Encoder 분류기 구현 + 토론 | Transformer Encoder 텍스트 분류 |
| 5 | BERT vs GPT 아키텍처 비교 시연 | BERT/GPT 활용 실습 + 성능 비교 토론 | BERT NER + GPT-2 텍스트 생성기 |
| 6 | 프롬프트 엔지니어링 + API 시연 | API 활용 시스템 구축 + 전략 토론 | 도메인 특화 텍스트 분석 시스템 |
| **7** | **중간고사 대비** | **중간고사 (객관식)** | **1~6주차 범위** |
| 8 | 토픽 모델링 이론 + BERTopic 시연 | BERTopic 실습 + 결과 해석 토론 | 토픽 모델링 분석 보고서 |
| 9 | Full Fine-tuning 원리 + Trainer 시연 | BERT 파인튜닝 실습 + 학습 전략 토론 | 전공 분야 모델 파인튜닝 |
| 10 | PEFT/LoRA 원리 + QLoRA 시연 | LoRA 실습 + Full FT 비교 토론 | LoRA vs Full FT 비교 보고서 |
| 11 | RAG 아키텍처 설계 + 파이프라인 시연 | RAG Q&A 시스템 구축 + 검색 전략 토론 | 전공 문서 기반 RAG Q&A 시스템 |
| 12 | AI Agent 개념 + LangGraph 시연 | Agent 프로토타입 개발 + 설계 토론 | 도메인 특화 AI Agent |
| 13 | 모델 배포 + 최적화 시연 | FastAPI 배포 + 평가 실습 + 토론 | FastAPI 배포 + 평가 리포트 |
| 14 | 개인 프로젝트 워크숍 A (가이드라인 + 개발) | 개인 프로젝트 워크숍 B (개발 + 피드백) | 프로토타입 완성 |
| **15** | **개인 프로젝트 발표** | **기말고사 (객관식)** | **종합 평가** |

---

## 1주차: AI 시대의 개막과 개발 환경 구축

> **미션**: 수업이 끝나면 PyTorch로 첫 번째 딥러닝 모델을 돌려본다

### 학습목표

1. AI, 머신러닝, 딥러닝의 관계를 설명할 수 있다
2. PyTorch 개발 환경을 구축하고 GPU를 활용할 수 있다
3. Tensor 연산과 Autograd 자동 미분을 이해하고 구현할 수 있다
4. VS Code + GitHub Copilot 실습 환경을 설정할 수 있다

---

### A회차: AI와 PyTorch 기초

#### 수업 타임라인

| 시간 | 내용 |
|------|------|
| 00:00~00:05 | 오늘의 질문 + 빠른 진단 |
| 00:05~00:55 | 이론 강의 |
| 00:55~01:25 | 라이브 코딩 시연 |
| 01:25~01:28 | 핵심 정리 + B회차 과제 스펙 공개 |
| 01:28~01:30 | Exit ticket |

##### 1.1 인공지능과 자연어처리 개요

**직관적 이해**: AI, 머신러닝, 딥러닝은 러시아 인형(마트료시카) 같다. 가장 큰 인형이 AI, 그 안에 머신러닝, 그 안에 딥러닝이 들어있다. ChatGPT가 "다음에 올 말"을 예측하는 단순한 원리로 어떻게 그렇게 똑똑해 보이는지 그 비밀을 풀어본다.

- AI, 머신러닝, 딥러닝의 관계와 각각의 역할
- 자연어처리(NLP)의 정의와 주요 응용 분야 (번역, 챗봇, 감성 분석, 요약 등)
- 언어 모델의 발전사: 통계 → 신경망 → Transformer → LLM
- 2024-2026 NLP 산업 동향과 취업 시장: 어떤 역량이 필요한가

##### 1.2 개발 환경 구축

> **라이브 코딩 시연**: 교수가 `scripts/setup_env.py` 스크립트를 실행하면서 GPU 감지, PyTorch 설치, 환경 검증 과정을 보여준다.

- **자동 환경 설정 스크립트** (`scripts/setup_env.py`):
  - Python 가상환경 자동 생성 (venv)
  - GPU 사양 자동 감지 (NVIDIA GPU 모델, VRAM, 드라이버 버전, 지원 CUDA 버전)
  - GPU 사양에 맞는 PyTorch + CUDA 버전 자동 설치 (GPU 없으면 CPU 버전 설치)
  - 15주차 전체 실습 패키지 일괄 설치 (`requirements.txt`)
  - 설치 결과 검증 및 GPU 벤치마크 (행렬 연산 CPU vs GPU 속도 비교)
- VS Code 설치 + GitHub Copilot / Copilot Chat 확장 설치
- Hugging Face 생태계 소개 (Hub, Transformers, Datasets)
- Google Colab / Kaggle GPU 환경 설정 (GPU가 없는 학생용 대안)
- Git/GitHub 기본 사용법

##### 1.3 Python 딥러닝 기초

**직관적 이해**: Tensor는 "숫자를 담는 다차원 상자"이다. 스칼라(점) → 벡터(선) → 행렬(면) → 텐서(공간)로 차원이 늘어나는 것을 시각적으로 이해한다. Autograd는 "수학 시험에서 풀이 과정을 자동으로 역추적하는 기능"이다.

- NumPy 행렬 연산 복습
- Tensor 기본 조작 (생성, 연산, 인덱싱, GPU 이동)
- Autograd를 이용한 자동 미분: 왜 미분이 필요한가, 어떻게 자동화되는가

---

### B회차: 환경 구축 + Tensor 실습

#### 수업 타임라인

| 시간 | 내용 |
|------|------|
| 00:00~00:05 | 2인1조 편성 + 과제 안내 |
| 00:05~00:10 | A회차 핵심 리캡 + Copilot 가이드 |
| 00:10~00:55 | Copilot 활용 조별 구현 (체크포인트 3개) |
| 00:55~01:00 | Google Classroom 제출 |
| 01:00~01:20 | 조별 결과 공유 + 토론 |
| 01:20~01:28 | 교수 종합 피드백 |
| 01:28~01:30 | 다음 주 예고 |

> **Copilot 활용**: Copilot에게 "PyTorch Tensor를 생성하고 GPU로 이동하는 코드를 작성해줘"와 같은 프롬프트로 시작하되, 생성된 코드의 각 줄이 무엇을 하는지 반드시 이해한다.

##### 1.4 실습

- **자동 환경 설정 스크립트 실행**: `python scripts/setup_env.py` 한 번으로 전체 환경 구축
- GPU 사양 확인 및 CUDA 설정 검증 (`torch.cuda.is_available()`, `torch.cuda.get_device_name()`)
- CPU vs GPU 행렬 연산 속도 비교 벤치마크 체험
- GitHub Copilot Pro 학생 무료 등록 + VS Code 설정
- PyTorch Tensor 연산 실습 (GPU 이동 포함)
- 간단한 선형 회귀 모델 구현 (Autograd 활용, GPU에서 학습)

**과제**: 자동 환경 설정 실행 결과 캡처 + PyTorch Tensor 조작 과제 (GPU/CPU 성능 비교 포함)

---

## 2주차: 딥러닝 핵심 원리와 PyTorch 실전

> **미션**: 수업이 끝나면 텍스트 분류 모델을 직접 학습시킨다

### A회차: 신경망 기본 구조 + PyTorch 패턴

#### 수업 타임라인

| 시간 | 내용 |
|------|------|
| 00:00~00:05 | 오늘의 질문 + 빠른 진단 |
| 00:05~00:55 | 이론 강의 |
| 00:55~01:25 | 라이브 코딩 시연 |
| 01:25~01:28 | 핵심 정리 + B회차 과제 스펙 공개 |
| 01:28~01:30 | Exit ticket |

##### 2.1 신경망 기본 구조

**직관적 이해**: 신경망은 "레고 블록 조립"과 같다. 뉴런 하나는 단순한 판단기(입력 × 가중치 → 활성화)이지만, 이를 수백만 개 쌓으면 복잡한 패턴을 인식한다. 학습은 "시험을 보고 틀린 문제를 복습하는 과정"이다 — 오답(손실)을 줄이는 방향으로 가중치를 조금씩 고친다.

- 퍼셉트론 → 다층 퍼셉트론(MLP): 단층의 한계와 은닉층이 해결하는 것
- 활성화 함수 (ReLU, GELU, Softmax): 왜 비선형성이 필요한가?
  - 비유: 활성화 함수 없는 신경망은 아무리 깊어도 "직선 하나"밖에 못 그린다
- 손실 함수 (Cross-Entropy, MSE): "정답과 얼마나 다른가"를 숫자로 표현
- 경사 하강법과 역전파 알고리즘: 산에서 가장 빨리 내려가는 길 찾기

##### 2.2 PyTorch 모델 개발 패턴

> **라이브 코딩 시연**: 교수가 nn.Module 기반 MLP 모델을 한 줄씩 작성하며, forward() 메서드와 손실 함수 계산 과정을 시연한다.

- `nn.Module`을 활용한 모델 정의: 레고 블록을 클래스로 만들기
- `Dataset`과 `DataLoader`로 데이터 파이프라인 구성
- **GPU 활용 학습 패턴**: `model.to(device)`, `data.to(device)`로 모델과 데이터를 GPU로 이동
- 옵티마이저 (SGD, Adam, AdamW): 산을 내려가는 전략의 차이
- 학습률 스케줄러 (CosineAnnealing, ReduceLROnPlateau): 처음엔 크게, 나중엔 세밀하게

##### 2.3 모델 학습과 평가

**직관적 이해**: 과적합은 "시험 기출문제만 외운 학생"과 같다. 기출은 100점이지만 새 문제를 못 푼다. Dropout은 "랜덤으로 뉴런을 쉬게 하여" 특정 뉴런에 의존하지 않게 만드는 기법이다.

- Training/Validation Loop 구현 패턴
- 과적합 방지: Dropout, Weight Decay, Early Stopping
- 분류 평가 지표: Accuracy, Precision, Recall, F1-Score
  - 비유: 암 진단에서 Precision은 "양성 판정 중 실제 양성 비율", Recall은 "실제 양성 중 잡아낸 비율"
- Confusion Matrix: 모델이 어디서 헷갈리는지 한눈에 보기
- 학습 과정 시각화 (Loss/Accuracy Curve)

---

### B회차: MLP 텍스트 분류 실습

#### 수업 타임라인

| 시간 | 내용 |
|------|------|
| 00:00~00:05 | 2인1조 편성 + 과제 안내 |
| 00:05~00:10 | A회차 핵심 리캡 + Copilot 가이드 |
| 00:10~00:55 | Copilot 활용 조별 구현 (체크포인트 3개) |
| 00:55~01:00 | Google Classroom 제출 |
| 01:00~01:20 | 조별 결과 공유 + 토론 |
| 01:20~01:28 | 교수 종합 피드백 |
| 01:28~01:30 | 다음 주 예고 |

> **Copilot 활용**: "PyTorch nn.Module로 3층 MLP 텍스트 분류 모델을 작성해줘"로 시작하고, Copilot이 생성한 forward() 메서드의 각 층이 하는 역할을 분석한다.

##### 2.4 실습

- 텍스트 전처리 (토큰화, 어휘 사전, 패딩)
- Bag-of-Words / TF-IDF 벡터화
- MLP 기반 감성 분석 모델 구현
- 하이퍼파라미터 튜닝 실험

**과제**: IMDb 영화 리뷰 감성 분류 MLP 모델 구현 + 성능 분석 보고서

---

## 3주차: 시퀀스 모델에서 Transformer로

> **미션**: 수업이 끝나면 Attention이 문장의 어디에 집중하는지 시각화한다

### A회차: 임베딩 + RNN → Attention 원리

#### 수업 타임라인

| 시간 | 내용 |
|------|------|
| 00:00~00:05 | 오늘의 질문 + 빠른 진단 |
| 00:05~00:55 | 이론 강의 |
| 00:55~01:25 | 라이브 코딩 시연 |
| 01:25~01:28 | 핵심 정리 + B회차 과제 스펙 공개 |
| 01:28~01:30 | Exit ticket |

##### 3.1 순차 데이터와 단어 임베딩

**직관적 이해**: "왕 - 남자 + 여자 = 여왕"이 성립하는 마법 같은 공간이 임베딩이다. 단어를 사전의 번호(one-hot)로 표현하면 "개"와 "강아지"가 전혀 다른 숫자이지만, 임베딩 공간에서는 가까이 위치한다. Word2Vec은 "비슷한 문맥에 등장하는 단어는 비슷한 의미를 갖는다"는 직관을 수학으로 구현한 것이다.

- 순차 데이터의 특성과 표현 방법
- Word2Vec (CBOW, Skip-gram) 원리: 주변 단어로 중심 단어 예측 / 중심에서 주변 예측
- 사전학습 임베딩 활용 (GloVe, FastText)
- 임베딩 공간의 의미적 특성과 시각화

##### 3.2 RNN/LSTM/GRU (개념 중심)

**직관적 이해**: RNN은 "기억력 있는 신경망"이다. 일반 신경망이 한 장의 사진만 보는 것이라면, RNN은 영화의 장면을 순서대로 보면서 이전 내용을 기억한다. 그러나 영화가 길어지면 앞부분을 잊는다(장기 의존성 문제). LSTM은 이를 해결한 "선택적 기억장치"로, 중요한 정보는 오래 기억하고 불필요한 정보는 잊는다.

- RNN의 구조와 장기 의존성 문제
- LSTM: Cell State(컨베이어 벨트), Forget/Input/Output Gate(정보 선별 밸브)
- GRU: LSTM의 간소화 버전 (Gate 2개로 축소)
- Seq2Seq 모델과 Encoder-Decoder 구조
- **RNN 계열의 한계**: 순차 처리(병렬화 불가), 긴 문맥 처리의 어려움

##### 3.3 Attention 메커니즘

**직관적 이해**: 시험 공부할 때 교과서 전체를 같은 비중으로 읽지 않는다. 중요한 부분에 밑줄을 긋고 더 집중한다. Attention도 마찬가지다. "이 단어를 이해하려면 문장의 어떤 부분에 집중해야 하는가?"를 모델이 스스로 결정한다. Query는 "질문", Key는 "후보 답의 라벨", Value는 "실제 답의 내용"으로 비유할 수 있다.

- Attention의 기본 개념과 동기: 왜 모든 정보를 같은 비중으로 볼 수 없는가
- Query, Key, Value의 이해
- Scaled Dot-Product Attention 수식과 구현
  - Attention(Q, K, V) = softmax(QKᵀ / √dₖ) · V
- Attention Weights의 의미와 해석: 어떤 단어가 어떤 단어에 주목하는가

##### 3.4 Self-Attention과 Multi-Head Attention

**직관적 이해**: Self-Attention은 "문장 안에서 단어들이 서로를 바라보는 것"이다. "나는 은행에서 돈을 찾았다"에서 "은행"이 "돈"에 높은 Attention을 주면 금융기관, "강"에 높은 Attention을 주면 강둑이 된다. Multi-Head는 "여러 관점에서 동시에 바라보기"로, 한 Head는 문법적 관계를, 다른 Head는 의미적 관계를 포착한다.

- Self-Attention: 문장 내 단어 간 관계 모델링 (모든 쌍 비교)
- Multi-Head Attention: 다양한 관점의 Attention 병렬 수행
- Concatenation과 Linear Projection: 여러 관점을 합치는 방법

---

### B회차: Self-Attention 구현 + 시각화

#### 수업 타임라인

| 시간 | 내용 |
|------|------|
| 00:00~00:05 | 2인1조 편성 + 과제 안내 |
| 00:05~00:10 | A회차 핵심 리캡 + Copilot 가이드 |
| 00:10~00:55 | Copilot 활용 조별 구현 (체크포인트 3개) |
| 00:55~01:00 | Google Classroom 제출 |
| 01:00~01:20 | 조별 결과 공유 + 토론 |
| 01:20~01:28 | 교수 종합 피드백 |
| 01:28~01:30 | 다음 주 예고 |

> **Copilot 활용**: Attention Score 계산 코드를 직접 작성해본 뒤, Copilot에게 "이 Attention 구현을 Multi-Head로 확장해줘"와 같이 점진적으로 요청한다.

##### 3.5 실습

- Word2Vec 모델 로드 및 유사도 측정
- Attention Score 직접 계산 (NumPy/PyTorch)
- Self-Attention 메커니즘 단계별 구현
- Attention Weights 시각화

**과제**: Self-Attention 모듈 구현 + Attention 시각화 리포트

---

## 4주차: Transformer 아키텍처 심층 분석

> **미션**: 수업이 끝나면 Transformer Encoder를 밑바닥부터 구현한다

### A회차: Transformer 전체 구조 + 구현 기초

#### 수업 타임라인

| 시간 | 내용 |
|------|------|
| 00:00~00:05 | 오늘의 질문 + 빠른 진단 |
| 00:05~00:55 | 이론 강의 |
| 00:55~01:25 | 라이브 코딩 시연 |
| 01:25~01:28 | 핵심 정리 + B회차 과제 스펙 공개 |
| 01:28~01:30 | Exit ticket |

##### 4.1 Transformer 전체 구조

**직관적 이해**: Transformer는 "동시통역 시스템"과 같다. Encoder는 원문을 깊이 이해하는 역할이고, Decoder는 이해한 내용을 바탕으로 한 단어씩 출력을 생성한다. RNN이 "한 글자씩 순서대로 읽는 사람"이라면, Transformer는 "문장 전체를 한눈에 보고 관계를 파악하는 사람"이다. 이 덕분에 병렬 처리가 가능해 훈련 속도가 비약적으로 빨라졌다.

- "Attention is All You Need" 논문 핵심: RNN 없이 Attention만으로 충분하다
- Encoder 구조: Self-Attention + Feed-Forward + Residual + LayerNorm
- Decoder 구조: Masked Self-Attention + Cross-Attention + Feed-Forward
- Positional Encoding: 순서 정보를 주입하는 방법
  - 비유: Transformer는 단어 순서를 모르기에, "나는 1번, 오늘 2번, 밥을 3번..." 같은 번호표를 붙여준다
  - Sinusoidal vs Learned Positional Encoding

##### 4.2 Transformer Encoder 구현 (PyTorch)

> **라이브 코딩 시연**: 교수가 Encoder Block을 한 줄씩 구현하며 각 구성 요소의 역할을 설명한다.

- Single Encoder Block 구현: Attention → Add & Norm → FFN → Add & Norm
- Multi-Layer Encoder 스택 구성
- Residual Connection: 왜 입력을 출력에 더하는가?
  - 비유: "원본을 보존하면서 변화량만 학습한다" — 새 내용을 배우되 기존 지식은 유지
- Layer Normalization: 각 층의 출력을 안정화
- Feed-Forward Network 구현

##### 4.3 Transformer Decoder 구현

- Causal Masking: 미래 토큰을 못 보게 가리기
  - 비유: 번역할 때 아직 생성하지 않은 단어를 미리 볼 수 없다 (커닝 방지)
- Cross-Attention: Decoder가 Encoder의 출력을 참조하는 메커니즘
- Encoder-Decoder 연결

##### 4.4 Tokenization 심화

**직관적 이해**: "unbelievable"을 통째로 단어로 쓰면 어휘 사전이 폭발한다. 대신 "un" + "believe" + "able"로 쪼개면 적은 조각으로 모든 단어를 표현할 수 있다. BPE는 "가장 자주 함께 등장하는 글자 쌍을 반복 병합"하는 알고리즘이다.

- BPE (Byte Pair Encoding) 알고리즘: 단계별 병합 과정
- WordPiece Tokenization: BERT가 사용하는 방식
- SentencePiece / Unigram Model
- Hugging Face Tokenizer 실습: 토크나이저 간 차이 비교

---

### B회차: Transformer Encoder 구현 실습

#### 수업 타임라인

| 시간 | 내용 |
|------|------|
| 00:00~00:05 | 2인1조 편성 + 과제 안내 |
| 00:05~00:10 | A회차 핵심 리캡 + Copilot 가이드 |
| 00:10~00:55 | Copilot 활용 조별 구현 (체크포인트 3개) |
| 00:55~01:00 | Google Classroom 제출 |
| 01:00~01:20 | 조별 결과 공유 + 토론 |
| 01:20~01:28 | 교수 종합 피드백 |
| 01:28~01:30 | 다음 주 예고 |

> **Copilot 활용**: Encoder Block의 기본 골격을 직접 작성한 뒤, Copilot에게 "LayerNorm과 Residual Connection을 추가해줘"와 같이 부분적으로 확장을 요청한다.

##### 4.5 실습

- Transformer Encoder Block 밑바닥 구현
- Positional Encoding 구현 및 시각화
- 간단한 Transformer 기반 텍스트 분류기 구현 (**GPU 학습으로 속도 체감**)
- Tokenizer 비교 실험 (BPE vs WordPiece vs SentencePiece)

**과제**: Transformer Encoder로 텍스트 분류 모델 구현 + 성능 분석 (GPU/CPU 학습 시간 비교 포함)

---

## 5주차: LLM 아키텍처: BERT와 GPT

> **미션**: 수업이 끝나면 BERT와 GPT를 직접 돌려보고 차이를 체감한다

### A회차: 사전학습 + BERT/GPT 아키텍처

#### 수업 타임라인

| 시간 | 내용 |
|------|------|
| 00:00~00:05 | 오늘의 질문 + 빠른 진단 |
| 00:05~00:55 | 이론 강의 |
| 00:55~01:25 | 라이브 코딩 시연 |
| 01:25~01:28 | 핵심 정리 + B회차 과제 스펙 공개 |
| 01:28~01:30 | Exit ticket |

##### 5.1 사전학습 패러다임

**직관적 이해**: 사전학습은 "의대 본과"와 같고, 파인튜닝은 "전문의 수련"과 같다. BERT/GPT는 인터넷의 방대한 텍스트로 "언어 자체"를 먼저 배운다(사전학습). 이후 특정 태스크(감성 분석, 번역 등)에 맞춰 소량의 데이터로 전문성을 더한다(파인튜닝). 처음부터 전문의를 양성하는 것보다 훨씬 효율적이다.

- Pre-training → Fine-tuning 전략
- Transfer Learning in NLP: 왜 처음부터 학습하지 않는가
- Encoder-only vs Decoder-only vs Encoder-Decoder: 세 가지 설계 철학

##### 5.2 BERT 아키텍처

**직관적 이해**: BERT는 "빈칸 채우기 달인"이다. "나는 오늘 [MASK]에 갔다"에서 빈칸에 들어갈 단어를 맞추는 훈련을 수억 번 반복하면서 언어를 이해하게 된다. 핵심은 "양방향"이다. GPT는 왼쪽에서 오른쪽으로만 읽지만, BERT는 양쪽 문맥을 모두 본다.

- Bidirectional Context: 왼쪽과 오른쪽을 모두 보는 이해력
- 사전학습 목표:
  - MLM (Masked Language Model): 15% 토큰을 가리고 맞추기
  - NSP (Next Sentence Prediction): 두 문장이 이어지는지 판단
- BERT의 Layer 구조 (Token + Segment + Position Embedding)
- BERT-Base (110M) vs BERT-Large (340M) 비교
- BERT 변형: RoBERTa(더 많이 학습), ALBERT(파라미터 공유로 경량화), DistilBERT(지식 증류로 축소), DeBERTa(Attention 개선)

##### 5.3 GPT 아키텍처

**직관적 이해**: GPT는 "소설 이어쓰기 달인"이다. 앞에 쓰인 내용만 보고 다음 단어를 예측하는 훈련을 반복한다. 이 단순한 "다음 단어 예측"을 수조 개의 토큰으로 훈련하면, 번역·요약·코딩까지 할 수 있는 범용 지능이 나타난다 — 이것이 스케일링의 마법이다.

- Autoregressive Language Modeling: Next Token Prediction
- Decoder-only 구조와 Causal Self-Attention (미래를 못 보는 Attention)
- GPT-1 → GPT-2 → GPT-3 → GPT-4 발전 과정: 파라미터 수와 능력의 관계
- 텍스트 생성 전략:
  - Greedy Search: 매번 확률 최대 토큰 선택 (안전하지만 단조로움)
  - Beam Search: 여러 후보를 동시에 추적
  - Top-k Sampling: 상위 k개 중 랜덤 선택 (다양성 확보)
  - Top-p (Nucleus) Sampling: 누적 확률 p까지의 토큰 중 선택
  - Temperature: 확률 분포의 날카로움 조절 (낮으면 보수적, 높으면 창의적)
- Zero-shot / Few-shot / In-Context Learning: 예시만으로 새 태스크 수행

##### 5.4 Hugging Face Transformers 실전

> **라이브 코딩 시연**: 교수가 Pipeline API로 3줄로 감성 분석을 수행하고, AutoModel로 단계별로 확장하는 과정을 보여준다.

- Pipeline API로 빠른 추론: 3줄로 감성 분석, NER, 요약 수행
- AutoModel, AutoTokenizer, AutoConfig: 모델 자동 로드
- **GPU 가속 추론**: `model.to(device)`와 `pipeline(device=0)`으로 대형 모델 추론 속도 확보
- 모델 로드, 추론, 임베딩 추출
- Model Hub 탐색 및 모델 선택 기준

---

### B회차: BERT/GPT 활용 실습

#### 수업 타임라인

| 시간 | 내용 |
|------|------|
| 00:00~00:05 | 2인1조 편성 + 과제 안내 |
| 00:05~00:10 | A회차 핵심 리캡 + Copilot 가이드 |
| 00:10~00:55 | Copilot 활용 조별 구현 (체크포인트 3개) |
| 00:55~01:00 | Google Classroom 제출 |
| 01:00~01:20 | 조별 결과 공유 + 토론 |
| 01:20~01:28 | 교수 종합 피드백 |
| 01:28~01:30 | 다음 주 예고 |

> **Copilot 활용**: Hugging Face Pipeline 코드를 Copilot에게 요청하되, AutoModel vs Pipeline의 차이를 직접 비교한다. "BERT로 감성 분석하는 코드를 AutoModel 방식으로 작성해줘"로 시작한다.

##### 5.5 실습

- BERT Tokenizer 사용법 + 임베딩 추출
- BERT로 감성 분석 / NER / 유사도 계산
- GPT-2 텍스트 생성 + 디코딩 전략 비교 (Temperature, Top-k, Top-p 조합 실험)
- Hugging Face Pipeline 활용 실습

**과제**: BERT 기반 NER 모델 + GPT-2 텍스트 생성기 구현

---

## 6주차: LLM API 활용과 프롬프트 엔지니어링

> **미션**: 수업이 끝나면 LLM API로 나만의 AI 앱 프로토타입을 만든다

### A회차: LLM API + 프롬프트 엔지니어링 기초

#### 수업 타임라인

| 시간 | 내용 |
|------|------|
| 00:00~00:05 | 오늘의 질문 + 빠른 진단 |
| 00:05~00:55 | 이론 강의 |
| 00:55~01:25 | 라이브 코딩 시연 |
| 01:25~01:28 | 핵심 정리 + B회차 과제 스펙 공개 |
| 01:28~01:30 | Exit ticket |

##### 6.1 상용 LLM API 생태계

**직관적 이해**: 직접 자동차 엔진을 만들 필요 없이 택시를 타면 된다. API는 "거대한 LLM 모델을 인터넷으로 빌려 쓰는 것"이다. 내 컴퓨터에 수천억 파라미터 모델을 올릴 수 없지만, API 한 줄 호출로 GPT-4나 Claude를 사용할 수 있다.

- OpenAI API (GPT-4o, o1, o3)
- Anthropic Claude API (Claude 4.5 Sonnet, Claude Opus 4)
- Google Gemini API
- 오픈소스 LLM: Llama 4, Mistral, Qwen, DeepSeek
- API Key 관리, 비용 관리(토큰 단위 과금), Rate Limiting

##### 6.2 프롬프트 엔지니어링

**직관적 이해**: 프롬프트 엔지니어링은 "AI에게 일 잘 시키는 기술"이다. 같은 사람에게 "이거 해줘"와 "당신은 10년 경력의 데이터 분석가입니다. 아래 데이터를 분석해서 3가지 인사이트를 표 형태로 정리해주세요"는 결과가 완전히 다르다. LLM도 마찬가지다.

- 효과적인 프롬프트 작성 원칙: 명확성, 구체성, 구조화
- Zero-shot Prompting: "이 리뷰가 긍정인지 부정인지 분류해줘"
- Few-shot Prompting: "예시를 보여주고 따라하게 하기"
- Chain-of-Thought (CoT) Prompting: "단계별로 생각하게 하기"
  - 비유: 수학 문제에서 "답만 말해"보다 "풀이 과정을 보여줘"라고 하면 정확도가 올라간다
- Self-Consistency, Tree of Thoughts
- System Prompt 설계와 Role Prompting

##### 6.3 Structured Output과 Function Calling

**직관적 이해**: LLM의 출력은 기본적으로 "자유 형식 텍스트"이다. 하지만 프로그램에서 쓰려면 JSON 같은 구조화된 데이터가 필요하다. Structured Output은 "AI 답변을 정해진 형식으로 받는 것"이고, Function Calling은 "AI가 외부 도구(검색, 계산기, DB)를 직접 호출할 수 있게 하는 것"이다.

- JSON Mode / Structured Output
- Function Calling 원리와 구현
- Tool Use 패턴
- 출력 파싱 및 검증

##### 6.4 LLM 평가 기초

- Perplexity: 모델이 다음 단어를 얼마나 잘 예측하는가 (낮을수록 좋다)
- BLEU, ROUGE: 생성된 텍스트가 정답과 얼마나 비슷한가
- LLM-as-a-Judge: LLM이 다른 LLM의 출력을 평가하는 패턴
- 정성적 평가 vs 정량적 평가
- Hallucination 탐지: AI가 그럴듯한 거짓말을 하는 것을 어떻게 잡는가

---

### B회차: API 활용 + 프롬프팅 실습

#### 수업 타임라인

| 시간 | 내용 |
|------|------|
| 00:00~00:05 | 2인1조 편성 + 과제 안내 |
| 00:05~00:10 | A회차 핵심 리캡 + Copilot 가이드 |
| 00:10~00:55 | Copilot 활용 조별 구현 (체크포인트 3개) |
| 00:55~01:00 | Google Classroom 제출 |
| 01:00~01:20 | 조별 결과 공유 + 토론 |
| 01:20~01:28 | 교수 종합 피드백 |
| 01:28~01:30 | 다음 주 예고 |

> **Copilot 활용**: Copilot에게 "OpenAI API로 감성 분석 Function Calling을 구현해줘"와 같이 요청하고, 생성된 코드의 API 버전과 파라미터가 최신인지 검증한다. 이때 Context7 MCP가 있으면 최신 문서를 자동으로 참조할 수 있다.

##### 6.5 실습

- OpenAI / Claude API 호출 및 응답 처리
- 다양한 프롬팅 기법 비교 실험
- Function Calling으로 날씨/검색 도구 연동
- Structured Output으로 데이터 추출

**과제**: 도메인 특화 텍스트 분석 시스템 구축

---

## 7주차: 중간고사

**A회차: 중간고사 대비**
- 신경망 원리 + Transformer 구조 이론 복습
- PyTorch 구현 패턴 실습
- BERT/GPT 비교 분석

**B회차: 중간고사 실시 (객관식)**

**평가 범위** (1~6주차):
- 신경망 원리, PyTorch 패턴
- 임베딩, RNN/LSTM/GRU, Attention, Self-Attention
- Transformer 아키텍처, Positional Encoding
- BERT vs GPT 아키텍처 비교
- 프롬프트 엔지니어링, LLM API 활용

**시험 형식**: 객관식, 80분, Copilot/인터넷 사용 금지

---

## 8주차: 텍스트 속 숨겨진 주제 찾기: 토픽 모델링

> **미션**: 수업이 끝나면 뉴스 기사 수만 건에서 숨겨진 주제를 자동으로 찾아낸다

### A회차: 토픽 모델링 원리 + BERTopic

#### 수업 타임라인

| 시간 | 내용 |
|------|------|
| 00:00~00:05 | 오늘의 질문 + 빠른 진단 |
| 00:05~00:55 | 이론 강의 |
| 00:55~01:25 | 라이브 코딩 시연 |
| 01:25~01:28 | 핵심 정리 + B회차 과제 스펙 공개 |
| 01:28~01:30 | Exit ticket |

##### 8.1 토픽 모델링 개요

**직관적 이해**: 뉴스 기사 10만 건이 있다. 하나씩 읽을 수 없으니, "이 문서들은 대략 어떤 주제들에 대해 이야기하고 있는가?"를 자동으로 파악하는 것이 토픽 모델링이다. 도서관에서 수만 권의 책을 자동으로 분류하는 사서와 같다.

- 토픽 모델링의 정의와 응용 분야
- LDA (Latent Dirichlet Allocation): 확률론적 접근
  - 비유: 각 문서는 여러 주제(토픽)의 혼합물이고, 각 토픽은 단어들의 확률 분포로 정의된다
- 신경망 기반 토픽 모델링의 진화
- BERTopic: 사전학습 임베딩 + 클러스터링을 활용한 현대적 접근

##### 8.2 BERTopic 아키텍처

**직관적 이해**: BERTopic은 "문서를 먼저 BERT로 벡터화한 뒤, 가까운 문서끼리 그룹화하여 주제를 찾는" 방식이다. LDA와 달리 확률 수학 없이, 거리 기반 클러스터링으로 직관적이면서도 강력하다.

- Document Embedding: 각 문서를 BERT로 벡터 변환
- Clustering: UMAP(차원 축소) + HDBSCAN(밀도 기반 클러스터링)
- Topic Representation: 각 클러스터의 핵심 단어 추출 (c-TF-IDF)
- Hierarchical Topic Clustering: 주제 간 관계 파악
- Dynamic Topic Modeling: 시간에 따른 주제 변화 추적

##### 8.3 고급 기능과 시각화

> **라이브 코딩 시연**: 교수가 BERTopic 파이프라인을 처음부터 끝까지 실행하고, 주제 시각화(Topic Distribution, Heatmap) 결과를 보여준다.

- Custom Vectorizer: 도메인 특화 임베딩 활용
- Merging Topics: 유사 주제 병합
- Dynamic Updates: 새 문서로 모델 확장
- Topic Diversity Metrics: 주제의 다양성 평가

---

### B회차: BERTopic 실습 + 결과 해석

#### 수업 타임라인

| 시간 | 내용 |
|------|------|
| 00:00~00:05 | 2인1조 편성 + 과제 안내 |
| 00:05~00:10 | A회차 핵심 리캡 + Copilot 가이드 |
| 00:10~00:55 | Copilot 활용 조별 구현 (체크포인트 3개) |
| 00:55~01:00 | Google Classroom 제출 |
| 01:00~01:20 | 조별 결과 공유 + 토론 |
| 01:20~01:28 | 교수 종합 피드백 |
| 01:28~01:30 | 다음 주 예고 |

> **Copilot 활용**: "BERTopic으로 뉴스 기사 데이터셋의 주제를 추출해줘"로 시작하고, 결과 해석과 시각화 코드를 점진적으로 추가한다.

##### 8.4 실습

- 실제 뉴스 데이터 로드 및 전처리
- BERTopic 모델 생성 및 학습
- 주제 시각화 (barchart, heatmap, network)
- 주제 동적 추적 (시계열 분석)

**과제**: 토픽 모델링 분석 보고서 (주제 해석 + 트렌드 분석 포함)

---

## 9주차: LLM 파인튜닝 (1) — Full Fine-tuning

> **미션**: 수업이 끝나면 나만의 데이터로 BERT 모델을 파인튜닝한다

### A회차: 파인튜닝 원리 + Trainer API

#### 수업 타임라인

| 시간 | 내용 |
|------|------|
| 00:00~00:05 | 오늘의 질문 + 빠른 진단 |
| 00:05~00:55 | 이론 강의 |
| 00:55~01:25 | 라이브 코딩 시연 |
| 01:25~01:28 | 핵심 정리 + B회차 과제 스펙 공개 |
| 01:28~01:30 | Exit ticket |

##### 9.1 파인튜닝의 이해

**직관적 이해**: 사전학습은 "대학 교양과목"이고 파인튜닝은 "전공 심화과정"이다. 기초를 다진 뒤 특정 분야에 깊이를 더하는 방식이 훨씬 효율적이다. 처음부터 모든 것을 학습하려면 수조 토큰이 필요하지만, 파인튜닝은 수천 문장으로 충분하다.

- Full Fine-tuning vs PEFT의 개념적 차이
- 파인튜닝의 학습 곡선: 얼마나 빠르게 성능이 향상되는가
- 과적합 방지: Validation Set 설정, Early Stopping
- Learning Rate Warmup & Decay: 왜 학습률을 점진적으로 조절하는가

##### 9.2 Hugging Face Trainer API

**직관적 이해**: Trainer는 "훈련 코치"와 같다. 모델 정의, 데이터 로드, 학습, 평가, 저장을 모두 자동으로 처리해주므로, 핵심 설정에만 집중할 수 있다.

- Trainer 초기화: model, args, train_dataset, eval_dataset
- TrainingArguments: 학습률, 배치 크기, 에포크, 저장 전략 등
- Datasets 라이브러리: 데이터 로드 및 전처리
- 분산 학습 (Multi-GPU, TPU) 지원
- Checkpoint & Resume: 학습 중단 후 재개

##### 9.3 Fine-tuning 평가와 성능 분석

> **라이브 코딩 시연**: 교수가 작은 데이터셋으로 BERT를 파인튜닝하고, 학습 곡선과 검증 지표를 실시간으로 보여준다.

- 다양한 평가 지표 구성 (정확도, F1, AUC 등)
- Confusion Matrix 기반 오류 분석
- 클래스 불균형 처리 (가중치 조정, Focal Loss)
- 신뢰도 평가 (Confidence Calibration)

---

### B회차: Full Fine-tuning 실습

#### 수업 타임라인

| 시간 | 내용 |
|------|------|
| 00:00~00:05 | 2인1조 편성 + 과제 안내 |
| 00:05~00:10 | A회차 핵심 리캡 + Copilot 가이드 |
| 00:10~00:55 | Copilot 활용 조별 구현 (체크포인트 3개) |
| 00:55~01:00 | Google Classroom 제출 |
| 01:00~01:20 | 조별 결과 공유 + 토론 |
| 01:20~01:28 | 교수 종합 피드백 |
| 01:28~01:30 | 다음 주 예고 |

> **Copilot 활용**: "Hugging Face Trainer로 BERT를 감성 분류 데이터셋에 파인튜닝해줘"로 요청하고, 실행 결과를 기반으로 성능 분석을 추가한다.

##### 9.4 실습

- Datasets 라이브러리로 데이터 로드 및 전처리
- Trainer로 모델 파인튜닝
- 학습 곡선 시각화 및 성능 분석
- 최적 체크포인트 선택

**과제**: 전공 분야 도메인 데이터로 모델 파인튜닝 + 성능 보고서

---

## 10주차: LLM 파인튜닝 (2) — PEFT와 LoRA

> **미션**: 수업이 끝나면 GPU 메모리 1/10로 거대 모델을 파인튜닝한다

### A회차: PEFT/LoRA 원리 + QLoRA

#### 수업 타임라인

| 시간 | 내용 |
|------|------|
| 00:00~00:05 | 오늘의 질문 + 빠른 진단 |
| 00:05~00:55 | 이론 강의 |
| 00:55~01:25 | 라이브 코딩 시연 |
| 01:25~01:28 | 핵심 정리 + B회차 과제 스펙 공개 |
| 01:28~01:30 | Exit ticket |

##### 10.1 PEFT (Parameter-Efficient Fine-Tuning)

**직관적 이해**: 거대 모델을 "고정"하고 "덧붙이는" 작은 어댑터만 학습하는 방식이다. 원본 책은 두지 않고, 스티커 메모만 붙여가며 정보를 추가하는 것 같다. Full Fine-tuning은 모든 페이지를 다시 쓰지만, PEFT는 여백에만 필기한다.

- Adapter Layers: 각 MHA/FFN 후에 작은 병렬 모듈 추가
- Prefix Tuning: 입력 토큰 앞에 학습 가능한 프리픽스 추가
- Prompt Tuning: 소프트 프롬프트 벡터 학습
- LoRA (Low-Rank Adaptation): 가중치 변화를 저랭크 분해로 근사

##### 10.2 LoRA의 수학과 직관

**직관적 이해**: LoRA는 "눈 수술 대신 안경을 끼우는 것"과 같다. 거대 모델(눈)의 구조를 건드리지 않고, 작은 어댑터(안경)로 입출력을 조정한다.

- 원본 가중치: W ∈ ℝ^(m×n)
- LoRA 적용: W' = W + ΔW = W + AB (A ∈ ℝ^(m×r), B ∈ ℝ^(r×n), r << min(m,n))
- 파라미터 축소 효과: 전체 파라미터의 0.1% 학습으로 Full FT와 유사한 성능
- Rank 선택 기준: r값에 따른 성능/메모리 트레이드오프
- LoRA 초기화: Gaussian init for A, Zero init for B (처음엔 추가 효과 없음)

##### 10.3 QLoRA와 양자화

**직관적 이해**: 사진을 "압축"하면 파일 크기가 작아지지만 질은 거의 같다. QLoRA는 모델을 저정밀도(4-bit)로 압축하고, LoRA 어댑터만 고정밀도로 학습한다.

- 4-bit Quantization: 32-bit float → 4-bit int (메모리 1/8 축소)
- NF4 (Normalized Float 4-bit): 오버플로우 방지
- Double Quantization: 스케일 값까지 양자화
- bitsandbytes 라이브러리: GPU에서 효율적인 연산

##### 10.4 LoRA 설정과 최적화

> **라이브 코딩 시연**: 교수가 70B 모델을 8GB GPU에 QLoRA로 파인튜닝하는 과정을 보여준다.

- r (rank) 선택: r=8, 16, 32 중 선택
- lora_alpha: r과의 비율에 따른 스케일링 (보통 lora_alpha = 2*r)
- target_modules: 어느 모듈에 LoRA를 적용할 것인가
- Gradient Checkpointing: 메모리 절약하며 역전파 계산

---

### B회차: LoRA/QLoRA 실습

#### 수업 타임라인

| 시간 | 내용 |
|------|------|
| 00:00~00:05 | 2인1조 편성 + 과제 안내 |
| 00:05~00:10 | A회차 핵심 리캡 + Copilot 가이드 |
| 00:10~00:55 | Copilot 활용 조별 구현 (체크포인트 3개) |
| 00:55~01:00 | Google Classroom 제출 |
| 01:00~01:20 | 조별 결과 공유 + 토론 |
| 01:20~01:28 | 교수 종합 피드백 |
| 01:28~01:30 | 다음 주 예고 |

> **Copilot 활용**: "PEFT와 bitsandbytes로 70B 모델을 QLoRA 파인튜닝해줘"로 요청하고, Full Fine-tuning과의 성능 비교 코드를 추가한다.

##### 10.5 실습

- LoRA 설정 및 적용 (PEFT 라이브러리)
- QLoRA 환경 설정 (bitsandbytes)
- 대형 모델 파인튜닝 (Llama, Mistral 등)
- Full FT vs LoRA vs QLoRA 성능/메모리 비교

**과제**: LoRA vs Full FT 비교 보고서 (성능, 학습 시간, 메모리 포함)

---

## 11주차: RAG 시스템 구축

> **미션**: 수업이 끝나면 도메인 문서 기반 Q&A 시스템을 배포한다

### A회차: RAG 아키텍처 설계 + 파이프라인

#### 수업 타임라인

| 시간 | 내용 |
|------|------|
| 00:00~00:05 | 오늘의 질문 + 빠른 진단 |
| 00:05~00:55 | 이론 강의 |
| 00:55~01:25 | 라이브 코딩 시연 |
| 01:25~01:28 | 핵심 정리 + B회차 과제 스펙 공개 |
| 01:28~01:30 | Exit ticket |

##### 11.1 RAG의 개념과 필요성

**직관적 이해**: RAG는 "오픈북 시험"과 같다. LLM이 모든 것을 암기하고 있을 수 없으니, 필요할 때 관련 문서를 찾아서 함께 LLM에 제공하는 방식이다.

- LLM의 한계: 학습 데이터 이후 정보를 모르고, 폐쇄 지식에 의존
- RAG의 핵심: 검색(Retrieval) + 증강(Augmentation) + 생성(Generation)
- vs Fine-tuning: 파인튜닝은 파라미터에 지식을 저장, RAG는 외부 저장소에서 동적 조회

##### 11.2 RAG 아키텍처

**직관적 이해**: RAG는 도서관 사서와 같다. 사용자의 질문(쿼리)을 이해하고, 관련 책(문서)을 빨리 찾아 사용자와 함께 읽으며 답변을 만든다.

- Document Chunking: 긴 문서를 작은 청크로 분할
- Embedding: 청크와 질문을 벡터화 (BERT, Sentence-Transformers)
- Vector Database: 빠른 유사도 검색 (FAISS, Chroma, Weaviate, Pinecone)
- Retriever: 상위 K개 관련 청크 선택
- Reranker: 검색된 청크의 순서 재정렬 (선택사항)
- Prompt Construction: 검색된 컨텍스트 + 질문을 LLM에 전달
- Generator (LLM): 답변 생성

##### 11.3 고급 RAG 기법

> **라이브 코딩 시연**: 교수가 PDF 문서를 벡터 DB에 저장하고, 자연어 질문으로 답변을 생성하는 전체 파이프라인을 구현한다.

- Hybrid Search: 키워드 + 의미 기반 검색 혼합
- Multi-Turn RAG: 대화 맥락 유지
- Query Expansion: 원본 질문을 여러 변형으로 확장
- Chain-of-Thought RAG: 추론 과정을 단계별로 수행

---

### B회차: RAG 시스템 구축 실습

#### 수업 타임라인

| 시간 | 내용 |
|------|------|
| 00:00~00:05 | 2인1조 편성 + 과제 안내 |
| 00:05~00:10 | A회차 핵심 리캡 + Copilot 가이드 |
| 00:10~00:55 | Copilot 활용 조별 구현 (체크포인트 3개) |
| 00:55~01:00 | Google Classroom 제출 |
| 01:00~01:20 | 조별 결과 공유 + 토론 |
| 01:20~01:28 | 교수 종합 피드백 |
| 01:28~01:30 | 다음 주 예고 |

> **Copilot 활용**: "LangChain과 FAISS로 PDF 기반 RAG Q&A 시스템을 만들어줘"로 요청하고, 쿼리 확장과 리랭킹을 추가한다.

##### 11.4 실습

- PDF/텍스트 문서 로드 및 청킹
- Embedding 모델로 벡터화
- Vector DB 구축 (FAISS, Chroma)
- LangChain으로 RAG 파이프라인 구성
- Retrieval 성능 평가

**과제**: 전공 문서 기반 RAG Q&A 시스템 구축 + 성능 평가

---

## 12주차: AI Agent 개발

> **미션**: 수업이 끝나면 자율적으로 태스크를 수행하는 AI Agent를 만든다

### A회차: AI Agent 개념 + LangGraph

#### 수업 타임라인

| 시간 | 내용 |
|------|------|
| 00:00~00:05 | 오늘의 질문 + 빠른 진단 |
| 00:05~00:55 | 이론 강의 |
| 00:55~01:25 | 라이브 코딩 시연 |
| 01:25~01:28 | 핵심 정리 + B회차 과제 스펙 공개 |
| 01:28~01:30 | Exit ticket |

##### 12.1 AI Agent의 개념

**직관적 이해**: Agent는 "자율적으로 생각하고 행동하는 AI"이다. 사용자의 명령을 받으면, 스스로 문제를 분해하고, 필요한 도구(검색, 계산, DB 조회)를 호출하며, 결과를 통합하여 최종 답변을 만든다.

- Reactive Agent: 각 스텝에서 다음 행동을 결정
- Agentic Loop: Thought → Action → Observation → (반복)
- Tool Use: 외부 함수/API를 호출하는 능력
- Agent vs Chatbot: Agent는 목표 지향적, Chatbot은 대화 지향적

##### 12.2 ReAct (Reasoning + Acting)

**직관적 이해**: ReAct는 "생각하고 행동하고 관찰하기"를 반복한다. 학생이 숙제를 할 때, 문제를 읽고(Thought), 계산기를 쓰고(Action), 결과를 확인하고(Observation) 다시 생각하는 것처럼.

- Thought: 현재 상황 이해, 다음 단계 계획
- Action: 도구 호출 (웹 검색, 파이썬 실행 등)
- Observation: 행동의 결과 확인
- 반복 종료: 최종 답변 도출

##### 12.3 Tool Design과 Integration

> **라이브 코딩 시연**: 교수가 웹 검색, 계산, DB 조회 도구를 정의하고, Agent가 이를 자동으로 조합하여 복잡한 질문에 답하는 과정을 보여준다.

- Tool 정의: 함수 이름, 파라미터, 설명
- Tool Calling: LLM이 JSON 형식으로 도구 호출 요청
- Error Handling: 도구 실행 오류 처리
- Tool Composition: 여러 도구를 순차적/병렬적으로 조합

##### 12.4 LangGraph를 이용한 Agent 개발

- StateGraph: Agent의 상태 정의 (메시지, 도구 결과 등)
- Node & Edge: 각 스텝과 전환 조건
- Conditional Logic: 도구 호출 후 분기 결정
- Error Recovery: 오류 시 재시도 전략

---

### B회차: Agent 프로토타입 개발

#### 수업 타임라인

| 시간 | 내용 |
|------|------|
| 00:00~00:05 | 2인1조 편성 + 과제 안내 |
| 00:05~00:10 | A회차 핵심 리캡 + Copilot 가이드 |
| 00:10~00:55 | Copilot 활용 조별 구현 (체크포인트 3개) |
| 00:55~01:00 | Google Classroom 제출 |
| 01:00~01:20 | 조별 결과 공유 + 토론 |
| 01:20~01:28 | 교수 종합 피드백 |
| 01:28~01:30 | 다음 주 예고 |

> **Copilot 활용**: "LangGraph로 웹 검색 + 데이터 분석을 수행하는 Agent를 만들어줘"로 요청하고, ReAct 루프를 추가한다.

##### 12.5 실습

- Tool 함수 정의 (Python 함수 또는 API)
- LangGraph StateGraph 구성
- Agent 루프 실행 및 로그 분석
- 오류 처리 및 재시도 로직

**과제**: 도메인 특화 AI Agent 프로토타입 개발 (2-3개 도구 포함)

---

## 13주차: 모델 배포와 프로덕션

> **미션**: 수업이 끝나면 AI 모델을 FastAPI 서버에 배포한다

### A회차: 배포 원리 + FastAPI/Docker

#### 수업 타임라인

| 시간 | 내용 |
|------|------|
| 00:00~00:05 | 오늘의 질문 + 빠른 진단 |
| 00:05~00:55 | 이론 강의 |
| 00:55~01:25 | 라이브 코딩 시연 |
| 01:25~01:28 | 핵심 정리 + B회차 과제 스펙 공개 |
| 01:28~01:30 | Exit ticket |

##### 13.1 모델 배포의 개념

**직관적 이해**: 학습한 모델을 내 컴퓨터에만 두면 다른 사람이 쓸 수 없다. 배포는 "모델을 24시간 켜진 서버에 올려서, 누구나 인터넷으로 접근할 수 있게 하는 것"이다.

- 로컬 개발 vs 프로덕션 배포
- 스케일링: 동시 요청 처리
- 모니터링: 성능과 오류 추적
- 버전 관리: 모델 업데이트 전략

##### 13.2 FastAPI 기초

**직관적 이해**: FastAPI는 "모델을 HTTP 엔드포인트로 노출하는" 웹 프레임워크다. 클라이언트가 POST 요청으로 입력을 보내면, 서버가 모델 추론을 수행하고 결과를 JSON으로 반환한다.

- GET/POST Endpoint 정의
- Request/Response 모델 (Pydantic)
- 자동 문서화 (Swagger UI)
- 동기 vs 비동기 처리 (async)
- 미들웨어와 예외 처리

##### 13.3 모델 최적화와 서빙

> **라이브 코딩 시연**: 교수가 BERT 모델을 FastAPI 서버로 감싸고, 배치 추론과 동시성 제어를 구현한다.

- 모델 로드 최적화: ONNX Runtime, TensorRT
- 배치 처리: 여러 요청을 모아서 한 번에 추론
- 캐싱: 반복되는 입력에 대한 빠른 응답
- Rate Limiting: 서버 부하 관리

##### 13.4 Docker와 컨테이너화

**직관적 이해**: Docker는 "애플리케이션 + 환경(라이브러리, Python 버전 등)을 담은 상자"를 배포한다. 어떤 컴퓨터에서 열어도 같은 환경에서 실행된다.

- Dockerfile 작성: 이미지 빌드 레시피
- 레이어 구조: 베이스 이미지 → 의존성 설치 → 코드 복사 → 실행
- Image vs Container: 설계도 vs 실행 중인 프로그램
- Docker Compose: 여러 서비스(모델, DB, 프론트엔드) 조합

---

### B회차: FastAPI 배포 + 평가

#### 수업 타임라인

| 시간 | 내용 |
|------|------|
| 00:00~00:05 | 2인1조 편성 + 과제 안내 |
| 00:05~00:10 | A회차 핵심 리캡 + Copilot 가이드 |
| 00:10~00:55 | Copilot 활용 조별 구현 (체크포인트 3개) |
| 00:55~01:00 | Google Classroom 제출 |
| 01:00~01:20 | 조별 결과 공유 + 토론 |
| 01:20~01:28 | 교수 종합 피드백 |
| 01:28~01:30 | 다음 주 예고 |

> **Copilot 활용**: "FastAPI와 Docker로 BERT 모델을 배포하는 전체 코드를 작성해줘"로 요청하고, 성능 테스트 코드를 추가한다.

##### 13.5 실습

- FastAPI 엔드포인트 구현 (입력 검증, 출력 포매팅)
- Dockerfile 작성 및 이미지 빌드
- 로컬 테스트 및 성능 프로파일링
- 배포 (AWS, GCP, Hugging Face Spaces 등)

**과제**: FastAPI 배포 + 성능 평가 리포트 (응답 시간, 처리량 포함)

---

## 14주차: 개인 프로젝트 개발

> **미션**: 수업이 끝나면 완성도 높은 AI 프로토타입을 개인 포트폴리오로 배포한다

### A회차: 프로젝트 워크숍 A — 가이드라인 + 개발 시작

**A회차 내용**:
- 개인 프로젝트 주제 선정 및 스코프 정의
- 데이터 수집 및 전처리 계획
- 모델/시스템 아키텍처 설계
- 개발 로드맵 수립 (14~15주차 일정)

### B회차: 프로젝트 워크숍 B — 개발 + 피드백

**B회차 내용**:
- 실제 개발 진행
- 개별 면담 및 교수 피드백
- 문제 해결 및 최적화
- 배포 준비

**프로젝트 평가 기준**:
- 기술적 구현의 깊이와 적절성 (30%)
- 모델 성능 및 결과 분석 (20%)
- 시스템 완성도 (배포, UI, API) (15%)
- 문제 정의의 명확성과 창의성 (15%)
- 발표 및 의사소통 (10%)
- 코드 품질 및 문서화 (10%)

---

## 15주차: 최종 평가

### A회차: 개인 프로젝트 최종 발표

#### 수업 타임라인

| 시간 | 내용 |
|------|------|
| 00:00~01:30 | 개인 최종 발표 (5~7분 × N명) |

**발표 구성 (5~7분)**:
1. 문제 정의와 동기 (1분)
2. 시스템 아키텍처 및 방법론 (2분)
3. 실험 결과 및 데모 시연 (2분)
4. 한계점 및 향후 과제 (1분)
5. 질의응답 (1분)

### B회차: 기말고사 (객관식)

#### 수업 타임라인

| 시간 | 내용 |
|------|------|
| 00:00~01:20 | 기말고사 (80분) |
| 01:20~01:30 | 정리 및 종강 |

**평가 범위** (8~13주차):
- 토픽 모델링 (BERTopic)
- Full Fine-tuning 원리, Trainer API
- PEFT/LoRA/QLoRA 원리 및 파라미터 계산
- RAG 시스템 설계
- AI Agent 아키텍처 (LangGraph)
- 모델 배포 및 최적화 전략 (FastAPI, Docker)

**시험 형식**: 객관식, 80분, Copilot/인터넷 사용 금지

**개인 프로젝트 제출물** (15주차 A회차 발표 시):
- 프로젝트 보고서 (PDF)
- 소스 코드 (GitHub Repository)
- 발표 자료 (PPT/PDF)
- 배포된 서비스 URL 또는 데모 영상

---

## 평가 방식

| 항목 | 비중 | 비고 |
|------|------|------|
| 중간고사 | 20% | 객관식 (1~6주차 범위) |
| 기말고사 | 20% | 객관식 (8~13주차 범위) |
| 주별 과제 | 30% | 2인1조 협업, 개별 제출, Google Classroom, 약 10회 |
| 개인 프로젝트 | 30% | 14~15주차, 구현 + 배포 + 발표 |

**출석 정책**:
- 결석 1회당 총점에서 **2점 차감**
- 지각 1회당 총점에서 **1점 차감**
- 출석은 별도 가점 없이 감점제로만 운영

**과제 운영**:
- B회차 2인1조 실습은 Copilot 활용 협업으로 진행하되, 제출은 개별로 한다
- 제출물: 코드 + 실행 결과 + 분석 리포트 (Google Classroom)
- 7주차(중간고사), 14~15주차(프로젝트)를 제외한 약 10회 과제

**시험 형식**:
- 중간고사·기말고사 모두 객관식으로 출제
- 시험 시간: 80분
- 시험 중 Copilot/인터넷 사용 금지

---

## 주차별 기술 스택 요약

| 주차 | 핵심 도구 | Copilot 활용 | 난이도 |
|------|----------|-------------|--------|
| 1 | Python, PyTorch, Colab | 환경 설정 보조 | ★★☆☆☆ |
| 2 | PyTorch (nn.Module, DataLoader) | MLP 구현 보조 | ★★★☆☆ |
| 3 | PyTorch, NumPy (Attention 구현) | Attention 확장 | ★★★☆☆ |
| 4 | PyTorch (Transformer 구현) | Encoder Block 확장 | ★★★★☆ |
| 5 | Hugging Face Transformers | Pipeline 코드 생성 | ★★★☆☆ |
| 6 | OpenAI/Claude API, LangChain | API 호출 코드 생성 | ★★☆☆☆ |
| 8 | BERTopic, UMAP, HDBSCAN | 파이프라인 코드 생성 | ★★★☆☆ |
| 9 | HF Trainer, Datasets | Trainer 설정 보조 | ★★★☆☆ |
| 10 | HF PEFT, bitsandbytes | LoRA 설정 보조 | ★★★★☆ |
| 11 | LangChain, FAISS, ChromaDB | RAG 파이프라인 생성 | ★★★☆☆ |
| 12 | LangGraph, OpenAI Agents SDK | Agent 그래프 생성 | ★★★★☆ |
| 13 | FastAPI, Docker, Gradio | API/Docker 코드 생성 | ★★★★☆ |

---

## 참고 자료

### 필수 문서
- [Hugging Face Documentation](https://huggingface.co/docs)
- [PyTorch Documentation](https://pytorch.org/docs)
- [LangChain Documentation](https://python.langchain.com/)
- [PEFT Documentation](https://huggingface.co/docs/peft)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)

### 추천 도서
- "Natural Language Processing with Transformers" (Lewis Tunstall et al.)
- "Dive into Deep Learning" (Aston Zhang et al.) - [온라인 무료](https://d2l.ai/)
- "Build a Large Language Model (From Scratch)" (Sebastian Raschka)

### 주요 논문
- "Attention Is All You Need" (Vaswani et al., 2017)
- "BERT: Pre-training of Deep Bidirectional Transformers" (Devlin et al., 2018)
- "Language Models are Few-Shot Learners" (GPT-3, Brown et al., 2020)
- "LoRA: Low-Rank Adaptation of Large Language Models" (Hu et al., 2021)
- "QLoRA: Efficient Finetuning of Quantized LLMs" (Dettmers et al., 2023)
- "ReAct: Synergizing Reasoning and Acting in Language Models" (Yao et al., 2022)
- "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (Lewis et al., 2020)

### 직관적 이해를 위한 추천 블로그/영상
- "The Illustrated Transformer" (Jay Alammar) — Transformer를 시각적으로 이해
- "The Illustrated BERT" (Jay Alammar) — BERT 구조를 그림으로 설명
- "LoRA Explained" (Lightning AI Blog) — LoRA 원리를 그림으로
- 3Blue1Brown "Neural Networks" 시리즈 — 신경망의 직관적 이해
- Andrej Karpathy "Let's build GPT" — GPT를 처음부터 구현하며 이해

---

## 기대 효과

본 강의를 수료한 학생은 다음 역량을 갖추게 된다:

1. **LLM 아키텍처 이해**: Transformer, BERT, GPT의 내부 구조를 구현 수준으로 이해
2. **효율적 파인튜닝**: LoRA/QLoRA로 제한된 자원에서 대형 모델 커스터마이징
3. **RAG 시스템 구축**: 도메인 특화 지식 기반 Q&A 시스템 설계 및 구현
4. **AI Agent 개발**: LLM 기반 자율적 태스크 수행 시스템 개발
5. **모델 배포**: FastAPI + Docker로 모델을 프로덕션 서비스로 배포
6. **AI 도구 활용**: VS Code + Copilot을 활용한 효율적 개발 능력
7. **포트폴리오**: GitHub에 공개 가능한 엔드-투-엔드 AI 프로젝트 경험
