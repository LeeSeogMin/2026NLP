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
- **1-2교시**: Copilot 없이 이론과 원리를 먼저 학습
- **3교시**: Copilot을 활용하여 실습 코드 작성 + 디버깅

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

---

## 수업 운영 구조 (3교시제)

매 수업은 3교시로 운영하며, 교시별 역할은 다음과 같다:

| 교시 | 시간 | 내용 | Copilot 사용 |
|------|------|------|-------------|
| **1교시** | 00:00~00:50 | 이론 + 직관적 비유 | 사용 안 함 |
| 쉬는시간 | 00:50~01:00 | | |
| **2교시** | 01:00~01:50 | 라이브 코딩 시연 + 코드 분석 | 교수 시연 시 선택 사용 |
| 쉬는시간 | 01:50~02:00 | | |
| **3교시** | 02:00~02:50 | Copilot 활용 실습 + 과제 | 적극 사용 |

---

## 15주 운영 계획

| 주차 | 주제 | 과제 (수업 중 제출) |
|:---:|------|-------------------|
| 1 | AI 시대의 개막과 개발 환경 구축 | GPU 환경 구축 + PyTorch Tensor 과제 |
| 2 | 딥러닝 핵심 원리와 PyTorch 실전 | IMDb 감성 분류 MLP 모델 + 분석 |
| 3 | 시퀀스 모델에서 Transformer로 | Self-Attention 모듈 구현 + 시각화 |
| 4 | Transformer 아키텍처 심층 분석 | Transformer Encoder 텍스트 분류 |
| 5 | LLM 아키텍처: BERT와 GPT | BERT NER + GPT-2 텍스트 생성기 |
| 6 | LLM API 활용과 프롬프트 엔지니어링 | 도메인 특화 텍스트 분석 시스템 |
| **7** | **중간고사** | **이론 + 코딩 + 서술** |
| 8 | 토픽 모델링 | 토픽 모델링 분석 보고서 |
| 9 | LLM 파인튜닝 (1) — Full Fine-tuning | 전공 분야 모델 파인튜닝 |
| 10 | LLM 파인튜닝 (2) — PEFT와 LoRA | LoRA vs Full FT 비교 보고서 |
| 11 | RAG 시스템 구축 | 전공 문서 기반 RAG Q&A 시스템 |
| 12 | AI Agent 개발 | 도메인 특화 AI Agent 프로토타입 |
| 13 | 모델 배포와 프로덕션 | FastAPI 배포 + 평가 리포트 |
| 14 | 최종 프로젝트 개발 | 프로젝트 프로토타입 완성 |
| **15** | **기말고사 + 프로젝트 발표** | **종합 평가 + 발표** |

---

## 1주차: AI 시대의 개막과 개발 환경 구축

> **미션**: 수업이 끝나면 PyTorch로 첫 번째 딥러닝 모델을 돌려본다

### 학습목표

1. AI, 머신러닝, 딥러닝의 관계를 설명할 수 있다
2. PyTorch 개발 환경을 구축하고 GPU를 활용할 수 있다
3. Tensor 연산과 Autograd 자동 미분을 이해하고 구현할 수 있다
4. VS Code + GitHub Copilot 실습 환경을 설정할 수 있다

### 수업 타임라인

| 시간 | 구분 | 내용 |
|------|------|------|
| 00:00~00:50 | **1교시** | AI/ML/DL 개요 + NLP 발전사 |
| 00:50~01:00 | 쉬는시간 | |
| 01:00~01:50 | **2교시** | 개발 환경 구축 + Python 딥러닝 기초 |
| 01:50~02:00 | 쉬는시간 | |
| 02:00~02:50 | **3교시** | Copilot 환경 설정 + Tensor 실습 + 과제 |

---

#### 1교시: AI와 자연어처리 개요

##### 1.1 인공지능과 자연어처리 개요

**직관적 이해**: AI, 머신러닝, 딥러닝은 러시아 인형(마트료시카) 같다. 가장 큰 인형이 AI, 그 안에 머신러닝, 그 안에 딥러닝이 들어있다. ChatGPT가 "다음에 올 말"을 예측하는 단순한 원리로 어떻게 그렇게 똑똑해 보이는지 그 비밀을 풀어본다.

- AI, 머신러닝, 딥러닝의 관계와 각각의 역할
- 자연어처리(NLP)의 정의와 주요 응용 분야 (번역, 챗봇, 감성 분석, 요약 등)
- 언어 모델의 발전사: 통계 → 신경망 → Transformer → LLM
- 2024-2026 NLP 산업 동향과 취업 시장: 어떤 역량이 필요한가

---

#### 2교시: 개발 환경과 PyTorch 기초

##### 1.2 개발 환경 구축
- Python 가상환경 설정 (Anaconda/venv)
- PyTorch 설치 및 GPU 환경 확인
- VS Code 설치 + GitHub Copilot / Copilot Chat 확장 설치
- Hugging Face 생태계 소개 (Hub, Transformers, Datasets)
- Google Colab / Kaggle GPU 환경 설정
- Git/GitHub 기본 사용법

##### 1.3 Python 딥러닝 기초

**직관적 이해**: Tensor는 "숫자를 담는 다차원 상자"이다. 스칼라(점) → 벡터(선) → 행렬(면) → 텐서(공간)로 차원이 늘어나는 것을 시각적으로 이해한다. Autograd는 "수학 시험에서 풀이 과정을 자동으로 역추적하는 기능"이다.

- NumPy 행렬 연산 복습
- Tensor 기본 조작 (생성, 연산, 인덱싱, GPU 이동)
- Autograd를 이용한 자동 미분: 왜 미분이 필요한가, 어떻게 자동화되는가

---

#### 3교시: Copilot 활용 실습

> **Copilot 활용**: Copilot에게 "PyTorch Tensor를 생성하고 GPU로 이동하는 코드를 작성해줘"와 같은 프롬프트로 시작하되, 생성된 코드의 각 줄이 무엇을 하는지 반드시 이해한다.

##### 1.4 실습
- GitHub Copilot Pro 학생 무료 등록 + VS Code 설정
- GPU 환경 구축 및 확인
- PyTorch Tensor 연산 실습
- 간단한 선형 회귀 모델 구현 (Autograd 활용)

**과제**: 개발 환경 구축 + PyTorch Tensor 조작 과제 제출

---

## 2주차: 딥러닝 핵심 원리와 PyTorch 실전

> **미션**: 수업이 끝나면 텍스트 분류 모델을 직접 학습시킨다

### 수업 타임라인

| 시간 | 구분 | 내용 |
|------|------|------|
| 00:00~00:50 | **1교시** | 신경망 기본 구조 + 역전파 |
| 00:50~01:00 | 쉬는시간 | |
| 01:00~01:50 | **2교시** | PyTorch 모델 개발 패턴 + 학습/평가 |
| 01:50~02:00 | 쉬는시간 | |
| 02:00~02:50 | **3교시** | MLP 텍스트 분류 실습 + 과제 |

---

#### 1교시: 신경망 기본 구조

##### 2.1 신경망 기본 구조

**직관적 이해**: 신경망은 "레고 블록 조립"과 같다. 뉴런 하나는 단순한 판단기(입력 × 가중치 → 활성화)이지만, 이를 수백만 개 쌓으면 복잡한 패턴을 인식한다. 학습은 "시험을 보고 틀린 문제를 복습하는 과정"이다 — 오답(손실)을 줄이는 방향으로 가중치를 조금씩 고친다.

- 퍼셉트론 → 다층 퍼셉트론(MLP): 단층의 한계와 은닉층이 해결하는 것
- 활성화 함수 (ReLU, GELU, Softmax): 왜 비선형성이 필요한가?
  - 비유: 활성화 함수 없는 신경망은 아무리 깊어도 "직선 하나"밖에 못 그린다
- 손실 함수 (Cross-Entropy, MSE): "정답과 얼마나 다른가"를 숫자로 표현
- 경사 하강법과 역전파 알고리즘: 산에서 가장 빨리 내려가는 길 찾기

---

#### 2교시: PyTorch 모델 개발과 평가

##### 2.2 PyTorch 모델 개발 패턴
- `nn.Module`을 활용한 모델 정의: 레고 블록을 클래스로 만들기
- `Dataset`과 `DataLoader`로 데이터 파이프라인 구성
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

#### 3교시: 텍스트 분류 실습

> **Copilot 활용**: "PyTorch nn.Module로 3층 MLP 텍스트 분류 모델을 작성해줘"로 시작하고, Copilot이 생성한 forward() 메서드의 각 층이 하는 역할을 분석한다.

##### 2.4 실습: 텍스트 분류 파이프라인
- 텍스트 전처리 (토큰화, 어휘 사전, 패딩)
- Bag-of-Words / TF-IDF 벡터화
- MLP 기반 감성 분석 모델 구현
- 하이퍼파라미터 튜닝 실험

**과제**: IMDb 영화 리뷰 감성 분류 MLP 모델 구현 + 성능 분석 보고서

---

## 3주차: 시퀀스 모델에서 Transformer로

> **미션**: 수업이 끝나면 Attention이 문장의 어디에 집중하는지 시각화한다

### 수업 타임라인

| 시간 | 구분 | 내용 |
|------|------|------|
| 00:00~00:50 | **1교시** | 임베딩 + RNN/LSTM 개념 |
| 00:50~01:00 | 쉬는시간 | |
| 01:00~01:50 | **2교시** | Attention + Self-Attention + Multi-Head |
| 01:50~02:00 | 쉬는시간 | |
| 02:00~02:50 | **3교시** | Attention 구현 + 시각화 실습 + 과제 |

---

#### 1교시: 순차 데이터와 시퀀스 모델

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

---

#### 2교시: Attention 메커니즘

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

#### 3교시: Attention 구현 실습

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

### 수업 타임라인

| 시간 | 구분 | 내용 |
|------|------|------|
| 00:00~00:50 | **1교시** | Transformer 전체 구조 + Positional Encoding |
| 00:50~01:00 | 쉬는시간 | |
| 01:00~01:50 | **2교시** | Encoder/Decoder 구현 + Tokenization |
| 01:50~02:00 | 쉬는시간 | |
| 02:00~02:50 | **3교시** | Transformer 구현 실습 + 과제 |

---

#### 1교시: Transformer 전체 구조

##### 4.1 Transformer 전체 구조

**직관적 이해**: Transformer는 "동시통역 시스템"과 같다. Encoder는 원문을 깊이 이해하는 역할이고, Decoder는 이해한 내용을 바탕으로 한 단어씩 출력을 생성한다. RNN이 "한 글자씩 순서대로 읽는 사람"이라면, Transformer는 "문장 전체를 한눈에 보고 관계를 파악하는 사람"이다. 이 덕분에 병렬 처리가 가능해 훈련 속도가 비약적으로 빨라졌다.

- "Attention is All You Need" 논문 핵심: RNN 없이 Attention만으로 충분하다
- Encoder 구조: Self-Attention + Feed-Forward + Residual + LayerNorm
- Decoder 구조: Masked Self-Attention + Cross-Attention + Feed-Forward
- Positional Encoding: 순서 정보를 주입하는 방법
  - 비유: Transformer는 단어 순서를 모르기에, "나는 1번, 오늘 2번, 밥을 3번..." 같은 번호표를 붙여준다
  - Sinusoidal vs Learned Positional Encoding

---

#### 2교시: Transformer 구현과 Tokenization

> **라이브 코딩 시연**: 교수가 Encoder Block을 한 줄씩 구현하며 각 구성 요소의 역할을 설명한다.

##### 4.2 Transformer Encoder 구현 (PyTorch)
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

#### 3교시: Transformer 구현 실습

> **Copilot 활용**: Encoder Block의 기본 골격을 직접 작성한 뒤, Copilot에게 "LayerNorm과 Residual Connection을 추가해줘"와 같이 부분적으로 확장을 요청한다.

##### 4.5 실습
- Transformer Encoder Block 밑바닥 구현
- Positional Encoding 구현 및 시각화
- 간단한 Transformer 기반 텍스트 분류기 구현
- Tokenizer 비교 실험 (BPE vs WordPiece vs SentencePiece)

**과제**: Transformer Encoder로 텍스트 분류 모델 구현 + 성능 분석

---

## 5주차: LLM 아키텍처: BERT와 GPT

> **미션**: 수업이 끝나면 BERT와 GPT를 직접 돌려보고 차이를 체감한다

### 수업 타임라인

| 시간 | 구분 | 내용 |
|------|------|------|
| 00:00~00:50 | **1교시** | 사전학습 패러다임 + BERT 아키텍처 |
| 00:50~01:00 | 쉬는시간 | |
| 01:00~01:50 | **2교시** | GPT 아키텍처 + Hugging Face 실전 |
| 01:50~02:00 | 쉬는시간 | |
| 02:00~02:50 | **3교시** | BERT/GPT 활용 실습 + 과제 |

---

#### 1교시: 사전학습과 BERT

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

---

#### 2교시: GPT와 Hugging Face

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
- Pipeline API로 빠른 추론: 3줄로 감성 분석, NER, 요약 수행
- AutoModel, AutoTokenizer, AutoConfig: 모델 자동 로드
- 모델 로드, 추론, 임베딩 추출
- Model Hub 탐색 및 모델 선택 기준

---

#### 3교시: BERT/GPT 활용 실습

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

### 수업 타임라인

| 시간 | 구분 | 내용 |
|------|------|------|
| 00:00~00:50 | **1교시** | 상용 LLM API + 프롬프트 엔지니어링 |
| 00:50~01:00 | 쉬는시간 | |
| 01:00~01:50 | **2교시** | Structured Output + Function Calling + 평가 |
| 01:50~02:00 | 쉬는시간 | |
| 02:00~02:50 | **3교시** | API 활용 실습 + 과제 |

---

#### 1교시: LLM API와 프롬프트 엔지니어링

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

---

#### 2교시: Structured Output과 LLM 평가

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

#### 3교시: API 활용 실습

> **Copilot 활용**: Copilot에게 "OpenAI API로 감성 분석 Function Calling을 구현해줘"와 같이 요청하고, 생성된 코드의 API 버전과 파라미터가 최신인지 검증한다. 이때 Context7 MCP가 있으면 최신 문서를 자동으로 참조할 수 있다.

##### 6.5 실습
- OpenAI / Claude API 호출 및 응답 처리
- 다양한 프롬프팅 기법 비교 실험
- Function Calling으로 날씨/검색 도구 연동
- Structured Output으로 데이터 추출

**프로젝트**: 프롬프트 엔지니어링으로 도메인 특화 텍스트 분석 시스템 구축

---

## 7주차: 중간고사

**평가 내용**
- 이론: 신경망 원리, Transformer 구조, BERT/GPT 비교, Attention 메커니즘
- 코딩: PyTorch 모델 구현, Attention 계산, Hugging Face 활용
- 서술: 아키텍처 비교 분석, 프롬프트 엔지니어링 전략 설계

---

## 8주차: 텍스트 속 숨겨진 주제 찾기: 토픽 모델링

> **미션**: 수업이 끝나면 뉴스 기사 수만 건에서 숨겨진 주제를 자동으로 찾아낸다

### 수업 타임라인

| 시간 | 구분 | 내용 |
|------|------|------|
| 00:00~00:50 | **1교시** | 토픽 모델링 개요 + BERTopic 아키텍처 |
| 00:50~01:00 | 쉬는시간 | |
| 01:00~01:50 | **2교시** | 토픽 표현 + 고급 기능 시연 |
| 01:50~02:00 | 쉬는시간 | |
| 02:00~02:50 | **3교시** | BERTopic 실습 + 과제 |

---

#### 1교시: 토픽 모델링의 이해

##### 8.1 토픽 모델링 개요

**직관적 이해**: 뉴스 기사 10만 건이 있다. 하나씩 읽을 수 없으니, "이 문서들은 대략 어떤 주제들에 대해 이야기하고 있는가?"를 자동으로 파악하는 것이 토픽 모델링이다. 도서관에서 수만 권의 책을 자동으로 분류하는 사서와 같다.

- 토픽 모델링의 정의와 목적
- 전통적 방법: LDA (Latent Dirichlet Allocation) 원리와 한계
  - 비유: LDA는 "각 문서는 여러 주제의 혼합, 각 주제는 여러 단어의 혼합"이라고 가정
- Transformer 시대의 토픽 모델링: 의미 기반 접근

##### 8.2 BERTopic 아키텍처

**직관적 이해**: BERTopic은 5단계 파이프라인으로, 각 단계가 명확한 역할을 한다. ① 문서를 벡터로 변환(의미 파악) → ② 고차원 벡터를 2~3차원으로 압축(시각화 가능하게) → ③ 가까운 문서끼리 묶기(클러스터링) → ④ 각 클러스터의 대표 키워드 추출 → ⑤ 선택적 미세 조정

- 5단계 파이프라인: Embedding → UMAP → HDBSCAN → c-TF-IDF → Fine-tuning
- Sentence Transformers (all-MiniLM-L6-v2, BGE 등)
- UMAP 차원 축소: 고차원 → 저차원, 가까운 점은 가까이 유지
- HDBSCAN 밀도 기반 클러스터링: 밀도 높은 영역을 자동 탐지
- c-TF-IDF: "이 클러스터에서 특히 많이 등장하는 단어"를 대표 키워드로

---

#### 2교시: 토픽 표현과 고급 기능

> **라이브 코딩 시연**: 교수가 BERTopic 파이프라인을 단계별로 실행하며 각 단계의 중간 결과를 시각화한다.

##### 8.3 토픽 표현과 고급 기능
- 토픽별 키워드 추출 및 레이블링
- Outlier 처리: 어디에도 속하지 않는 문서 다루기
- 토픽 간 유사도 분석
- Dynamic Topic Modeling: 시간에 따라 토픽이 어떻게 변하는가
- Guided / Hierarchical Topic Modeling
- LLM을 활용한 토픽 레이블 생성: "키워드 나열" 대신 자연어 이름 자동 부여

---

#### 3교시: BERTopic 실습

> **Copilot 활용**: Copilot에게 "BERTopic으로 한국어 뉴스 기사를 토픽 모델링하는 코드를 작성해줘"라고 요청하고, 임베딩 모델과 UMAP 파라미터를 직접 조정하며 결과 변화를 관찰한다.

##### 8.4 실습
- BERTopic으로 뉴스 기사 토픽 모델링
- 토픽 시각화 (Intertopic Distance Map, Topic Hierarchy)
- 시간별 토픽 트렌드 분석
- LLM 기반 토픽 레이블 자동 생성

**프로젝트**: 자신의 관심 분야 문서로 토픽 모델링 수행 및 분석 보고서

---

## 9주차: LLM 파인튜닝 (1) — Full Fine-tuning

> **미션**: 수업이 끝나면 BERT를 내 데이터로 파인튜닝한다

### 수업 타임라인

| 시간 | 구분 | 내용 |
|------|------|------|
| 00:00~00:50 | **1교시** | 전이 학습 전략 + 데이터셋 준비 |
| 00:50~01:00 | 쉬는시간 | |
| 01:00~01:50 | **2교시** | Trainer API + 학습 모니터링 |
| 01:50~02:00 | 쉬는시간 | |
| 02:00~02:50 | **3교시** | BERT 파인튜닝 실습 + 과제 |

---

#### 1교시: 전이 학습과 데이터 준비

##### 9.1 전이 학습과 파인튜닝 전략

**직관적 이해**: 영어를 잘하는 사람이 스페인어를 배우면, 처음부터 배우는 사람보다 훨씬 빠르다. 이미 "언어란 무엇인가"를 아는 뇌가 있기 때문이다. 전이 학습도 같다. BERT가 이미 "언어"를 이해하고 있으므로, 여기에 소량의 감성 분석 데이터만 추가하면 감성 분석 전문가가 된다.

- Pre-training → Fine-tuning 패러다임
- Feature Extraction(모델 고정, 마지막 층만 학습) vs Fine-tuning(전체 조정)
- Full Fine-tuning vs Partial Fine-tuning
- 파인튜닝 태스크: 분류, NER, QA, 요약, 번역

##### 9.2 데이터셋 준비
- 데이터 수집, 정제, 포맷팅 (CSV, JSON, Parquet)
- Train/Validation/Test Split 전략: 왜 세 개로 나누는가
  - 비유: Train은 교과서, Validation은 모의고사, Test는 수능
- 데이터 불균형 처리 (Oversampling, Class Weights)
- Hugging Face Datasets 라이브러리

---

#### 2교시: Trainer API와 학습 모니터링

> **라이브 코딩 시연**: 교수가 TrainingArguments를 설정하고 Trainer를 실행하며, Loss Curve를 실시간으로 관찰한다.

##### 9.3 Hugging Face Trainer API
- Trainer 클래스와 TrainingArguments
- 핵심 하이퍼파라미터:
  - Learning Rate: 너무 크면 발산, 너무 작으면 수렴 안 함
  - Batch Size: 메모리와 학습 안정성의 트레이드오프
  - Epochs: 데이터를 몇 바퀴 볼 것인가
  - Warmup: 처음에 학습률을 천천히 올려 안정적 출발
- Gradient Accumulation: GPU 메모리가 부족할 때 큰 배치 효과 내기
- Mixed Precision (fp16/bf16): 정밀도를 약간 낮춰 속도 2배
- Compute Metrics 함수 정의
- Callbacks (EarlyStoppingCallback, TensorBoard)

##### 9.4 학습 모니터링과 디버깅
- Loss Curve 분석: 학습이 잘 되고 있는지 판단하는 법
  - Train Loss만 떨어지고 Val Loss가 올라가면 → 과적합
  - 둘 다 안 떨어지면 → 학습률 확인, 데이터 확인
- 과적합 방지: Dropout, Label Smoothing, Weight Decay
- TensorBoard / Weights & Biases 활용
- Gradient Norm Monitoring

---

#### 3교시: 파인튜닝 실습

> **Copilot 활용**: Copilot에게 "Hugging Face Trainer로 BERT 감성 분석 파인튜닝 코드를 작성해줘"라고 요청하고, TrainingArguments의 각 파라미터가 무엇을 의미하는지 분석한다.

##### 9.5 실습
- BERT 파인튜닝으로 금융 뉴스 분류
- 커스텀 Dataset 클래스 작성
- Trainer API 활용 학습 + 시각화
- 파인튜닝 전후 성능 비교 분석

**프로젝트**: 자신의 전공 분야 데이터로 모델 파인튜닝 (1단계)

---

## 10주차: LLM 파인튜닝 (2) — PEFT와 LoRA

> **미션**: 수업이 끝나면 LoRA로 대형 모델을 단일 GPU에서 튜닝한다

### 수업 타임라인

| 시간 | 구분 | 내용 |
|------|------|------|
| 00:00~00:50 | **1교시** | Full FT 한계 + LoRA 원리 |
| 00:50~01:00 | 쉬는시간 | |
| 01:00~01:50 | **2교시** | QLoRA + 기타 PEFT + HF PEFT 라이브러리 |
| 01:50~02:00 | 쉬는시간 | |
| 02:00~02:50 | **3교시** | LoRA 실습 + 비교 실험 + 과제 |

---

#### 1교시: LoRA의 원리

##### 10.1 Full Fine-tuning의 한계

**직관적 이해**: BERT-Large(340M 파라미터)를 파인튜닝하려면 약 4GB GPU 메모리가 필요하다. 그런데 Llama-3 70B는? 280GB가 필요하다. A100 GPU 4장이 있어야 한다. 현실적으로 대부분의 팀은 이런 자원이 없다. "모델 전체를 수정하지 말고, 핵심 부분만 아주 조금 수정하면 어떨까?"라는 질문에서 PEFT가 탄생했다.

- 메모리/계산 비용 구체적 분석 (모델 크기별)
- 대형 모델의 파인튜닝 현실적 어려움
- Parameter-Efficient Fine-Tuning (PEFT) 필요성

##### 10.2 LoRA 심화

**직관적 이해**: LoRA는 "안경"과 같다. 사람의 눈(모델)을 수술(Full Fine-tuning)하는 대신, 안경(LoRA Adapter)을 끼워서 시력을 교정한다. 안경은 작고 가볍고, 필요에 따라 교체할 수 있다. 수학적으로는 "거대한 가중치 행렬의 변화량을 두 개의 작은 행렬(Low-Rank)로 근사"하는 것이다.

- Low-Rank Matrix Decomposition 원리
  - 가중치 변화 ΔW를 직접 학습하면 파라미터가 너무 많다
  - ΔW = B × A (큰 행렬 = 얇은 행렬 두 개의 곱)으로 근사
  - Rank r = 안경의 도수. 높을수록 세밀하지만 파라미터도 늘어남
- LoRA 하이퍼파라미터:
  - Rank (r): 4, 8, 16, 32 중 선택. 보통 8-16이면 충분
  - Alpha (α): 스케일링 팩터. 보통 rank의 2배로 설정
  - Target Modules: 어떤 층에 LoRA를 적용할 것인가
  - Dropout: LoRA 층의 과적합 방지
- LoRA Adapter 구조와 Merging: 학습 후 원래 모델에 합치기

---

#### 2교시: QLoRA와 PEFT 생태계

##### 10.3 QLoRA

**직관적 이해**: LoRA가 "안경"이라면, QLoRA는 "모델 자체를 먼저 압축(양자화)하고 안경을 끼는 것"이다. 70B 모델을 4-bit로 양자화하면 약 35GB → 약 9GB로 줄어든다. 여기에 LoRA를 추가하면 단일 GPU로도 대형 모델 파인튜닝이 가능해진다.

- 양자화(Quantization) 개념: FP32(4바이트) → FP16(2바이트) → INT8(1바이트) → INT4(0.5바이트)
- NormalFloat4 (NF4) 데이터 타입: 정보 손실을 최소화하는 4-bit 표현
- QLoRA = 4-bit Quantization + LoRA
- 메모리 효율성 분석: 모델별 필요 메모리 비교표

##### 10.4 기타 PEFT 기법
- Prefix Tuning: 각 층의 앞에 학습 가능한 벡터를 추가
- Adapter Layers: 기존 층 사이에 작은 병목 층을 삽입
- P-tuning / Prompt Tuning: 입력에 학습 가능한 가상 토큰 추가
- (IA)³: 활성화 값에 학습 가능한 벡터를 곱하기
- PEFT 기법 비교: 정확도, 파라미터 수, 학습 속도 일목 요연 표

##### 10.5 Hugging Face PEFT 라이브러리
- LoraConfig 설정 및 적용
- get_peft_model(): 기존 모델에 LoRA를 입히는 한 줄
- print_trainable_parameters(): 전체 vs 학습 파라미터 수 확인 (99% 감소 확인)
- 모델 저장/로드/Adapter Merging

---

#### 3교시: LoRA 실습

> **Copilot 활용**: LoraConfig 설정과 get_peft_model() 적용을 Copilot이 도와주되, rank/alpha 값의 의미와 trade-off는 학생이 직접 실험하며 체감한다.

##### 10.6 실습
- BERT/GPT-2에 LoRA 적용
- Full Fine-tuning vs LoRA 성능/비용/시간 비교
- Rank/Alpha 값 변경 실험
- QLoRA로 대형 모델 (7B) 튜닝 (Google Colab T4에서 가능)
- Adapter 저장 및 재사용

**프로젝트**: LoRA 파인튜닝 + Full Fine-tuning 성능 비교 보고서

---

## 11주차: RAG 시스템 구축

> **미션**: 수업이 끝나면 내 문서를 검색하고 답변하는 RAG 시스템을 만든다

### 수업 타임라인

| 시간 | 구분 | 내용 |
|------|------|------|
| 00:00~00:50 | **1교시** | LLM 한계 + RAG 파이프라인 설계 |
| 00:50~01:00 | 쉬는시간 | |
| 01:00~01:50 | **2교시** | Vector DB + LangChain + RAG 최적화 |
| 01:50~02:00 | 쉬는시간 | |
| 02:00~02:50 | **3교시** | RAG 시스템 구축 실습 + 과제 |

---

#### 1교시: LLM의 한계와 RAG

##### 11.1 LLM의 한계와 RAG

**직관적 이해**: ChatGPT에게 "우리 회사의 올해 매출은?"이라고 물으면 답을 모른다. 학습 데이터에 없는 정보이기 때문이다. RAG는 이 문제를 해결한다. "먼저 관련 문서를 검색하고, 그 문서를 참고하여 답변을 생성한다." 오픈북 시험과 같은 원리이다.

- Hallucination: AI가 모르는 것을 그럴듯하게 지어내는 현상
- 지식 한계: Training Cutoff 이후 정보 부재
- RAG vs Fine-tuning: 언제 무엇을 선택하는가
  - RAG: 최신 정보, 자주 변하는 데이터, 출처가 중요할 때
  - Fine-tuning: 특정 스타일/형식, 도메인 지식을 모델에 내재화할 때
- RAG의 장점: 최신 정보, 출처 추적 가능, Hallucination 감소

##### 11.2 RAG 파이프라인 설계

**직관적 이해 — 도서관 비유**:
1. **청킹** = 책을 페이지별로 나누기 (너무 크면 검색이 부정확, 너무 작으면 문맥이 끊김)
2. **임베딩** = 각 페이지에 도서 분류 번호 매기기 (의미를 숫자 벡터로 변환)
3. **Vector DB 저장** = 서가에 꽂기
4. **질문 임베딩 + 검색** = 사서에게 질문하면 가장 관련 있는 페이지를 찾아줌
5. **LLM 생성** = 찾은 페이지를 참고하여 답변 작성

- 문서 수집 → 청킹 → 임베딩 → Vector DB → 검색 → 생성
- 청킹 전략: 고정 크기, 문단 기반, 의미 기반 — 각각의 장단점
- Embedding 모델 선택 (OpenAI, Sentence-BERT, BGE)

---

#### 2교시: Vector DB와 LangChain

> **라이브 코딩 시연**: 교수가 PDF → 청킹 → 임베딩 → FAISS 저장 → 질의응답까지 전체 파이프라인을 시연한다.

##### 11.3 Vector Database
- Vector Embedding과 유사도 검색 (Cosine Similarity, L2 Distance)
  - Cosine Similarity: 벡터의 "방향"이 얼마나 비슷한가 (길이 무시)
- FAISS: Meta가 만든 로컬 벡터 검색 라이브러리 (빠르고 무료)
- ChromaDB: 개발 친화적 경량 벡터 DB
- Pinecone, Weaviate: 프로덕션 벡터 DB (관리형 서비스)

##### 11.4 LangChain 프레임워크
- Document Loaders: PDF, HTML, CSV 등 다양한 문서 로드
- Text Splitters: RecursiveCharacterTextSplitter — 청킹 자동화
- Vector Stores: FAISS, ChromaDB 등과 통합
- Retriever + LLM Chain 구성: 검색 → 프롬프트 → 생성 파이프라인
- Conversational RAG: 대화 맥락을 유지하며 검색+생성

##### 11.5 RAG 최적화
- Retrieval 품질 개선: Reranking(검색 결과 재정렬), Hybrid Search(키워드+벡터 결합)
- Chunk 크기/오버랩 최적화
- 프롬프트 템플릿 최적화: "아래 문맥만 참고하여 답변하세요"
- RAG 평가: Faithfulness(사실에 충실한가), Relevance(관련 문서를 잘 찾았는가), Answer Quality

---

#### 3교시: RAG 시스템 구축 실습

> **Copilot 활용**: LangChain의 API가 자주 바뀌므로, Copilot 사용 시 반드시 최신 문서를 확인한다. Context7 MCP가 있으면 최신 LangChain 문서를 자동 참조할 수 있다.

##### 11.6 실습
- FAISS + LangChain으로 문서 기반 Q&A 시스템 구축
- PDF 문서 로드 → 청킹 → 임베딩 → 검색 → 답변 생성
- ChromaDB 활용 버전 구축
- RAG 성능 평가 파이프라인 구현

**프로젝트**: 전공 분야 문서 기반 RAG Q&A 시스템 구축

---

## 12주차: AI Agent 개발

> **미션**: 수업이 끝나면 스스로 생각하고 도구를 사용하는 AI Agent를 만든다

### 수업 타임라인

| 시간 | 구분 | 내용 |
|------|------|------|
| 00:00~00:50 | **1교시** | AI Agent 개요 + Tool Use |
| 00:50~01:00 | 쉬는시간 | |
| 01:00~01:50 | **2교시** | Agent 프레임워크 + 메모리/상태 관리 |
| 01:50~02:00 | 쉬는시간 | |
| 02:00~02:50 | **3교시** | Agent 구축 실습 + 과제 |

---

#### 1교시: AI Agent의 이해

##### 12.1 AI Agent 개요

**직관적 이해**: ChatGPT는 "대화"만 한다. 물어보면 답하지만 직접 "행동"하지 못한다. AI Agent는 다르다. "서울 날씨 검색해서, 비가 오면 우산 알림 보내줘"라고 하면 실제로 날씨 API를 호출하고, 조건을 판단하고, 알림을 보낸다. Agent = **생각하는 두뇌(LLM) + 행동하는 손(Tools) + 기억(Memory) + 계획(Planning)**이다.

- AI Agent의 정의와 유형
- ReAct 패턴: Reasoning(생각) + Acting(행동)을 번갈아 수행
  - "문제를 분석한다(Think) → 도구를 호출한다(Act) → 결과를 본다(Observe) → 다시 생각한다(Think)"
- 단일 Agent vs 멀티 Agent 시스템: 혼자 일하기 vs 팀으로 일하기

##### 12.2 Tool Use와 Function Calling 심화
- Tool 정의 및 스키마 설계: LLM이 도구를 이해할 수 있는 형식
- LLM의 Tool 선택 메커니즘: "이 질문에는 어떤 도구가 필요한가"를 스스로 판단
- 도구 예시: 웹 검색, 코드 실행, 데이터베이스 쿼리, 파일 읽기/쓰기
- 에러 핸들링과 재시도 전략: 도구가 실패했을 때 어떻게 할 것인가

---

#### 2교시: Agent 프레임워크와 메모리

> **라이브 코딩 시연**: 교수가 LangGraph로 상태 기반 Agent를 단계별로 구축하며 ReAct 패턴이 실제로 작동하는 모습을 보여준다.

##### 12.3 Agent 프레임워크
- LangGraph: 상태 기반 Agent 워크플로우
  - 비유: Agent를 "상태 머신(State Machine)"으로 설계 — 각 상태에서 어떤 행동을 할지 정의
- OpenAI Agents SDK
- CrewAI: 멀티 Agent 협업 — "리서처 Agent가 조사하고, 작가 Agent가 글을 쓴다"
- Agent 디자인 패턴: Router(분기), Orchestrator(조율), Evaluator(검증)

##### 12.4 Agent 메모리와 상태 관리
- Short-term Memory: 현재 대화 맥락 (몇 턴 전까지 기억)
- Long-term Memory: 벡터 DB에 저장하는 장기 기억
- 상태 관리와 체크포인팅: 중간에 중단해도 이어서 실행

---

#### 3교시: Agent 구축 실습

> **Copilot 활용**: LangGraph의 그래프 정의 코드를 Copilot이 생성하고, 학생은 노드(행동)와 엣지(분기 조건)를 직접 설계한다.

##### 12.5 실습
- LangGraph로 웹 검색 + 문서 분석 Agent 구축
- 멀티 Tool Agent 구현 (검색 + 계산 + 코드 실행)
- RAG Agent: 문서 검색 기반 질의응답 Agent
- 간단한 멀티 Agent 시스템 프로토타입

**프로젝트**: 도메인 특화 AI Agent 프로토타입 개발

---

## 13주차: 모델 배포와 프로덕션

> **미션**: 수업이 끝나면 내 모델을 API로 배포하고 웹에서 접근한다

### 수업 타임라인

| 시간 | 구분 | 내용 |
|------|------|------|
| 00:00~00:50 | **1교시** | 모델 서빙 + 최적화 |
| 00:50~01:00 | 쉬는시간 | |
| 01:00~01:50 | **2교시** | 배포 인프라 + LLM 평가 + 윤리 |
| 01:50~02:00 | 쉬는시간 | |
| 02:00~02:50 | **3교시** | FastAPI 배포 + Gradio 실습 + 과제 |

---

#### 1교시: 모델 서빙과 최적화

##### 13.1 모델 서빙 기초

**직관적 이해**: 지금까지는 Jupyter Notebook에서 모델을 실행했다. 하지만 실무에서는 "다른 사람이 웹/앱에서 클릭하면 모델이 응답하는" 시스템이 필요하다. 이것이 모델 서빙이다. 레스토랑 비유로, 지금까지 우리는 "주방에서 요리 연습"을 했고, 이제 "손님에게 서빙"하는 단계이다.

- 추론 파이프라인 설계: 입력 → 전처리 → 모델 추론 → 후처리 → 응답
- FastAPI로 모델 API 서버 구축
- 요청/응답 스키마 설계 (Pydantic): 입력/출력 형식을 명확히 정의
- 비동기 처리와 배치 추론: 여러 요청을 효율적으로 처리

##### 13.2 모델 최적화
- 추론 속도 최적화: ONNX Runtime, TensorRT
  - ONNX: 모델을 범용 포맷으로 변환해 최적화된 런타임에서 실행
- 양자화를 통한 경량화 (INT8, INT4): 정밀도를 낮춰 속도와 메모리 절약
- 모델 캐싱 전략: 자주 쓰는 결과를 저장
- GPU vs CPU 추론 트레이드오프: 비용 vs 속도

---

#### 2교시: 배포 인프라와 평가

##### 13.3 배포 인프라

**직관적 이해**: Docker는 "이사할 때 짐을 상자에 포장하는 것"과 같다. 내 컴퓨터에서 되는 코드가 서버에서도 똑같이 되도록, 코드+환경+의존성을 하나의 컨테이너에 넣는다.

- Docker 기초: Dockerfile 작성, 이미지 빌드, 컨테이너 실행
- Gradio / Streamlit으로 데모 UI 구축: 코드 몇 줄로 웹 인터페이스
- Hugging Face Spaces 배포: 무료 호스팅
- 클라우드 배포 개요 (AWS, GCP): 프로덕션 서비스

##### 13.4 LLM 앱 평가 체계
- 자동 평가 vs 인간 평가: 각각의 장단점
- LLM-as-a-Judge 구현: GPT-4가 다른 모델의 출력을 채점
- Hallucination 탐지 파이프라인
- A/B 테스트 기초: 두 버전 중 어느 것이 더 좋은가
- 프로덕션 모니터링: 서비스 운영 중 품질 추적

##### 13.5 윤리와 안전
- Bias와 Fairness: 학습 데이터의 편향이 결과에 미치는 영향
- 프라이버시 (PII 필터링): 개인정보가 모델에 들어가지 않도록
- 책임 있는 AI 개발
- AI 규제 동향 (EU AI Act 등)

---

#### 3교시: 배포 실습

> **Copilot 활용**: FastAPI 라우터 코드와 Dockerfile을 Copilot이 생성하고, 학생은 API 엔드포인트 설계와 Pydantic 스키마를 직접 결정한다.

##### 13.6 실습
- FastAPI로 파인튜닝 모델 API 서버 구축
- Docker로 컨테이너화
- Gradio 데모 UI 연동
- LLM-as-a-Judge 평가 파이프라인 구현

**과제**: 파인튜닝 모델을 API로 배포 + 평가 리포트

---

## 14주차: 최종 프로젝트 개발 및 발표 준비

> **미션**: 수업이 끝나면 최종 프로젝트의 프로토타입이 완성된다

### 수업 타임라인

| 시간 | 구분 | 내용 |
|------|------|------|
| 00:00~00:50 | **1교시** | 프로젝트 가이드라인 + 평가 기준 |
| 00:50~01:00 | 쉬는시간 | |
| 01:00~01:50 | **2교시** | 프로젝트 개발 + 중간 점검 |
| 01:50~02:00 | 쉬는시간 | |
| 02:00~02:50 | **3교시** | 보고서/발표 자료 작성 + 피드백 |

---

#### 1교시: 프로젝트 가이드라인

##### 14.1 프로젝트 가이드라인
- 프로젝트 구조 및 요구사항
- 평가 기준 안내
- 발표 형식 (10-15분)

---

#### 2교시: 프로젝트 개발

> **Copilot 활용**: 프로젝트 전반에서 Copilot Agent 모드를 적극 활용한다. copilot-instructions.md에 프로젝트 기술 스택과 코딩 규칙을 정의하면 일관된 코드 생성이 가능하다.

##### 14.2 프로젝트 개발
- 개인/팀별 프로젝트 집중 개발
- 중간 점검 및 피드백 세션
- 디버깅 및 성능 최적화
- 코드 정리 및 문서화

---

#### 3교시: 보고서 및 발표 자료

##### 14.3 보고서 및 발표 자료 작성
- 정량적/정성적 평가 결과 정리
- Error Analysis: 모델이 틀린 사례를 분석하여 개선 방향 도출
- Ablation Study: 각 구성요소의 기여도 확인 (LoRA 빼면? 청킹 전략 바꾸면?)
- 시각화 (학습 곡선, Confusion Matrix, t-SNE)
- 발표 슬라이드 작성 + 데모 준비

**프로젝트 주제 예시**
1. **도메인 특화 챗봇**: RAG + Agent로 전공 분야 Q&A 시스템
2. **금융 뉴스 감성 분석 API**: LoRA 파인튜닝 + FastAPI 배포
3. **논문 요약/분석 Agent**: 논문 검색 + 요약 + 비교 분석 자동화
4. **다국어 번역 시스템**: 파인튜닝 기반 특화 번역기
5. **코드 리뷰 Agent**: GitHub PR 자동 리뷰 시스템
6. **법률/의료 문서 분석기**: RAG 기반 도메인 특화 검색
7. **실시간 SNS 여론 분석**: 스트리밍 감성 분석 대시보드
8. **AI 튜터**: 교육 콘텐츠 기반 개인 맞춤 학습 Agent

---

## 15주차: 기말고사 및 프로젝트 최종 발표

### 수업 타임라인

| 시간 | 구분 | 내용 |
|------|------|------|
| 00:00~01:20 | **전반부** | 기말고사 |
| 01:20~01:30 | 쉬는시간 | |
| 01:30~02:50 | **후반부** | 프로젝트 최종 발표 |

---

#### 전반부: 기말고사

##### 15.1 기말고사

**이론 평가**
- LLM 아키텍처 (BERT, GPT) 비교 분석
- 파인튜닝 기법 (Full vs PEFT/LoRA) 원리
- LoRA 수학적 원리 및 파라미터 계산
- RAG 시스템 설계
- AI Agent 아키텍처
- 모델 배포 및 최적화 전략

**문제 유형**
- 단답형: 개념 정의, 수식 해석
- 서술형: 아키텍처 설계, 시스템 비교
- 계산 문제: LoRA 파라미터 수, Attention 계산
- 설계 문제: 주어진 요구사항에 맞는 시스템 아키텍처 설계

---

#### 후반부: 프로젝트 최종 발표

##### 15.2 프로젝트 최종 발표

**발표 구성 (10-15분)**
1. 문제 정의와 동기 (2분)
2. 시스템 아키텍처 및 방법론 (4분)
3. 실험 결과 및 분석 (5분)
4. 데모 시연 (2분)
5. 한계점 및 향후 과제 (1분)
6. 질의응답 (3분)

**평가 기준**
- 기술적 구현의 깊이와 적절성 (30%)
- 모델 성능 및 결과 분석 (20%)
- 시스템 완성도 (배포, UI, API) (15%)
- 문제 정의의 명확성과 창의성 (15%)
- 발표 및 의사소통 (10%)
- 코드 품질 및 문서화 (10%)

**제출물**
- 프로젝트 보고서 (PDF)
- 소스 코드 (GitHub Repository)
- 발표 자료 (PPT/PDF)
- 배포된 서비스 URL 또는 데모 영상

---

## 평가 방식

| 항목 | 비중 | 비고 |
|------|------|------|
| 주차별 과제 | 20% | 8-10회 실습 과제 |
| 중간고사 | 25% | 이론 + 코딩 |
| 기말고사 | 20% | 종합 이론 평가 |
| 최종 프로젝트 | 35% | 구현 + 배포 + 발표 |

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
