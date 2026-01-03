# 딥러닝 자연어처리 (2025) - 상세 강의계획서

## 인문사회과학을 위한 AI 언어 모델의 이론과 응용: LLM 파인튜닝 중심

---

## 1주차: AI 시대의 개막과 개발 환경 준비

### 1.1 인공지능의 이해
- 인공지능(AI)의 정의와 역사적 발전
- 인공지능의 세 가지 수준: 약한 AI, 강한 AI, 초지능
- AI, 머신러닝, 딥러닝의 관계와 차이점
- 지도학습, 비지도학습, 강화학습 개요

### 1.2 자연어처리와 언어 모델
- 자연어처리(NLP)란 무엇인가
- NLP의 주요 응용 분야 (번역, 챗봇, 감성 분석 등)
- 언어 모델의 개념과 중요성
- 언어 모델의 발전 과정: 통계 → 신경망 → Transformer → LLM

### 1.3 AI 개발 생태계
- 주요 딥러닝 프레임워크 소개 (TensorFlow, PyTorch, JAX)
- Hugging Face 생태계 소개
- 클라우드 GPU 서비스 (Google Colab, Kaggle, AWS)

### 1.4 실습: 개발 환경 구축
- Python 설치 및 가상환경 설정 (Anaconda/Miniconda)
- 필수 라이브러리 설치 (NumPy, Pandas, Matplotlib)
- PyTorch 설치 및 GPU 환경 확인
- Google Colab 사용법 및 GPU/TPU 설정
- Jupyter Notebook 기본 사용법

**과제**: 개인 개발 환경 구축 및 간단한 Python 코드 실행 스크린샷 제출

---

## 2주차: 언어 모델의 진화: 통계에서 신경망까지

### 2.1 언어 모델의 기초
- 언어 모델의 역할과 목적
- 조건부 확률과 언어 모델
- 언어 모델 평가 지표: Perplexity

### 2.2 통계 기반 언어 모델
- N-gram 모델의 원리
- Unigram, Bigram, Trigram 모델
- 희소성 문제(Sparsity Problem)와 스무딩(Smoothing) 기법
- N-gram 모델의 한계점

### 2.3 신경망 기반 언어 모델로의 전환
- 신경망 언어 모델의 등장 배경
- 분산 표현(Distributed Representation)의 개념
- Word Embedding의 필요성

### 2.4 단어 임베딩
- Word2Vec (CBOW, Skip-gram)
- GloVe (Global Vectors)
- FastText
- 임베딩 공간의 의미적 특성

### 2.5 실습
- N-gram 모델 직접 구현
- 간단한 텍스트 생성 실습
- 텍스트 데이터 전처리 (토큰화, 정제, 불용어 제거)
- 사전 학습된 Word2Vec 모델 로드 및 단어 유사도 측정

**과제**: 뉴스 기사 데이터로 Bigram 모델 구현 및 Perplexity 계산

---

## 3주차: 딥러닝의 핵심: 신경망과 학습 원리

### 3.1 인공 신경망의 기본 구조
- 생물학적 뉴런과 인공 뉴런
- 퍼셉트론(Perceptron)의 구조와 한계
- 다층 퍼셉트론(MLP)과 은닉층
- 가중치(Weights)와 편향(Bias)의 역할

### 3.2 활성화 함수
- 활성화 함수의 필요성
- Sigmoid, Tanh 함수
- ReLU, Leaky ReLU, GELU
- 출력층 활성화 함수 (Softmax)

### 3.3 손실 함수
- 손실 함수의 개념
- 회귀 문제: MSE, MAE
- 분류 문제: Cross-Entropy Loss
- 손실 함수와 최적화의 관계

### 3.4 최적화 알고리즘
- 경사 하강법(Gradient Descent)의 원리
- Batch GD vs Mini-batch GD vs Stochastic GD
- 역전파(Backpropagation) 알고리즘
- Chain Rule과 미분 계산

### 3.5 실습: PyTorch 기초
- Tensor 기본 조작 (생성, 연산, 인덱싱)
- Autograd를 이용한 자동 미분
- 간단한 선형 회귀 모델 구현
- MLP 모델 직접 설계 및 학습

**과제**: MNIST 손글씨 숫자 분류를 위한 MLP 모델 구현

---

## 4주차: PyTorch 기반 딥러닝 모델 개발 프로세스

### 4.1 PyTorch 핵심 구성 요소
- `torch.nn.Module`을 활용한 모델 정의
- `torch.nn`의 주요 레이어 (Linear, Conv2d, etc.)
- 모델 파라미터 관리 및 초기화

### 4.2 데이터 처리 파이프라인
- `torch.utils.data.Dataset` 클래스
- `torch.utils.data.DataLoader` 활용
- 데이터 전처리와 증강(Augmentation)
- 배치(Batch) 처리의 이해

### 4.3 옵티마이저와 학습률 스케줄러
- 다양한 옵티마이저 (SGD, Adam, AdamW)
- 학습률(Learning Rate)의 중요성
- 학습률 스케줄러 (StepLR, ReduceLROnPlateau, CosineAnnealing)

### 4.4 모델 학습 루프
- Training Loop 구현
- Validation Loop 구현
- 과적합(Overfitting)과 일반화(Generalization)
- 정규화 기법 (L1, L2, Dropout, Batch Normalization)

### 4.5 모델 평가
- 분류 문제 평가 지표 (Accuracy, Precision, Recall, F1-Score)
- Confusion Matrix
- 학습 과정 시각화 (Loss Curve, Accuracy Curve)

### 4.6 실습: 텍스트 분류
- IMDb 영화 리뷰 데이터셋 로드 및 전처리
- 텍스트를 벡터로 변환 (Bag-of-Words, TF-IDF)
- MLP 기반 감성 분석 모델 구현
- 모델 학습 및 성능 평가
- 하이퍼파라미터 튜닝

**과제**: 다양한 하이퍼파라미터 조합으로 모델 성능 비교 분석

---

## 5주차: 순차 데이터 처리: RNN과 LSTM/GRU

### 5.1 순차 데이터의 이해
- 순차 데이터(Sequential Data)의 특성
- 시계열 데이터, 자연어, 오디오 신호
- Feedforward 네트워크의 한계

### 5.2 순환 신경망(RNN)
- RNN의 기본 구조와 작동 원리
- Hidden State의 개념
- RNN의 순전파(Forward Propagation)
- BPTT (Backpropagation Through Time)

### 5.3 RNN의 문제점
- 장기 의존성(Long-term Dependency) 문제
- 기울기 소실(Vanishing Gradient)
- 기울기 폭주(Exploding Gradient)

### 5.4 LSTM (Long Short-Term Memory)
- LSTM의 구조: Cell State, Hidden State
- Forget Gate, Input Gate, Output Gate
- LSTM의 정보 흐름
- LSTM이 장기 의존성을 해결하는 방법

### 5.5 GRU (Gated Recurrent Unit)
- GRU의 구조와 LSTM과의 차이
- Reset Gate, Update Gate
- LSTM vs GRU: 언제 무엇을 사용할 것인가

### 5.6 Sequence-to-Sequence 모델
- Encoder-Decoder 구조
- 기계 번역에의 적용
- Seq2Seq의 한계점

### 5.7 실습
- PyTorch로 RNN/LSTM/GRU 구현
- 간단한 시퀀스 데이터 학습
- 문자 단위(Character-level) LSTM 언어 모델
- 텍스트 생성 실습

**과제**: 셰익스피어 텍스트로 Character-level LSTM 모델 학습 및 텍스트 생성

---

## 6주차: 혁신의 중심: Transformer 아키텍처

### 6.1 Transformer 등장 배경
- RNN/LSTM의 한계 (순차 처리, 병렬화 어려움)
- "Attention is All You Need" 논문 소개
- Transformer의 혁신성

### 6.2 Attention 메커니즘
- Attention의 기본 개념
- Query, Key, Value의 이해
- Attention Score 계산
- Scaled Dot-Product Attention
- Attention Weights의 의미

### 6.3 Self-Attention
- Self-Attention의 개념
- 문장 내 단어 간 관계 모델링
- Self-Attention 계산 과정
- Self-Attention의 장점

### 6.4 Multi-Head Attention
- Multi-Head Attention의 필요성
- 여러 관점에서의 Attention
- Concatenation과 Linear Projection
- Head 수 결정

### 6.5 Positional Encoding
- Transformer에서의 위치 정보
- Sinusoidal Positional Encoding
- Learned Positional Encoding

### 6.6 Transformer 구조
- Encoder의 구조 (Self-Attention + Feed-Forward)
- Decoder의 구조 (Masked Self-Attention + Cross-Attention + Feed-Forward)
- Residual Connection과 Layer Normalization
- Encoder-Decoder Attention

### 6.7 실습
- Self-Attention 메커니즘 단계별 구현
- Attention Weights 시각화
- 간단한 Transformer Encoder 블록 구현
- Positional Encoding 구현 및 시각화

**과제**: Transformer Encoder로 간단한 텍스트 분류 모델 구현

---

## 7주차: 중간고사

**평가 내용**
- 이론 평가 (AI 기초, 신경망, RNN/LSTM, Transformer 원리)
- 코딩 실습 (PyTorch 기본, 모델 구현, 데이터 처리)
- 서술형 문제 (Attention 메커니즘 설명, RNN vs Transformer 비교)

---

## 8주차: 텍스트 속 숨겨진 주제 찾기: 임베딩 기반 토픽 모델링

### 8.1 토픽 모델링 개요
- 토픽 모델링의 정의와 목적
- 전통적 토픽 모델링: LDA (Latent Dirichlet Allocation)
- LDA의 원리와 한계점
- Transformer 시대의 토픽 모델링

### 8.2 BERTopic 소개
- BERTopic의 등장 배경
- BERTopic vs LDA
- BERTopic의 강점과 활용 분야

### 8.3 BERTopic 아키텍처
- 5단계 파이프라인 이해
  1. Document Embedding (Sentence-BERT)
  2. Dimensionality Reduction (UMAP)
  3. Clustering (HDBSCAN)
  4. Topic Representation (c-TF-IDF)
  5. Fine-tuning (Optional)

### 8.4 주요 구성 요소 심화
- Sentence Transformers: all-MiniLM-L6-v2, BGE, etc.
- UMAP: 차원 축소의 원리와 하이퍼파라미터
- HDBSCAN: 밀도 기반 클러스터링
- c-TF-IDF: 클래스 기반 TF-IDF의 이해

### 8.5 토픽 표현 및 해석
- 토픽별 주요 키워드 추출
- 토픽 레이블링 및 명명
- Outlier 처리 방법
- 토픽 간 유사도 분석

### 8.6 고급 기능
- Dynamic Topic Modeling (시간에 따른 토픽 변화)
- Guided Topic Modeling (시드 단어 활용)
- Hierarchical Topic Modeling
- LLM을 활용한 토픽 레이블 생성

### 8.7 실습
- BERTopic 라이브러리 설치 및 기본 사용법
- 뉴스 기사 데이터셋으로 토픽 모델링
- 토픽 시각화 (Intertopic Distance Map, Topic Hierarchy)
- 시간별 토픽 트렌드 분석
- 토픽 모델 저장 및 로드

**프로젝트**: 자신의 관심 분야 문서(논문, 뉴스, SNS 등)로 토픽 모델링 수행 및 분석 보고서 작성

---

## 9주차: LLM 시대 (1) - BERT 아키텍처와 활용

### 9.1 사전학습 언어 모델의 패러다임
- Pre-training과 Fine-tuning 전략
- Transfer Learning in NLP
- GPT vs BERT: Decoder-only vs Encoder-only
- BERT의 혁신성

### 9.2 BERT 아키텍처 심층 분석
- Encoder-only 구조의 의미
- Bidirectional Context Understanding
- BERT의 Layer 구조
  - Embedding Layer (Token + Segment + Position)
  - Multi-Layer Transformer Encoders
  - Pooling Layer
- BERT-Base vs BERT-Large (파라미터 비교)

### 9.3 BERT 사전학습
- Masked Language Model (MLM)
  - Masking 전략 (15% masking rule)
  - [MASK] 토큰의 역할
  - MLM Loss 계산
- Next Sentence Prediction (NSP)
  - NSP의 목적과 한계
  - [CLS], [SEP] 토큰
- 사전학습 데이터셋 (BooksCorpus, Wikipedia)

### 9.4 BERT Tokenization
- WordPiece Tokenization
- Subword Tokenization의 장점
- Special Tokens ([CLS], [SEP], [MASK], [PAD], [UNK])
- Tokenizer 실습

### 9.5 BERT 변형 모델들
- RoBERTa: BERT의 개선 버전
- ALBERT: 파라미터 효율성
- DistilBERT: 지식 증류(Knowledge Distillation)
- DeBERTa: Disentangled Attention

### 9.6 Hugging Face Transformers 라이브러리
- Hugging Face Hub 소개
- Pipeline API: 쉬운 시작
- AutoModel, AutoTokenizer, AutoConfig
- 사전 학습된 모델 로드 및 추론

### 9.7 BERT 활용 사례
- Text Classification (감성 분석, 주제 분류)
- Named Entity Recognition (NER)
- Question Answering
- Sentence Similarity
- Text Embedding 추출

### 9.8 실습
- Hugging Face Transformers 기본 사용법
- Pipeline으로 빠른 추론 (Sentiment Analysis)
- BERT Tokenizer 사용법
- 사전 학습된 BERT 모델로 텍스트 분류
- BERT로 개체명 인식 (NER)
- BERT 임베딩 추출 및 유사도 계산

**프로젝트**: 전공 분야 텍스트 데이터로 BERT 기반 분류 모델 구축

---

## 10주차: LLM 시대 (2) - GPT 아키텍처와 생성 모델

### 10.1 자기회귀 언어 모델
- Autoregressive Language Modeling
- Next Token Prediction의 개념
- GPT의 등장과 의의

### 10.2 GPT 아키텍처
- Decoder-only 구조
- Causal Self-Attention (Masked Self-Attention)
- GPT vs BERT: 단방향 vs 양방향
- GPT-1, GPT-2, GPT-3의 발전 과정

### 10.3 GPT 아키텍처 상세
- Transformer Decoder 블록
- Causal Masking의 원리
- Layer 구성 및 파라미터 수
- GPT-2 구조 분석 (124M, 355M, 774M, 1.5B)

### 10.4 텍스트 생성 메커니즘
- Greedy Search
- Beam Search
- Sampling 기법
  - Temperature Sampling
  - Top-k Sampling
  - Top-p (Nucleus) Sampling
- 생성 전략 비교 및 선택

### 10.5 GPT의 능력
- Zero-shot Learning
- Few-shot Learning
- In-Context Learning
- Emergent Abilities (창발적 능력)

### 10.6 프롬프트 엔지니어링 기초
- 프롬프트의 중요성
- 효과적인 프롬프트 작성법
- Instruction Following
- Chain-of-Thought Prompting 소개

### 10.7 텍스트 생성 평가
- Perplexity
- BLEU Score
- ROUGE Score
- 정성적 평가의 중요성

### 10.8 실습
- GPT-2 모델 로드 및 텍스트 생성
- 다양한 디코딩 전략 비교 실험
- Temperature, Top-k, Top-p 파라미터 조정
- Zero-shot, Few-shot 프롬프팅 실습
- 텍스트 완성 및 요약 실습
- 생성 품질 평가

**프로젝트**: 도메인 특화 텍스트 생성기 프로토타입 개발 (예: 논문 초록 생성, 뉴스 기사 생성)

---

## 11주차: LLM 파인튜닝 (1) - 전이 학습과 Full Fine-tuning

### 11.1 전이 학습(Transfer Learning)
- 전이 학습의 개념과 필요성
- 컴퓨터 비전에서의 전이 학습
- NLP에서의 전이 학습
- Pre-training vs Fine-tuning

### 11.2 파인튜닝 전략
- Feature Extraction (특징 추출)
- Fine-tuning (미세 조정)
- Full Fine-tuning vs Partial Fine-tuning
- Layer-wise Fine-tuning

### 11.3 파인튜닝 태스크
- Sequence Classification (문장/문서 분류)
- Token Classification (NER, POS Tagging)
- Question Answering
- Summarization
- Translation

### 11.4 데이터셋 준비
- 데이터 수집 및 정제
- 데이터 포맷 (CSV, JSON, Parquet)
- Train/Validation/Test Split
- 데이터 불균형 처리
- Hugging Face Datasets 라이브러리 활용

### 11.5 Hugging Face Trainer API
- Trainer 클래스의 구조
- TrainingArguments 설정
  - Learning Rate, Batch Size, Epochs
  - Evaluation Strategy
  - Logging and Checkpointing
- Compute Metrics 함수 정의
- Callbacks 활용

### 11.6 하이퍼파라미터 튜닝
- Learning Rate 선택
- Batch Size의 영향
- Warmup Steps
- Weight Decay
- Gradient Accumulation

### 11.7 과적합 방지
- Early Stopping
- Dropout
- Label Smoothing
- Data Augmentation for NLP
- Regularization Techniques

### 11.8 학습 모니터링
- Loss Curve 분석
- Validation Metrics 추적
- TensorBoard / Weights & Biases 활용
- Gradient Norm Monitoring

### 11.9 실습
- 금융 뉴스 분류를 위한 BERT 파인튜닝
- Hugging Face Datasets로 데이터 로드
- Tokenization 및 데이터 전처리
- Trainer API를 활용한 모델 학습
- 학습 과정 시각화 및 분석
- 검증 데이터로 모델 평가
- 파인튜닝 전후 성능 비교

**프로젝트**: 자신의 전공 분야 데이터로 특화 모델 파인튜닝 (1단계)

---

## 12주차: LLM 파인튜닝 (2) - PEFT와 효율적 튜닝

### 12.1 Full Fine-tuning의 한계
- 메모리 요구사항
- 계산 비용
- 학습 시간
- 모델 배포의 어려움

### 12.2 Parameter-Efficient Fine-Tuning (PEFT)
- PEFT의 개념과 필요성
- PEFT 방법론 분류
  - Additive Methods (Adapter, Prefix Tuning)
  - Selective Methods (BitFit)
  - Reparameterization Methods (LoRA)
  - Hybrid Methods

### 12.3 LoRA (Low-Rank Adaptation) 심화
- LoRA의 핵심 아이디어
- Low-Rank Matrix Decomposition
- LoRA의 수학적 원리
  - Weight Matrix Update: ΔW = BA
  - Rank r의 의미
- LoRA Adapter 구조

### 12.4 LoRA 하이퍼파라미터
- Rank (r): 학습 파라미터 수 결정
- Alpha (α): Scaling Factor
- Target Modules 선택
  - Query, Key, Value Projection
  - Attention vs Feed-Forward
- LoRA Dropout
- Bias 처리 방식

### 12.5 QLoRA (Quantized LoRA)
- 양자화(Quantization)의 개념
- 4-bit Quantization
- NormalFloat4 (NF4) 데이터 타입
- QLoRA = Quantization + LoRA
- QLoRA의 메모리 효율성

### 12.6 기타 PEFT 기법
- Prefix Tuning: 학습 가능한 Prefix 추가
- Adapter Layers: 병목 레이어 삽입
- P-tuning / Prompt Tuning
- (IA)³: Infused Adapter

### 12.7 PEFT 성능 분석
- Full Fine-tuning vs LoRA 비교
  - 정확도
  - 학습 시간
  - 메모리 사용량
  - 파라미터 수
- Rank 값 변화에 따른 성능
- Target Modules 선택의 영향

### 12.8 Hugging Face PEFT 라이브러리
- PEFT 라이브러리 설치 및 구조
- LoraConfig 설정
- get_peft_model() 함수
- print_trainable_parameters()로 파라미터 확인
- 모델 저장 및 로드
- Adapter Merging

### 12.9 실습
- LoRA Config 설정 및 적용
- BERT/GPT-2 모델에 LoRA 적용
- Full Fine-tuning vs LoRA 성능 비교
- 학습 파라미터 수 비교 (99% 감소 확인)
- Rank와 Alpha 값 실험
- QLoRA를 활용한 대형 모델 튜닝
- 다양한 Target Modules 조합 실험
- Adapter 저장 및 재사용

**프로젝트**: 자신의 전공 분야 데이터로 LoRA 파인튜닝 (2단계) 및 Full Fine-tuning과의 성능 비교 보고서

---

## 13주차: LLM 고급 응용 - RAG와 프롬프트 엔지니어링

### 13.1 LLM의 한계
- Hallucination (환각) 문제
- 지식의 한계 (Training Cutoff Date)
- 도메인 특화 지식 부족
- 실시간 정보 부재
- 출처 검증의 어려움

### 13.2 검색 증강 생성 (RAG) 개요
- RAG의 정의와 목적
- RAG vs Fine-tuning: 언제 무엇을 사용할 것인가
- RAG의 장점
  - 최신 정보 활용
  - 도메인 지식 주입
  - Hallucination 감소
  - 출처 추적 가능

### 13.3 RAG 시스템 구성 요소
- Retriever: 관련 문서 검색
- Generator: 답변 생성
- Knowledge Base: 문서 저장소

### 13.4 Vector Database 기초
- Vector Embedding의 개념
- 유사도 검색 (Similarity Search)
- FAISS (Facebook AI Similarity Search)
- ChromaDB
- Pinecone, Weaviate 소개

### 13.5 RAG 파이프라인
1. 문서 수집 및 청킹 (Chunking)
2. 임베딩 생성
3. Vector DB에 저장
4. 쿼리 임베딩
5. 유사 문서 검색 (Top-k Retrieval)
6. Context와 함께 LLM에 전달
7. 답변 생성

### 13.6 LangChain 소개
- LangChain 프레임워크 개요
- Document Loaders
- Text Splitters
- Vector Stores
- Retrievers
- Chains

### 13.7 프롬프트 엔지니어링 심화
- Zero-shot Prompting
- Few-shot Prompting (In-Context Learning)
- Chain-of-Thought (CoT) Prompting
- Self-Consistency
- Tree of Thoughts
- System Prompt 설계
- Role Prompting

### 13.8 API 기반 LLM 활용
- OpenAI API (GPT-4, GPT-3.5)
- Anthropic Claude API
- Google Gemini API
- API Key 관리 및 보안
- Rate Limiting 및 비용 관리
- Streaming Response

### 13.9 실습
- LangChain 기본 사용법
- 문서 로드 및 청킹
- FAISS Vector Database 구축
- 간단한 RAG 시스템 구현
  - 문서 기반 Q&A
- 프롬프트 엔지니어링 실험
  - Zero-shot vs Few-shot 비교
  - CoT Prompting 실습
- OpenAI/Claude API 활용
  - API 호출 및 응답 처리
  - 프롬프트 최적화

**프로젝트 (선택)**: 자신의 전공 분야 문서로 RAG 기반 Q&A 시스템 구축

---

## 14주차: 최종 프로젝트 개발 및 발표 준비

### 14.1 프로젝트 발표 가이드라인
- 프로젝트 구조 및 요구사항
- 평가 기준 안내
- 발표 시간 및 형식

### 14.2 모델 평가 및 보고서 작성
- 정량적 평가 지표 선택 및 계산
- 정성적 분석 방법
- Error Analysis (오류 분석)
- Ablation Study
- 보고서 작성 템플릿

### 14.3 시각화 및 결과 해석
- 학습 곡선 시각화
- Confusion Matrix
- Attention Visualization
- t-SNE/UMAP을 통한 임베딩 시각화
- 결과 해석 및 인사이트 도출

### 14.4 실무 적용 시 고려사항
- 모델 배포 전략
- 추론 속도 최적화
- 모델 경량화
- API 서버 구축 기초
- 윤리적 고려사항 (Bias, Fairness, Privacy)

### 14.5 프로젝트 개발
- 개인/팀별 프로젝트 개발
- 중간 점검 및 피드백 세션
- 문제 해결 및 디버깅
- 코드 정리 및 문서화

### 14.6 발표 자료 준비
- 슬라이드 작성 가이드
- 데모 준비
- 스토리텔링

**프로젝트 주제 예시**
1. **정책 문서 요약 생성기**: 긴 정책 문서를 핵심 요약문으로 변환
2. **다국어 번역 및 요약**: 외국 문헌 번역 후 핵심 내용 요약
3. **SNS 여론 분석 대시보드**: 트위터, 뉴스 댓글 등의 감성 분석
4. **FAQ 자동 응답 시스템**: 전공 분야 질문에 자동 답변
5. **역사 문헌 개체명 인식**: 역사 문서에서 인물, 장소, 사건 추출
6. **법률 문서 분석기**: 판례, 법령 문서 분석 및 유사 판례 검색
7. **의료 기록 분류 시스템**: 환자 기록 자동 분류 및 요약
8. **금융 뉴스 감성 분석**: 뉴스 기사 기반 시장 심리 분석
9.  **소설/시나리오 자동 생성기**: 창작 텍스트 생성

---

## 15주차: 기말고사 및 프로젝트 최종 발표

### 15.1 기말고사 (전반부)
**이론 평가**
- LLM 아키텍처 (BERT, GPT)
- 파인튜닝 기법 (Full Fine-tuning vs PEFT)
- LoRA 원리 및 하이퍼파라미터
- RAG 시스템 구성
- 프롬프트 엔지니어링

**문제 유형**
- 단답형: 개념 정의, 수식 해석
- 서술형: 아키텍처 비교, 기법 설명
- 계산 문제: LoRA 파라미터 수 계산
- 사례 분석: 적절한 방법론 선택

### 15.2 프로젝트 최종 발표 (후반부)
**발표 구성 (10-15분)**
1. 문제 정의 (2분)
   - 해결하고자 하는 문제
   - 데이터셋 소개
2. 방법론 (4분)
   - 모델 선택 이유
   - 전처리 과정
   - 파인튜닝/PEFT 전략
3. 실험 및 결과 (5분)
   - 실험 설계
   - 정량적 결과
   - 정성적 분석
4. 결론 및 향후 과제 (2분)
   - 한계점
   - 개선 방향
5. 질의응답 (3분)

**평가 기준**
- 문제 정의의 명확성 (15%)
- 기술적 구현의 적절성 (25%)
- 모델 성능 및 결과 분석 (25%)
- 창의성 및 실용성 (20%)
- 발표 및 의사소통 (10%)
- 코드 품질 및 문서화 (5%)

**제출물**
- 프로젝트 보고서 (PDF)
- 소스 코드 (GitHub Repository)
- 발표 자료 (PPT/PDF)
- (선택) 데모 동영상

---

## 주요 변경사항 요약

### 제거된 내용
1. **7주차: 시계열 분석** - LLM 중심 강의로 재편
2. **13주차: LangGraph Agent** - 고급 주제로 과도한 난이도

### 강화된 내용
1. **9-10주차**: BERT와 GPT 아키텍처 각 1주씩 심화
2. **11-12주차**: 파인튜닝 이론과 PEFT 각 1주씩 강화
3. **각 주차**: 구체적인 세부 목차와 실습 내용 명시

### 교육 철학
- **"넓고 얕게"에서 "좁고 깊게"로**
- **이론 중심에서 프로젝트 중심으로**
- **핵심 기술(파인튜닝) 마스터에 집중**

---

## 평가 방식

- **과제 (20%)**: 주차별 실습 과제 (8-10회)
- **중간고사 (30%)**: 이론 및 코딩 시험
- **기말고사 (20%)**: 종합 이론 평가
- **최종 프로젝트 (30%)**: 개인/팀 LLM 응용 프로젝트

---

## 참고 자료

### 필수 문서
- [Hugging Face Documentation](https://huggingface.co/docs)
- [PyTorch Documentation](https://pytorch.org/docs)
- [BERTopic Documentation](https://maartengr.github.io/BERTopic/)
- [PEFT Documentation](https://huggingface.co/docs/peft)

### 추천 도서
- "Natural Language Processing with Transformers" (Lewis Tunstall et al.)
- "Dive into Deep Learning" (Aston Zhang et al.) - [온라인 무료](https://d2l.ai/)
- "Speech and Language Processing" (Dan Jurafsky, James H. Martin) - [온라인](https://web.stanford.edu/~jurafsky/slp3/)

### 온라인 강좌
- [Hugging Face NLP Course](https://huggingface.co/learn/nlm-course)
- [Stanford CS224N: NLP with Deep Learning](http://web.stanford.edu/class/cs224n/)
- [Fast.ai NLP Course](https://www.fast.ai/)

### 주요 논문
- "Attention Is All You Need" (Vaswani et al., 2017)
- "BERT: Pre-training of Deep Bidirectional Transformers" (Devlin et al., 2018)
- "Language Models are Few-Shot Learners" (GPT-3, Brown et al., 2020)
- "LoRA: Low-Rank Adaptation of Large Language Models" (Hu et al., 2021)
- "QLoRA: Efficient Finetuning of Quantized LLMs" (Dettmers et al., 2023)

---

## 기대 효과

본 강의를 수료한 학생은 다음 역량을 갖추게 된다:

1. **LLM 아키텍처 이해**: BERT, GPT 등 최신 언어 모델의 구조와 작동 원리 완벽 이해
2. **Hugging Face 생태계 활용**: 모델 로드, 파인튜닝, 배포 등 실무 능력 보유
3. **효율적 파인튜닝 능력**: 제한된 자원에서 PEFT(LoRA)를 활용한 모델 커스터마이징
4. **프로젝트 수행 경험**: 실제 문제를 AI로 해결하는 엔드-투-엔드 개발 경험
5. **도메인 특화 모델 개발**: 자신의 전공 분야 데이터로 실용적인 AI 시스템 구축