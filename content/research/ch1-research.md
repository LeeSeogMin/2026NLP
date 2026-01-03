# 제1장 리서치 결과

## 조사 일자: 2026-01-02

---

## 1. AI/ML/DL 역사 및 관계

### 인공지능의 역사
- **1956년**: 다트머스 회의에서 "인공지능(Artificial Intelligence)" 용어 탄생
  - John McCarthy, Marvin Minsky 등 참여
- **1970-80년대**: 첫 번째 AI 겨울 (기대 대비 성과 부족)
- **2012년**: AlexNet이 ImageNet 대회 우승 → 딥러닝 부흥
- **2017년**: Transformer 아키텍처 발표 ("Attention Is All You Need")
- **2022년**: ChatGPT 출시로 LLM 대중화

### AI/ML/DL 관계
- **AI (인공지능)**: 인간 지능을 모방하는 모든 기술
- **ML (머신러닝)**: 데이터로부터 학습하는 AI의 하위 분야
- **DL (딥러닝)**: 심층 신경망을 사용하는 ML의 하위 분야
- 포함 관계: AI ⊃ ML ⊃ DL

---

## 2. 딥러닝 프레임워크 비교 (2025년 기준)

### 시장 점유율
- TensorFlow: 약 38% (산업계)
- PyTorch: 약 23% (연구계 55%+ 지배)
- 신규 연구 논문의 75%+ 가 PyTorch 사용

### PyTorch 장점
- 동적 계산 그래프 (디버깅 용이)
- Pythonic한 코드 스타일
- Hugging Face 등 최신 라이브러리 우선 지원
- `torch.compile()`로 20-25% 속도 향상
- GPT, LLaMA, Stable Diffusion 등 학습에 사용

### TensorFlow 장점
- 프로덕션 배포 도구 (TF Serving, TFLite)
- TPU 지원 최적화
- 엔터프라이즈 환경 안정성
- XLA 컴파일러로 15-20% 속도 향상

### 권장 사항
- **연구/학습**: PyTorch (본 교재 선택)
- **프로덕션 배포**: TensorFlow
- **둘 다 배우는 것이 이상적**

출처: [OpenCV Blog](https://opencv.org/blog/pytorch-vs-tensorflow/), [Udacity](https://www.udacity.com/blog/2025/06/tensorflow-vs-pytorch-which-framework-should-you-learn-in-2025.html)

---

## 3. Hugging Face 생태계 (2025년 기준)

### Transformers 라이브러리 현황
- **설치 수**: 일 300만+ (pip 기준)
- **총 설치**: 12억+ 회
- **지원 모델 아키텍처**: 400+
- **Hub 모델 체크포인트**: 100만+

### Transformers v5 주요 변화
- PyTorch 단일 백엔드 집중 (Flax/TensorFlow 지원 종료)
- 양자화(Quantization) 기본 지원 (8-bit, 4-bit)
- GGUF 파일 호환성 (llama.cpp 연동)
- Python 3.9+, PyTorch 2.1+ 요구

### 주요 구성 요소
- **Transformers**: 사전학습 모델 정의 및 추론
- **Hub**: 모델/데이터셋 공유 플랫폼
- **Datasets**: 데이터셋 로드/처리
- **Tokenizers**: 고속 토큰화
- **PEFT**: 효율적 파인튜닝 (LoRA 등)
- **Accelerate**: 분산 학습

출처: [Hugging Face GitHub](https://github.com/huggingface/transformers), [Transformers Docs](https://huggingface.co/docs/transformers/en/index)

---

## 4. 언어 모델 발전사

### 통계적 언어 모델 (1990년대)
- N-gram 모델
- 조건부 확률 기반
- 희소성(Sparsity) 문제

### 신경망 언어 모델 (2010년대)
- Word2Vec (2013): 단어 임베딩
- RNN/LSTM: 순차 처리
- 장기 의존성 문제

### Transformer 시대 (2017-)
- **2017**: "Attention Is All You Need" (Vaswani et al.)
- Self-Attention 메커니즘
- 병렬 처리 가능

### GPT 계보
| 모델 | 연도 | 파라미터 |
|------|------|----------|
| GPT-1 | 2018 | 1.17억 |
| GPT-2 | 2019 | 15억 |
| GPT-3 | 2020 | 1750억 |
| GPT-4 | 2023 | ~1조 (추정) |

### BERT 계보
| 모델 | 연도 | 특징 |
|------|------|------|
| BERT | 2018 | 양방향 인코더, 3.4억 파라미터 |
| RoBERTa | 2019 | BERT 개선, NSP 제거 |
| ALBERT | 2019 | 파라미터 공유로 경량화 |
| DeBERTa | 2020 | Disentangled Attention |

### 2024-2025 최신 동향
- **추론 모델(Reasoning Models)**: OpenAI o1 (2024.09)
  - 수학 올림피아드: GPT-4o 13% → o1 83%
- **오픈소스 모델**: DeepSeek-R1 (2025.01)
  - 6710억 파라미터, o1 수준 성능

출처: [Wikipedia LLM](https://en.wikipedia.org/wiki/Large_language_model), [Medium LLM History](https://medium.com/@lmpo/a-brief-history-of-lmms-from-transformers-2017-to-deepseek-r1-2025-dae75dd3f59a)

---

## 5. 클라우드 GPU 서비스

### Google Colab
- 무료 티어: T4 GPU (제한적)
- Pro: A100, 더 긴 런타임
- 장점: 빠른 시작, Google Drive 연동

### Kaggle Notebooks
- 무료 GPU: 주 30시간
- P100/T4 GPU
- 데이터셋 접근 용이

### 클라우드 서비스
- AWS: EC2 GPU 인스턴스, SageMaker
- GCP: Vertex AI, TPU 지원
- Azure: Azure ML

---

## 6. 개발 환경 요구사항

### Python 버전
- Python 3.10+ 권장
- 3.9 최소 지원

### PyTorch 설치
- CPU 버전: `pip install torch`
- GPU 버전: CUDA 버전에 맞게 선택
- 공식 사이트: https://pytorch.org/get-started/locally/

### 필수 라이브러리
```
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
torch>=2.1.0
transformers>=4.40.0
```

---

## 참고문헌

1. Vaswani, A. et al. (2017). Attention Is All You Need. *NeurIPS*.
2. Devlin, J. et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers. *arXiv*.
3. Brown, T. et al. (2020). Language Models are Few-Shot Learners. *NeurIPS*.
4. Jurafsky, D. & Martin, J.H. (2024). Speech and Language Processing (3rd ed.). https://web.stanford.edu/~jurafsky/slp3/
5. PyTorch Documentation. https://pytorch.org/docs/
6. Hugging Face Documentation. https://huggingface.co/docs
