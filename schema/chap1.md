# 제1장 집필계획서: AI 시대의 개막과 개발 환경 준비

## 개요

| 항목 | 내용 |
|------|------|
| **장 제목** | AI 시대의 개막과 개발 환경 준비 |
| **대상 독자** | 딥러닝과 자연어처리를 처음 배우는 학부생 (3~4학년) |
| **장 유형** | 기초 개념 장 |
| **이론:실습 비율** | 70:30 |
| **예상 분량** | 500-600줄 (약 30쪽) |

---

## 학습 목표

이 장을 마치면 다음을 수행할 수 있다:

- 인공지능, 머신러닝, 딥러닝의 관계를 설명할 수 있다
- 자연어처리의 개념과 주요 응용 분야를 이해한다
- 언어 모델의 발전 과정을 설명할 수 있다
- Python 기반 딥러닝 개발 환경을 직접 구축할 수 있다

---

## 절 구성

### 1.1 인공지능의 이해 (~120줄)

**핵심 내용**:

1. **인공지능(AI)의 정의와 역사적 발전**
   - AI 용어 탄생 (1956년 다트머스 회의)
   - AI 겨울과 부흥의 역사 (1970s-80s 겨울, 2012 딥러닝 부흥)

2. **인공지능의 세 가지 수준**
   - 약한 AI (Narrow AI): 특정 작업만 수행 (현재 대부분의 AI)
   - 강한 AI (AGI): 범용 인간 수준 지능
   - 초지능 (ASI): 인간을 초월한 지능

3. **AI, 머신러닝, 딥러닝의 관계**
   - 포함 관계: AI ⊃ ML ⊃ DL
   - 각 분야의 특징과 차이점

4. **학습 패러다임 개요**
   - 지도학습: 정답이 있는 데이터로 학습
   - 비지도학습: 정답 없이 패턴 발견
   - 강화학습: 보상을 통한 행동 학습

**필수 다이어그램**: AI/ML/DL 포함 관계도 (Mermaid)

---

### 1.2 자연어처리와 언어 모델 (~120줄)

**핵심 내용**:

1. **자연어처리(NLP)란 무엇인가**
   - 자연어 vs 인공어 (프로그래밍 언어)
   - NLP의 목표: 기계가 인간 언어를 이해하고 생성
   - 주요 도전 과제: 모호성, 맥락 의존성

2. **NLP의 주요 응용 분야**
   - 기계번역 (Google Translate, DeepL)
   - 챗봇/대화 시스템 (ChatGPT, Claude)
   - 감성분석 (제품 리뷰, SNS 분석)
   - 개체명인식 (NER)
   - 질의응답 시스템

3. **언어 모델의 개념**
   - 언어 모델이란: 다음 단어를 예측하는 확률 모델
   - P(w₁, w₂, ..., wₙ) 확률 분포

4. **언어 모델의 발전 과정**
   - 통계적 언어 모델: N-gram (1990s)
   - 신경망 언어 모델: RNN/LSTM (2010s)
   - Transformer 아키텍처 (2017)
   - 대규모 언어 모델 LLM (2020s): GPT, BERT

**필수 다이어그램**: 언어 모델 발전 타임라인 (Mermaid)

---

### 1.3 AI 개발 생태계 (~100줄)

**핵심 내용**:

1. **주요 딥러닝 프레임워크**
   - TensorFlow: Google 개발, 산업계 표준, 정적 그래프
   - PyTorch: Meta(Facebook) 개발, 연구계 표준, 동적 그래프
   - JAX: Google 개발, 고성능 수치 연산, 함수형 패러다임
   - 프레임워크 선택 기준 (본 교재는 PyTorch 사용)

2. **Hugging Face 생태계**
   - Transformers 라이브러리: 사전학습 모델 접근
   - Hub: 모델/데이터셋 공유 플랫폼
   - Datasets, Tokenizers, PEFT 등 도구

3. **클라우드 GPU 서비스**
   - Google Colab: 무료 GPU 접근 (T4, 제한적)
   - Kaggle Notebooks: 무료 GPU (주당 30시간)
   - AWS/GCP/Azure: 유료 클라우드 서비스

**필수 표**: 딥러닝 프레임워크 비교표

---

### 1.4 실습: 개발 환경 구축 (~160줄)

**핵심 내용**:

1. **Python 설치 및 가상환경 설정**
   - Python 3.10+ 권장
   - Anaconda/Miniconda 설치 방법
   - 가상환경 생성: `python -m venv` 또는 `conda create`
   - 가상환경 활성화 (Windows/macOS/Linux)

2. **필수 라이브러리 설치**
   - NumPy: 수치 연산
   - Pandas: 데이터 처리
   - Matplotlib: 시각화
   - 설치 확인 코드

3. **PyTorch 설치**
   - CPU 버전 vs GPU(CUDA) 버전
   - 공식 사이트 설치 명령어 안내
   - 설치 확인 및 버전 출력

4. **Google Colab 사용법**
   - Colab 접속 및 노트북 생성
   - GPU/TPU 런타임 설정 방법
   - 드라이브 마운트

5. **Jupyter Notebook 기본 사용법**
   - 셀 실행, 마크다운 작성
   - 단축키 안내

**실습 코드**:
- `1-4-환경설정.py`: 라이브러리 버전 확인
- `1-4-pytorch확인.py`: PyTorch 설치 및 GPU 확인

---

## 생성할 파일 목록

### 문서
| 파일 경로 | 설명 |
|-----------|------|
| `schema/chap1.md` | 본 집필계획서 |
| `content/research/ch1-research.md` | 리서치 결과 |
| `content/drafts/ch1-draft.md` | 초안 |
| `docs/ch1.md` | 최종 완성본 |

### 실습 코드
| 파일 경로 | 설명 |
|-----------|------|
| `practice/chapter1/code/1-4-환경설정.py` | 라이브러리 설치 확인 |
| `practice/chapter1/code/1-4-pytorch확인.py` | PyTorch/GPU 확인 |
| `practice/chapter1/code/requirements.txt` | 의존성 목록 |

### 그래픽 (Mermaid)
| 파일 경로 | 설명 |
|-----------|------|
| `content/graphics/ch1/fig-1-1-ai-ml-dl.mmd` | AI/ML/DL 관계도 |
| `content/graphics/ch1/fig-1-2-lm-evolution.mmd` | 언어 모델 발전사 |

---

## 참고문헌 (검증 필요)

1. McCarthy, J. et al. (1956). A Proposal for the Dartmouth Summer Research Project on Artificial Intelligence.
2. Vaswani, A. et al. (2017). Attention Is All You Need. *NeurIPS*.
3. Jurafsky, D. & Martin, J.H. (2024). Speech and Language Processing (3rd ed.). https://web.stanford.edu/~jurafsky/slp3/
4. PyTorch 공식 문서: https://pytorch.org/docs/
5. Hugging Face 공식 문서: https://huggingface.co/docs

---

**작성일**: 2026-01-02
**상태**: 승인됨
