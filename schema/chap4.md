# 제4장 집필계획서: Transformer 아키텍처 심층 분석

## 기본 정보

| 항목 | 내용 |
|------|------|
| **주차** | 4주차 |
| **제목** | Transformer 아키텍처 심층 분석 |
| **미션** | 수업이 끝나면 Transformer Encoder를 밑바닥부터 구현한다 |
| **장 유형** | 핵심 기술 장 |
| **목표 분량** | 600-700줄 (이론:실습 = 60:40) |
| **난이도** | ★★★★☆ |
| **참고 자산** | 구 ch6 (Transformer 아키텍처) 심화 확장 |

---

## 학습 목표

1. Transformer의 전체 구조(Encoder-Decoder)를 설명하고 각 구성 요소의 역할을 이해한다
2. Positional Encoding(Sinusoidal vs Learned)의 필요성과 원리를 설명할 수 있다
3. PyTorch로 Transformer Encoder Block을 밑바닥부터 구현할 수 있다
4. Causal Masking과 Cross-Attention을 이해하고 Decoder 구조를 설명할 수 있다
5. BPE, WordPiece 등 서브워드 토크나이제이션 알고리즘의 원리를 이해한다

---

## 수업 타임라인

| 시간 | 구분 | 내용 | Copilot |
|------|------|------|---------|
| 00:00~00:50 | **1교시** | Transformer 전체 구조 + Positional Encoding | 사용 안 함 |
| 00:50~01:00 | 쉬는시간 | | |
| 01:00~01:50 | **2교시** | Encoder/Decoder 구현 + Tokenization | 교수 시연 |
| 01:50~02:00 | 쉬는시간 | | |
| 02:00~02:50 | **3교시** | Transformer 구현 실습 + 과제 | 적극 사용 |

---

## 절 구성

### 1교시: Transformer 전체 구조

#### 4.1 Transformer 전체 구조
- "Attention is All You Need" 논문 핵심 메시지
- 직관적 비유: "동시통역 시스템" (Encoder=원문 이해, Decoder=순차 생성)
- Encoder 구조: Self-Attention + FFN + Residual + LayerNorm
- Decoder 구조: Masked Self-Attention + Cross-Attention + FFN
- RNN 대비 Transformer의 혁신: 병렬 처리, O(1) 경로 길이, 확장성
- Mermaid: Transformer 전체 아키텍처 다이어그램

#### 4.2 Positional Encoding
- 위치 정보가 필요한 이유 (Self-Attention의 순서 무관성)
- Sinusoidal Positional Encoding 수식과 직관
- Learned Positional Encoding (BERT, GPT)
- 두 방식 비교 표
- 코드: PE 구현 + 위치 간 유사도 시각화

### 2교시: Transformer 구현과 Tokenization

#### 4.3 Transformer Encoder 구현 (PyTorch)
- Residual Connection 직관: "원본을 보존하면서 변화량만 학습"
- Layer Normalization: 각 층의 출력 안정화
- Pre-LN vs Post-LN 비교
- Feed-Forward Network 구현
- Single Encoder Block 구현: MHA → Add&Norm → FFN → Add&Norm
- Multi-Layer Encoder 스택
- 라이브 코딩 시연 표시

#### 4.4 Transformer Decoder 구현
- Causal Masking: 미래 토큰을 못 보게 가리기
- Cross-Attention: Decoder가 Encoder 출력을 참조
- Encoder-Decoder 연결 구조

#### 4.5 Tokenization 심화
- BPE 알고리즘: 단계별 병합 과정 + 직관적 비유
- WordPiece: BERT가 사용하는 방식
- SentencePiece / Unigram Model
- Hugging Face Tokenizer 실습: 토크나이저 간 차이 비교

### 3교시: Transformer 구현 실습

#### 4.6 실습
- Transformer Encoder Block 밑바닥 구현
- Positional Encoding 구현 및 시각화
- Transformer 기반 텍스트 분류기 (IMDb 감성 분류)
- Tokenizer 비교 실험 (BPE vs WordPiece)
- 과제: Transformer Encoder로 텍스트 분류 모델 구현 + 성능 분석

---

## 실습 코드 계획

| 파일 | 절 | 내용 |
|------|-----|------|
| `4-1-트랜스포머구조.py` | 4.1, 4.2 | Positional Encoding 구현 + 유사도 시각화, 구조 비교 |
| `4-3-인코더구현.py` | 4.3, 4.4 | Transformer Encoder/Decoder 밑바닥 구현 + 분류 모델 |
| `4-5-토크나이저.py` | 4.5 | BPE/WordPiece 비교 실험 + Hugging Face Tokenizer |

---

## 그래픽 계획

| 파일명 | 유형 | 내용 |
|--------|------|------|
| `fig-4-1-transformer-architecture.mmd` | Mermaid | Transformer 전체 구조 (Encoder-Decoder) |
| `fig-4-2-encoder-block.mmd` | Mermaid | Encoder Block 내부 구조 |
| `fig-4-3-decoder-block.mmd` | Mermaid | Decoder Block 내부 구조 |
| `fig-4-4-bpe-process.mmd` | Mermaid | BPE 병합 과정 |

---

## 3장과의 연결

- 3장에서 다룬 Self-Attention, Multi-Head Attention을 **구성 요소**로 사용
- 3장의 Causal Mask 소개를 4장에서 **구현**
- 3장의 감성 분류 모델(Attention 기반)을 4장에서 **Transformer Encoder 기반**으로 업그레이드

## 5장으로의 연결

- 4장에서 구현한 Transformer Encoder → 5장에서 BERT의 기반
- 4장의 Decoder → 5장에서 GPT의 기반
- 4장의 Tokenization → 5장에서 BERT/GPT의 토크나이저와 연결

---

## 참고문헌 (검증 필요)

1. Vaswani, A. et al. (2017). Attention Is All You Need. *NeurIPS*.
2. Ba, J. L. et al. (2016). Layer Normalization. *arXiv*.
3. He, K. et al. (2016). Deep Residual Learning for Image Recognition. *CVPR*.
4. Sennrich, R. et al. (2016). Neural Machine Translation of Rare Words with Subword Units. *ACL*.
5. Schuster, M. & Nakajima, K. (2012). Japanese and Korean voice search. *ICASSP*.
6. Kudo, T. & Richardson, J. (2018). SentencePiece. *EMNLP*.

---

**작성일**: 2026-02-11
