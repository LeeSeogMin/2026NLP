# 5장 집필계획서: LLM 아키텍처 — BERT와 GPT

## 기본 정보
- **주차**: 5주차
- **주제**: LLM 아키텍처: BERT와 GPT
- **미션**: 수업이 끝나면 BERT와 GPT를 직접 돌려보고 차이를 체감한다
- **목표 분량**: 600~700줄 (핵심 기술 장)
- **참고 자산**: 구 ch9 (BERT) + 구 ch10 (GPT)

## 학습 목표 (5개)
1. 사전학습-파인튜닝 패러다임과 Transfer Learning의 원리를 설명할 수 있다
2. BERT의 양방향 아키텍처와 MLM/NSP 사전학습 방법을 이해한다
3. GPT의 자기회귀 아키텍처와 Causal Self-Attention을 설명할 수 있다
4. 다양한 텍스트 생성 전략(Greedy/Beam/Top-k/Top-p/Temperature)을 비교할 수 있다
5. Hugging Face Transformers를 사용하여 BERT/GPT 모델을 활용할 수 있다

## 교시 구성

### 1교시: 사전학습과 BERT (00:00~00:50)
- 5.1 사전학습 패러다임
  - 직관적 이해: 의대 본과 vs 전문의 수련
  - Pre-training → Fine-tuning 전략
  - Transfer Learning
  - Encoder-only vs Decoder-only vs Encoder-Decoder
  - Mermaid: fig-5-1-pretrain-finetune
- 5.2 BERT 아키텍처
  - 직관적 이해: 빈칸 채우기 달인
  - Bidirectional Context
  - MLM (15% 마스킹) + NSP
  - Embedding 3요소 (Token + Segment + Position)
  - BERT-Base vs BERT-Large
  - BERT 변형: RoBERTa, ALBERT, DistilBERT, DeBERTa
  - Mermaid: fig-5-2-bert-architecture

### 2교시: GPT와 Hugging Face (01:00~01:50)
- 5.3 GPT 아키텍처
  - 직관적 이해: 소설 이어쓰기 달인
  - Autoregressive Language Modeling
  - Decoder-only + Causal Self-Attention
  - GPT-1→2→3→4 발전사
  - 텍스트 생성 전략 (Greedy/Beam/Top-k/Top-p/Temperature)
  - Zero-shot / Few-shot / In-Context Learning
  - Mermaid: fig-5-3-bert-vs-gpt, fig-5-4-gpt-generation
  - 라이브 코딩 시연: GPT-2 텍스트 생성 전략 비교
- 5.4 Hugging Face Transformers 실전
  - Pipeline API (3줄로 감성분석/NER/요약)
  - AutoModel, AutoTokenizer, AutoConfig
  - Model Hub 탐색 및 모델 선택 기준

### 3교시: BERT/GPT 활용 실습 (02:00~02:50)
- 5.5 실습
  - Copilot 활용 안내
  - BERT Tokenizer + 임베딩 추출
  - BERT 감성분석 / NER / 유사도
  - GPT-2 텍스트 생성 + 디코딩 전략 비교
  - Hugging Face Pipeline 활용
- 과제: BERT 기반 NER 모델 + GPT-2 텍스트 생성기 구현

## 실습 코드
| 파일 | 내용 |
|------|------|
| 5-1-bert-basics.py | BERT 토크나이저 + MLM + 임베딩 추출 + Pipeline |
| 5-3-gpt-generation.py | GPT-2 로드/토큰화/텍스트 생성 전략 비교 |
| 5-5-bert-gpt-practice.py | 3교시 실습 종합 |

## 그래픽
| 파일 | 내용 |
|------|------|
| fig-5-1-pretrain-finetune.mmd | Pre-training → Fine-tuning 패러다임 흐름도 |
| fig-5-2-bert-architecture.mmd | BERT 내부 구조 |
| fig-5-3-bert-vs-gpt.mmd | BERT vs GPT Attention 비교 |
| fig-5-4-gpt-generation.mmd | GPT 자기회귀 생성 + 디코딩 전략 |
