# 13장 집필계획서: LLM 고급 응용 - RAG와 프롬프트 엔지니어링

## 개요

**장 제목**: LLM 고급 응용 - RAG와 프롬프트 엔지니어링
**대상 독자**: 딥러닝과 자연어처리를 처음 배우는 학부생 (3~4학년)
**장 유형**: 실습 중심 장 (이론:실습 = 40:60)
**예상 분량**: 700-800줄

---

## 학습 목표

이 장을 마치면 다음을 수행할 수 있다:
- LLM의 한계(Hallucination, 지식 한계)를 이해하고 해결 방안을 설명할 수 있다
- RAG(검색 증강 생성)의 개념과 구성 요소를 이해한다
- Vector Database의 원리와 FAISS를 활용할 수 있다
- LangChain을 활용하여 간단한 RAG 시스템을 구현할 수 있다
- 효과적인 프롬프트 엔지니어링 기법을 적용할 수 있다

---

## 절 구성

### 13.1 LLM의 한계 (~60줄)

**핵심 내용**:
- Hallucination (환각) 문제
  - 사실과 다른 정보 생성
  - 신뢰성 문제
- 지식의 한계 (Training Cutoff Date)
- 도메인 특화 지식 부족
- 실시간 정보 부재
- 출처 검증의 어려움

### 13.2 RAG 개요 (~80줄)

**핵심 내용**:
- RAG(Retrieval-Augmented Generation) 정의
- RAG vs Fine-tuning: 언제 무엇을 사용할 것인가
- RAG의 장점
  - 최신 정보 활용
  - 도메인 지식 주입
  - Hallucination 감소
  - 출처 추적 가능

**다이어그램**: RAG 시스템 개요

### 13.3 RAG 시스템 구성 요소 (~60줄)

**핵심 내용**:
- Retriever: 관련 문서 검색
- Generator: 답변 생성
- Knowledge Base: 문서 저장소
- 각 구성 요소의 역할

### 13.4 Vector Database (~100줄)

**핵심 내용**:
- Vector Embedding의 개념
- 유사도 검색 (Similarity Search)
  - Cosine Similarity
  - Euclidean Distance
- FAISS (Facebook AI Similarity Search)
- ChromaDB 소개
- Pinecone, Weaviate 간략 소개

**코드**: FAISS 기본 사용법

### 13.5 RAG 파이프라인 (~100줄)

**핵심 내용**:
1. 문서 수집 및 청킹 (Chunking)
2. 임베딩 생성
3. Vector DB에 저장
4. 쿼리 임베딩
5. 유사 문서 검색 (Top-k Retrieval)
6. Context와 함께 LLM에 전달
7. 답변 생성

**다이어그램**: RAG 파이프라인

### 13.6 LangChain 소개 (~80줄)

**핵심 내용**:
- LangChain 프레임워크 개요
- Document Loaders
- Text Splitters
- Vector Stores
- Retrievers
- Chains

**코드**: LangChain 기본 사용법

### 13.7 프롬프트 엔지니어링 심화 (~100줄)

**핵심 내용**:
- Zero-shot Prompting
- Few-shot Prompting (In-Context Learning)
- Chain-of-Thought (CoT) Prompting
- Self-Consistency
- System Prompt 설계
- Role Prompting

**표**: 프롬프트 기법 비교

### 13.8 API 기반 LLM 활용 (~60줄)

**핵심 내용**:
- OpenAI API (GPT-4, GPT-4o)
- Anthropic Claude API
- API Key 관리 및 보안
- Rate Limiting 및 비용 관리

### 13.9 실습: RAG 시스템 구현 (~120줄)

**핵심 내용**:
- 문서 로드 및 청킹
- FAISS Vector Database 구축
- 간단한 RAG 시스템 구현
- 문서 기반 Q&A
- 프롬프트 엔지니어링 실험

---

## 생성할 파일 목록

### 문서
- `schema/chap13.md`: 집필계획서 (본 파일)
- `content/research/ch13-research.md`: 리서치 결과
- `content/drafts/ch13-draft.md`: 초안
- `docs/ch13.md`: 최종 완성본

### 실습 코드
- `practice/chapter13/code/13-4-vector-database.py`: Vector DB 기초
- `practice/chapter13/code/13-6-langchain-basics.py`: LangChain 기본
- `practice/chapter13/code/13-9-rag-system.py`: RAG 시스템 구현
- `practice/chapter13/code/requirements.txt`

### 그래픽
- `content/graphics/ch13/fig-13-1-llm-limitations.mmd`: LLM 한계
- `content/graphics/ch13/fig-13-2-rag-overview.mmd`: RAG 개요
- `content/graphics/ch13/fig-13-3-rag-pipeline.mmd`: RAG 파이프라인
- `content/graphics/ch13/fig-13-4-prompt-techniques.mmd`: 프롬프트 기법

---

## 12장과의 연계

- 12장에서 배운 PEFT/LoRA는 모델 자체 개선
- 13장 RAG는 외부 지식 활용으로 LLM 보완
- RAG + Fine-tuning 조합으로 최적의 시스템 구축 가능

---

## 핵심 개념 정리

| 개념 | 설명 |
|------|------|
| RAG | 검색 증강 생성, 외부 문서 검색 + LLM 생성 결합 |
| Vector Database | 임베딩 벡터 저장 및 유사도 검색 |
| FAISS | Facebook의 효율적 유사도 검색 라이브러리 |
| LangChain | LLM 애플리케이션 개발 프레임워크 |
| CoT Prompting | 단계별 추론을 유도하는 프롬프트 기법 |

---

## 마지막 업데이트

2026-01-03
