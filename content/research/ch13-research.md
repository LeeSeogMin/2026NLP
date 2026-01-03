# 13장 리서치: RAG와 프롬프트 엔지니어링

## 1. RAG (Retrieval-Augmented Generation)

### 1.1 개념
- 검색 증강 생성: 외부 지식 검색 + LLM 생성 결합
- LLM의 한계(Hallucination, 지식 단절) 해결
- 2020년 Meta AI 연구팀 발표

### 1.2 핵심 구성 요소
1. **Retriever**: 관련 문서 검색
2. **Augmentation**: 컨텍스트 보강
3. **Generator**: LLM 기반 답변 생성

### 1.3 RAG vs Fine-tuning
| 항목 | RAG | Fine-tuning |
|------|-----|-------------|
| 지식 업데이트 | 실시간 가능 | 재학습 필요 |
| 비용 | 저비용 | 고비용 |
| 도메인 지식 | 문서 추가로 해결 | 데이터 수집 필요 |
| 출처 추적 | 가능 | 불가능 |

### 1.4 RAG 파이프라인
1. 문서 수집 및 청킹 (Chunking)
2. 임베딩 생성 (Sentence Transformers)
3. Vector DB 저장
4. 쿼리 임베딩
5. 유사 문서 검색 (Top-k)
6. Context + Query를 LLM에 전달
7. 답변 생성

---

## 2. Vector Database

### 2.1 FAISS (Facebook AI Similarity Search)
- Meta AI Research 개발
- C++로 작성, Python 인터페이스 제공
- GPU 가속 지원
- 수십억 벡터 처리 가능

### 2.2 주요 인덱스 타입
- **IndexFlatL2**: L2 거리 기반, 정확하지만 느림
- **IndexIVFFlat**: Voronoi 셀 기반 분할, 빠른 근사 검색
- **IndexHNSW**: Hierarchical Navigable Small World, 균형 잡힌 성능

### 2.3 기타 Vector DB
- **ChromaDB**: 경량, 쉬운 사용
- **Pinecone**: 클라우드 관리형
- **Weaviate**: 오픈소스, GraphQL 지원
- **Milvus**: 대규모 분산 처리

### 2.4 유사도 측정
- Cosine Similarity: 방향 유사도
- Euclidean Distance (L2): 거리 기반
- Inner Product: 내적 기반

---

## 3. LangChain

### 3.1 개요
- LLM 애플리케이션 개발 프레임워크
- 체인 기반 모듈 조합
- 2023년 등장, 2025년 LangGraph 추가

### 3.2 핵심 컴포넌트
- **Document Loaders**: 다양한 형식 문서 로드 (PDF, HTML, TXT)
- **Text Splitters**: 문서 청킹 (RecursiveCharacterTextSplitter)
- **Embeddings**: 임베딩 모델 래퍼
- **Vector Stores**: FAISS, Chroma 등 통합
- **Retrievers**: 검색기 인터페이스
- **Chains**: RetrievalQA, ConversationalRetrievalChain

### 3.3 2025 업데이트
- LangGraph: 복잡한 워크플로우 제어
- 향상된 캐싱 및 메모리 관리
- 고급 에러 핸들링

---

## 4. 프롬프트 엔지니어링

### 4.1 Zero-shot Prompting
- 예시 없이 직접 질문
- 간단한 태스크에 적합

### 4.2 Few-shot Prompting
- 몇 개의 예시 제공
- In-Context Learning 활용
- 복잡한 태스크 성능 향상

### 4.3 Chain-of-Thought (CoT) Prompting
- 단계별 추론 유도
- "Let's think step by step"
- 수학, 논리 문제에 효과적
- 100B+ 파라미터 모델에서 효과적

### 4.4 고급 기법
- **Self-Consistency**: 여러 추론 경로의 일관성 확인
- **Tree of Thoughts**: 트리 구조 탐색
- **Auto-CoT**: 자동 CoT 예시 생성

### 4.5 2025 연구 동향
- 강력한 LLM에서 few-shot CoT는 주로 출력 형식 정렬 역할
- Zero-shot CoT가 few-shot 성능과 유사하거나 능가
- 모델 어텐션이 예시보다 지시사항에 집중

---

## 5. API 기반 LLM

### 5.1 주요 API
- **OpenAI**: GPT-4, GPT-4o
- **Anthropic**: Claude 3.5, Claude 3 Opus
- **Google**: Gemini Pro

### 5.2 API 사용 시 고려사항
- API Key 보안 관리
- Rate Limiting
- 비용 최적화 (토큰 수 관리)
- 스트리밍 응답

---

## 참고문헌

- Lewis, P., et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. NeurIPS.
- LangChain Documentation: https://python.langchain.com/docs
- FAISS Documentation: https://faiss.ai
- Prompt Engineering Guide: https://promptingguide.ai
- Wei, J., et al. (2022). Chain-of-Thought Prompting Elicits Reasoning in Large Language Models. NeurIPS.

---

**리서치 완료**: 2026-01-03
