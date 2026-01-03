"""
13-9-rag-system.py
간단한 RAG 시스템 구현

이 코드는 LangChain과 로컬 모델을 활용하여
문서 기반 Q&A 시스템을 구현한다.
"""

import os
import tempfile
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch

print("=" * 60)
print("RAG 시스템 구현 실습")
print("=" * 60)

# ============================================================
# 1. 지식 베이스 문서 준비
# ============================================================
print("\n[1] 지식 베이스 문서 준비")
print("-" * 50)

# 예시 지식 베이스 문서
knowledge_base = """
# 딥러닝 자연어처리 FAQ

## Q1: Transformer 아키텍처란?
Transformer는 2017년 "Attention is All You Need" 논문에서 소개된 신경망 아키텍처이다.
Self-Attention 메커니즘을 핵심으로 하여 입력 시퀀스의 모든 위치 간 관계를 모델링한다.
RNN과 달리 병렬 처리가 가능하여 학습 속도가 빠르다.
BERT, GPT 등 현대 언어 모델의 기반이 된다.

## Q2: BERT와 GPT의 차이점은?
BERT(Bidirectional Encoder Representations from Transformers)는 양방향 인코더 구조를 사용한다.
Masked Language Modeling(MLM)으로 사전 학습하며, 문맥 이해와 분류 태스크에 적합하다.

GPT(Generative Pre-trained Transformer)는 자기회귀 디코더 구조를 사용한다.
다음 토큰 예측으로 사전 학습하며, 텍스트 생성에 특화되어 있다.

BERT는 양방향 문맥을 고려하고, GPT는 왼쪽에서 오른쪽 단방향으로 처리한다.

## Q3: 파인튜닝이란?
파인튜닝(Fine-tuning)은 사전 학습된 모델을 특정 태스크에 적응시키는 과정이다.
사전 학습으로 일반적인 언어 이해 능력을 습득한 모델을
소량의 태스크별 데이터로 추가 학습하여 특정 문제를 해결한다.

Full Fine-tuning은 모든 파라미터를 학습하고,
PEFT(Parameter-Efficient Fine-Tuning)는 일부 파라미터만 효율적으로 학습한다.

## Q4: LoRA란?
LoRA(Low-Rank Adaptation)는 PEFT의 대표적인 기법이다.
가중치 변화량을 저랭크 행렬로 분해하여 표현한다.
ΔW = B × A 형태로, 원본 가중치의 1% 미만으로 유사한 성능을 달성한다.
메모리 효율적이며, 태스크별 어댑터만 저장하면 된다.

## Q5: RAG란?
RAG(Retrieval-Augmented Generation)는 검색과 생성을 결합한 기법이다.
외부 문서에서 관련 정보를 검색하여 LLM의 입력에 추가한다.
이를 통해 Hallucination을 줄이고, 최신 정보나 도메인 지식을 활용할 수 있다.

RAG 파이프라인: 문서 청킹 → 임베딩 → 벡터 저장 → 쿼리 검색 → 컨텍스트와 함께 생성

## Q6: Attention 메커니즘이란?
Attention은 입력의 어떤 부분에 집중할지 결정하는 메커니즘이다.
Query, Key, Value 벡터를 사용하여 Attention Score를 계산한다.
Self-Attention은 같은 시퀀스 내 토큰들 간의 관계를 모델링한다.
Multi-Head Attention은 여러 관점에서 Attention을 병렬로 계산한다.
"""

# 임시 파일에 저장
temp_dir = tempfile.mkdtemp()
kb_file = os.path.join(temp_dir, "knowledge_base.txt")
with open(kb_file, "w", encoding="utf-8") as f:
    f.write(knowledge_base)

print(f"지식 베이스 문서 크기: {len(knowledge_base)} 문자")

# ============================================================
# 2. 문서 처리 파이프라인
# ============================================================
print("\n[2] 문서 처리 파이프라인")
print("-" * 50)

# 1. 문서 로드
loader = TextLoader(kb_file, encoding="utf-8")
documents = loader.load()
print(f"1. 문서 로드 완료: {len(documents)}개")

# 2. 청킹
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
    separators=["\n\n", "\n", ".", " "]
)
chunks = text_splitter.split_documents(documents)
print(f"2. 청킹 완료: {len(chunks)}개 청크")

# 3. 임베딩 생성
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
print(f"3. 임베딩 모델 로드 완료: all-MiniLM-L6-v2")

# 4. Vector Store 생성
vectorstore = FAISS.from_documents(chunks, embeddings)
print(f"4. Vector Store 생성 완료: {len(chunks)}개 문서 색인")

# ============================================================
# 3. Retriever 설정
# ============================================================
print("\n[3] Retriever 설정")
print("-" * 50)

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)
print(f"Retriever 설정: Top-3 유사도 검색")

# ============================================================
# 4. 간단한 RAG 시스템
# ============================================================
print("\n[4] RAG 시스템 테스트")
print("-" * 50)


def simple_rag(query, retriever, top_k=3):
    """간단한 RAG 시스템"""
    # 1. 관련 문서 검색
    docs = retriever.invoke(query)

    # 2. 컨텍스트 구성
    context = "\n\n".join([doc.page_content for doc in docs[:top_k]])

    # 3. 프롬프트 구성
    prompt = f"""다음 문맥을 참고하여 질문에 답변하세요.

문맥:
{context}

질문: {query}

답변:"""

    return context, prompt, docs


# 테스트 질문들
questions = [
    "BERT와 GPT의 차이점은 무엇인가요?",
    "LoRA가 무엇인지 설명해주세요.",
    "RAG 시스템은 어떻게 작동하나요?",
]

print("RAG 검색 테스트:")
for q in questions:
    context, prompt, docs = simple_rag(q, retriever)
    print(f"\n질문: {q}")
    print(f"검색된 문서 수: {len(docs)}")
    print(f"관련 문맥 (일부):")
    print(f"  {context[:200]}...")

# ============================================================
# 5. 프롬프트 엔지니어링 예시
# ============================================================
print("\n[5] 프롬프트 엔지니어링 예시")
print("-" * 50)

# Zero-shot Prompt
zero_shot = """질문: Transformer의 핵심 메커니즘은 무엇인가요?
답변:"""

# Few-shot Prompt
few_shot = """다음은 딥러닝 개념에 대한 Q&A 예시입니다.

Q: CNN이란 무엇인가요?
A: CNN(Convolutional Neural Network)은 이미지 처리에 특화된 신경망으로,
   합성곱 연산을 사용하여 공간적 특징을 추출합니다.

Q: RNN이란 무엇인가요?
A: RNN(Recurrent Neural Network)은 순차 데이터 처리를 위한 신경망으로,
   이전 상태의 정보를 현재 처리에 활용하는 순환 구조를 가집니다.

Q: Transformer의 핵심 메커니즘은 무엇인가요?
A:"""

# Chain-of-Thought Prompt
cot_prompt = """Transformer 아키텍처가 RNN보다 효과적인 이유를 단계별로 설명하겠습니다.

1단계: RNN의 한계 분석
   - RNN은 순차적으로 처리하여 병렬화가 어렵습니다.
   - 긴 시퀀스에서 기울기 소실 문제가 발생합니다.

2단계: Transformer의 해결책
   - Self-Attention으로 모든 위치 간 직접 연결이 가능합니다.
   - 병렬 처리가 가능하여 학습 속도가 빠릅니다.

3단계: 결론
   - Transformer는 장거리 의존성을 효과적으로 모델링합니다.
   - 이로 인해 BERT, GPT 등 현대 언어 모델의 기반이 되었습니다.

따라서, Transformer의 핵심은 Self-Attention 메커니즘입니다."""

print("프롬프트 기법 비교:")
print(f"\n1. Zero-shot Prompt ({len(zero_shot)} 문자)")
print(f"   - 예시 없이 직접 질문")

print(f"\n2. Few-shot Prompt ({len(few_shot)} 문자)")
print(f"   - 유사한 Q&A 예시 제공")

print(f"\n3. Chain-of-Thought Prompt ({len(cot_prompt)} 문자)")
print(f"   - 단계별 추론 과정 포함")

# ============================================================
# 6. RAG 프롬프트 템플릿
# ============================================================
print("\n[6] RAG 프롬프트 템플릿")
print("-" * 50)

query = "LoRA를 사용하면 어떤 장점이 있나요?"
context, prompt, docs = simple_rag(query, retriever)

rag_prompt_template = f"""당신은 딥러닝 전문가입니다.
제공된 문맥을 기반으로 정확하고 상세하게 답변하세요.
문맥에 없는 내용은 추측하지 마세요.

### 문맥:
{context}

### 질문:
{query}

### 답변:
LoRA의 주요 장점은 다음과 같습니다:
1. 메모리 효율성 - 원본 파라미터의 1% 미만으로 학습
2. 저장 효율성 - 태스크별 어댑터만 저장
3. 카타스트로픽 포겟팅 방지 - 원본 가중치 보존
4. 빠른 학습 - 적은 파라미터로 빠른 수렴"""

print(f"RAG 프롬프트 예시:")
print("-" * 30)
print(rag_prompt_template)

# 정리
import shutil
shutil.rmtree(temp_dir)

# ============================================================
# 7. 핵심 요약
# ============================================================
print("\n" + "=" * 60)
print("핵심 요약")
print("=" * 60)
print("""
RAG 시스템 구현 단계:

1. 문서 준비
   - 지식 베이스 문서 수집
   - TextLoader로 로드

2. 문서 처리
   - RecursiveCharacterTextSplitter로 청킹
   - 적절한 chunk_size와 overlap 설정

3. 벡터화 및 저장
   - Sentence Transformers로 임베딩
   - FAISS Vector Store 구축

4. 검색
   - Retriever로 관련 문서 검색
   - Top-k 결과 반환

5. 생성
   - 검색된 문맥과 질문을 LLM에 전달
   - 정확한 답변 생성

프롬프트 엔지니어링:
- Zero-shot: 예시 없이 직접 질문
- Few-shot: 유사 예시 제공
- Chain-of-Thought: 단계별 추론 유도
""")

print("=" * 60)
print("RAG 시스템 구현 실습 완료")
print("=" * 60)
