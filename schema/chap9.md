# 9장 집필계획서: LLM 시대 (1) - BERT 아키텍처와 활용

## 개요

**장 제목**: LLM 시대 (1) - BERT 아키텍처와 활용
**대상 독자**: 딥러닝과 자연어처리를 처음 배우는 학부생 (3~4학년)
**장 유형**: 핵심 기술 장 (이론:실습 = 60:40)
**예상 분량**: 600-700줄

---

## 학습 목표

이 장을 마치면 다음을 수행할 수 있다:
- 사전학습(Pre-training)과 미세조정(Fine-tuning) 패러다임을 이해한다
- BERT의 Encoder-only 구조와 양방향 문맥 이해의 의미를 설명할 수 있다
- MLM(Masked Language Model)과 NSP(Next Sentence Prediction) 학습 방식을 이해한다
- BERT 토크나이저(WordPiece)의 작동 원리를 설명할 수 있다
- Hugging Face Transformers 라이브러리로 BERT를 활용할 수 있다
- BERT를 활용한 텍스트 분류, NER, 임베딩 추출을 수행할 수 있다

---

## 절 구성

### 9.1 사전학습 언어 모델의 패러다임 (~80줄)

**핵심 내용**:
- Pre-training과 Fine-tuning 전략
  - 대규모 코퍼스로 사전학습
  - 특정 태스크에 미세조정
- Transfer Learning in NLP
  - CV에서의 전이 학습 성공
  - NLP로의 확장
- GPT vs BERT 비교
  - Decoder-only vs Encoder-only
  - 단방향 vs 양방향
- BERT의 혁신성
  - 2018년 GLUE 벤치마크 석권
  - NLP 분야 패러다임 전환

**다이어그램**: Pre-train → Fine-tune 워크플로우

### 9.2 BERT 아키텍처 심층 분석 (~100줄)

**핵심 내용**:
- Encoder-only 구조의 의미
  - 입력 전체를 양방향으로 참조
  - Self-Attention으로 문맥 이해
- Bidirectional Context Understanding
  - 좌-우 문맥 동시 고려
  - GPT의 단방향 vs BERT의 양방향
- BERT의 Layer 구조
  - Embedding Layer (Token + Segment + Position)
  - Multi-Layer Transformer Encoders
  - Pooler Layer
- BERT-Base vs BERT-Large
  - Base: 12 layers, 768 hidden, 12 heads, 110M params
  - Large: 24 layers, 1024 hidden, 16 heads, 340M params

**다이어그램**: BERT 아키텍처 상세

### 9.3 BERT 사전학습 (~90줄)

**핵심 내용**:
- Masked Language Model (MLM)
  - 15% 토큰 마스킹
  - [MASK], 랜덤 토큰, 원본 유지 비율
  - MLM Loss 계산
- Next Sentence Prediction (NSP)
  - 문장 쌍 관계 학습
  - [CLS] 토큰으로 분류
  - NSP의 한계점
- 사전학습 데이터셋
  - BooksCorpus (800M words)
  - English Wikipedia (2,500M words)
- 사전학습 설정
  - 256 시퀀스 길이, 40 epochs
  - 배치 크기, 학습률

### 9.4 BERT Tokenization (~70줄)

**핵심 내용**:
- WordPiece Tokenization
  - Subword 기반 토큰화
  - OOV 문제 해결
  - ## 접두사의 의미
- Special Tokens
  - [CLS]: 분류 태스크용
  - [SEP]: 문장 구분
  - [MASK]: MLM 학습용
  - [PAD]: 패딩
  - [UNK]: 미등록 단어
- Tokenizer 사용법
  - encode(), decode()
  - attention_mask, token_type_ids

### 9.5 BERT 변형 모델들 (~70줄)

**핵심 내용**:
- RoBERTa: BERT의 개선
  - NSP 제거
  - 동적 마스킹
  - 더 큰 배치, 더 많은 데이터
- ALBERT: 파라미터 효율성
  - Factorized Embedding
  - Cross-layer Parameter Sharing
  - SOP (Sentence Order Prediction)
- DistilBERT: 지식 증류
  - 6 layers (BERT-Base의 절반)
  - 97% 성능, 60% 크기, 2배 속도
- DeBERTa: Disentangled Attention
  - Content + Position 분리
  - Enhanced Mask Decoder

**비교표**: BERT 변형 모델 비교

### 9.6 Hugging Face Transformers 라이브러리 (~80줄)

**핵심 내용**:
- Hugging Face Hub 소개
  - 200,000+ 모델
  - 커뮤니티 기반 생태계
- Pipeline API
  - 간단한 추론
  - sentiment-analysis, ner, question-answering
- AutoModel, AutoTokenizer
  - 자동 모델/토크나이저 선택
  - from_pretrained() 메서드
- 모델 로드 및 추론
  - 입력 준비
  - 출력 해석

### 9.7 BERT 활용 사례 (~60줄)

**핵심 내용**:
- Text Classification
  - [CLS] 토큰 활용
  - 감성 분석, 주제 분류
- Named Entity Recognition (NER)
  - 토큰별 분류
  - B-PER, I-PER, O 태깅
- Question Answering
  - 지문 내 답변 위치 찾기
  - Start/End 위치 예측
- Sentence Similarity
  - [CLS] 임베딩 비교
  - 코사인 유사도
- Text Embedding 추출
  - 마지막 은닉 상태 활용
  - 평균 풀링

### 9.8 실습 (~150줄)

**핵심 내용**:
- Hugging Face Transformers 기본 사용법
- Pipeline으로 감성 분석
- BERT Tokenizer 실습
- 텍스트 분류 추론
- NER (개체명 인식) 실습
- BERT 임베딩 추출 및 유사도 계산

**실습 코드**:
- `9-1-bert-pipeline.py`: Pipeline API 활용
- `9-4-bert-tokenizer.py`: 토크나이저 실습
- `9-7-bert-applications.py`: 다양한 활용

---

## 생성할 파일 목록

### 문서
- `schema/chap9.md`: 집필계획서 (현재 파일)
- `content/research/ch9-research.md`: 리서치 결과
- `content/drafts/ch9-draft.md`: 초안
- `docs/ch9.md`: 최종 완성본

### 실습 코드
- `practice/chapter9/code/9-1-bert-pipeline.py`
- `practice/chapter9/code/9-4-bert-tokenizer.py`
- `practice/chapter9/code/9-7-bert-applications.py`
- `practice/chapter9/code/requirements.txt`

### 그래픽
- `content/graphics/ch9/fig-9-1-pretrain-finetune.mmd`
- `content/graphics/ch9/fig-9-2-bert-architecture.mmd`
- `content/graphics/ch9/fig-9-3-mlm-nsp.mmd`
- `content/graphics/ch9/fig-9-4-bert-variants.mmd`

---

## 핵심 개념

1. **Pre-training + Fine-tuning 패러다임**:
   - 대규모 코퍼스로 언어 이해력 학습 (Pre-training)
   - 소규모 태스크별 데이터로 미세조정 (Fine-tuning)

2. **BERT의 핵심 아이디어**:
   - Encoder-only: 입력 전체를 양방향으로 참조
   - MLM: 마스킹된 토큰 예측으로 양방향 학습
   - [CLS] 토큰: 문장 수준 표현

3. **BERT 입력 구성**:
   Token Embedding + Segment Embedding + Position Embedding

4. **MLM 마스킹 전략 (15% 토큰)**:
   - 80%: [MASK]로 대체
   - 10%: 랜덤 토큰으로 대체
   - 10%: 원본 유지

5. **WordPiece Tokenization**:
   - "playing" → ["play", "##ing"]
   - OOV 문제 해결, 어휘 크기 축소

---

## 7단계 워크플로우 실행 계획

### 1단계: 집필계획서 작성 ✓
- `schema/chap9.md` 작성 완료

### 2단계: 자료 조사
- BERT 원 논문 (Devlin et al., 2018)
- RoBERTa, ALBERT, DistilBERT 논문
- Hugging Face 공식 문서
- WordPiece Tokenization

### 3단계: 정보 구조화
- 핵심 개념 정리
- 아키텍처 다이어그램 설계
- 실습 시나리오 구성

### 4단계: 구현 및 문서화
- 실습 코드 작성 및 실행
- 본문 초안 작성
- Mermaid 다이어그램 제작

### 5단계: 최적화
- 6장(Transformer)과의 연결성 확인
- 문체 일관성 검토
- 용어 통일

### 6단계: 품질 검증
- Multi-LLM Review (GPT-4o + grok-4-1-fast-reasoning)
- `docs/ch9.md`로 최종 저장

### 7단계: MS Word 변환
- `npm run convert:chapter 9` 실행

---

## 참고문헌

- Devlin, J., et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. arXiv:1810.04805.
- Liu, Y., et al. (2019). RoBERTa: A Robustly Optimized BERT Pretraining Approach. arXiv:1907.11692.
- Lan, Z., et al. (2019). ALBERT: A Lite BERT for Self-supervised Learning of Language Representations.
- Sanh, V., et al. (2019). DistilBERT, a distilled version of BERT. arXiv:1910.01108.
- He, P., et al. (2020). DeBERTa: Decoding-enhanced BERT with Disentangled Attention.
- Hugging Face Documentation: https://huggingface.co/docs/transformers/
