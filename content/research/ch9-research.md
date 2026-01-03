# 9장 리서치 결과: LLM 시대 (1) - BERT 아키텍처와 활용

**조사일**: 2026-01-03
**조사 주제**: BERT, Pre-training, MLM, WordPiece, Hugging Face

---

## 1. BERT 개요

### 1.1 BERT란?
- **B**idirectional **E**ncoder **R**epresentations from **T**ransformers
- Google AI Language팀 (Devlin et al., 2018)이 발표
- Encoder-only Transformer 아키텍처
- 양방향 문맥 이해 (Bidirectional Context)

### 1.2 BERT의 혁신성
- 2018년 GLUE 벤치마크 11개 태스크 중 9개에서 SOTA 달성
- Pre-training + Fine-tuning 패러다임 확립
- NLP 분야 패러다임 전환

### 1.3 모델 크기
| 모델 | Layers | Hidden | Heads | Parameters |
|------|--------|--------|-------|------------|
| BERT-Base | 12 | 768 | 12 | 110M |
| BERT-Large | 24 | 1024 | 16 | 340M |

---

## 2. BERT 사전학습

### 2.1 Masked Language Model (MLM)
- 입력 토큰의 15% 무작위 마스킹
- 마스킹 전략:
  - 80%: [MASK] 토큰으로 대체
  - 10%: 랜덤 토큰으로 대체
  - 10%: 원본 유지
- 마스킹된 토큰의 원래 단어 예측

**예시**:
- 입력: "The cat sat on the [MASK]"
- 예측: "mat"

### 2.2 Next Sentence Prediction (NSP)
- 두 문장이 연속인지 판단
- [CLS] 토큰으로 이진 분류
- 50%: 실제 연속 문장 (IsNext)
- 50%: 랜덤 문장 (NotNext)

### 2.3 사전학습 데이터
- BooksCorpus: 800M words
- English Wikipedia: 2,500M words
- 총 약 3.3B words

### 2.4 학습 설정
- 시퀀스 길이: 512 (처음 90%: 128)
- 배치 크기: 256
- 학습률: 1e-4
- Warmup: 10,000 steps
- 총 1,000,000 steps

---

## 3. BERT 입력 구조

### 3.1 세 가지 임베딩의 합
```
Input = Token Embedding + Segment Embedding + Position Embedding
```

### 3.2 Special Tokens
| 토큰 | 역할 |
|------|------|
| [CLS] | 분류 태스크용 (문장 수준 표현) |
| [SEP] | 문장 구분자 |
| [MASK] | MLM 학습용 마스킹 |
| [PAD] | 패딩 |
| [UNK] | 미등록 단어 |

### 3.3 입력 예시
```
토큰: [CLS] 나는 학교에 간다 [SEP] 오늘 날씨가 좋다 [SEP]
세그먼트: 0 0 0 0 0 1 1 1 1 1
위치: 0 1 2 3 4 5 6 7 8 9
```

---

## 4. WordPiece Tokenization

### 4.1 개요
- 서브워드 기반 토큰화 알고리즘
- BERT, DistilBERT, Electra에서 사용
- OOV(Out-of-Vocabulary) 문제 해결

### 4.2 동작 원리
- Greedy longest-match-first 전략
- 어휘에서 가장 긴 매칭 접두사 찾기
- ## 접두사: 단어 내부 토큰 표시

### 4.3 예시
```
"unbreakable" → ["un", "##break", "##able"]
"playing" → ["play", "##ing"]
"revolutionized" → ["revolution", "##ized"]
```

### 4.4 어휘 크기
- BERT: 30,522 tokens
- GPT-2: 50,257 tokens

### 4.5 BPE와의 차이
- BPE: 가장 빈번한 쌍 병합
- WordPiece: 학습 데이터 likelihood 최대화하는 쌍 병합

---

## 5. BERT 변형 모델

### 5.1 RoBERTa (2019)
- Facebook AI Research
- BERT 학습 최적화
- 주요 변경:
  - NSP 제거
  - 동적 마스킹 (배치마다 새로운 마스킹)
  - 더 큰 배치 (8K)
  - 더 많은 데이터 (160GB)
  - 더 긴 학습

### 5.2 ALBERT (2019)
- Google Research
- 파라미터 효율성 개선
- 주요 기법:
  - Factorized Embedding: E=128 (vs 768)
  - Cross-layer Parameter Sharing
  - SOP (Sentence Order Prediction): NSP 대체
- BERT-Large 대비 18x 적은 파라미터

### 5.3 DistilBERT (2019)
- Hugging Face
- 지식 증류 (Knowledge Distillation)
- 특징:
  - 6 layers (BERT-Base의 절반)
  - 97% 성능 유지
  - 60% 크기 감소
  - 2배 빠른 속도

### 5.4 DeBERTa (2020)
- Microsoft
- Disentangled Attention
  - Content와 Position 분리
- Enhanced Mask Decoder
- 2021년 SuperGLUE에서 인간 수준 달성

### 5.5 비교표
| 모델 | 파라미터 | 특징 | 성능 (GLUE) |
|------|----------|------|-------------|
| BERT-Base | 110M | 기본 | 79.6 |
| RoBERTa | 125M | 학습 최적화 | 88.5 |
| ALBERT-xxlarge | 235M | 파라미터 공유 | 89.4 |
| DistilBERT | 66M | 지식 증류 | 77.0 |
| DeBERTa-Large | 390M | Disentangled | 90.3 |

---

## 6. BERT 활용 태스크

### 6.1 Text Classification
- [CLS] 토큰 임베딩 사용
- 분류 헤드 추가
- 예: 감성 분석, 스팸 탐지, 주제 분류

### 6.2 Named Entity Recognition (NER)
- 토큰별 분류
- BIO 태깅: B-PER, I-PER, O
- 예: 인물, 조직, 장소 추출

### 6.3 Question Answering
- 지문 내 답변 위치 찾기
- Start/End 위치 예측
- SQuAD 데이터셋

### 6.4 Sentence Similarity
- [CLS] 임베딩 비교
- 코사인 유사도 계산
- 예: 문장 매칭, 의미적 유사도

### 6.5 Text Embedding
- 마지막 은닉 상태 추출
- 평균 풀링 또는 [CLS] 사용
- 다운스트림 태스크 입력

---

## 7. Hugging Face Transformers

### 7.1 주요 클래스
```python
from transformers import (
    BertTokenizer,           # 토크나이저
    BertModel,               # 기본 모델
    BertForSequenceClassification,  # 분류
    BertForTokenClassification,     # NER
    BertForQuestionAnswering,       # QA
    AutoTokenizer,           # 자동 토크나이저
    AutoModel,               # 자동 모델
    pipeline                 # 파이프라인 API
)
```

### 7.2 Pipeline API
```python
# 감성 분석
classifier = pipeline("sentiment-analysis")
result = classifier("I love this movie!")

# NER
ner = pipeline("ner")
result = ner("Hugging Face is based in New York")

# QA
qa = pipeline("question-answering")
result = qa(question="Where is Paris?", context="Paris is in France.")
```

### 7.3 모델 로드
```python
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# 입력 준비
inputs = tokenizer("Hello world", return_tensors="pt")
outputs = model(**inputs)
```

---

## 8. 2024년 최신 연구 동향

### 8.1 BPDec (2024.01)
- MLM Decoder 재설계
- Encoder 후 추가 Transformer 블록
- Fine-tuning 비용 증가 없이 성능 향상

### 8.2 NextLevelBERT (2024.02)
- 토큰이 아닌 청크(문장) 단위 MLM
- 긴 문서 처리 개선

### 8.3 ModernBERT-Large-Instruct
- 0.4B 파라미터 인코더 모델
- MLM 헤드를 생성적 분류에 활용
- FLAN 데이터셋 활용

---

## 9. 참고문헌

- Devlin, J., et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers. arXiv:1810.04805
- Liu, Y., et al. (2019). RoBERTa: A Robustly Optimized BERT Pretraining Approach
- Lan, Z., et al. (2019). ALBERT: A Lite BERT for Self-supervised Learning
- Sanh, V., et al. (2019). DistilBERT, a distilled version of BERT
- He, P., et al. (2020). DeBERTa: Decoding-enhanced BERT with Disentangled Attention
- Hugging Face Transformers: https://huggingface.co/docs/transformers/
- Hugging Face Hub: https://huggingface.co/models
