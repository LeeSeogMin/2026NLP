# 14장 리서치 결과: 최종 프로젝트 개발 및 발표 준비

## 1. NLP 모델 평가 지표

### 1.1 분류 평가 지표

**Precision, Recall, F1-Score**
- Precision = TP / (TP + FP): 예측한 것 중 맞은 비율
- Recall = TP / (TP + FN): 실제 중 맞춘 비율
- F1 = 2 × (Precision × Recall) / (Precision + Recall)
- Macro/Micro/Weighted 평균

**Confusion Matrix**
- 예측 vs 실제 레이블 매트릭스
- 클래스별 오분류 패턴 파악
- scikit-learn의 confusion_matrix, ConfusionMatrixDisplay

### 1.2 생성 평가 지표

**BLEU (Bilingual Evaluation Understudy)**
- 기계 번역 평가의 표준
- n-gram precision 기반
- Brevity Penalty로 짧은 문장 페널티
- 범위: 0~1 (1이 완벽 일치)

**ROUGE (Recall-Oriented Understudy for Gisting Evaluation)**
- 텍스트 요약 평가
- Recall 중심
- ROUGE-1, ROUGE-2, ROUGE-L 변형
- ROUGE-L: Longest Common Subsequence

**BERTScore**
- BERT 임베딩 기반 유사도
- 문맥적 의미 고려
- Precision, Recall, F1 제공

### 1.3 언제 무엇을 사용할지

| 태스크 | 권장 지표 |
|--------|-----------|
| 텍스트 분류 | Accuracy, F1-Score, Confusion Matrix |
| NER/토큰 분류 | Entity-level F1, Token F1 |
| 기계 번역 | BLEU, BERTScore |
| 텍스트 요약 | ROUGE-1, ROUGE-2, ROUGE-L |
| 텍스트 생성 | Perplexity, BLEU, Human Evaluation |

---

## 2. 시각화 기법

### 2.1 학습 과정 시각화

**Loss/Accuracy Curve**
- matplotlib으로 학습/검증 곡선 비교
- 과적합 패턴 식별
- Early Stopping 시점 시각화

### 2.2 Confusion Matrix 히트맵

```python
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
plt.show()
```

### 2.3 임베딩 시각화

**t-SNE (t-Distributed Stochastic Neighbor Embedding)**
- 로컬 구조 보존에 집중
- perplexity 파라미터 조정 필요
- 고차원 → 2D/3D

**UMAP (Uniform Manifold Approximation and Projection)**
- t-SNE보다 빠름
- 글로벌 구조도 어느 정도 보존
- n_neighbors, min_dist 파라미터

**권장 사항**
- 고차원 데이터는 먼저 PCA로 50차원 정도로 축소
- 속도와 대규모 데이터셋: UMAP
- 클러스터 분리 시각화: t-SNE

### 2.4 Attention 시각화

**BertViz 라이브러리**
- Head별 Attention 패턴
- Layer별 비교
- Interactive 시각화

---

## 3. 모델 배포

### 3.1 Gradio

**장점**
- 몇 줄 코드로 데모 생성
- Python만 알면 됨
- Hugging Face Spaces 무료 호스팅

**기본 사용법**
```python
import gradio as gr

def predict(text):
    # 모델 추론
    return result

demo = gr.Interface(fn=predict, inputs="text", outputs="text")
demo.launch()
```

**배포 옵션**
- `launch(share=True)`: 72시간 임시 공유 링크
- Hugging Face Spaces: 영구 무료 호스팅
- Docker 컨테이너화

### 3.2 FastAPI

**장점**
- 고성능 비동기 처리
- 자동 API 문서화 (Swagger)
- 타입 힌트 기반 검증

**기본 구조**
```python
from fastapi import FastAPI
from transformers import pipeline

app = FastAPI()
classifier = pipeline("sentiment-analysis")

@app.post("/predict")
async def predict(text: str):
    return classifier(text)
```

### 3.3 추론 최적화

**양자화 (Quantization)**
- FP32 → INT8/INT4
- 모델 크기 2-4배 감소
- 속도 향상

**ONNX 변환**
- 프레임워크 독립적 포맷
- ONNX Runtime으로 최적화 추론
- Hugging Face Optimum 라이브러리

**배치 처리**
- 여러 입력 동시 처리
- GPU 활용 극대화

---

## 4. 윤리적 고려사항

### 4.1 Bias와 Fairness
- 학습 데이터의 편향 검토
- 성별, 인종, 연령 등 민감 속성
- Fairness 지표 측정

### 4.2 Privacy
- 개인정보 포함 데이터 주의
- 모델에 학습 데이터 노출 위험
- Differential Privacy 기법

### 4.3 책임 있는 AI
- 모델 한계 명시
- Hallucination 경고
- 사용 가이드라인 제공

---

## 5. 발표 가이드

### 5.1 발표 구조 (10-15분)
1. 문제 정의 (2분): 왜 이 문제가 중요한가
2. 방법론 (4분): 데이터, 모델, 학습 전략
3. 실험 결과 (5분): 정량적/정성적 분석
4. 결론 (2분): 기여점, 한계, 향후 연구
5. 질의응답 (3분)

### 5.2 효과적인 슬라이드
- 한 슬라이드 = 하나의 핵심 메시지
- 시각 자료 적극 활용
- 코드는 핵심만 발췌
- 결과 표/그래프 명확히

### 5.3 데모 준비
- Gradio/Streamlit으로 인터랙티브 데모
- 실시간 추론 시연
- 예상 질문에 대한 답변 준비

---

## 참고 자료

- GeeksforGeeks: Understanding BLEU and ROUGE score for NLP evaluation
- Towards Data Science: Visualizing Your Embeddings (January 2025)
- DataCamp: Building User Interfaces For AI Applications with Gradio
- freeCodeCamp: How to Deploy a Machine Learning Model Using Gradio
- Hugging Face: Inference Endpoints Documentation
- XAI: Dimensionality Reduction in NLP - Visualizing Sentence Embeddings
