# 14장 집필계획서: 최종 프로젝트 개발 및 발표 준비

## 개요

**장 제목**: 최종 프로젝트 개발 및 발표 준비
**대상 독자**: 딥러닝과 자연어처리를 학습한 학부생 (3~4학년)
**장 유형**: 응용/실습 중심 장 (이론:실습 = 50:50)
**예상 분량**: 550-650줄

---

## 학습 목표

이 장을 마치면 다음을 수행할 수 있다:
- NLP 프로젝트의 평가 지표를 적절히 선택하고 계산할 수 있다
- 학습 결과를 효과적으로 시각화하고 해석할 수 있다
- 모델 배포와 최적화의 기본 개념을 이해한다
- 체계적인 프로젝트 보고서와 발표 자료를 작성할 수 있다

---

## 절 구성

### 14.1 프로젝트 발표 가이드라인 (~80줄)

**핵심 내용**:
- NLP 프로젝트 구조
  - 문제 정의 → 데이터 수집 → 모델 선택 → 실험 → 평가 → 결론
- 프로젝트 유형별 접근법
  - 텍스트 분류 프로젝트
  - 텍스트 생성 프로젝트
  - RAG 기반 Q&A 프로젝트
- 평가 기준 안내
  - 기술적 구현 (40%)
  - 결과 분석 (30%)
  - 문서화/발표 (30%)

### 14.2 모델 평가 및 보고서 작성 (~120줄)

**핵심 내용**:
- 정량적 평가 지표
  - 분류: Accuracy, Precision, Recall, F1-Score
  - 생성: Perplexity, BLEU, ROUGE
  - Confusion Matrix 해석
- 정성적 분석 방법
  - Error Analysis (오류 유형 분류)
  - Case Study (대표 사례 분석)
- Ablation Study
  - 구성 요소별 성능 기여도 분석
- 보고서 작성 템플릿
  - 서론, 관련 연구, 방법론, 실험, 결과, 결론

**다이어그램**: 모델 평가 워크플로우

### 14.3 시각화 및 결과 해석 (~120줄)

**핵심 내용**:
- 학습 과정 시각화
  - Loss Curve, Accuracy Curve
  - Learning Rate 변화 추적
- 모델 분석 시각화
  - Confusion Matrix 히트맵
  - Classification Report
  - ROC Curve, PR Curve
- 임베딩 시각화
  - t-SNE, UMAP 차원 축소
  - 클러스터 분석
- Attention 시각화
  - Attention Weights 히트맵
  - BertViz 라이브러리
- 결과 해석 및 인사이트 도출

**실습 코드**: `14-3-visualization.py`

### 14.4 실무 적용 시 고려사항 (~100줄)

**핵심 내용**:
- 모델 배포 전략
  - REST API (FastAPI, Flask)
  - Hugging Face Spaces
  - Gradio 데모
- 추론 속도 최적화
  - 배치 처리
  - 모델 캐싱
  - GPU 활용
- 모델 경량화
  - 양자화 (Quantization)
  - 지식 증류 (Knowledge Distillation)
  - Pruning
- 윤리적 고려사항
  - Bias와 Fairness
  - Privacy 보호
  - 책임 있는 AI 사용

**다이어그램**: 모델 배포 파이프라인

### 14.5 프로젝트 개발 가이드 (~80줄)

**핵심 내용**:
- 프로젝트 주제 예시
  - 도메인 특화 감성 분석
  - 문서 자동 요약
  - Q&A 챗봇
  - 개체명 인식
- 개발 체크리스트
  - 데이터 준비
  - 모델 학습
  - 평가 및 분석
  - 문서화
- 문제 해결 팁
  - 과적합 대응
  - 성능 개선 전략
  - 디버깅 기법

### 14.6 발표 자료 준비 (~80줄)

**핵심 내용**:
- 발표 구조 (10-15분)
  - 문제 정의 (2분)
  - 방법론 (4분)
  - 실험 및 결과 (5분)
  - 결론 (2분)
  - 질의응답 (3분)
- 슬라이드 작성 가이드
  - 시각적 요소 활용
  - 핵심 메시지 전달
- 데모 준비
  - Gradio/Streamlit 활용
  - 실시간 추론 시연
- 스토리텔링
  - 문제 → 해결 → 영향

**다이어그램**: 발표 구조 플로우

---

## 생성할 파일 목록

### 문서
- `schema/chap14.md`: 집필계획서
- `content/research/ch14-research.md`: 리서치 결과
- `content/drafts/ch14-draft.md`: 초안
- `docs/ch14.md`: 최종 완성본

### 실습 코드
- `practice/chapter14/code/14-3-visualization.py`: 시각화 실습
- `practice/chapter14/code/14-4-deployment-demo.py`: Gradio 데모 예시
- `practice/chapter14/code/requirements.txt`

### 그래픽
- `content/graphics/ch14/fig-14-1-project-workflow.mmd`: 프로젝트 워크플로우
- `content/graphics/ch14/fig-14-2-evaluation-metrics.mmd`: 평가 지표 체계
- `content/graphics/ch14/fig-14-3-deployment-pipeline.mmd`: 배포 파이프라인
- `content/graphics/ch14/fig-14-4-presentation-structure.mmd`: 발표 구조

---

## 7단계 워크플로우 실행 계획

### 1단계: 집필계획서 작성 ✓
- `schema/chap14.md` 작성

### 2단계: 자료 조사
- NLP 프로젝트 평가 지표 조사
- 모델 시각화 도구 조사
- 모델 배포/경량화 기법 조사
- 효과적인 기술 발표 방법 조사

### 3단계: 정보 구조화
- 핵심 개념 정리
- 다이어그램 설계
- 실습 시나리오 구성

### 4단계: 구현 및 문서화
- 시각화 코드 작성 및 실행
- Gradio 데모 코드 작성
- 본문 초안 작성
- Mermaid 다이어그램 제작

### 5단계: 최적화
- 문체 일관성 검토
- 분량 조정
- 용어 통일

### 6단계: 품질 검증
- Multi-LLM Review
- `docs/ch14.md`로 최종 저장

### 7단계: MS Word 변환
- `npm run convert:chapter 14` 실행
