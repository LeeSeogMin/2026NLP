## 14주차 B회차: 프로젝트 워크숍 B — 개발 + 피드백

> **미션**: 개인 프로젝트의 핵심 기능을 완성하고, 성능 최적화와 배포 준비를 마칠 수 있다

### 수업 타임라인

| 시간 | 내용 | 성격 |
|------|------|------|
| 00:00~00:05 | 진행 상황 공유 및 브리핑 | 정보 공유 |
| 00:05~01:00 | 개발 시간 + 교수 순회 피드백 | 실행 + 조언 |
| 01:00~01:20 | 개별 면담 (문제 해결, 최적화 방향 제시) | 개별 지도 |
| 01:20~01:28 | 배포 체크리스트 확인 + 발표 자료 안내 | 최종 점검 |
| 01:28~01:30 | 15주차 최종 발표 안내 및 마무리 | 안내 |

---

### A회차 핵심 리캡

**개인 프로젝트의 목표**:
- LLM 기술을 활용한 **실제 문제 해결** 시스템 완성
- 데이터 수집·전처리에서 모델 배포까지 **엔드-투-엔드(end-to-end) 구현**
- **포트폴리오 수준의 완성도**를 갖춘 AI 프로토타입 개발

**14주차 A회차에서 확립한 내용**:
- 프로젝트 주제 선정 및 상세 스코프 정의
- 데이터 수집·전처리 계획 수립 (공개 데이터/자체 생성/크롤링)
- 모델·시스템 아키텍처 설계 (아키텍처 다이어그램 포함)
- 개발 로드맵 (14~15주차 체크포인트 정의)

**14주차 B회차의 역할**:
- A회차 설계에 따라 **실제 개발 진행**
- 개별 면담을 통한 **문제 해결 및 최적화**
- 배포 가능한 **MVP(최소 기능 제품) 완성**
- 15주차 발표를 위한 **자료 및 데모 준비**

---

### 과제 스펙

**과제**: 개인 프로젝트 핵심 기능 개발 + 배포 체크리스트 작성

**제출 형태**: 개인 제출, Google Classroom 업로드 + GitHub Repository

**필수 산출물**:
1. **구현 코드** (Python 스크립트 또는 Jupyter 노트북)
   - 데이터 로드 및 전처리 코드
   - 모델 학습/추론 코드
   - 성능 평가 코드
2. **배포 준비 파일**
   - `requirements.txt` (의존성 명시)
   - `app.py` 또는 `main.py` (FastAPI/Streamlit 엔트리포인트) — 배포할 경우
   - `Dockerfile` — Docker 배포할 경우
3. **중간 결과 보고서** (1~2페이지)
   - 현재 완료도 (%) 및 다음 주 계획
   - 발생한 문제 및 해결 방법
   - 성능 지표 (정확도, F1, BLEU 등)

**검증 기준**:
- ✓ 데이터 로드/전처리 완료
- ✓ 모델 학습 또는 파인튜닝 실행
- ✓ 성능 평가 지표 계산 및 기록
- ✓ 배포 환경 설정 (requirements.txt, Dockerfile 작성 시작)
- ✓ API 엔드포인트 또는 UI 프로토타입 작동 확인
- ✓ GitHub Repository에 코드 푸시

---

### 14주차 B회차 운영 방식

#### 전체 흐름

**00:00~00:05 | 진행 상황 공유 및 브리핑**

- 교수가 전체 학생의 프로젝트 주제 및 진행률 확인
- A회차 이후 진행 사항 질문 (데이터 수집 완료 여부, 모델 선정 결정 등)
- 공통적인 문제점 미리 안내 (예: GPU 메모리 부족, 데이터셋 불일치 등)
- 배포 일정 및 발표 시간 배정 안내

**예시 브리핑**:
> "A회차 이후 모두들 데이터 수집을 마쳤나요? 만약 크롤링 중인 학생이 있다면, 오늘 면담 시간에 개별 피드백을 받으세요. 또한 Hugging Face 모델을 파인튜닝하려는 경우, GPU 메모리 부족 문제가 발생할 수 있으니, 필요하면 QLoRA나 LoRA를 사용해서 메모리를 절약하세요. 오늘 특히 배포 준비(요구사항 파일, Dockerfile)를 시작하시고, 다음 주 발표를 위해 슬라이드도 준비하기 시작하세요."

**00:05~01:00 | 개발 시간 + 교수 순회 피드백**

교수는 교실을 순회하며 각 학생의 진행 상황을 확인하고 즉각적인 조언을 제공한다.

**교수 순회의 목표**:
- 학생들의 코드를 확인하고 오류 디버깅 지원
- 설계 재검토 및 개선 방향 제시
- 기술적 선택(모델, 라이브러리 등)의 타당성 검증
- 배포 가능성 평가 및 단순화 방향 제시

**예상 문제와 교수의 즉각적 조언**:

1. **GPU 메모리 부족**
   - 증상: `RuntimeError: CUDA out of memory`
   - 조언: 배치 크기 축소, 그래디언트 누적(Gradient Accumulation), QLoRA 활용
   - 코드 예시:
   ```python
   # 배치 크기 축소
   train_args = TrainingArguments(
       per_device_train_batch_size=2,  # 4 → 2로 축소
       gradient_accumulation_steps=8,  # 대신 누적해서 효과 유지
   )
   ```

2. **데이터셋 불일치 오류**
   - 증상: `ValueError: Expected 512 tokens, got 400`
   - 조언: 토크나이저 설정 재확인 (padding, truncation), 데이터 전처리 로직 점검
   - 코드 예시:
   ```python
   inputs = tokenizer(
       texts,
       padding='max_length',   # 반드시 설정
       truncation=True,
       max_length=512
   )
   ```

3. **모델 로드 실패**
   - 증상: `ConnectionError: Failed to download model`
   - 조언: 인터넷 연결 확인, Hugging Face 캐시 디렉토리 확인, 대체 모델 제안
   - 코드 예시:
   ```python
   import os
   os.environ['HF_HOME'] = '/path/to/cache'  # 캐시 위치 명시
   model = AutoModel.from_pretrained('model-name')
   ```

4. **검증 성능이 학습 성능보다 좋음 (Underfitting)**
   - 증상: 학습 손실 5.0, 검증 손실 4.5
   - 조언: 모델이 학습 부족 상태. 에포크 수 증가, 학습률 조정, 정규화 감소
   - 코드 예시:
   ```python
   train_args = TrainingArguments(
       num_train_epochs=5,      # 3 → 5로 증가
       learning_rate=2e-4,      # 작은 값으로 안정적 학습
       weight_decay=0.01,       # 정규화 감소
   )
   ```

5. **추론 속도가 너무 느림 (Latency 문제)**
   - 증상: 한 문장 추론에 2초 이상 소요
   - 조언: 배치 추론, 모델 양자화(Quantization), ONNX 최적화, 캐싱 추가
   - 코드 예시:
   ```python
   from onnxruntime import InferenceSession
   session = InferenceSession('model.onnx')  # ONNX 모델로 3~5배 고속화
   ```

6. **API 요청 시 타임아웃**
   - 증상: FastAPI 엔드포인트가 30초 이상 응답 없음
   - 조언: 비동기 처리(async), 배치 처리, 캐싱 추가
   - 코드 예시:
   ```python
   @app.post("/predict")
   async def predict(request: PredictRequest):
       # 동기 → 비동기로 변경
       result = await asyncio.to_thread(model.predict, request.text)
       return result
   ```

7. **Docker 이미지 크기가 너무 큼 (1GB 이상)**
   - 증상: `docker build`에 1시간 이상 소요, 이미지 크기 2GB
   - 조언: 다단계 빌드(Multi-stage build), 불필요한 패키지 제거, 최소화된 베이스 이미지 선택
   - Dockerfile 예시:
   ```dockerfile
   # 1단계: 빌드
   FROM python:3.10-slim as builder
   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt

   # 2단계: 실행 (빌드 파일 제외)
   FROM python:3.10-slim
   COPY --from=builder /usr/local/lib /usr/local/lib
   COPY app.py .
   CMD ["python", "app.py"]
   ```

8. **모델이 특정 언어에만 잘 작동**
   - 증상: 한국어는 정확도 92%, 영어는 65%
   - 조언: 다국어 모델 사용(mBERT, XLM-RoBERTa), 언어별 파인튜닝
   - 코드 예시:
   ```python
   model = AutoModel.from_pretrained('xlm-roberta-base')  # 다국어 지원
   ```

9. **배포 후 실시간 모델 업데이트 불가능**
   - 증상: 새 모델로 교체하려면 서버 재시작 필요
   - 조언: 모델 버저닝, 핫 리로드(Hot Reload), 블루-그린 배포(Blue-Green Deployment)
   - 코드 예시:
   ```python
   class ModelLoader:
       _model = None
       @classmethod
       def load_model(cls, model_name):
           cls._model = AutoModel.from_pretrained(model_name)
       @classmethod
       def get_model(cls):
           return cls._model
   ```

10. **평가 지표 해석 불명확**
    - 증상: Accuracy 0.89지만 F1은 0.65 (클래스 불균형)
    - 조언: Accuracy는 속임수. 클래스 불균형 시 F1, Precision, Recall을 함께 봐야 함
    - 코드 예시:
    ```python
    from sklearn.metrics import classification_report
    print(classification_report(y_true, y_pred, digits=4))
    # 각 클래스별로 Precision, Recall, F1 확인
    ```

**교수 순회 효율성 팁**:
- 학생 10~15명당 순회 시간 약 45분
- 각 학생당 2~3분씩 면담하며 진행 상황 확인 + 한 가지 핵심 조언 제공
- 개발 중인 학생은 계속 진행하도록 격려하고, 명백한 오류는 즉시 수정하도록 지도
- 완료도가 많이 뒤처진 학생(데이터 수집 미완료 등)은 01:00~01:20 개별 면담 시간으로 이동

**01:00~01:20 | 개별 면담 (문제 해결, 최적화 방향 제시)**

교수가 각 학생과 1:1로 만나 심화 상담을 진행한다. 순회 시간에 해결하지 못한 기술적 문제나 설계 재검토가 필요한 학생들을 우선으로 한다.

**면담 가이드 (학생 1명당 5~10분)**:

① **현황 파악** (1분)
- 현재 진행 상황을 간단하게 설명하게 함
- 예: "데이터 수집 완료, 모델 학습 중, API 구현 시작"

② **문제점 식별** (2분)
- 진행 중 막혔던 부분 파악
- 예: "배치 크기 8일 때 CUDA 메모리 부족", "검증 정확도가 50%로 너무 낮음"

③ **원인 분석 + 해결책 제시** (3~4분)
- 왜 그런 문제가 발생했는지 설명하고, 구체적 해결 방법 제시
- 코드 스니펫 또는 논문 링크 제공
- 학생이 이해했는지 확인 (설명을 다시 말하도록 함)

④ **배포 준비 상태 점검** (2분)
- 배포 준비 체크리스트 검토
- 부족한 부분(Dockerfile, API 엔드포인트 등) 마무리 일정 논의

⑤ **격려 + 다음 목표** (1분)
- 15주차 발표 준비 안내
- 예: "지금 진도면 충분합니다. 다음 주는 발표 슬라이드를 만들고, 데모를 3~5번 반복해서 연습하세요."

**면담 체크리스트 (학생용 서식)**:

```
[ ] 데이터 수집 완료 (공개 데이터 또는 자체 생성)
[ ] 전처리 파이프라인 구현 완료
[ ] 모델 선정 및 학습 시작
[ ] 성능 평가 지표 계산 (Accuracy, F1, BLEU 등)
[ ] API 엔드포인트 구현 (FastAPI/Streamlit)
[ ] Docker 설정 시작 (Dockerfile 작성)
[ ] GitHub에 코드 푸시
[ ] 배포 테스트 (로컬 또는 테스트 서버)
[ ] 15주차 발표 슬라이드 틀 작성
[ ] 데모 비디오 또는 라이브 데모 준비
```

**01:20~01:28 | 배포 체크리스트 확인 + 발표 자료 안내**

**배포 준비 체크리스트** (모든 학생이 확인해야 함):

**1. 코드 정리 및 문서화**
- [ ] 함수/클래스에 docstring 추가
- [ ] 주석은 "무엇(What)"이 아닌 "왜(Why)"에 집중
- [ ] 변수명이 명확한가? (x, y 대신 input_text, predictions)
- [ ] 불필요한 코드 정리 (테스트 코드, 프린트문 제거)

**2. 의존성 관리**
- [ ] `requirements.txt` 또는 `environment.yml` 작성
  ```
  torch==2.0.1
  transformers==4.35.0
  fastapi==0.104.0
  pydantic==2.5.0
  pandas==2.0.0
  ```
- [ ] 각 패키지의 버전을 명시 (최신 버전 x, 검증된 버전 o)
- [ ] 로컬 환경에서 `pip install -r requirements.txt` 실행하여 재설치 테스트

**3. 데이터 처리**
- [ ] 데이터 경로가 상대 경로(relative path)인가?
  ```python
  # ❌ 잘못된 예
  data_path = '/home/user/data/dataset.csv'

  # ✅ 올바른 예
  data_path = os.path.join(os.path.dirname(__file__), 'data', 'dataset.csv')
  ```
- [ ] 데이터 크기가 배포 환경에 맞는가? (너무 크면 클라우드 스토리지 활용)
- [ ] 프라이버시 고려: 민감 정보 제거 또는 익명화

**4. 모델 저장 및 로드**
- [ ] 학습한 모델이 저장되어 있는가? (재학습 필요 없음)
  ```python
  model.save_pretrained('./model_checkpoint')
  tokenizer.save_pretrained('./model_checkpoint')
  ```
- [ ] 배포 시 모델을 로드하는 코드가 있는가?
  ```python
  model = AutoModel.from_pretrained('./model_checkpoint')
  ```

**5. API 엔드포인트 (FastAPI 사용 시)**
- [ ] `/health` 또는 `/` 엔드포인트로 서버 상태 확인 가능
- [ ] 입력 검증 (Pydantic 모델)
- [ ] 예외 처리 (HTTPException으로 에러 반환)
- [ ] Swagger UI로 API 문서 자동 생성 확인 (`http://localhost:8000/docs`)

**6. Docker 설정 (배포할 경우)**
- [ ] `Dockerfile` 작성 완료
- [ ] `docker build` 성공
- [ ] `docker run` 후 API 또는 UI 접근 가능
- [ ] 이미지 크기 최적화 (1GB 이상 시 다단계 빌드 검토)

**7. 성능 및 최적화**
- [ ] 추론 시간 측정 (1개 예제당 소요 시간)
- [ ] 배치 추론 지원 (여러 입력 동시 처리)
- [ ] 메모리 사용량 측정 (학습, 추론 각각)
- [ ] 필요 시 양자화 또는 ONNX 최적화 적용

**8. 테스트**
- [ ] 로컬 환경에서 전체 파이프라인 실행 성공
- [ ] 여러 입력값으로 테스트 (정상 데이터 + 엣지 케이스)
  ```python
  # 엣지 케이스 예시
  test_inputs = [
      "",                                  # 빈 입력
      "a" * 10000,                         # 초과 길이
      "한글 🔥 English mixed",             # 다국어 혼합
  ]
  ```
- [ ] 오류 발생 시 graceful하게 처리 (crash하지 않음)

**9. 배포 플랫폼 선택 및 설정**
- [ ] Hugging Face Spaces (권장: 가장 간단)
- [ ] AWS, GCP, Azure (더 많은 제어 필요)
- [ ] Heroku (무료 플랜 종료 예정)
- [ ] 자체 서버 (고급)

**10. 문서화 및 README**
- [ ] `README.md` 작성 (프로젝트 설명, 사용법, 결과)
- [ ] 예제 입력/출력 포함
- [ ] 설치 방법 명시
- [ ] 배포된 URL 또는 데모 링크 포함

---

**발표 자료 준비 안내** (15주차 A회차 발표용)

**발표 시간**: 1명당 5~7분 (질의응답 포함)

**필수 포함 요소** (슬라이드 수 가이드):

1. **제목 슬라이드** (1장)
   - 프로젝트 제목, 학번, 이름, 제출일

2. **문제 정의 및 동기** (1~2장)
   - "이 프로젝트를 왜 했는가?" 를 명확하게
   - 예: "한국어 감정 분석 모델이 이미 많지만, 특정 도메인(예: 영화 리뷰)에 맞게 파인튜닝된 모델은 부족하다"
   - 해결할 문제를 데이터로 보여주기 (정확도가 낮은 예시 캡처)

3. **시스템 아키텍처** (1~2장)
   - 데이터 흐름도 (Mermaid 또는 그림)
   - 모델 아키텍처 (코드 또는 다이어그램)
   - 배포 구조 (API, 프론트엔드, DB 관계)

4. **데이터 및 방법론** (1~2장)
   - 사용한 데이터셋 크기, 클래스 분포, 출처
   - 전처리 방법
   - 사용한 모델 (BERT, GPT, LoRA 등) 및 선택 이유
   - 하이퍼파라미터 (batch size, learning rate, epochs)

5. **실험 결과 및 성능** (1~2장)
   - 핵심 정량 결과 (표 또는 차트)
     - 정확도, F1, BLEU 등
     - 학습 곡선 (손실 그래프)
   - 정성적 결과 (예시 입력/출력)
   - 한계점 명시 (예: 특정 언어는 성능이 떨어짐)

6. **데모 또는 스크린샷** (1~2장)
   - 배포된 서비스 라이브 데모 또는 영상
   - API Swagger UI 스크린샷
   - 웹 인터페이스 스크린샷

7. **기술적 도전 및 해결** (1장)
   - 개발 중 겪었던 문제와 해결 방법
   - 예: "초기 GPU 메모리 부족 → QLoRA 도입으로 해결"

8. **향후 개선 방향** (1장)
   - 더 할 수 있는 작업들 (시간 부족으로 미완료)
   - 예: "다국어 지원", "모바일 앱 개발", "리얼타임 파인튜닝"

9. **결론** (1장)
   - 프로젝트 요약 (3~4문장)
   - 배운 점

10. **참고자료** (1장, 필요시)
    - 사용한 논문, 오픈소스, 튜토리얼 링크

**발표 슬라이드 제작 팁**:
- 텍스트는 최소화 (키워드 위주)
- 그래프, 표, 다이어그램 활용 (이해도 향상)
- 폰트는 크게 (교실 뒤에서도 읽을 수 있어야 함)
- 색상은 최대 3가지 (일관성 유지)
- 애니메이션은 최소 (분산 주의)

**01:28~01:30 | 15주차 최종 발표 안내 및 마무리**

**15주차 A회차 발표 시간표** (학생 수에 따라 조정):
- 총 30명 기준: 1명당 5~7분 × 30 = 150~210분 (2~3 시간)
- 시간이 부족하면 학생 당 4~5분으로 단축 가능

**발표 순서**:
- 교수가 프로젝트 주제 난이도 또는 기술 스택을 고려하여 순서 정함
- 각 학생은 준비 시간 5분 제공 (프로젝터 설정, 데모 테스트 등)

**평가 기준** (100점 만점):

| 항목 | 배점 | 평가 요소 |
|------|------|----------|
| 문제 정의의 명확성 및 창의성 | 15점 | 문제가 명확한가? 해결 가치가 있는가? |
| 기술적 구현의 깊이 | 30점 | 모델 아키텍처, 파인튜닝, 최적화 등 |
| 모델 성능 및 분석 | 20점 | 정량 결과, 정성적 평가, 한계점 인식 |
| 시스템 완성도 (배포, UI/API) | 15점 | 실제 배포 가능한가? 사용자 경험이 있는가? |
| 발표 및 의사소통 | 10점 | 설명이 명확한가? 청중 이해도가 높은가? |
| 코드 품질 및 문서화 | 10점 | README, docstring, 코드 정리 수준 |

**제출 마감**: 15주차 A회차 발표 30분 전까지 다음 파일 제출
- 발표 슬라이드 (PDF 또는 PPT)
- 프로젝트 최종 보고서 (PDF, 2~3페이지)
- GitHub Repository URL (코드 공개)

**마무리 안내**:
> "여러분의 프로젝트들이 정말 좋습니다. 14주 동안 배운 모든 기술을 직접 활용해서 완성도 있는 시스템을 만들었습니다. 다음 주 발표에서는 여러분의 노력과 창의성을 마음껏 보여주세요. 질문이 있으면 메일이나 오피스 아워로 연락 주세요. 화이팅!"

---

### 개발 시 자주 겪는 문제와 체크포인트 (심화)

개별 면담이나 순회 시 다음 체크포인트를 통해 학생들을 지도할 수 있다:

**체크포인트 1: 데이터 전처리 검증**
```python
# 데이터 로드 후 반드시 확인해야 할 항목
print("Dataset shape:", df.shape)                    # 크기 확인
print("Missing values:\n", df.isnull().sum())        # 결측치 확인
print("Class distribution:\n", df['label'].value_counts())  # 클래스 분포
print("Sample text:", df['text'].iloc[0][:100])     # 샘플 텍스트 확인
print("Text length stats:", df['text'].str.len().describe())  # 길이 분포
```

**체크포인트 2: 모델 로드 및 토크나이저 검증**
```python
# 모델과 토크나이저 호환성 확인
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained('klue/bert-base')
model = AutoModel.from_pretrained('klue/bert-base')

# 토크나이저 검증
sample_text = "한국어 텍스트 테스트"
tokens = tokenizer.tokenize(sample_text)
input_ids = tokenizer.encode(sample_text)
print(f"Tokens: {tokens}")
print(f"Input IDs: {input_ids}")
print(f"Token count: {len(input_ids)}")  # 512를 초과하지 않아야 함
```

**체크포인트 3: 학습 과정 모니터링**
```python
# 손실이 감소하는가? 과적합이 발생하는가?
# Hugging Face Trainer 사용 시 자동으로 training_loss, eval_loss 기록
# → TensorBoard로 시각화 가능
# tensorboard --logdir=./results

# 또는 수동으로 기록
train_losses = []
eval_losses = []
for epoch in range(num_epochs):
    train_loss = train_one_epoch()
    eval_loss = evaluate()
    train_losses.append(train_loss)
    eval_losses.append(eval_loss)
    print(f"Epoch {epoch}: train_loss={train_loss:.4f}, eval_loss={eval_loss:.4f}")

# 그래프로 시각화
import matplotlib.pyplot as plt
plt.plot(train_losses, label='Train Loss')
plt.plot(eval_losses, label='Eval Loss')
plt.legend()
plt.savefig('training_curve.png')
```

**체크포인트 4: 성능 평가 지표 계산**
```python
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import numpy as np

# 분류 모델의 경우
y_true = [1, 0, 1, 1, 0]
y_pred = [1, 0, 1, 0, 0]
y_proba = [0.9, 0.1, 0.8, 0.3, 0.2]

print(classification_report(y_true, y_pred))  # Precision, Recall, F1 상세 출력
print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
print("ROC-AUC:", roc_auc_score(y_true, y_proba))

# NLP 생성 모델의 경우 (BLEU, ROUGE 등)
from rouge_score import rouge_scorer
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
scores = scorer.score("생성한 텍스트", "참조 텍스트")
print(scores)  # rouge1, rouge2, rougeL 점수 출력
```

**체크포인트 5: API 엔드포인트 테스트**
```python
# FastAPI 서버 실행 후 테스트
import requests

url = "http://localhost:8000/predict"
payload = {"text": "이 영화 정말 좋아요"}
response = requests.post(url, json=payload)
print(response.json())

# 배치 요청 테스트
batch_payload = {
    "texts": ["영화 1", "영화 2", "영화 3"]
}
response = requests.post(url, json=batch_payload)
print(response.json())
```

---

### 흔한 배포 문제와 해결책

**문제 1: 로컬에서는 되는데 배포 후 안 됨**

원인: 파일 경로, 환경 변수, 모델 캐시 위치 등이 다를 수 있음

해결책:
```python
# ❌ 절대 경로 사용 (배포 시 실패)
model_path = '/home/myuser/models/model.pt'

# ✅ 상대 경로 또는 환경 변수 사용
import os
model_dir = os.getenv('MODEL_DIR', './models')
model_path = os.path.join(model_dir, 'model.pt')

# Dockerfile에서:
ENV MODEL_DIR=/app/models
COPY ./models /app/models
```

**문제 2: 배포 후 메모리 누수**

원인: 모델을 요청마다 로드하거나, 입력 데이터를 정리하지 않음

해결책:
```python
from fastapi import FastAPI
import torch

app = FastAPI()

# 모델을 한 번만 로드 (앱 시작 시)
model = None

@app.on_event("startup")
async def startup_event():
    global model
    model = AutoModel.from_pretrained('klue/bert-base')
    model.eval()

@app.post("/predict")
async def predict(request: PredictRequest):
    with torch.no_grad():  # 그래디언트 계산 안 함 (메모리 절약)
        outputs = model(**inputs)
    return outputs

@app.on_event("shutdown")
async def shutdown_event():
    global model
    del model
    torch.cuda.empty_cache()
```

**문제 3: 동시 요청 처리 불가**

원인: 모델이 한 요청씩만 처리 (배치 미지원)

해결책:
```python
import asyncio
from typing import List

request_queue = asyncio.Queue()
batch_size = 8

async def batch_process():
    """배치 단위로 모아서 한 번에 처리"""
    while True:
        batch = []
        for _ in range(batch_size):
            item = await request_queue.get()
            batch.append(item)

        # 배치 추론
        results = model.batch_predict(batch)
        for item, result in zip(batch, results):
            item['future'].set_result(result)

@app.post("/predict")
async def predict(request: PredictRequest):
    future = asyncio.Future()
    await request_queue.put({'request': request, 'future': future})
    result = await future
    return result
```

---

### 15주차 최종 발표를 위한 준비물 체크리스트

학생들이 14주차 B회차를 끝내면서 다음을 확인하도록 안내:

- [ ] GitHub Repository에 전체 코드 푸시 완료
- [ ] `README.md` 작성 완료 (설치 방법, 사용법, 결과 포함)
- [ ] 배포된 서비스 URL 또는 데모 영상 준비
- [ ] 발표 슬라이드 초안 완성
- [ ] 로컬 환경에서 전체 파이프라인 재실행 테스트 (재현 가능성 확인)
- [ ] 발표 리허설 최소 1회 완료 (5~7분 내 마무리 가능 확인)
- [ ] 최종 프로젝트 보고서 작성 (2~3페이지)

---

### 교수 평가 및 피드백 요점

**B회차 평가 기준** (채점은 15주차 발표 후 종합하지만, 14주차 B회차에서 미리 파악):

| 평가 항목 | 좋음 | 보통 | 개선 필요 |
|----------|------|------|----------|
| 개발 진행도 | 70% 이상 완료 | 40~70% 완료 | 40% 미만 |
| 코드 품질 | docstring, 주석, 명확한 변수명 | 일부 정리됨 | 정리 필요 |
| 문제 해결 능력 | 오류 발생 시 스스로 디버깅 | 일부 조언 필요 | 많은 지도 필요 |
| 배포 준비 | 배포 경로 명확, 체크리스트 진행 중 | 기본 설정만 완료 | 미계획 |
| 시간 관리 | 일정 진도 유지 | 다소 뒤처짐 | 심각한 지연 |

**격려 메시지** (B회차 마지막):
> "지난 14주 동안 LLM의 원리부터 배포까지 정말 많은 것을 배웠습니다. 개인 프로젝트를 통해 여러분이 실무 수준의 AI 엔지니어로 성장했다고 느낍니다. 다음 주 발표에서 자신감 있게 여러분의 프로젝트를 보여주세요. 무엇보다 중요한 것은 완벽한 결과가 아니라, 어떤 과정을 거쳐서 그 결과에 도달했는지를 설명하는 것입니다. 화이팅!"

---

**마지막 업데이트**: 2026-02-25
**작성자**: Claude Code
**버전**: 1.0
