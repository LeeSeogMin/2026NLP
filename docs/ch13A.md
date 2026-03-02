## 13주차 A회차: 모델 배포와 프로덕션

> **미션**: 수업이 끝나면 학습한 NLP 모델을 FastAPI 서버에 배포하고, Docker로 컨테이너화하여 프로덕션 환경에서 운영하는 원리를 이해한다

### 학습목표

이 회차를 마치면 다음을 수행할 수 있다:

1. 로컬 개발과 프로덕션 배포의 차이를 설명하고, 배포 과정의 핵심 단계(스케일링, 모니터링, 버전 관리)를 이해할 수 있다
2. FastAPI를 사용하여 GET/POST 엔드포인트를 정의하고, Pydantic으로 요청/응답을 검증할 수 있다
3. 비동기(async) 처리와 Rate Limiting으로 동시 요청을 효율적으로 처리할 수 있다
4. ONNX Runtime과 배치 처리로 모델 추론 성능을 최적화할 수 있다
5. Dockerfile을 작성하고 Docker 이미지를 빌드·실행하여 환경 독립적으로 모델을 배포할 수 있다
6. Docker Compose를 사용하여 모델, 캐시, 로깅 등 여러 서비스를 조합할 수 있다

### 수업 타임라인

| 시간        | 내용                                                        | Copilot 사용                  |
| ----------- | ----------------------------------------------------------- | ----------------------------- |
| 00:00~00:05 | 오늘의 질문 + 빠른 진단(퀴즈 1문항)                         | 사용 안 함                    |
| 00:05~00:55 | 이론 강의 (배포 개념 → FastAPI → 최적화 → Docker)           | 사용 안 함                    |
| 00:55~01:25 | 라이브 코딩 시연 (BERT → FastAPI 래핑 → 배치 처리 → 동시성) | 직접 실습 또는 시연 영상 참고 |
| 01:25~01:28 | 핵심 정리 + B회차 과제 스펙 공개                            |                               |
| 01:28~01:30 | Exit ticket (1문항)                                         |                               |

---

### 오늘의 질문 + 빠른 진단

**오늘의 질문**: "여러분이 만든 감정 분류 모델이 좋은 성능을 냈다. 이제 누구나 사용할 수 있게 하려면 어떻게 해야 할까? 하루에 1000명이 동시에 모델을 써야 한다면?"

**빠른 진단 (1문항)**:

다음 중 "프로덕션 배포"의 핵심 조건으로 가장 중요한 것은?

① 모델의 정확도가 95% 이상이어야 한다
② 누구나 언제든지 사용할 수 있고, 여러 명의 동시 요청도 처리할 수 있어야 한다
③ 모델을 학습한 학생이 직접 관리해야 한다
④ 가장 최신의 하드웨어가 필요하다

정답: **②** — 배포는 모델의 정확도도 중요하지만, "서비스 가능성(Availability)", "동시성(Concurrency)", "신뢰성(Reliability)"이 더 중요하다.

---

### 이론 강의

#### 13.1 배포의 개념과 왜 필요한가

##### 로컬 개발과 프로덕션의 차이

당신이 감정 분류 모델을 학습했다고 하자. 로컬 컴퓨터에서 Jupyter Notebook을 열어 다음 코드를 실행한다:

```python
model = BertForSequenceClassification.from_pretrained('klue/bert-base')
text = "이 영화 정말 좋다!"
inputs = tokenizer(text, return_tensors='pt')
outputs = model(**inputs)
```

이는 **로컬 개발(Local Development)**이다. 문제가 하나 있다: **당신의 컴퓨터가 켜져 있을 때만** 모델이 작동한다. 당신이 잠들면 누구도 모델을 사용할 수 없다.

**프로덕션 배포(Production Deployment)**는 다르다. 24시간 켜진 서버에 모델을 올려두고, 누구나 웹 요청으로 접근할 수 있게 한다.

```
로컬 개발:
사용자 → (직접) → 내 컴퓨터의 Python 코드 → 결과

프로덕션 배포:
사용자 → (HTTP) → 웹 서버(내 코드) → 결과
```

**직관적 이해**: 당신이 맛있는 음식을 집에서 만들었다. 가족만 먹을 수 있다(로컬). 이제 식당을 차려서 누구나 돈을 내고 음식을 주문하고 먹을 수 있게 한다(배포). 식당은 24시간 오픈하고, 여러 고객을 동시에 처리하며, 주문이 잘못되지 않도록 체크하고, 건강상 문제가 없도록 점검한다.

배포에서 중요한 네 가지 요소가 있다:

1. **서비스 가능성(Availability)**: "언제 들어와도 켜져 있는가"
2. **동시성(Concurrency)**: "여러 사용자를 동시에 처리할 수 있는가"
3. **신뢰성(Reliability)**: "오류가 발생해도 서비스가 중단되지 않는가"
4. **성능(Performance)**: "응답이 빠른가"

#### 13.2 FastAPI로 모델을 HTTP 서비스로 만들기

##### HTTP 통신의 기초

웹 서비스는 **HTTP(Hypertext Transfer Protocol)**로 통신한다. HTTP에는 여러 메서드(Method)가 있다:

- **GET**: 서버로부터 데이터를 읽는다 (입력 없이 조회만 함)
- **POST**: 서버에 데이터를 보낸다 (입력 데이터를 포함해서 처리)
- **PUT**: 기존 데이터를 수정한다
- **DELETE**: 데이터를 삭제한다

모델 추론은 **POST**를 사용한다. 왜? 매번 다른 텍스트를 보내기 때문이다.

```
POST /predict HTTP/1.1
Content-Type: application/json

{
  "text": "이 영화 정말 좋다!"
}
```

서버는 이 JSON을 받아서 모델 추론을 수행한 뒤 결과를 JSON으로 반환한다:

```
HTTP/1.1 200 OK
Content-Type: application/json

{
  "text": "이 영화 정말 좋다!",
  "label": "긍정",
  "score": 0.987
}
```

##### FastAPI 프레임워크

**FastAPI**는 Python의 현대적 웹 프레임워크이다. 특징:

- **빠른 성능**: ASGI(비동기) 기반으로 매우 빠르다
- **자동 검증**: Pydantic을 사용하여 입력 데이터를 자동 검증한다
- **자동 문서화**: Swagger UI가 자동으로 생성된다
- **비동기 지원**: async/await로 I/O 대기 시간을 최소화한다

**직관적 이해**: FastAPI는 당신의 Python 함수를 "HTTP 엔드포인트"로 변환해준다. 함수를 정의하면 자동으로 웹 서버가 만들어진다.

간단한 예시:

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class PredictRequest(BaseModel):
    text: str

class PredictResponse(BaseModel):
    label: str
    score: float

@app.post("/predict")
def predict(request: PredictRequest):
    # 모델 추론
    outputs = model(request.text)
    return PredictResponse(
        label=outputs.label,
        score=outputs.score
    )
```

이 코드의 의미를 분해하면:

1. **PredictRequest**: 클라이언트가 보낼 JSON의 형식을 정의한다. text 필드가 반드시 있어야 한다.
2. **PredictResponse**: 서버가 반환할 JSON의 형식을 정의한다.
3. **@app.post("/predict")**: "/predict" 경로에 POST 요청이 들어오면 predict() 함수를 실행한다.
4. **자동 검증**: FastAPI가 자동으로 JSON을 PredictRequest로 파싱하고, 필드 타입이 맞는지 검증한다.

FastAPI는 요청이 들어올 때마다 다음을 수행한다:

```
클라이언트 JSON → Pydantic으로 검증 → Python 함수 실행 → 결과 JSON으로 변환 → 클라이언트에게 반환
```

> **쉽게 말해서**: Pydantic은 "입력 데이터가 올바른 형식인지 검사하는 경호원"이다. 잘못된 형식이면 400 Bad Request 에러를 던진다.

##### Swagger UI와 자동 문서화

FastAPI를 시작하면 자동으로 문서가 생성된다:

```bash
uvicorn main:app --reload
# http://127.0.0.1:8000/docs 열기
```

이 주소에서 Swagger UI가 나타나서, 웹 브라우저에서 직접 API를 테스트할 수 있다. 각 엔드포인트의 입력/출력 형식이 자동으로 문서화된다.

> **그래서 무엇이 달라지는가?** 로컬에서는 Jupyter에서 Python 함수를 직접 호출했다. 배포에서는 HTTP를 통해 원격으로 접근한다. FastAPI가 이 변환을 자동으로 처리해준다. 클라이언트는 Python 환경이 없어도 된다. 웹 브라우저만 있으면 충분하다.

#### 13.3 동시성 처리와 Rate Limiting

##### 동기(Sync) vs 비동기(Async)

1000명이 동시에 모델을 사용한다고 하자. 각 추론에 1초가 걸린다면?

**동기 처리의 문제**:

```python
@app.post("/predict")
def predict(request: PredictRequest):  # 동기 함수
    outputs = model(request.text)  # 1초 대기
    return {"label": outputs.label}
```

이 코드는 요청이 들어올 때마다 1초를 기다린다. 만약 두 명이 동시에 요청하면?

```
요청 1: 0초 시작 → 1초 끝
요청 2: 1초 시작 → 2초 끝  (1초 대기!)
```

요청 2는 요청 1이 끝날 때까지 기다려야 한다. 1000명이 요청하면 마지막 사람은 1000초를 기다린다.

**비동기 처리의 개선**:

```python
@app.post("/predict")
async def predict(request: PredictRequest):  # 비동기 함수
    outputs = await model_async(request.text)
    return {"label": outputs.label}
```

비동기 함수는 I/O 대기 중에 다른 요청을 처리할 수 있다:

```
요청 1: 0초 시작, I/O 대기 진입
요청 2: 0초 시작, I/O 대기 진입 (요청 1이 대기 중이니 바로 처리)
요청 3: 0초 시작, I/O 대기 진입
...
모든 요청: 약 1초 후 완료
```

**직관적 이해**: 음식점에 손님들이 줄 서 있다. 동기 방식은 한 명을 완전히 처리한 후 다음 명을 받는다(음식 준비 완료 → 영수증 발행 → 다음 손님). 비동기 방식은 음식 준비 중에 다른 손님의 주문을 받는다(손님 A의 음식이 조리 중 → 손님 B의 주문 받기 → 손님 A의 음식 완성 및 영수증 발행 → 손님 B로 돌아가기).

실제로 모델 추론은 GPU에서 일어나므로, CPU는 추론 중에 다른 요청을 받을 수 있다.

##### Rate Limiting: 서버 보호

만약 누군가 악의적으로 1초마다 1000개의 요청을 보낸다면? 서버의 메모리와 GPU가 터진다.

**Rate Limiting**은 "초당 최대 N개 요청"을 제한한다:

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.post("/predict")
@limiter.limit("10/minute")  # 분당 10개 요청만 허용
async def predict(request: PredictRequest):
    outputs = model(request.text)
    return {"label": outputs.label}
```

이제 한 사용자가 분당 10개 이상 요청하면 429 Too Many Requests 에러를 받는다.

> **그래서 무엇이 달라지는가?** Rate Limiting 없으면 악의적 사용자가 서버를 마비시킬 수 있다(DDoS 공격). 제한이 있으면 공정하게 자원을 배분할 수 있다.

#### 13.4 모델 최적화와 배치 처리

##### 배치 처리: 여러 요청을 모아서 한 번에 처리

신경망의 성능을 생각해보자. 1개 샘플을 처리하는 시간과 32개 샘플을 처리하는 시간이 같다면? 배치 처리가 훨씬 효율적이다.

```
배치 크기 1: 각 1ms → 100개 처리하면 100ms
배치 크기 32: 각 10ms → 100개를 3.125개 배치로 처리 → 약 32ms (3배 빠름!)
```

요청이 들어올 때마다 즉시 처리하지 말고, 조금 기다렸다가 여러 요청을 모아서 한 번에 배치 처리하면 처리량을 크게 늘릴 수 있다.

```python
import asyncio
from queue import Queue

batch_queue = Queue()
batch_results = {}

async def batch_processor():
    while True:
        # 최대 32개 요청을 모을 때까지 대기, 또는 100ms 타임아웃
        batch = []
        while len(batch) < 32:
            try:
                req_id, text = batch_queue.get(timeout=0.1)
                batch.append((req_id, text))
            except:
                break

        if batch:
            # 배치 추론
            texts = [text for _, text in batch]
            outputs = model(texts)  # 모든 텍스트를 한 번에 처리

            # 결과 저장
            for (req_id, _), output in zip(batch, outputs):
                batch_results[req_id] = output

@app.post("/predict")
async def predict(request: PredictRequest):
    req_id = uuid.uuid4()
    batch_queue.put((req_id, request.text))

    # 결과가 나올 때까지 대기
    while req_id not in batch_results:
        await asyncio.sleep(0.01)

    return batch_results[req_id]
```

**직관적 이해**: 은행의 자동 환전기를 생각해보자. 거래마다 거래비가 붙으면 비효율적이다. 여러 거래를 모아서 한 번에 처리하는 게 낫다.

##### 캐싱: 반복되는 입력에 대한 빠른 응답

같은 텍스트가 여러 번 입력되는 경우가 많다. 이미 한 번 추론했다면 결과를 캐시해두고 바로 반환하는 게 빠르다:

```python
from functools import lru_cache

@lru_cache(maxsize=10000)
def cached_predict(text: str):
    outputs = model(text)
    return outputs.label

@app.post("/predict")
async def predict(request: PredictRequest):
    label = cached_predict(request.text)
    return {"label": label}
```

첫 요청은 1초, 두 번째 요청(같은 텍스트)은 1ms가 된다.

**표 13.1** 최적화 기법 비교

| 기법                 | 시간 절감                 | 적용 난도 | 병렬성                  |
| -------------------- | ------------------------- | --------- | ----------------------- |
| 캐싱                 | 중간 (반복 요청)          | 낮음      | 높음 (읽기 공유)        |
| 배치 처리            | 높음 (처리량 증가)        | 중간      | 높음 (모아서 처리)      |
| ONNX Runtime         | 중간~높음 (인퍼런스 속도) | 높음      | 중간 (모델 변환 필요)   |
| 양자화(Quantization) | 높음 (메모리·속도)        | 높음      | 중간 (정확도 손실 가능) |

> **쉽게 말해서**: 최적화는 "자주 하는 일은 빠르게, 많은 일은 한 번에"를 의미한다.

#### 13.5 Docker와 컨테이너화

##### 환경 문제: 내 컴퓨터에선 되는데 남의 컴퓨터에서 안 된다

당신이 만든 모델 서버를 다른 사람에게 준다고 하자. 그 사람이 설치한다:

```bash
git clone myrepo
python main.py
```

에러가 난다:

```
ModuleNotFoundError: No module named 'transformers'
```

"아, requirements.txt를 설치해야 해!"

```bash
pip install -r requirements.txt
```

또 에러:

```
ImportError: libcuda.so.11.2: cannot open shared object file
```

"아, CUDA 11.2를 설치해야 해!"

Python, PyTorch, CUDA, transformers 등 모든 버전이 정확히 맞아야 한다. 이를 **환경 문제(Dependency Hell)**라 한다.

**직관적 이해**: 당신이 요리 레시피를 준다고 하자. "이 음식을 만드는 데 필요한 재료: 밀가루, 계란, 우유". 그런데 사람마다 집에 있는 재료가 다르다. 누군가는 계란이 없고, 누군가는 밀가루 종류가 다르다. 완벽한 결과를 원한다면, "정확히 이 재료들을 이 양으로 섞은 반죽을 받으세요"라고 하면 된다.

**Docker**는 정확히 그런 역할을 한다. "이 코드가 실행되는 정확한 환경" 전체를 패키징한다.

##### Dockerfile과 이미지 빌드

Dockerfile은 "이 Docker 이미지를 어떻게 만들지"를 정의하는 레시피이다:

```dockerfile
# 베이스 이미지: 기본 OS와 Python 3.10
FROM python:3.10-slim-bullseye

# 작업 디렉토리 설정
WORKDIR /app

# 의존성 파일 복사
COPY requirements.txt .

# 의존성 설치
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드 복사
COPY . .

# 포트 8000 노출
EXPOSE 8000

# 서버 실행
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

이 파일을 빌드하면 Docker 이미지가 만들어진다:

```bash
docker build -t my-model:1.0 .
```

그러면 누구든 이 이미지를 받아서 실행할 수 있다:

```bash
docker run -p 8000:8000 my-model:1.0
```

> **그래서 무엇이 달라지는가?** 설치 문제로 인한 "내 컴퓨터에선 되는데"라는 말이 사라진다. Docker가 환경 전체를 캡슐화하므로, 어떤 컴퓨터에서든 정확히 같은 환경에서 실행된다.

##### 도커 이미지의 레이어 구조

Docker 이미지는 **레이어(Layer)**로 이루어져 있다:

```
Dockerfile 내용:
1. FROM python:3.10-slim-bullseye      → 베이스 레이어 (약 200MB)
2. COPY requirements.txt .             → 의존성 파일 레이어 (1KB)
3. RUN pip install -r requirements.txt → 설치 레이어 (약 500MB)
4. COPY . .                            → 코드 레이어 (몇 MB)
5. EXPOSE 8000                         → 메타데이터
6. CMD ["uvicorn", "main:app", ...]   → 메타데이터

최종 이미지 크기: 약 700MB
```

레이어는 **캐싱**된다. 코드만 변경했다면, 1~3번 레이어는 재사용되고 4번만 다시 빌드된다.

**표 13.2** 이미지 vs 컨테이너

| 항목 | 이미지        | 컨테이너             |
| ---- | ------------- | -------------------- |
| 개념 | 정적 설계도   | 실행 중인 프로세스   |
| 저장 | 디스크에 저장 | 메모리에서 실행      |
| 상태 | 변하지 않음   | 계속 변함            |
| 비유 | 요리 레시피   | 실제로 요리하는 과정 |
| 명령 | docker build  | docker run           |

##### Docker Compose: 여러 서비스 조합

실제 서비스는 모델 서버 하나만으로는 부족하다:

- **모델 서버**: FastAPI로 추론 처리
- **캐시 레이어**: Redis로 결과 캐싱
- **로깅 시스템**: 모든 요청을 기록
- **데이터베이스**: 사용자 정보 저장

각각을 별도의 Docker 컨테이너로 실행하면, 독립적으로 관리할 수 있다. **Docker Compose**는 이 여러 서비스를 한 번에 관리한다:

```yaml
# docker-compose.yml
version: "3.8"

services:
  model-server:
    build: .
    ports:
      - "8000:8000"
    environment:
      REDIS_URL: redis://cache:6379
    depends_on:
      - cache

  cache:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  logs:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
```

한 명령으로 모든 서비스를 띄운다:

```bash
docker-compose up
```

**직관적 이해**: 오케스트라를 생각해보자. 바이올린, 첼로, 플루트, 드럼이 독립적으로 악기를 연주한다. Docker Compose는 지휘자처럼 모두를 조화시킨다.

---

### 라이브 코딩 시연

> **학습 가이드**: BERT 감정 분류 모델을 FastAPI 서버로 감싸고, 배치 처리와 동시성 제어를 구현하여 실제 부하를 견딜 수 있는 서버를 직접 실습하거나 시연 영상을 참고하여 따라가 보자.

#### [단계 1] BERT 모델 로드와 기본 FastAPI 엔드포인트

```python
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

app = FastAPI()

# 모델과 토크나이저 로드 (앱 시작 시 1회만)
model_name = "klue/bert-base-multilingual-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# 모델을 GPU로 이동
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# 요청/응답 스키마
class PredictRequest(BaseModel):
    text: str

class PredictResponse(BaseModel):
    text: str
    label: int
    logits: list
    confidence: float

# 기본 엔드포인트
@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    # 토크나이징
    inputs = tokenizer(
        request.text,
        return_tensors='pt',
        truncation=True,
        max_length=128,
        padding=True
    )

    # 모델로 이동
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # 추론
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits[0].cpu().numpy().tolist()
    pred_label = int(torch.argmax(outputs.logits, dim=1)[0])
    confidence = float(torch.max(torch.softmax(outputs.logits, dim=1), dim=1)[0])

    return PredictResponse(
        text=request.text,
        label=pred_label,
        logits=logits,
        confidence=confidence
    )

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
```

**테스트**:

```bash
uvicorn main:app --reload
# http://127.0.0.1:8000/docs 열기 → Swagger UI에서 /predict 테스트
```

JSON 요청:

```json
{
  "text": "이 영화는 정말 재미있었다!"
}
```

응답:

```json
{
  "text": "이 영화는 정말 재미있었다!",
  "label": 1,
  "logits": [0.234, 0.766],
  "confidence": 0.766
}
```

#### [단계 2] 배치 처리와 요청 큐

```python
import asyncio
import uuid
from queue import Queue
from threading import Thread
import numpy as np

# 배치 처리 큐와 결과 저장소
batch_queue = asyncio.Queue(maxsize=100)
request_results = {}

async def batch_processor():
    """배치 요청을 모아서 한 번에 처리"""
    while True:
        batch = []
        batch_ids = []

        # 최대 16개 요청을 모을 때까지 대기 (또는 500ms 타임아웃)
        timeout = asyncio.get_event_loop().time() + 0.5

        while len(batch) < 16:
            try:
                wait_time = timeout - asyncio.get_event_loop().time()
                if wait_time <= 0:
                    break

                req_id, text = await asyncio.wait_for(
                    batch_queue.get(),
                    timeout=wait_time
                )
                batch.append(text)
                batch_ids.append(req_id)
            except asyncio.TimeoutError:
                break

        if not batch:
            await asyncio.sleep(0.01)
            continue

        # 배치 토크나이징
        inputs = tokenizer(
            batch,
            return_tensors='pt',
            truncation=True,
            max_length=128,
            padding='longest'
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        # 배치 추론
        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits.cpu().numpy()
        labels = np.argmax(logits, axis=1)
        confidences = np.max(torch.softmax(outputs.logits, dim=1).cpu().numpy(), axis=1)

        # 결과 저장
        for req_id, text, logit, label, conf in zip(
            batch_ids, batch, logits, labels, confidences
        ):
            request_results[req_id] = {
                'text': text,
                'label': int(label),
                'logits': logit.tolist(),
                'confidence': float(conf)
            }

# 배치 처리 태스크 시작
@app.on_event("startup")
async def startup():
    asyncio.create_task(batch_processor())

@app.post("/predict/batch", response_model=PredictResponse)
async def predict_batch(request: PredictRequest):
    req_id = str(uuid.uuid4())

    # 큐에 요청 추가
    await batch_queue.put((req_id, request.text))

    # 결과가 나올 때까지 대기 (최대 10초)
    wait_count = 0
    while req_id not in request_results and wait_count < 100:
        await asyncio.sleep(0.1)
        wait_count += 1

    if req_id not in request_results:
        return {"error": "Processing timeout"}

    result = request_results.pop(req_id)
    return PredictResponse(**result)
```

이 배치 처리 방식은 다음과 같이 작동한다:

1. 요청들이 들어오면 큐에 추가된다
2. 배치 프로세서가 최대 16개를 모으거나 500ms를 기다린다
3. 모인 요청들을 한 번에 배치 추론한다 (GPU 병렬 처리)
4. 결과를 각 요청자에게 반환한다

**성능 비교**:

```
단일 처리: 16개 요청 × 1초 = 16초
배치 처리: 배치 크기 16 × 1초 = 약 1.5초 (단일 배치 내 병렬화 + 오버헤드)

처리량: 단일 1개/초 vs 배치 약 10개/초 (10배 개선!)
```

#### [단계 3] Rate Limiting과 에러 처리

```python
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi import HTTPException

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request, exc):
    return {
        "error": "Rate limit exceeded",
        "detail": str(exc.detail)
    }

@app.post("/predict", response_model=PredictResponse)
@limiter.limit("30/minute")  # 분당 30개 요청 제한
async def predict(request: PredictRequest):
    # 입력 검증
    if not request.text or len(request.text) == 0:
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    if len(request.text) > 5000:
        raise HTTPException(status_code=400, detail="Text too long (max 5000 chars)")

    # ... 나머지 코드

    try:
        # 추론 수행
        inputs = tokenizer(request.text, return_tensors='pt', truncation=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
        # ...
    except Exception as e:
        # 예외 로깅
        print(f"Error during inference: {str(e)}")
        raise HTTPException(status_code=500, detail="Inference failed")
```

#### [단계 4] 캐싱과 성능 모니터링

```python
from functools import lru_cache
import time
from typing import Dict

# 간단한 메모리 캐시 (프로덕션에는 Redis 권장)
@lru_cache(maxsize=10000)
def cached_tokenize(text: str):
    return tokenizer.encode(text, truncation=True, max_length=128)

# 성능 메트릭 수집
class MetricsCollector:
    def __init__(self):
        self.total_requests = 0
        self.total_time = 0.0
        self.cache_hits = 0
        self.batch_sizes = []

    def record(self, duration: float, cache_hit: bool = False):
        self.total_requests += 1
        self.total_time += duration
        if cache_hit:
            self.cache_hits += 1

    def get_stats(self) -> Dict:
        avg_time = self.total_time / self.total_requests if self.total_requests > 0 else 0
        cache_hit_rate = (self.cache_hits / self.total_requests * 100) if self.total_requests > 0 else 0

        return {
            'total_requests': self.total_requests,
            'avg_latency_ms': avg_time * 1000,
            'cache_hit_rate_percent': cache_hit_rate
        }

metrics = MetricsCollector()

@app.post("/predict", response_model=PredictResponse)
@limiter.limit("30/minute")
async def predict(request: PredictRequest):
    start_time = time.time()

    # 캐시 확인
    cached_ids = cached_tokenize(request.text)

    # ... 추론 로직 ...

    duration = time.time() - start_time
    metrics.record(duration, cache_hit=False)

    return PredictResponse(...)

@app.get("/metrics")
async def get_metrics():
    """성능 메트릭 확인"""
    return metrics.get_stats()
```

테스트 결과:

```
http://127.0.0.1:8000/metrics

{
  "total_requests": 150,
  "avg_latency_ms": 245,
  "cache_hit_rate_percent": 12.3
}
```

#### [단계 5] Dockerfile 작성

```dockerfile
# Dockerfile

# 베이스 이미지: CUDA 11.8 + Python 3.10
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

WORKDIR /app

# Python 설치
RUN apt-get update && apt-get install -y python3.10 python3-pip && rm -rf /var/lib/apt/lists/*

# 의존성 파일 복사
COPY requirements.txt .

# Python 의존성 설치 (캐시 미사용으로 용량 절감)
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드 복사
COPY . .

# 포트 8000 노출
EXPOSE 8000

# 헬스 체크
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3 -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# uvicorn으로 서버 실행
CMD ["python3", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

`requirements.txt`:

```
fastapi==0.104.1
uvicorn[standard]==0.24.0
torch==2.1.0
transformers==4.34.0
slowapi==0.1.9
pydantic==2.5.0
```

빌드 및 실행:

```bash
# 이미지 빌드
docker build -t bert-model:1.0 .

# 컨테이너 실행
docker run -p 8000:8000 --gpus all bert-model:1.0

# 접근
curl http://localhost:8000/health
```

#### [단계 6] Docker Compose로 전체 시스템 구성

```yaml
# docker-compose.yml
version: "3.8"

services:
  model-server:
    build: .
    container_name: bert-api
    ports:
      - "8000:8000"
    environment:
      MODEL_NAME: klue/bert-base-multilingual-cased
      BATCH_SIZE: 16
      CACHE_SIZE: 10000
    volumes:
      - ./logs:/app/logs
      - ./models:/app/models
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    depends_on:
      - redis
      - prometheus

  redis:
    image: redis:7-alpine
    container_name: bert-cache
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  prometheus:
    image: prom/prometheus:latest
    container_name: bert-monitoring
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus

volumes:
  redis_data:
  prometheus_data:
```

한 명령으로 전체 시스템 시작:

```bash
docker-compose up -d

# 서비스 확인
docker-compose ps

# 로그 확인
docker-compose logs -f model-server

# 중지
docker-compose down
```

---

### 핵심 정리 + B회차 과제 스펙

#### 이 회차의 핵심 내용

- **배포란** 학습한 모델을 24시간 켜진 서버에 올려서 누구나 웹으로 접근할 수 있게 하는 것이며, 서비스 가능성(Availability), 동시성(Concurrency), 신뢰성(Reliability), 성능(Performance)이 핵심이다.

- **FastAPI**는 Python 함수를 HTTP 엔드포인트로 변환해주는 프레임워크로, Pydantic으로 입력 검증을 자동화하고 Swagger UI로 문서화한다.

- **비동기(async) 처리**는 I/O 대기 중에 다른 요청을 처리하여 동시성을 크게 향상시킨다. 동기 방식보다 같은 시간에 훨씬 많은 요청을 처리할 수 있다.

- **Rate Limiting**으로 초당 최대 요청 수를 제한하여 서버 부하를 관리하고, 악의적 사용자로부터 보호한다.

- **배치 처리**는 여러 요청을 모아서 한 번에 추론하여 처리량을 크게 향상시킨다. 일반적으로 3~10배의 처리량 개선을 기대할 수 있다.

- **캐싱**으로 반복되는 입력에 대한 응답 시간을 1ms 수준으로 줄일 수 있다.

- **Docker**는 애플리케이션과 환경(라이브러리, Python 버전 등)을 담은 컨테이너로 배포하여, "내 컴퓨터에선 되는데"라는 문제를 완전히 해결한다.

- **Docker 이미지는 레이어로 구성**되며, 변경 부분만 재빌드되어 빌드 시간을 단축할 수 있다. 이미지는 정적 설계도이고, 컨테이너는 실행 중인 프로세스이다.

- **Docker Compose**는 모델 서버, 캐시, 로깅, 모니터링 등 여러 서비스를 한 번에 관리하여 프로덕션 환경을 쉽게 구성할 수 있다.

#### B회차 과제 스펙

**B회차 (90분) — 실습 + 토론**: FastAPI로 BERT 모델 배포 + Docker 컨테이너화

**과제 목표**:

- FastAPI로 감정 분류 모델을 웹 서비스화한다
- 배치 처리와 캐싱으로 성능을 최적화한다
- Dockerfile을 작성하여 Docker 이미지로 배포한다
- Rate Limiting과 헬스 체크로 신뢰성을 확보한다

**과제 구성** (3단계, 30~40분 완결):

- **체크포인트 1 (12분)**: FastAPI 엔드포인트 구현 (GET /health, POST /predict)
- **체크포인트 2 (15분)**: 배치 처리와 캐싱 추가, 성능 테스트
- **체크포인트 3 (10분)**: Dockerfile 작성 + 로컬 도커 빌드 및 실행 테스트

**제출 형식**:

- 완성된 코드 파일 (`practice/chapter13/code/13-1-fastapi-deployment.py`)
- Dockerfile (`practice/chapter13/code/Dockerfile`)
- requirements.txt (`practice/chapter13/code/requirements.txt`)
- 성능 리포트 (응답 시간, 처리량, 캐시 히트율 포함, 1~2문단)

**Copilot 활용 가이드**:

- 기본: "FastAPI로 BERT 감정 분류 모델을 감싸는 코드를 작성해줘"
- 심화: "배치 처리와 Rate Limiting을 추가해줄 수 있어?"
- 검증: "이 Dockerfile이 올바른지 확인해줄래? 모델 로드 시간을 줄이는 방법도 조언해줘"

---

### Exit ticket

**문제 (1문항)**:

배치 처리를 사용할 때, 최대 배치 크기를 16으로 설정하고 타임아웃을 500ms로 설정했다. 다음 중 일어날 수 있는 결과를 모두 선택하시오. (복수 선택)

```
Case A: 15개 요청 + 400ms 경과 → 즉시 처리? 대기?
Case B: 20개 요청 빠르게 들어옴 → 어떻게?
```

① Case A는 500ms 타임아웃까지 대기하여 추가 요청을 모은다
② Case A는 15개만 모아서 즉시 처리한다
③ Case B는 첫 16개를 처리하고 나머지 4개는 다음 배치에 포함된다
④ Case B는 20개 모두를 한 번에 처리한다

정답: **① 과 ③**

**설명**:

- Case A: 타임아웃이 목표인데, 타임아웃까지 도달하지 않았으므로 더 많은 요청이 올 때까지 대기한다. 500ms에 도달하거나 배치 크기 16에 도달하면 처리한다.
- Case B: 배치 크기가 16으로 제한되어 있으므로, 16개를 먼저 처리하고 나머지 4개는 다음 배치 주기에 포함된다. ④는 배치 크기 제한을 무시하므로 오답이다.

이 로직으로 인해 배치 처리는 처리량(Throughput)을 크게 향상시키면서도, 각 요청의 최대 지연(Tail Latency)을 제어할 수 있다.

---

## 더 알아보기

이 장의 내용을 더 깊이 학습하려면 다음 자료를 참고하라:

- Sebastián Ramírez. FastAPI Documentation. https://fastapi.tiangolo.com/
- Docker Official Documentation. https://docs.docker.com/
- Nick Janetakis. A Deep Dive into Docker Layers. https://nickjanetakis.com/blog/docker-layers-explained
- Chip Huyen. (2022). Machine Learning Systems Design. https://huyenchip.com/
- Jay Alammar. The Illustrated Docker. https://jalammar.github.io/illustrated-docker/

---

## 다음 장 예고

다음 회차(13주차 B회차)에서는 이 이론을 바탕으로 **실제로 BERT 감정 분류 모델을 FastAPI로 배포**하고, **배치 처리와 캐싱으로 성능을 최적화**한 뒤, **Dockerfile을 작성하여 Docker로 배포**한다. 마지막으로 **여러 요청을 동시에 보내서 성능을 측정**하고, 각 조별로 최적화 전략을 공유한다.

---

## 참고문헌

1. Ramírez, S. (2023). FastAPI - Modern, Fast Web Framework for Building APIs. https://fastapi.tiangolo.com/
2. Docker Inc. (2023). Docker Official Documentation. https://docs.docker.com/
3. Newman, S. (2015). Building Microservices: Designing Fine-Grained Systems. O'Reilly. ISBN 978-1491950357
4. Huyen, C. (2022). Machine Learning Systems Design. https://huyenchip.com/machine-learning-systems-design.pdf
5. Sculley, D. et al. (2015). Hidden Technical Debt in Machine Learning Systems. _NIPS_. https://papers.nips.cc/paper/2015/file/86df7dcfd896fcaf2674f757a2463eba-Paper.pdf
6. Kubernetes Community. (2023). Kubernetes Official Documentation. https://kubernetes.io/docs/
7. Janetakis, N. (2019). A Deep Dive into Docker Layers. https://nickjanetakis.com/blog/docker-layers-explained
