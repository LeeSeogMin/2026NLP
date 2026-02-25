## 13주차 B회차: FastAPI 배포 + 성능 평가

> **미션**: FastAPI로 NLP 모델 서빙 API를 구현하고, Docker로 컨테이너화하여 배포하며, 성능을 측정할 수 있다

### 수업 타임라인

| 시간 | 내용 | Copilot 사용 |
|------|------|-------------|
| 00:00~00:05 | 조 편성 + 역할 배분 (조원 A/B) | 사용 안 함 |
| 00:05~00:10 | A회차 핵심 리캡 + 과제 스펙 재확인 | 사용 안 함 |
| 00:10~00:55 | 2인1조 Copilot 구현 (체크포인트 3회) | 적극 사용 |
| 00:55~01:00 | Google Classroom 제출 (조별 1부) | 사용 안 함 |
| 01:00~01:20 | 결과 토론 (배포 전략 비교·성능 차이 분석) | 사용 안 함 |
| 01:20~01:28 | 교수 피드백 + 핵심 정리 | 사용 안 함 |
| 01:28~01:30 | 다음 주 예고 | 사용 안 함 |

---

### A회차 핵심 리캡

**배포의 네 가지 핵심 요소**:
- 서비스 가능성(Availability): 24시간 켜져 있어야 한다
- 동시성(Concurrency): 여러 사용자를 동시에 처리해야 한다
- 신뢰성(Reliability): 오류가 발생해도 서비스가 중단되지 않아야 한다
- 성능(Performance): 응답이 충분히 빨라야 한다

**FastAPI의 역할**:
- Python 함수를 HTTP 엔드포인트로 변환한다
- Pydantic으로 입력 데이터를 자동 검증한다
- Swagger UI로 API 문서를 자동 생성한다
- 비동기(async) 처리로 동시성을 최대화한다

**최적화 기법**:
- 배치 처리: 여러 요청을 모아서 한 번에 추론 (처리량 3~10배 개선)
- 캐싱: 반복되는 입력에 대해 이전 결과를 재사용 (응답 시간 1/1000 수준)
- Rate Limiting: 초당 최대 요청 수 제한으로 서버 보호

**Docker의 가치**:
- 애플리케이션과 환경을 컨테이너로 패키징하여 "내 컴퓨터에선 되는데" 문제 해결
- Docker 이미지는 레이어로 구성되며, 변경된 부분만 재빌드
- Docker Compose로 모델, 캐시, 모니터링 등 여러 서비스 조합 가능

**실습 연계**:
- 이번 회차에서는 이론을 바탕으로 실제 BERT 모델을 FastAPI로 감싸고, 배치 처리와 캐싱을 추가한다
- Dockerfile을 작성하여 Docker로 배포하고, 성능을 측정한다
- 여러 요청을 동시에 보내서 배포 효과를 직접 확인한다

---

### 과제 스펙

**과제**: FastAPI 배포 + Docker 컨테이너화 + 성능 평가 리포트

**제출 형태**: 조별 1부, Google Classroom 업로드

**파일 구성**:
- 구현 코드 파일 (`13-1-fastapi-deployment.py`)
- Dockerfile
- requirements.txt
- 성능 평가 리포트 (1-2페이지)

**검증 기준**:
- ✓ FastAPI 엔드포인트 구현 및 Swagger UI 확인
- ✓ Pydantic 모델로 입력/출력 검증
- ✓ 배치 처리 구현 및 처리량 측정
- ✓ 캐싱 추가 및 캐시 히트율 확인
- ✓ Dockerfile 작성 및 로컬 빌드 및 실행
- ✓ 성능 테스트 (응답 시간, 처리량, 캐시 히트율)

---

### 2인1조 실습

> **Copilot 활용**: FastAPI 엔드포인트를 먼저 수동으로 작성해본 뒤, Copilot에게 "이 엔드포인트에 배치 처리를 추가해줄래?", "Rate Limiting 코드 작성해줄 수 있어?", "Dockerfile을 작성해줄래?" 같이 단계적으로 요청한다. Copilot의 제안을 검토하고 수정하는 과정에서 배포 최적화의 원리를 깊이 있게 이해할 수 있다.

**역할 분담**:
- **조원 A (드라이버)**: 코드 작성 및 실행, 결과 확인, 성능 테스트
- **조원 B (네비게이터)**: 로직 검토, Copilot 프롬프트 설계, 오류 해석, 성능 분석
- **체크포인트마다 역할 교대**: 드라이버와 네비게이터를 번갈아가며 진행하여 두 명 모두 전체 구현을 이해한다

---

#### 체크포인트 1: FastAPI 엔드포인트 + Pydantic 모델 (15분)

**목표**: FastAPI로 BERT 감정 분류 모델을 래핑하는 기본 엔드포인트를 구현하고, Pydantic으로 입출력을 검증한다

**핵심 단계**:

① **기본 구조 설정** — FastAPI 앱과 모델 로드

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import time

app = FastAPI(
    title="BERT 감정 분류 API",
    description="KLUE BERT를 사용한 한국어 감정 분류 서비스",
    version="1.0.0"
)

# 모델과 토크나이저 로드 (앱 시작 시 1회만)
MODEL_NAME = "klue/bert-base-multilingual-cased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

# GPU 사용 가능 시 GPU로, 아니면 CPU로
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

print(f"모델 로드 완료: {device}")
print(f"모델 파라미터: {sum(p.numel() for p in model.parameters()):,}")
```

② **Pydantic 모델 정의** — 요청과 응답 형식 명시

```python
class PredictRequest(BaseModel):
    """감정 분류 요청"""
    text: str

    class Config:
        example = {"text": "이 영화는 정말 재미있었다!"}

class PredictResponse(BaseModel):
    """감정 분류 응답"""
    text: str
    label: int  # 0: 부정, 1: 긍정
    label_name: str  # "부정" 또는 "긍정"
    confidence: float  # 신뢰도 (0~1)
    logits: list  # 원본 로짓값

    class Config:
        example = {
            "text": "이 영화는 정말 재미있었다!",
            "label": 1,
            "label_name": "긍정",
            "confidence": 0.987,
            "logits": [0.234, 0.766]
        }
```

③ **기본 엔드포인트 구현**

```python
@app.get("/health")
async def health_check():
    """서버 헬스 체크"""
    return {
        "status": "healthy",
        "model": MODEL_NAME,
        "device": str(device)
    }

@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """
    감정 분류 API
    - 입력: {"text": "분석할 텍스트"}
    - 출력: {"text": ..., "label": 0/1, "label_name": "부정/긍정", "confidence": float}
    """
    # 입력 검증
    if not request.text or len(request.text) == 0:
        raise HTTPException(status_code=400, detail="텍스트가 비어있습니다")

    if len(request.text) > 512:
        raise HTTPException(status_code=400, detail="텍스트가 너무 깁니다 (최대 512자)")

    try:
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

        # 결과 추출
        logits = outputs.logits[0].cpu().numpy().tolist()
        pred_label = int(torch.argmax(outputs.logits, dim=1)[0])
        probs = torch.softmax(outputs.logits, dim=1)[0].cpu().numpy()
        confidence = float(probs[pred_label])

        label_names = ["부정", "긍정"]

        return PredictResponse(
            text=request.text,
            label=pred_label,
            label_name=label_names[pred_label],
            confidence=confidence,
            logits=logits
        )

    except Exception as e:
        print(f"추론 중 오류: {str(e)}")
        raise HTTPException(status_code=500, detail="추론 실패")
```

④ **테스트**

```bash
uvicorn 13-1-fastapi-deployment:app --reload
# http://127.0.0.1:8000/docs 열기 (Swagger UI)
```

Swagger UI에서 "/predict" 엔드포인트 테스트:

```json
입력: {"text": "이 영화는 정말 재미있었다!"}

응답:
{
  "text": "이 영화는 정말 재미있었다!",
  "label": 1,
  "label_name": "긍정",
  "confidence": 0.987,
  "logits": [0.234, 0.766]
}
```

**검증 체크리스트**:
- [ ] FastAPI 앱이 시작되었는가? (터미널에 "Uvicorn running on...")
- [ ] Swagger UI가 보이는가? (http://127.0.0.1:8000/docs)
- [ ] /health 엔드포인트가 작동하는가?
- [ ] /predict 엔드포인트가 올바른 응답을 반환하는가?
- [ ] 입력 검증이 작동하는가? (빈 문자열, 512자 초과 입력 시 400 에러)

**Copilot 프롬프트 1**:
```
"FastAPI로 BERT 감정 분류 모델을 래핑하는 코드를 작성해줄래?
Pydantic으로 요청/응답 모델을 정의하고, /health와 /predict 엔드포인트를 만들어줘.
응답에는 text, label, label_name, confidence, logits를 포함해야 해."
```

**Copilot 프롬프트 2**:
```
"위의 코드에 입력 검증을 추가해줄 수 있어? 빈 문자열과 512자 초과를 체크하고,
오류가 발생하면 HTTPException을 던져야 해."
```

---

#### 체크포인트 2: 배치 처리 + 캐싱 (15분)

**목표**: 배치 처리와 캐싱을 추가하여 성능을 최적화하고, 성능 메트릭을 수집한다

**핵심 단계**:

① **메모리 캐시 추가**

```python
from functools import lru_cache

# 토크나이징 결과 캐시 (최대 10000개 항목)
@lru_cache(maxsize=10000)
def cached_tokenize(text: str):
    """텍스트를 토크나이징하고 결과를 캐시"""
    inputs = tokenizer(
        text,
        return_tensors='pt',
        truncation=True,
        max_length=128,
        padding=True
    )
    return inputs

# 전체 예측 결과 캐시 (메모리 기반, 프로덕션에는 Redis 권장)
prediction_cache = {}

def get_cached_prediction(text: str):
    """캐시에서 예측 결과 조회"""
    if text in prediction_cache:
        return prediction_cache[text], True  # 결과, 캐시 히트 여부
    return None, False

def cache_prediction(text: str, result: dict):
    """예측 결과를 캐시에 저장 (최대 5000개)"""
    if len(prediction_cache) > 5000:
        # 캐시 오버플로우 시 가장 오래된 항목 제거
        oldest_key = next(iter(prediction_cache))
        del prediction_cache[oldest_key]

    prediction_cache[text] = result
```

② **성능 메트릭 수집**

```python
from typing import Dict
from collections import deque

class MetricsCollector:
    """성능 메트릭 수집"""

    def __init__(self):
        self.total_requests = 0
        self.total_time = 0.0
        self.cache_hits = 0
        self.batch_requests = 0
        self.latencies = deque(maxlen=1000)  # 최근 1000개 요청의 응답 시간

    def record(self, duration: float, cache_hit: bool = False, batch_size: int = 1):
        """요청 기록"""
        self.total_requests += 1
        self.total_time += duration
        self.latencies.append(duration)

        if cache_hit:
            self.cache_hits += 1

        if batch_size > 1:
            self.batch_requests += 1

    def get_stats(self) -> Dict:
        """성능 통계 반환"""
        if not self.latencies:
            return {}

        latencies_list = sorted(list(self.latencies))

        return {
            'total_requests': self.total_requests,
            'cache_hits': self.cache_hits,
            'cache_hit_rate_percent': (self.cache_hits / self.total_requests * 100) if self.total_requests > 0 else 0,
            'avg_latency_ms': (self.total_time / self.total_requests * 1000) if self.total_requests > 0 else 0,
            'p50_latency_ms': latencies_list[len(latencies_list) // 2] * 1000 if latencies_list else 0,
            'p95_latency_ms': latencies_list[int(len(latencies_list) * 0.95)] * 1000 if latencies_list else 0,
            'p99_latency_ms': latencies_list[int(len(latencies_list) * 0.99)] * 1000 if latencies_list else 0,
            'batch_requests': self.batch_requests,
        }

metrics = MetricsCollector()
```

③ **캐싱이 포함된 /predict 엔드포인트**

```python
@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """감정 분류 API (캐싱 포함)"""
    start_time = time.time()

    # 입력 검증
    if not request.text or len(request.text) == 0:
        raise HTTPException(status_code=400, detail="텍스트가 비어있습니다")

    if len(request.text) > 512:
        raise HTTPException(status_code=400, detail="텍스트가 너무 깁니다")

    # 캐시 확인
    cached_result, cache_hit = get_cached_prediction(request.text)
    if cache_hit:
        duration = time.time() - start_time
        metrics.record(duration, cache_hit=True, batch_size=1)
        return PredictResponse(**cached_result)

    try:
        # 토크나이징 (캐시됨)
        inputs = cached_tokenize(request.text)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # 추론
        with torch.no_grad():
            outputs = model(**inputs)

        # 결과 추출
        logits = outputs.logits[0].cpu().numpy().tolist()
        pred_label = int(torch.argmax(outputs.logits, dim=1)[0])
        probs = torch.softmax(outputs.logits, dim=1)[0].cpu().numpy()
        confidence = float(probs[pred_label])

        label_names = ["부정", "긍정"]

        result = {
            "text": request.text,
            "label": pred_label,
            "label_name": label_names[pred_label],
            "confidence": confidence,
            "logits": logits
        }

        # 결과 캐시
        cache_prediction(request.text, result)

        duration = time.time() - start_time
        metrics.record(duration, cache_hit=False, batch_size=1)

        return PredictResponse(**result)

    except Exception as e:
        print(f"추론 중 오류: {str(e)}")
        raise HTTPException(status_code=500, detail="추론 실패")
```

④ **메트릭 조회 엔드포인트**

```python
@app.get("/metrics")
async def get_metrics():
    """성능 메트릭 조회"""
    return metrics.get_stats()

@app.get("/cache-info")
async def get_cache_info():
    """캐시 정보 조회"""
    return {
        "cache_size": len(prediction_cache),
        "cache_maxsize": 5000,
        "tokenize_cache_info": cached_tokenize.cache_info()._asdict()
    }
```

⑤ **성능 테스트 코드**

```python
import asyncio
import aiohttp
import json

async def performance_test():
    """성능 테스트: 100개 요청 (50% 캐시 히트 예상)"""

    test_texts = [
        "이 영화는 정말 재미있었다!",
        "최악의 경험이었다",
        "완벽한 서비스",
        "별로였음",
        "추천합니다",
    ]

    print("성능 테스트 시작...")
    start_time = time.time()

    async with aiohttp.ClientSession() as session:
        tasks = []
        for i in range(100):
            text = test_texts[i % len(test_texts)]
            payload = {"text": text}

            task = session.post(
                "http://127.0.0.1:8000/predict",
                json=payload
            )
            tasks.append(task)

        responses = await asyncio.gather(*tasks)

    total_time = time.time() - start_time

    # 결과 분석
    successful = sum(1 for r in responses if r.status == 200)
    failed = len(responses) - successful

    print(f"완료: {successful}/{len(responses)} 성공")
    print(f"총 시간: {total_time:.2f}초")
    print(f"처리량: {len(responses) / total_time:.1f} 요청/초")

    # 메트릭 확인
    stats = metrics.get_stats()
    print(f"\n메트릭:")
    print(f"  캐시 히트율: {stats['cache_hit_rate_percent']:.1f}%")
    print(f"  평균 응답 시간: {stats['avg_latency_ms']:.1f}ms")
    print(f"  P95 응답 시간: {stats['p95_latency_ms']:.1f}ms")

# 테스트 실행 (만약 스크립트를 직접 실행할 경우)
# asyncio.run(performance_test())
```

**검증 체크리스트**:
- [ ] 첫 번째 요청은 1초 이상, 같은 텍스트의 두 번째 요청은 1ms 미만인가?
- [ ] /metrics 엔드포인트가 캐시 히트율을 올바르게 보여주는가?
- [ ] 100개 요청의 처리 시간이 배치 크기 16 기준 약 10초 이내인가?
- [ ] /cache-info 엔드포인트가 캐시 상태를 보여주는가?

**Copilot 프롬프트 3**:
```
"캐싱을 추가해줄 수 있어? lru_cache를 사용해서 토크나이징 결과를 캐시하고,
메모리 딕셔너리로 전체 예측 결과도 캐시해야 해. 캐시 히트율을 추적해야 해."
```

**Copilot 프롬프트 4**:
```
"성능 메트릭을 수집하는 클래스를 만들어줄래? 평균 응답 시간, P95, P99 지연시간,
캐시 히트율을 계산하고, /metrics 엔드포인트로 조회할 수 있어야 해."
```

---

#### 체크포인트 3: Docker 컨테이너화 (15분)

**목표**: Dockerfile을 작성하여 모델 서버를 Docker 이미지로 패키징하고, 로컬에서 빌드 및 실행하여 배포 효과를 확인한다

**핵심 단계**:

① **requirements.txt 작성**

```
fastapi==0.104.1
uvicorn[standard]==0.24.0
torch==2.1.0
transformers==4.34.0
pydantic==2.5.0
aiohttp==3.9.0
```

② **Dockerfile 작성**

```dockerfile
# 베이스 이미지: Python 3.10
FROM python:3.10-slim-bullseye

# 작업 디렉토리 설정
WORKDIR /app

# 시스템 패키지 업데이트 (필요 시)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 의존성 파일 복사
COPY requirements.txt .

# Python 의존성 설치 (--no-cache-dir로 용량 절감)
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드 복사
COPY 13-1-fastapi-deployment.py .

# 포트 8000 노출
EXPOSE 8000

# 헬스 체크 (선택)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3 -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# uvicorn으로 서버 실행
CMD ["uvicorn", "13-1-fastapi-deployment:app", "--host", "0.0.0.0", "--port", "8000"]
```

③ **이미지 빌드**

```bash
# Docker 이미지 빌드 (태그: bert-model:1.0)
docker build -t bert-model:1.0 .

# 빌드 진행 상황 확인
# → FROM python:3.10-slim-bullseye (베이스 이미지 다운로드)
# → RUN pip install ... (의존성 설치, 약 5~10분)
# → COPY ... (코드 복사)
# → 완료: Successfully tagged bert-model:1.0
```

④ **이미지 확인 및 정보 조회**

```bash
# 빌드된 이미지 목록 확인
docker images

# 출력 예시:
# REPOSITORY    TAG    IMAGE ID       CREATED        SIZE
# bert-model    1.0    abc123def456   5 minutes ago   2.1GB

# 이미지 크기 상세 정보
docker image inspect bert-model:1.0 | grep Size
```

⑤ **컨테이너 실행**

```bash
# 컨테이너 실행 (포트 8000을 로컬 8000으로 매핑)
docker run -p 8000:8000 bert-model:1.0

# 출력 예시:
# INFO:     Uvicorn running on http://0.0.0.0:8000
# INFO:     Application startup complete
```

⑥ **API 테스트 (새 터미널에서)**

```bash
# 헬스 체크
curl http://localhost:8000/health

# 예측 요청
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "이 영화는 정말 재미있었다!"}'

# 응답 예시:
# {
#   "text": "이 영화는 정말 재미있었다!",
#   "label": 1,
#   "label_name": "긍정",
#   "confidence": 0.987,
#   "logits": [0.234, 0.766]
# }
```

⑦ **Docker 컨테이너 관리**

```bash
# 실행 중인 컨테이너 확인
docker ps

# 컨테이너 로그 확인
docker logs <container_id>

# 컨테이너 중지
docker stop <container_id>

# 컨테이너 재시작
docker restart <container_id>

# 컨테이너 제거
docker rm <container_id>
```

⑧ **Docker Compose (선택)**

`docker-compose.yml`:

```yaml
version: '3.8'

services:
  model-api:
    build: .
    container_name: bert-api
    ports:
      - "8000:8000"
    environment:
      MODEL_NAME: klue/bert-base-multilingual-cased
    volumes:
      - ./logs:/app/logs
    restart: always
```

한 명령으로 시작:

```bash
docker-compose up -d
docker-compose logs -f
docker-compose down
```

**검증 체크리스트**:
- [ ] Dockerfile이 올바른 문법인가? (docker build 성공)
- [ ] 빌드된 이미지 크기가 합리적인가? (약 2~3GB)
- [ ] 컨테이너가 실행되는가? (docker run 성공)
- [ ] 컨테이너 내부에서 /health 엔드포인트가 응답하는가?
- [ ] /predict 엔드포인트가 올바른 결과를 반환하는가?
- [ ] Swagger UI에 접근할 수 있는가? (http://localhost:8000/docs)

**Copilot 프롬프트 5**:
```
"Dockerfile을 작성해줄 수 있어? Python 3.10 기반이고,
requirements.txt를 설치하고, FastAPI 앱을 uvicorn으로 실행해야 해.
HEALTHCHECK도 추가해줘."
```

**Copilot 프롬프트 6**:
```
"Docker 빌드와 실행 과정을 설명해줄 수 있어?
docker build 명령과 docker run 명령의 각 옵션이 무엇을 하는지 알려줘."
```

---

### 제출 안내 (Google Classroom)

**제출 방법**:
- Google Classroom의 "13주차 B회차" 과제에 조별 1부 제출
- 파일명 형식: `group_{조번호}_ch13B.zip`

**포함할 파일**:
```
group_{조번호}_ch13B/
├── 13-1-fastapi-deployment.py      # 전체 구현 코드 (FastAPI + 캐싱)
├── Dockerfile                       # Docker 이미지 정의
├── requirements.txt                 # Python 의존성
├── docker-compose.yml               # Docker Compose (선택)
└── performance_report.md            # 성능 평가 리포트
```

**리포트 포함 항목** (performance_report.md):
- 각 체크포인트의 구현 과정 및 어려웠던 점 (3~4문장)
- 캐싱 효과: "첫 요청 vs 캐시 히트 요청의 응답 시간" (2~3문장)
- 성능 메트릭 결과: 평균 응답 시간, P95/P99 지연시간, 캐시 히트율 (테이블 포함)
- Docker 배포 효과: "로컬 vs Docker 컨테이너 환경에서의 차이" (2~3문장)
- Copilot 사용 경험: 어떤 프롬프트가 효과적이었는가? (2문장)

**성능 리포트 예시**:

```markdown
## 성능 평가 리포트

### 1. 구현 과정

FastAPI 기본 엔드포인트 구현은 Swagger UI가 자동으로 생성되어 매우 직관적이었다.
캐싱을 추가할 때는 lru_cache와 메모리 딕셔너리를 조합하여
첫 요청과 반복 요청 간 응답 시간 차이를 100배 이상 개선할 수 있었다.
Docker 빌드는 처음에는 의존성 크기 때문에 예상보다 오래 걸렸으나,
--no-cache-dir 옵션으로 용량을 줄일 수 있었다.

### 2. 캐싱 효과

| 항목 | 첫 요청 | 캐시 히트 | 개선율 |
|------|--------|----------|--------|
| 응답 시간 | 1,250ms | 5ms | 250배 |
| 텍스트 | "이 영화는 정말 재미있었다!" | 동일 | - |

캐시 히트율이 처음에는 10%였으나, 같은 문장 100개를 반복 입력하면 50% 이상으로 상승한다.

### 3. 성능 메트릭

100개 요청 처리 결과:
- 총 시간: 8.5초
- 처리량: 11.8 요청/초
- 평균 응답 시간: 245ms
- P95 응답 시간: 450ms
- P99 응답 시간: 980ms
- 캐시 히트율: 45.3%

### 4. Docker 배포 효과

로컬 환경에서는 Python, PyTorch, transformers 버전이 개발자 컴퓨터에 따라 다를 수 있다.
Docker로 패키징하면 이 모든 환경이 고정되어,
어떤 컴퓨터에서든 정확히 같은 환경에서 실행된다.
프로덕션 서버 배포 시 "내 컴퓨터에선 되는데" 문제가 완전히 해결된다.

### 5. Copilot 사용 경험

"배치 처리를 추가해줄 수 있어?"라는 프롬프트보다,
"100개 요청을 모아서 한 번에 처리하는 배치 프로세서를 만들어줄래?"
같이 구체적인 요구사항을 제시할 때 더 나은 코드가 생성되었다.
Dockerfile 작성 시 "HEALTHCHECK 추가"까지 명시하니 더 완성된 파일이 나왔다.
```

**제출 마감**: 수업 종료 후 24시간 이내

---

### 결과 토론 가이드

> **토론 가이드**: 조별 구현 결과를 공유하며, 캐싱과 배치 처리의 효과를 비교하고 Docker 배포의 실무적 가치를 함께 논의한다

**토론 주제**:

① **캐싱 전략의 차이**
- 각 조가 사용한 캐싱 방식이 다른가? (메모리 vs Redis)
- 캐시 크기 설정에 따른 히트율 차이는?
- 실제로 50%는 캐시 히트율이 나왔는가?
- "자주 쓰는 문장"을 캐싱하는 게 효율적일까?

② **배치 처리의 트레이드오프**
- 배치 크기와 타임아웃 설정이 응답 시간에 미치는 영향
- 배치 크기가 크면 처리량은 많지만 개별 요청의 지연시간이 커진다
- 어느 정도 크기가 실무 환경에 적절할까?

③ **Docker 이미지 최적화**
- 이미지 크기가 얼마나 되었는가? (2GB 이상이 정상)
- 베이스 이미지 선택 (python:3.10-slim vs full) 영향
- 멀티스테이지 빌드로 이미지 크기를 더 줄일 수 있을까?

④ **성능 메트릭 해석**
- P95와 P99 지연시간이 평균과 얼마나 다른가?
- 이는 "느린 요청"이 얼마나 있는지 보여준다
- 실무에서는 P99를 기준으로 SLA(Service Level Agreement)를 정의한다

⑤ **실무적 배포 고려사항**
- 캐시 크기 설정: 메모리 부족 시 어떻게?
- 모델 업데이트: 새 모델 버전은 어떻게 배포?
- 여러 인스턴스 실행: Docker Compose에서 여러 서비스를 띄울 수 있나?
- 모니터링: 프로덕션에서는 어떻게 API 상태를 추적할까?

**발표 형식**:
- 각 조 4~6분 발표 (구현 전략 + 성능 결과)
- 다른 조의 질문에 답변 (2~3개 질문)
- 교수의 보충 설명 및 피드백

---

### 교수 피드백 포인트

**강화할 점**:
- 캐싱과 배치 처리는 "더 빠른 알고리즘"이 아니라 "같은 일을 더 효율적으로 하는 전략"임을 강조한다. 모델 자체는 변하지 않지만, 요청 처리 방식을 최적화했다.
- Docker의 진정한 가치는 "환경 일관성"이다. 개발 환경과 프로덕션 환경이 정확히 같으므로, 오류 추적과 디버깅이 훨씬 쉬워진다.
- 각 조의 성능 메트릭이 다를 수 있다는 점을 확인한다. 캐시 크기, 배치 타임아웃, 하드웨어 차이 등이 영향을 미친다. 이는 배포 최적화가 "정해진 답"이 아니라 "트레이드오프"임을 보여준다.

**주의할 점**:
- "더 많은 캐시 = 더 좋은 성능"이 아니다. 캐시 히트율과 메모리 사용의 균형을 맞춰야 한다.
- Docker는 "배포를 쉽게 한다"는 이점이 있지만, "성능을 향상시키지는 않는다". 로컬 실행과 Docker 컨테이너 실행의 성능은 거의 같다 (약간의 오버헤드만 있음).
- 학생들이 "왜 배치 처리가 필요한가?"라고 물을 때, GPU 병렬화의 관점에서 설명하는 것이 좋다. 한 개 샘플 vs 32개 샘플의 추론 시간은 비슷하지만, 32개 샘플이 훨씬 효율적이다.

**다음 학습으로의 연결**:
- 14주차에서는 실제 프로덕션 환경 배포(AWS, GCP, Azure)를 다룬다.
- 이 회차에서 배운 "환경 패키징"이 클라우드 배포의 기초가 된다.
- 마이크로서비스 아키텍처(여러 API 조합)로 확장할 때도 Docker가 핵심 도구가 된다.
- 성능 모니터링(Prometheus, Grafana)과 로깅(ELK Stack)도 본격적으로 배운다.

---

### 다음 주 예고

다음 주 14주차 A회차에서는 **프로덕션 배포 심화**를 다룬다.

**예고 내용**:
- Kubernetes(K8s): Docker 컨테이너를 자동으로 오케스트레이션하여 여러 인스턴스 관리
- CI/CD 파이프라인: 코드 변경 → 자동 테스트 → 자동 배포
- 모니터링과 로깅: Prometheus, Grafana로 API 성능 추적
- 무중단 배포(Blue-Green Deployment): 실행 중인 서비스를 중단하지 않고 업데이트

**사전 준비**:
- 13주차 내용 (특히 Docker와 배치 처리) 복습
- 여러 인스턴스가 필요한 이유 생각해보기
- 프로덕션 환경에서의 "업데이트"는 어떻게 이루어질까 고민

---

## 참고 자료

**실습 코드**:
- _전체 구현 코드는 practice/chapter13/code/13-1-fastapi-deployment.py 참고_
- _성능 테스트 코드는 practice/chapter13/code/13-2-performance-test.py 참고_

**권장 읽기**:
- Ramírez, S. (2023). FastAPI Documentation. https://fastapi.tiangolo.com/
- Docker Inc. (2023). Docker Official Documentation. https://docs.docker.com/
- Janetakis, N. (2019). A Deep Dive into Docker Layers. https://nickjanetakis.com/blog/docker-layers-explained
- Huyen, C. (2022). Machine Learning Systems Design. https://huyenchip.com/machine-learning-systems-design.pdf
- Fowler, M. (2014). Microservices. https://martinfowler.com/articles/microservices.html

---
