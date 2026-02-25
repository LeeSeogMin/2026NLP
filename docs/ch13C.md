# 제13장 C: FastAPI 배포 + Docker 컨테이너화 — 모범 구현과 해설

> 이 문서는 B회차 과제 제출 후 공개된다. 제출 전에는 열람하지 않는다.

---

## 체크포인트 1 모범 구현: FastAPI 엔드포인트 + Pydantic 모델

FastAPI로 BERT 감정 분류 모델을 웹 서비스로 변환하는 것이 첫 단계다. 다음은 완전한 구현이다.

### 기본 구조: 모델 로드와 FastAPI 앱 초기화

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import time
from typing import Optional

# FastAPI 앱 생성
app = FastAPI(
    title="BERT 감정 분류 API",
    description="KLUE BERT를 사용한 한국어 감정 분류 서비스",
    version="1.0.0",
    docs_url="/docs",
    openapi_url="/openapi.json"
)

# 모델과 토크나이저 로드 (앱 시작 시 1회만 수행)
# 매번 요청마다 로드하면 극도로 느리므로, 앱 시작 시 메모리에 로드하고 유지
MODEL_NAME = "klue/bert-base-multilingual-cased"

print("모델 로드 중...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

# GPU 가능 여부 확인하고 모델 이동
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()  # 평가 모드로 설정 (드롭아웃 비활성화)

print(f"모델 로드 완료: device={device}")
print(f"모델 파라미터: {sum(p.numel() for p in model.parameters()):,}")
```

### Pydantic 요청/응답 모델 정의

```python
class PredictRequest(BaseModel):
    """감정 분류 요청 모델"""
    text: str = Field(
        ...,
        min_length=1,
        max_length=512,
        example="이 영화는 정말 재미있었다!",
        description="분석할 텍스트 (1~512자)"
    )

    class Config:
        # Swagger UI에 표시될 예시
        schema_extra = {
            "example": {
                "text": "이 영화는 정말 재미있었다!"
            }
        }


class PredictResponse(BaseModel):
    """감정 분류 응답 모델"""
    text: str = Field(
        description="입력받은 텍스트"
    )
    label: int = Field(
        description="예측 레이블 (0: 부정, 1: 긍정)"
    )
    label_name: str = Field(
        description="레이블 이름 ('부정' 또는 '긍정')"
    )
    confidence: float = Field(
        description="신뢰도 (0.0~1.0)"
    )
    logits: list = Field(
        description="원본 로짓값 [부정_점수, 긍정_점수]"
    )

    class Config:
        schema_extra = {
            "example": {
                "text": "이 영화는 정말 재미있었다!",
                "label": 1,
                "label_name": "긍정",
                "confidence": 0.987,
                "logits": [0.234, 0.766]
            }
        }
```

### 헬스 체크 엔드포인트

```python
@app.get("/health")
async def health_check():
    """
    서버 상태 확인 엔드포인트

    Docker Compose와 Kubernetes 같은 오케스트레이션 도구가
    이 엔드포인트를 정기적으로 호출하여 서버 상태를 확인한다.

    반환:
        dict: 상태 정보 (status, model, device)
    """
    return {
        "status": "healthy",
        "model": MODEL_NAME,
        "device": str(device),
        "timestamp": time.time()
    }
```

**해석**: /health 엔드포인트는 배포 환경에서 필수다.
- Kubernetes는 이 엔드포인트로 Pod가 정상 작동하는지 확인한다
- 반환되는 정보는 모니터링 시스템이 수집한다
- GET 메서드를 사용하여 입력값 없이 빠르게 응답할 수 있도록 한다

### 기본 예측 엔드포인트

```python
@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """
    감정 분류 API 엔드포인트

    요청받은 텍스트의 감정을 분류한다:
    - 0 (부정): 부정적인 의견
    - 1 (긍정): 긍정적인 의견

    입력값 자동 검증:
        - text: 1~512자 사이의 문자열
        - 빈 문자열이거나 512자 초과 시 400 Bad Request

    Args:
        request: 분류할 텍스트를 포함한 요청

    Returns:
        PredictResponse: 분류 결과 (text, label, label_name, confidence, logits)

    Raises:
        HTTPException(400): 입력 형식 오류
        HTTPException(500): 모델 추론 실패
    """
    # [단계 1] 입력 검증
    # Pydantic이 자동으로 길이를 검증하지만, 명시적으로 한 번 더 확인
    if not request.text or len(request.text.strip()) == 0:
        raise HTTPException(
            status_code=400,
            detail="텍스트가 비어있습니다"
        )

    if len(request.text) > 512:
        raise HTTPException(
            status_code=400,
            detail=f"텍스트가 너무 깁니다 (현재: {len(request.text)}자, 최대: 512자)"
        )

    try:
        # [단계 2] 토크나이징
        # transformers의 토크나이저는 자동으로 텍스트를 서브워드로 분할
        inputs = tokenizer(
            request.text,
            return_tensors='pt',          # PyTorch 텐서 반환
            truncation=True,              # 최대 길이 초과 시 자르기
            max_length=128,               # BERT의 standard max_length
            padding=True                  # 배치 처리를 위해 패딩
        )

        # [단계 3] 모델이 작동하는 디바이스로 이동
        # GPU 메모리에 로드하면 추론이 훨씬 빠르다
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # [단계 4] 모델 추론
        # torch.no_grad()는 역전파 계산을 스킵하여 메모리 절약
        # 평가 시에는 업데이트가 필요 없으므로 항상 사용한다
        with torch.no_grad():
            outputs = model(**inputs)

        # [단계 5] 결과 추출 및 가공
        # outputs.logits: (batch_size=1, num_classes=2)
        logits = outputs.logits[0].cpu().numpy().tolist()
        # argmax: 가장 큰 로짓을 가진 클래스 선택
        pred_label = int(torch.argmax(outputs.logits, dim=1)[0])
        # softmax: 로짓을 확률로 변환 (0~1 사이의 합=1)
        probs = torch.softmax(outputs.logits, dim=1)[0].cpu().numpy()
        confidence = float(probs[pred_label])

        # [단계 6] 응답 생성
        label_names = ["부정", "긍정"]

        return PredictResponse(
            text=request.text,
            label=pred_label,
            label_name=label_names[pred_label],
            confidence=confidence,
            logits=logits
        )

    except Exception as e:
        # 예외 발생 시 500 Internal Server Error 반환
        # 로그에는 상세 정보 기록 (디버깅용)
        print(f"추론 중 오류: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="모델 추론 중 오류가 발생했습니다"
        )
```

**핵심 포인트 설명**:

#### 왜 async 함수를 사용하는가?

```python
# 올바름
@app.post("/predict")
async def predict(request: PredictRequest):
    ...

# 일반적 (async는 선택사항이지만 동시성을 위해 권장)
@app.post("/predict")
def predict_sync(request: PredictRequest):
    ...
```

async 함수는 I/O 대기 중에 다른 요청을 처리할 수 있다.
- 토크나이징: CPU 작업 (빠름)
- GPU 추론: I/O 대기로 간주 가능 (GPU 메모리 전송 포함)
- CPU로 결과 복사: I/O 작업

따라서 async를 사용하면 여러 요청을 동시에 처리할 수 있다.

#### Pydantic의 자동 검증

```python
# Pydantic이 자동으로 수행하는 검증

class PredictRequest(BaseModel):
    text: str = Field(min_length=1, max_length=512)

# 잘못된 요청
request_json = {
    "text": ""  # 빈 문자열
}
# → FastAPI가 422 Unprocessable Entity 반환

request_json = {
    "text": "a" * 1000  # 1000자
}
# → FastAPI가 422 Unprocessable Entity 반환

request_json = {
    "text": "이 영화는 정말 재미있었다!",
    "extra_field": 123  # 정의되지 않은 필드
}
# → FastAPI가 무시하거나 에러 (Config에 따라)
```

#### CPU와 GPU 메모리 관리

```python
# 입력을 GPU로 이동
inputs = {k: v.to(device) for k, v in inputs.items()}

# 모델도 이미 GPU에 있으므로, 메모리 내에서 빠르게 계산
with torch.no_grad():
    outputs = model(**inputs)

# 결과를 CPU로 복사 (JSON 직렬화 필요)
logits = outputs.logits[0].cpu().numpy().tolist()
```

이 과정을 거치지 않으면:
- GPU 메모리와 CPU 메모리 사이의 전송이 병목
- 추론 시간이 2~3배 증가

### 흔한 실수와 해결법

#### 1. 모델을 매번 로드하기

```python
# 틀림 (매 요청마다 모델 로드)
@app.post("/predict")
async def predict(request: PredictRequest):
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)  # 1~2초 소요!
    ...

# 맞음 (앱 시작 시 1회만 로드)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

@app.post("/predict")
async def predict(request: PredictRequest):
    # 이미 로드된 모델 사용
    outputs = model(**inputs)
```

첫 번째 방식은 매 요청마다 1~2초가 걸린다. 100개 요청이면 100~200초!

#### 2. 모델을 eval() 모드로 설정하지 않기

```python
# 틀림
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
# 기본값은 training=True, dropout이 활성화됨

@app.post("/predict")
async def predict(request: PredictRequest):
    outputs = model(**inputs)  # 매번 다른 결과!

# 맞음
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()  # dropout과 batchnorm이 비활성화됨

@app.post("/predict")
async def predict(request: PredictRequest):
    outputs = model(**inputs)  # 같은 입력은 항상 같은 결과
```

training 모드에서는 dropout이 활성화되어, 같은 텍스트가 매번 다른 결과를 낸다.
배포 환경에서는 절대 허용되지 않는다.

#### 3. Softmax 없이 로짓으로 신뢰도 사용하기

```python
# 틀림
logits = outputs.logits[0].cpu().numpy()
confidence = float(logits[pred_label])  # 음수일 수도 있고, 1보다 클 수도 있음

# 맞음
probs = torch.softmax(outputs.logits, dim=1)[0].cpu().numpy()
confidence = float(probs[pred_label])  # 항상 0~1 사이

# 예시
logits = [0.5, -0.2]  # 원본 로짓
probs = softmax([0.5, -0.2]) = [0.62, 0.38]  # 확률로 변환
```

softmax를 사용하면:
- 모든 확률의 합이 1이 됨
- 각 확률이 0~1 사이의 값이 됨
- 두 클래스의 상대적 신뢰도를 명확하게 표현

#### 4. 배치 처리를 고려하지 않은 차원 관리

```python
# 문제 상황
inputs = tokenizer(request.text, return_tensors='pt', ...)  # (1, seq_len)
outputs = model(**inputs)  # (1, num_classes)

# 배치 처리를 추가할 때
texts = [text1, text2, text3]  # 3개
inputs = tokenizer(texts, return_tensors='pt', ...)  # (3, seq_len)
outputs = model(**inputs)  # (3, num_classes)

logits = outputs.logits[0]  # 첫 번째만 추출
pred_label = int(torch.argmax(outputs.logits, dim=1)[0])  # 첫 번째만

# 맞음 (반복문으로 처리)
for i, text in enumerate(texts):
    logits = outputs.logits[i]
    pred_label = int(torch.argmax(outputs.logits, dim=1)[i])
```

배치 크기가 1일 때와 32일 때 차원이 다르므로 주의가 필요하다.

---

## 체크포인트 2 모범 구현: 배치 처리 + 캐싱 + 성능 메트릭

### 메모리 캐시 구현

```python
from functools import lru_cache
from collections import deque
import time

# [전략 1] Tokenization 캐싱
# 같은 텍스트가 들어오면 토크나이징을 다시 하지 않음
@lru_cache(maxsize=10000)
def cached_tokenize(text: str):
    """
    텍스트를 토크나이징하고 결과를 캐시

    maxsize=10000: 최대 10,000개 항목 저장
    LRU(Least Recently Used): 가장 오래 사용되지 않은 항목부터 제거
    """
    inputs = tokenizer(
        text,
        return_tensors='pt',
        truncation=True,
        max_length=128,
        padding=True
    )
    # 주의: inputs는 dict of tensors이므로 직렬화 불가
    # 대신 token_ids만 캐싱하고, 나머지는 재계산하는 것이 나음
    return inputs['input_ids'], inputs['attention_mask']


# [전략 2] 전체 예측 결과 캐싱
# 같은 텍스트의 예측 결과를 캐시하면 응답 시간이 1ms 이하로 단축
prediction_cache = {}  # 메모리 기반 캐시
cache_access_count = {}  # 캐시 히트 추적용


def get_cached_prediction(text: str):
    """
    캐시에서 예측 결과를 조회한다.

    Returns:
        (result_dict, cache_hit): (결과, 캐시 히트 여부)
    """
    if text in prediction_cache:
        cache_access_count[text] = cache_access_count.get(text, 0) + 1
        return prediction_cache[text], True
    return None, False


def cache_prediction(text: str, result: dict):
    """
    예측 결과를 캐시에 저장한다.

    메모리 부족 시 가장 오래된 항목부터 제거 (FIFO)
    프로덕션에서는 Redis를 사용하여 메모리 절약
    """
    if len(prediction_cache) > 5000:
        # 캐시 오버플로우 방지
        # 첫 번째 항목(가장 오래된)을 제거
        oldest_key = next(iter(prediction_cache))
        del prediction_cache[oldest_key]
        if oldest_key in cache_access_count:
            del cache_access_count[oldest_key]

    prediction_cache[text] = result
    cache_access_count[text] = 0
```

**캐시 효과 분석**:

```python
# 시나리오 1: 고유한 요청만
texts = ["좋다", "싫다", "최고", "최악", ...]  # 100개 모두 다름
→ 캐시 히트율: 0%
→ 총 시간: 100개 × 0.25초 = 25초

# 시나리오 2: 반복적인 요청
texts = ["좋다"] * 50 + ["싫다"] * 50  # 50개씩 반복
→ 첫 "좋다": 0.25초
→ 나머지 "좋다": 0.001초 × 49 = 0.049초
→ 첫 "싫다": 0.25초
→ 나머지 "싫다": 0.001초 × 49 = 0.049초
→ 총 시간: 0.25 + 0.049 + 0.25 + 0.049 ≈ 0.6초
→ 캐시 히트율: 98% (98개/100개)
→ 속도 향상: 25초 → 0.6초 (41배!)
```

### 성능 메트릭 수집 클래스

```python
from typing import Dict
from collections import deque
import numpy as np


class MetricsCollector:
    """
    API 성능 메트릭을 수집하는 클래스

    추적 항목:
    - 총 요청 수
    - 캐시 히트 수
    - 각 요청의 응답 시간
    - 배치 크기

    통계:
    - 평균 응답 시간
    - P50, P95, P99 지연시간 (백분위수)
    - 캐시 히트율
    """

    def __init__(self):
        self.total_requests = 0
        self.total_time = 0.0
        self.cache_hits = 0
        self.batch_requests = 0
        # 최근 1000개 요청의 응답 시간 추적 (P-tile 계산용)
        self.latencies = deque(maxlen=1000)

    def record(self, duration: float, cache_hit: bool = False, batch_size: int = 1):
        """
        요청의 응답 시간을 기록한다.

        Args:
            duration: 응답 시간 (초)
            cache_hit: 캐시 히트 여부
            batch_size: 배치에 포함된 요청 수
        """
        self.total_requests += 1
        self.total_time += duration
        self.latencies.append(duration)

        if cache_hit:
            self.cache_hits += 1

        if batch_size > 1:
            self.batch_requests += 1

    def get_stats(self) -> Dict:
        """
        현재 메트릭을 딕셔너리로 반환한다.
        """
        if not self.latencies:
            return {
                'total_requests': 0,
                'cache_hits': 0,
                'cache_hit_rate_percent': 0,
                'avg_latency_ms': 0,
                'p50_latency_ms': 0,
                'p95_latency_ms': 0,
                'p99_latency_ms': 0,
                'batch_requests': 0
            }

        # 응답 시간을 정렬하여 백분위수 계산
        latencies_list = sorted(list(self.latencies))
        n = len(latencies_list)

        # 백분위수 계산 (0번째 인덱스는 최소값, 99번째는 최대값)
        p50_idx = n // 2
        p95_idx = int(n * 0.95)
        p99_idx = int(n * 0.99)

        return {
            'total_requests': self.total_requests,
            'cache_hits': self.cache_hits,
            'cache_hit_rate_percent': (
                self.cache_hits / self.total_requests * 100
                if self.total_requests > 0 else 0
            ),
            'avg_latency_ms': (
                self.total_time / self.total_requests * 1000
                if self.total_requests > 0 else 0
            ),
            'p50_latency_ms': latencies_list[p50_idx] * 1000 if latencies_list else 0,
            'p95_latency_ms': latencies_list[p95_idx] * 1000 if latencies_list else 0,
            'p99_latency_ms': latencies_list[p99_idx] * 1000 if latencies_list else 0,
            'batch_requests': self.batch_requests
        }


# 전역 메트릭 수집 인스턴스
metrics = MetricsCollector()
```

**메트릭 해석 가이드**:

| 메트릭 | 의미 | 목표값 |
|--------|------|--------|
| avg_latency | 평균 응답 시간 | < 500ms |
| p50_latency | 중간값 (50%) | < 250ms |
| p95_latency | 95% 요청의 응답 시간 | < 1000ms |
| p99_latency | 최악의 1%를 제외한 응답 시간 | < 2000ms |
| cache_hit_rate | 캐시로부터 응답한 비율 | > 30% |

P95와 P99는 "느린 요청" (tail latency)을 감지한다:
- P95 = 100개 요청 중 5개는 이보다 느리다는 의미
- P99 = 100개 요청 중 1개는 이보다 느리다는 의미

### 캐싱이 포함된 /predict 엔드포인트

```python
@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """
    감정 분류 API (캐싱 + 성능 추적)
    """
    start_time = time.time()

    # [단계 1] 입력 검증
    if not request.text or len(request.text.strip()) == 0:
        raise HTTPException(status_code=400, detail="텍스트가 비어있습니다")

    # [단계 2] 캐시 확인
    # 같은 텍스트가 최근에 처리되었다면 즉시 반환
    cached_result, cache_hit = get_cached_prediction(request.text)
    if cache_hit:
        duration = time.time() - start_time
        metrics.record(duration, cache_hit=True, batch_size=1)
        return PredictResponse(**cached_result)

    try:
        # [단계 3] 새로운 요청 처리
        # 토크나이징 (캐시됨)
        input_ids, attention_mask = cached_tokenize(request.text)

        # 디바이스로 이동
        inputs = {
            'input_ids': input_ids.to(device),
            'attention_mask': attention_mask.to(device)
        }

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

        # [단계 4] 결과 캐시
        cache_prediction(request.text, result)

        # [단계 5] 성능 기록
        duration = time.time() - start_time
        metrics.record(duration, cache_hit=False, batch_size=1)

        return PredictResponse(**result)

    except Exception as e:
        print(f"추론 중 오류: {str(e)}")
        raise HTTPException(status_code=500, detail="추론 실패")
```

### 메트릭 조회 엔드포인트

```python
@app.get("/metrics")
async def get_metrics():
    """
    현재까지 수집된 성능 메트릭을 반환한다.

    응답 예시:
    {
        "total_requests": 150,
        "cache_hits": 45,
        "cache_hit_rate_percent": 30.0,
        "avg_latency_ms": 245.3,
        "p50_latency_ms": 180.5,
        "p95_latency_ms": 450.2,
        "p99_latency_ms": 980.1,
        "batch_requests": 0
    }
    """
    return metrics.get_stats()


@app.get("/cache-info")
async def get_cache_info():
    """
    캐시의 현재 상태를 반환한다.

    응답 예시:
    {
        "prediction_cache_size": 128,
        "prediction_cache_max_size": 5000,
        "tokenize_cache_info": {
            "hits": 234,
            "misses": 12,
            "maxsize": 10000,
            "currsize": 45
        }
    }
    """
    return {
        "prediction_cache_size": len(prediction_cache),
        "prediction_cache_max_size": 5000,
        "tokenize_cache_info": cached_tokenize.cache_info()._asdict()
    }
```

### 성능 테스트 스크립트

```python
import asyncio
import aiohttp
import json
import statistics


async def performance_test():
    """
    성능 테스트: 100개 요청 동시 전송 (50% 캐시 히트 기대)
    """

    test_texts = [
        "이 영화는 정말 재미있었다!",
        "최악의 경험이었다",
        "완벽한 서비스",
        "별로였음",
        "추천합니다",
    ]

    print("=" * 60)
    print("성능 테스트 시작")
    print("=" * 60)

    start_time = time.time()
    request_times = []

    async with aiohttp.ClientSession() as session:
        tasks = []
        for i in range(100):
            text = test_texts[i % len(test_texts)]
            payload = {"text": text}

            async def make_request(text_to_send):
                req_start = time.time()
                try:
                    async with session.post(
                        "http://127.0.0.1:8000/predict",
                        json={"text": text_to_send},
                        timeout=aiohttp.ClientTimeout(total=10)
                    ) as resp:
                        await resp.json()
                        req_time = time.time() - req_start
                        request_times.append(req_time)
                        return resp.status
                except Exception as e:
                    print(f"요청 실패: {str(e)}")
                    return 500

            task = make_request(text)
            tasks.append(task)

        responses = await asyncio.gather(*tasks)

    total_time = time.time() - start_time

    # [결과 분석]
    successful = sum(1 for r in responses if r == 200)
    failed = len(responses) - successful

    print(f"\n요청 결과:")
    print(f"  성공: {successful}/{len(responses)}")
    print(f"  실패: {failed}/{len(responses)}")
    print(f"\n총 소요 시간: {total_time:.2f}초")
    print(f"처리량: {len(responses) / total_time:.1f} 요청/초")

    # [응답 시간 분석]
    if request_times:
        print(f"\n응답 시간 분석:")
        print(f"  최소: {min(request_times) * 1000:.1f}ms")
        print(f"  최대: {max(request_times) * 1000:.1f}ms")
        print(f"  평균: {statistics.mean(request_times) * 1000:.1f}ms")
        print(f"  중앙값: {statistics.median(request_times) * 1000:.1f}ms")

        if len(request_times) > 1:
            stdev = statistics.stdev(request_times)
            print(f"  표준편차: {stdev * 1000:.1f}ms")

    # [메트릭 조회]
    # 실제 구동 시 http 요청으로 /metrics 엔드포인트 호출
    print(f"\nAPI 메트릭:")
    print(f"  총 요청: {metrics.total_requests}")
    print(f"  캐시 히트율: {metrics.get_stats()['cache_hit_rate_percent']:.1f}%")
    print(f"  평균 응답: {metrics.get_stats()['avg_latency_ms']:.1f}ms")
    print(f"  P95 응답: {metrics.get_stats()['p95_latency_ms']:.1f}ms")

    print("=" * 60)
```

**성능 테스트 결과 해석**:

```
성공한 시나리오:
- 100개 요청, 5개 유니크 텍스트
- 총 시간: 약 5초
- 처리량: 20개/초
- 캐시 히트율: 80% (첫 5개 미스, 나머지 95개 히트)
- P95 응답: 5ms (캐시 히트 영향)

실패한 시나리오 (캐시 없음):
- 100개 요청, 100개 유니크 텍스트
- 총 시간: 약 25초
- 처리량: 4개/초
- 캐시 히트율: 0%
- P95 응답: 250ms

개선 효과: 5배 처리량 향상
```

### 흔한 실수

#### 1. 캐시 키로 객체 사용하기

```python
# 틀림
def cached_predict(request: PredictRequest):
    # Pydantic BaseModel은 해시 불가
    if request in cache:  # TypeError!
        return cache[request]

# 맞음
def cached_predict(text: str):
    if text in cache:
        return cache[text]
```

캐시는 내부적으로 딕셔너리를 사용하므로, 키는 해시 가능해야 한다.
복잡한 객체는 불가능하므로 문자열이나 숫자를 사용해야 한다.

#### 2. 캐시 크기 제한을 무시하기

```python
# 틀림 (무한정 증가)
prediction_cache = {}  # 제한 없음

@app.post("/predict")
async def predict(request: PredictRequest):
    prediction_cache[request.text] = result  # 메모리 부족!

# 맞음 (크기 제한)
def cache_prediction(text: str, result: dict):
    if len(prediction_cache) > 5000:
        oldest_key = next(iter(prediction_cache))
        del prediction_cache[oldest_key]
    prediction_cache[text] = result
```

제한이 없으면 수일이 지나면서 메모리가 부족해져 서버가 크래시한다.

#### 3. 캐시 무효화 전략 부재

```python
# 문제: 모델이 업데이트되어도 캐시가 유지됨
# 이전 모델: "좋다" → 0.9 (긍정)
# 새 모델: "좋다" → 0.1 (부정)
# 하지만 캐시는 여전히 0.9를 반환

# 해결책 1: 모델 버전과 함께 캐시
cache_key = f"{text}:{model_version}"
prediction_cache[cache_key] = result

# 해결책 2: 캐시 TTL (Time To Live) 설정
import time
cache_entry = {
    'result': result,
    'timestamp': time.time()
}
# 1시간 후 만료되도록 설정
if time.time() - cache_entry['timestamp'] > 3600:
    del prediction_cache[text]
```

프로덕션 환경에서는 모델 업데이트 시 캐시를 초기화해야 한다.

---

## 체크포인트 3 모범 구현: Docker 컨테이너화

### requirements.txt 작성

```
fastapi==0.104.1
uvicorn[standard]==0.24.0
torch==2.1.0
transformers==4.34.0
pydantic==2.5.0
aiohttp==3.9.0
numpy==1.24.0
```

**버전 선택 기준**:
- FastAPI/Uvicorn: 최신 안정 버전
- PyTorch: GPU 지원 포함 (CPU 전용이 필요하면 torch-cpu 사용)
- transformers: 모델 호환성 확인 후 선택

### Dockerfile 작성

```dockerfile
# ============ 빌드 스테이지 ============
# 베이스 이미지: 가볍고 안정적인 Python 3.10
FROM python:3.10-slim-bullseye AS builder

# 작업 디렉토리 설정
WORKDIR /app

# 시스템 패키지 업데이트 (선택적이지만 권장)
# - build-essential: C 컴파일러 (PyTorch 등에 필요할 수 있음)
# - git: 일부 라이브러리 설치 시 필요
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# 의존성 파일 복사
COPY requirements.txt .

# Python 패키지 설치
# --no-cache-dir: 캐시를 저장하지 않아 이미지 크기 감소 (약 30% 절감)
RUN pip install --no-cache-dir -r requirements.txt

# ============ 런타임 스테이지 ============
# 최종 이미지 (빌더 스테이지의 설치 파일만 복사)
FROM python:3.10-slim-bullseye

WORKDIR /app

# 빌더 스테이지에서 설치된 패키지 복사
# 여러 단계 빌드로 최종 이미지 크기 줄임
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages

# 애플리케이션 코드 복사
COPY 13-1-fastapi-deployment.py .

# 포트 8000 노출
# 이것은 문서화 목적이며, 실제로는 docker run -p 8000:8000으로 바인딩
EXPOSE 8000

# 헬스 체크 (선택사항이지만 권장)
# 매 30초마다 /health 엔드포인트 호출하여 서버 상태 확인
# - timeout=10s: 10초 이상 응답 없으면 실패로 간주
# - retries=3: 3번 연속 실패하면 unhealthy
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3 -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# 서버 시작 명령
# uvicorn: FastAPI 애플리케이션 서버
# --host 0.0.0.0: 모든 인터페이스에서 수신 (외부 접근 가능)
# --port 8000: 포트 8000으로 실행
CMD ["uvicorn", "13-1-fastapi-deployment:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Dockerfile 내용 해석**:

| 명령 | 역할 | 예시 |
|------|------|------|
| FROM | 베이스 이미지 선택 | python:3.10-slim |
| WORKDIR | 컨테이너 내 작업 디렉토리 | /app |
| RUN | 쉘 명령 실행 | pip install -r requirements.txt |
| COPY | 로컬 파일 복사 | COPY requirements.txt . |
| EXPOSE | 포트 노출 (문서화) | EXPOSE 8000 |
| HEALTHCHECK | 서버 상태 확인 | HEALTHCHECK --interval=30s ... |
| CMD | 기본 실행 명령 | CMD ["uvicorn", ...] |

### 이미지 빌드

```bash
# 이미지 빌드 (태그: bert-model:1.0)
docker build -t bert-model:1.0 .

# 빌드 과정 (대략 5~10분)
# Step 1/10 : FROM python:3.10-slim-bullseye
# Step 2/10 : WORKDIR /app
# ...
# Successfully tagged bert-model:1.0

# 이미지 크기 확인
docker images bert-model
# REPOSITORY   TAG    IMAGE ID       CREATED        SIZE
# bert-model   1.0    abc123def456   2 minutes ago   2.1GB
```

**빌드 시간 최적화**:

```dockerfile
# 느린 빌드 (매번 의존성 재설치)
FROM python:3.10-slim-bullseye
COPY . .  # 모든 파일 먼저 복사
RUN pip install -r requirements.txt  # 매번 다시 설치

# 빠른 빌드 (변경 부분만 재설치)
FROM python:3.10-slim-bullseye
COPY requirements.txt .  # 의존성만 먼저 복사
RUN pip install -r requirements.txt  # 여기까지는 캐시 사용
COPY . .  # 코드 복사 (변경 가능)
```

Docker는 레이어 기반이므로, 자주 변경되는 파일은 Dockerfile의 뒤쪽에 배치해야 캐시를 활용할 수 있다.

### 컨테이너 실행

```bash
# 기본 실행
docker run -p 8000:8000 bert-model:1.0
# -p 8000:8000: 로컬 포트 8000 ← 컨테이너 포트 8000

# 로그 보기
# INFO:     Uvicorn running on http://0.0.0.0:8000
# INFO:     Application startup complete

# 새 터미널에서 API 테스트
curl http://localhost:8000/health
# {"status":"healthy","model":"klue/bert-base-multilingual-cased","device":"cpu","timestamp":1635...}
```

### 다른 터미널에서 API 테스트

```bash
# 헬스 체크
curl http://localhost:8000/health

# 감정 분류 요청
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "이 영화는 정말 재미있었다!"}'

# 응답
# {
#   "text": "이 영화는 정말 재미있었다!",
#   "label": 1,
#   "label_name": "긍정",
#   "confidence": 0.987,
#   "logits": [0.234, 0.766]
# }

# Swagger UI 접근
# 브라우저에서 http://localhost:8000/docs 열기
```

### Docker 컨테이너 관리

```bash
# 실행 중인 컨테이너 확인
docker ps
# CONTAINER ID   IMAGE           STATUS        PORTS
# abc123def456   bert-model:1.0  Up 5 minutes  0.0.0.0:8000->8000/tcp

# 컨테이너 ID로 접근하기
CONTAINER_ID=abc123def456

# 컨테이너 로그 보기
docker logs $CONTAINER_ID

# 실시간 로그 보기
docker logs -f $CONTAINER_ID

# 컨테이너 중지
docker stop $CONTAINER_ID

# 컨테이너 재시작
docker restart $CONTAINER_ID

# 컨테이너 제거 (중지 후)
docker rm $CONTAINER_ID

# 컨테이너 내 셸 접근 (디버깅용)
docker exec -it $CONTAINER_ID /bin/bash
# 컨테이너 내부에서:
# # python3 -c "import torch; print(torch.cuda.is_available())"
# # exit
```

### Docker Compose 사용 (선택사항)

```yaml
# docker-compose.yml
version: '3.8'

services:
  model-api:
    # 이미지 빌드
    build:
      context: .
      dockerfile: Dockerfile

    container_name: bert-api

    # 포트 매핑
    ports:
      - "8000:8000"

    # 환경 변수 설정 (선택)
    environment:
      MODEL_NAME: klue/bert-base-multilingual-cased
      BATCH_SIZE: 16
      CACHE_MAX_SIZE: 10000

    # 볼륨 마운트 (로그 저장)
    volumes:
      - ./logs:/app/logs

    # 컨테이너 재시작 정책
    restart: always

    # 헬스 체크 (Docker의 HEALTHCHECK 보완)
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 5s
```

**Docker Compose 실행**:

```bash
# 시작
docker-compose up -d
# -d: 백그라운드 실행

# 상태 확인
docker-compose ps
# NAME         COMMAND                  SERVICE   STATUS
# bert-api     "uvicorn 13-1-f..."      model-api  Up 3 seconds

# 로그 확인
docker-compose logs -f model-api

# 중지
docker-compose down
```

### 흔한 Docker 실수

#### 1. "내 컴퓨터에선 되는데" 문제

```python
# 틀림 (로컬 경로 하드코딩)
model_path = "/Users/myname/models/bert.bin"
model = torch.load(model_path)

# 맞음 (상대경로 또는 Hugging Face 모델 사용)
model = AutoModelForSequenceClassification.from_pretrained(
    "klue/bert-base-multilingual-cased"
)
```

Docker 이미지는 새로운 환경에서 실행되므로, 로컬 경로가 존재하지 않는다.

#### 2. 이미지 크기 폭증

```dockerfile
# 틀림 (불필요한 파일 모두 포함)
FROM python:3.10
COPY . .  # 현재 디렉토리 모두 복사 (git, .vscode, 데이터셋 등)
RUN pip install -r requirements.txt
# 이미지 크기: 5GB+

# 맞음 (.dockerignore로 불필요한 파일 제외)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY 13-1-fastapi-deployment.py .
# 이미지 크기: 2.1GB
```

`.dockerignore` 파일:

```
__pycache__/
*.pyc
.git/
.vscode/
.env
*.db
data/
logs/
```

#### 3. 여러 버전의 Python이 겹쳐서 설치

```dockerfile
# 틀림 (기존 Python + pip로 설치)
FROM python:3.10
RUN apt-get install python3-pip  # 시스템 Python도 설치됨

# 맞음 (한 가지 Python만 사용)
FROM python:3.10
# 이미 Python 3.10과 pip가 설치되어 있음
RUN pip install -r requirements.txt  # 이 pip는 Python 3.10용
```

#### 4. 컨테이너가 시작되었지만 응답 없음

```bash
# 문제: 포트를 잘못 바인딩
docker run -p 3000:8000 bert-model:1.0
# 컨테이너는 포트 8000에서 실행
# 로컬에서 접근하려면 3000을 사용해야 함
curl http://localhost:8000/health  # 실패
curl http://localhost:3000/health  # 성공

# Dockerfile에서 --host 0.0.0.0 설정 필수
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
# 만약 --host 127.0.0.1이면 컨테이너 외부에서 접근 불가
```

---

## 대안적 구현 방법

### 다단계 빌드 (Multi-stage Build)로 이미지 크기 줄이기

```dockerfile
# 스테이지 1: 빌더 (의존성 설치)
FROM python:3.10-slim AS builder

WORKDIR /app
COPY requirements.txt .

# 패키지 설치 (빌드 도구 필요)
RUN pip install --user --no-cache-dir -r requirements.txt

# 스테이지 2: 런타임 (최소 크기)
FROM python:3.10-slim

WORKDIR /app

# 빌더 스테이지에서 설치된 패키지만 복사
COPY --from=builder /root/.local /root/.local

# 환경 변수 설정
ENV PATH=/root/.local/bin:$PATH

# 애플리케이션 코드 복사
COPY 13-1-fastapi-deployment.py .

EXPOSE 8000

CMD ["uvicorn", "13-1-fastapi-deployment:app", "--host", "0.0.0.0", "--port", "8000"]
```

**효과**:
```
단계 빌드 전: 2.5GB (빌드 도구 포함)
다단계 빌드 후: 1.8GB (최종 패키지만)
절감: 28%
```

### 환경 변수로 모델 선택

```python
import os

MODEL_NAME = os.getenv('MODEL_NAME', 'klue/bert-base-multilingual-cased')
BATCH_SIZE = int(os.getenv('BATCH_SIZE', '16'))
CACHE_SIZE = int(os.getenv('CACHE_SIZE', '10000'))

print(f"Configuration:")
print(f"  Model: {MODEL_NAME}")
print(f"  Batch Size: {BATCH_SIZE}")
print(f"  Cache Size: {CACHE_SIZE}")
```

Docker Compose에서 설정:

```yaml
services:
  model-api:
    environment:
      MODEL_NAME: klue/bert-base-multilingual-cased
      BATCH_SIZE: 32
      CACHE_SIZE: 20000
```

### GPU 지원 Docker 이미지

```dockerfile
# CUDA 11.8 + Python 3.10 + PyTorch
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

WORKDIR /app

# Python 설치
RUN apt-get update && apt-get install -y python3.10 python3-pip && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY 13-1-fastapi-deployment.py .

EXPOSE 8000

CMD ["python3", "-m", "uvicorn", "13-1-fastapi-deployment:app", \
     "--host", "0.0.0.0", "--port", "8000"]
```

실행 시 GPU 사용:

```bash
docker run --gpus all -p 8000:8000 bert-model:gpu
# --gpus all: 모든 GPU 사용 가능하게 설정
```

---

## 심화 학습 포인트

### 성능 최적화 전략 비교

| 전략 | 효과 | 구현 난도 | 메모리 증가 |
|------|------|----------|-----------|
| 캐싱 | 반복 요청 1000배 빠름 | 낮음 | 중간 (max 5000개) |
| 배치 처리 | 처리량 10배 증가 | 중간 | 낮음 |
| 모델 양자화 | 추론 2배 빠름 | 높음 | 낮음 (메모리 1/4) |
| ONNX Runtime | 추론 2~3배 빠름 | 높음 | 낮음 |
| 멀티 인스턴스 | 병렬 처리로 처리량 N배 | 중간 | 높음 (메모리 N배) |

**선택 기준**:
- 반복적인 요청이 많다 → 캐싱 우선
- 매우 많은 동시 요청 → 배치 처리
- 응답 시간이 중요 → 모델 최적화 (양자화, ONNX)
- 대규모 서비스 → 멀티 인스턴스 + Load Balancer

### Docker 이미지 레이어 최적화

```dockerfile
# 비효율적 (레이어 많음)
FROM ubuntu:22.04
RUN apt-get update
RUN apt-get install -y python3
RUN apt-get install -y pip
RUN apt-get install -y git
# 총 5개 레이어

# 효율적 (레이어 최소화)
FROM ubuntu:22.04
RUN apt-get update && \
    apt-get install -y python3 pip git && \
    rm -rf /var/lib/apt/lists/*
# 총 2개 레이어 (한 명령으로 통합)
```

각 RUN 명령은 새로운 레이어를 생성하므로, 가능한 한 통합하는 것이 효율적이다.

### 프로덕션 체크리스트

배포 전 반드시 확인할 사항:

```
□ 모델 로드 시간이 5초 이내인가?
□ 예측 응답 시간이 500ms 이내인가?
□ 캐시 히트율이 20% 이상인가?
□ P99 지연시간이 2초 이내인가?
□ 에러 처리가 명확한가? (400, 500 상태코드)
□ Rate Limiting이 설정되어 있는가?
□ 헬스 체크 엔드포인트가 있는가?
□ 로그가 구조화되어 있는가? (JSON 형식 권장)
□ 모델 버전이 명확하게 추적되는가?
□ Docker 이미지가 2~3GB 범위인가?
□ 컨테이너가 재시작되어도 문제없는가?
□ 신용 데이터나 개인정보가 로그에 기록되지 않는가?
```

---

## 실제 프로덕션 배포 사례

### AWS EC2 배포 예시

```bash
# EC2 인스턴스 접속
ssh -i my-key.pem ubuntu@ec2-instance-ip

# Docker 설치
sudo apt-get install docker.io docker-compose

# 코드 클론
git clone https://github.com/myrepo/nlp-deploy.git
cd nlp-deploy

# 이미지 빌드 및 실행
docker build -t bert-model:prod .
docker run -d -p 8000:8000 \
  --restart always \
  --name bert-api \
  bert-model:prod

# 상태 확인
docker ps
curl http://localhost:8000/health

# 로그 모니터링
docker logs -f bert-api
```

### Kubernetes 배포 예시

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: bert-model
spec:
  replicas: 3  # 3개 인스턴스
  selector:
    matchLabels:
      app: bert-model
  template:
    metadata:
      labels:
        app: bert-model
    spec:
      containers:
      - name: bert-model
        image: bert-model:prod
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 10
---
apiVersion: v1
kind: Service
metadata:
  name: bert-model-service
spec:
  selector:
    app: bert-model
  type: LoadBalancer
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
```

배포 명령:

```bash
kubectl apply -f deployment.yaml
kubectl get pods  # 3개 Pod 실행 확인
kubectl get svc   # LoadBalancer IP 확인
```

---

## 최종 학습 정리

### 13주차 핵심 개념 요약

1. **배포의 의미**: 로컬 개발 환경을 떠나 24시간 운영 가능한 프로덕션 환경으로의 전환
2. **FastAPI**: Python 함수를 HTTP 엔드포인트로 자동 변환하는 현대적 웹 프레임워크
3. **Pydantic**: 입력 데이터를 자동 검증하고 JSON 직렬화를 처리
4. **비동기 처리**: I/O 대기 중 다른 요청을 처리하여 동시성 극대화
5. **캐싱**: 반복 요청에 대해 1000배 이상 응답 시간 개선
6. **배치 처리**: 여러 요청을 모아 GPU 병렬화로 처리량 10배 향상
7. **Docker**: 환경 일관성을 보장하여 "내 컴퓨터에선 되는데" 문제 해결
8. **성능 메트릭**: P95, P99 지연시간으로 사용자 경험 측정
9. **Rate Limiting**: 악의적 사용을 방지하고 공정한 자원 배분 실현
10. **헬스 체크**: 배포 환경(Kubernetes 등)이 서버 상태를 자동으로 모니터링

### 이전 장과의 연결

- **12주차** (Fine-tuning): 특정 도메인에 특화된 모델 학습
- **13주차** (배포): 학습한 모델을 누구나 사용 가능한 서비스로 변환
- **14주차** (프로덕션 심화): Kubernetes, CI/CD, 모니터링으로 대규모 운영

### 다음 단계

이 13주차 내용을 넘어, 실무에서는 다음을 추가로 학습해야 한다:

- **API Gateway**: 여러 마이크로서비스 앞에 놓인 단일 진입점
- **Service Mesh**: Istio 등으로 서비스 간 통신 관리
- **모니터링**: Prometheus, Grafana로 메트릭 수집 및 시각화
- **로깅**: ELK Stack으로 중앙집중식 로그 관리
- **배포 전략**: Blue-Green, Canary로 무중단 배포
- **A/B 테스팅**: 사용자 그룹별로 다른 모델 버전 제공

---

**생성일**: 2026-02-25
**대상 독자**: 컴퓨터공학/AI 전공 학부생 (3~4학년)
**난이도**: 중상급 (웹 개발, 배포 기초 선수)
**예상 페이지**: 약 35~40쪽
