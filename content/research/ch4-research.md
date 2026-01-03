# 제4장 리서치 노트: PyTorch 기반 딥러닝 모델 개발 프로세스

## 조사 일자: 2026-01-02

---

## 1. Dataset과 DataLoader

### 1.1 핵심 개념
- **Dataset**: 샘플과 레이블을 저장
- **DataLoader**: Dataset을 감싸서 배치, 셔플, 병렬 로딩 제공

### 1.2 커스텀 Dataset 작성
필수 메서드:
- `__init__`: 데이터 로드 및 초기화
- `__len__`: 데이터셋 크기 반환
- `__getitem__`: 인덱스로 샘플 접근

### 1.3 DataLoader 주요 파라미터
- `batch_size`: 배치 크기
- `shuffle`: 에폭마다 셔플 여부
- `num_workers`: 병렬 로딩 워커 수
- `collate_fn`: 배치 구성 함수 커스터마이징

### 1.4 Best Practices
- `num_workers > 0`: 병렬 데이터 로딩으로 병목 감소
- `if __name__ == '__main__':` 블록 사용 (멀티프로세싱)
- 작은 데이터셋: 단일 프로세스가 디버깅에 유리

**출처**: [PyTorch Data Tutorial](https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html)

---

## 2. Adam vs AdamW 옵티마이저

### 2.1 Adam (Adaptive Moment Estimation)
- Momentum + RMSprop 결합
- 각 파라미터별 적응적 학습률
- 대규모 데이터셋과 복잡한 모델에 효과적

### 2.2 AdamW
- **핵심 차이**: Weight Decay를 그래디언트 업데이트에서 분리
- Adam: L2 정규화가 그래디언트에 적용
- AdamW: Weight Decay가 파라미터에 직접 적용

### 2.3 성능 비교
- AdamW가 더 나은 일반화 성능
- 넓은 학습률 범위에서 안정적
- BERT, GPT, ViT 등 대규모 모델 파인튜닝에 권장

### 2.4 선택 가이드
- **Adam**: 작은 데이터셋, 덜 복잡한 모델
- **AdamW**: 대규모 모델, 과적합 위험 있는 태스크
- AdamW weight_decay 권장 범위: 0.005 ~ 0.02

**출처**: [Adam vs AdamW Comparison](https://blog.prodia.com/post/adam-w-vs-adam-key-differences-and-best-use-cases-for-developers)

---

## 3. 학습률 스케줄러

### 3.1 Cosine Annealing
- 코사인 함수로 학습률 감소
- 지역 최소점 탈출에 도움
- `CosineAnnealingLR`: 단일 감소
- `CosineAnnealingWarmRestarts`: 주기적 리셋

### 3.2 Warmup
- 초기 학습률을 낮게 시작하여 점진적 증가
- 초기 불안정성 방지
- 이동 평균, 편향 보정 안정화

### 3.3 Warmup + Cosine Annealing
- 표준 조합
- 초기 Warmup 후 Cosine 감소
- SCCA: 테스트 정확도 2-5% 향상 보고

### 3.4 Warmup-Stable-Decay (WSD)
- 연속 학습에 적합
- 고정된 스텝 수 불필요

### 3.5 PyTorch 스케줄러
- `StepLR`: 일정 에폭마다 학습률 감소
- `ExponentialLR`: 지수적 감소
- `ReduceLROnPlateau`: 성능 정체 시 감소
- `CosineAnnealingLR`: 코사인 감소

**출처**: [Cosine Learning Rate in PyTorch](https://medium.com/@utkrisht14/cosine-learning-rate-schedulers-in-pytorch-486d8717d541)

---

## 4. 과적합 방지 기법

### 4.1 Dropout
- 학습 시 무작위로 뉴런 비활성화
- 앙상블 효과
- 일반적으로 0.2~0.5 비율

### 4.2 Batch Normalization
- 각 배치에서 정규화
- 학습 안정화, 빠른 수렴
- Dropout과 함께 사용 시 주의

### 4.3 Weight Decay (L2 정규화)
- 큰 가중치에 패널티
- AdamW에서 분리된 구현

### 4.4 Early Stopping
- 검증 손실 정체 시 학습 중단
- 과적합 방지의 효과적 방법

---

## 5. 평가 지표

### 5.1 분류 지표
- **Accuracy**: 전체 정확도
- **Precision**: 양성 예측 중 실제 양성 비율
- **Recall**: 실제 양성 중 예측 양성 비율
- **F1-Score**: Precision과 Recall의 조화 평균

### 5.2 Confusion Matrix
- TP, TN, FP, FN 시각화
- 클래스별 성능 파악

---

## 참고문헌
1. PyTorch Documentation. (2025). Datasets & DataLoaders. https://docs.pytorch.org/
2. Loshchilov, I. & Hutter, F. (2019). Decoupled Weight Decay Regularization. ICLR.
3. Loshchilov, I. & Hutter, F. (2017). SGDR: Stochastic Gradient Descent with Warm Restarts. ICLR.
