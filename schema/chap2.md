# 집필계획서: 제2장 딥러닝 핵심 원리와 PyTorch 실전

## 기본 정보

| 항목 | 내용 |
|------|------|
| 장 번호 | 2 |
| 제목 | 딥러닝 핵심 원리와 PyTorch 실전 |
| 주차 | 2주차 |
| 미션 | 수업이 끝나면 텍스트 분류 모델을 직접 학습시킨다 |
| 분류 | 재작성 (구 ch3+ch4 합쳐서 재구성) |
| 예상 분량 | 600-700줄 (핵심 기술 장) |

## 참고 자산

| 구버전 소스 | 재활용 부분 |
|------------|------------|
| `_archive/old-syllabus/docs/ch4.md` | nn.Module, Dataset/DataLoader, Optimizer, Training Loop, 텍스트 분류 전체 구조 |
| `_archive/old-syllabus/practice/chapter4/code/` | 5개 파일 (모델정의, 데이터로더, 옵티마이저, 학습루프, 텍스트분류) |

## 수업 타임라인

| 시간 | 구분 | 내용 |
|------|------|------|
| 00:00~00:50 | **1교시** | 신경망 기본 구조 + 역전파 |
| 00:50~01:00 | 쉬는시간 | |
| 01:00~01:50 | **2교시** | PyTorch 모델 개발 패턴 + 학습/평가 |
| 01:50~02:00 | 쉬는시간 | |
| 02:00~02:50 | **3교시** | MLP 텍스트 분류 실습 + 과제 |

## 문서 구조

```
# 제2장: 딥러닝 핵심 원리와 PyTorch 실전
> 미션: 수업이 끝나면 텍스트 분류 모델을 직접 학습시킨다
## 학습 목표 (1-5)
### 수업 타임라인 (표)
---
#### 1교시: 신경망 기본 구조
  ## 2.1 신경망 기본 구조
    - 퍼셉트론 → 다층 퍼셉트론(MLP): 레고 블록 비유
    - 활성화 함수 (ReLU, GELU, Softmax): 비선형성이 없으면 직선밖에 못 그린다
    - 손실 함수 (Cross-Entropy, MSE): 정답과의 거리를 숫자로
    - 경사 하강법과 역전파: 산에서 내려가는 길 찾기
    - 그림 2.1: MLP 구조 다이어그램
    - 그림 2.2: 역전파 흐름
---
#### 2교시: PyTorch 모델 개발과 평가
  > 라이브 코딩 시연: nn.Module 정의 → DataLoader → Training Loop
  ## 2.2 PyTorch 모델 개발 패턴
    - nn.Module로 모델 정의 (레고 블록을 클래스로)
    - Dataset/DataLoader로 데이터 파이프라인
    - Optimizer (SGD, Adam, AdamW): 산을 내려가는 전략
    - LR Scheduler (CosineAnnealing, ReduceLROnPlateau)
  ## 2.3 모델 학습과 평가
    - Training/Validation Loop
    - 과적합 방지 (Dropout, Weight Decay, Early Stopping): 기출만 외운 학생 비유
    - 평가 지표 (Accuracy, Precision, Recall, F1): 암 진단 비유
    - Confusion Matrix
    - Loss/Accuracy Curve
---
#### 3교시: 텍스트 분류 실습
  > Copilot 활용: "PyTorch nn.Module로 3층 MLP 텍스트 분류 모델을 작성해줘"
  ## 2.4 실습: 텍스트 분류 파이프라인
    - 텍스트 전처리 (토큰화, 어휘 사전)
    - Bag-of-Words 벡터화
    - MLP 감성 분석 모델
    - 하이퍼파라미터 튜닝
---
## 핵심 정리
## 더 알아보기
## 다음 장 예고
## 참고문헌
```

## 생성 파일 목록

### 실습 코드
| 파일 | 절 | 설명 |
|------|-----|------|
| `2-1-신경망기초.py` | 2.1 | 퍼셉트론/MLP, 활성화 함수, 손실 함수, 역전파 데모 |
| `2-2-모델개발.py` | 2.2-2.3 | nn.Module, DataLoader, Optimizer, Training Loop, Early Stopping |
| `2-4-텍스트분류.py` | 2.4 | BoW 벡터화, MLP 감성 분석, 평가 지표 |

### 그래픽
| 파일 | 설명 |
|------|------|
| `fig-2-1-mlp-structure.mmd` | MLP 구조 다이어그램 |
| `fig-2-2-backprop-flow.mmd` | 역전파 흐름 다이어그램 |
| `fig-2-3-training-loop.mmd` | Training Loop 사이클 |
