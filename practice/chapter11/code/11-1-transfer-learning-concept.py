"""
11장 실습 코드 11-1: 전이 학습 개념 이해
- 사전학습 모델 로드 및 구조 분석
- Feature Extraction vs Fine-tuning 비교
"""

import torch
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification

def load_pretrained_model():
    """사전학습 BERT 모델 로드 및 구조 확인"""
    print("=" * 50)
    print("1. 사전학습 BERT 모델 로드")
    print("=" * 50)

    # 토크나이저와 모델 로드
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    # 모델 구조 요약
    print(f"\n모델: {model_name}")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"총 파라미터 수: {total_params:,} ({total_params/1e6:.1f}M)")
    print(f"학습 가능 파라미터: {trainable_params:,}")

    # 레이어 구조 확인
    print(f"\n레이어 구성:")
    print(f"  - Embeddings: 토큰 + 위치 + 세그먼트 임베딩")
    print(f"  - Encoder: {model.config.num_hidden_layers}개 Transformer 블록")
    print(f"  - 은닉 차원: {model.config.hidden_size}")
    print(f"  - 어텐션 헤드: {model.config.num_attention_heads}")

    return model, tokenizer


def feature_extraction_demo(model):
    """Feature Extraction: 사전학습 가중치 고정"""
    print("\n" + "=" * 50)
    print("2. Feature Extraction (특징 추출) 모드")
    print("=" * 50)

    # 모든 파라미터 동결
    for param in model.parameters():
        param.requires_grad = False

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)

    print(f"\n파라미터 동결 후:")
    print(f"  - 학습 가능 파라미터: {trainable_params:,}")
    print(f"  - 동결된 파라미터: {frozen_params:,}")
    print(f"\n특징:")
    print(f"  - 사전학습 지식 보존")
    print(f"  - 빠른 학습 가능")
    print(f"  - 적은 데이터에 적합")


def finetuning_demo():
    """Fine-tuning: 분류 헤드 추가 및 전체 학습"""
    print("\n" + "=" * 50)
    print("3. Fine-tuning (미세 조정) 모드")
    print("=" * 50)

    # 분류용 모델 로드 (랜덤 초기화된 분류 헤드 포함)
    model_name = "bert-base-uncased"
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2  # 이진 분류
    )

    # 파라미터 분석
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # 분류 헤드 파라미터
    classifier_params = sum(p.numel() for p in model.classifier.parameters())

    print(f"\n분류 모델 구조:")
    print(f"  - BERT Base: ~110M 파라미터")
    print(f"  - 분류 헤드: {classifier_params:,} 파라미터")
    print(f"  - 총 파라미터: {total_params:,}")
    print(f"  - 학습 가능 파라미터: {trainable_params:,} (100%)")

    print(f"\n특징:")
    print(f"  - 전체 모델 가중치 업데이트")
    print(f"  - 도메인 특화 학습 가능")
    print(f"  - 더 많은 계산 자원 필요")

    return model


def partial_finetuning_demo(model):
    """Partial Fine-tuning: 일부 레이어만 학습"""
    print("\n" + "=" * 50)
    print("4. Partial Fine-tuning (부분 미세 조정)")
    print("=" * 50)

    # 먼저 모든 파라미터 동결
    for param in model.parameters():
        param.requires_grad = False

    # 분류 헤드와 마지막 2개 인코더 레이어만 학습 가능하게 설정
    for param in model.classifier.parameters():
        param.requires_grad = True

    # 마지막 2개 레이어 학습 가능
    for i in range(10, 12):  # 레이어 10, 11 (0-indexed)
        for param in model.bert.encoder.layer[i].parameters():
            param.requires_grad = True

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    total_params = trainable_params + frozen_params

    print(f"\n마지막 2개 레이어 + 분류 헤드만 학습:")
    print(f"  - 학습 가능 파라미터: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")
    print(f"  - 동결된 파라미터: {frozen_params:,} ({100*frozen_params/total_params:.1f}%)")

    print(f"\n장점:")
    print(f"  - 하위 레이어의 일반적 특징 보존")
    print(f"  - 상위 레이어만 태스크에 맞게 조정")
    print(f"  - 과적합 위험 감소")


def compare_strategies():
    """파인튜닝 전략 비교 요약"""
    print("\n" + "=" * 50)
    print("5. 파인튜닝 전략 비교")
    print("=" * 50)

    print("""
┌─────────────────┬────────────────┬────────────────┬────────────────┐
│     전략        │ 학습 파라미터  │   데이터 요구  │    적합한 상황  │
├─────────────────┼────────────────┼────────────────┼────────────────┤
│ Feature Extract │ 분류 헤드만    │ 적음 (수백개)  │ 유사한 도메인  │
│ Partial Fine-   │ 상위 레이어    │ 중간           │ 약간 다른 도메인│
│ Full Fine-tune  │ 전체 모델      │ 많음 (수천개)  │ 다른 도메인    │
└─────────────────┴────────────────┴────────────────┴────────────────┘
""")


def main():
    print("=" * 60)
    print("11장: 전이 학습 개념 이해")
    print("=" * 60)

    # 1. 사전학습 모델 로드
    base_model, tokenizer = load_pretrained_model()

    # 2. Feature Extraction 데모
    feature_extraction_demo(base_model)

    # 3. Fine-tuning 데모
    classification_model = finetuning_demo()

    # 4. Partial Fine-tuning 데모
    partial_finetuning_demo(classification_model)

    # 5. 전략 비교
    compare_strategies()

    print("\n" + "=" * 60)
    print("실습 완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()
