"""
14장 실습: 모델 평가 및 시각화
- 분류 평가 지표 계산
- Confusion Matrix 시각화
- 학습 곡선 시각화
- 임베딩 시각화 (t-SNE, UMAP)
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, ConfusionMatrixDisplay
)
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정 (시스템에 따라 조정)
plt.rcParams['font.family'] = 'AppleGothic'  # macOS
plt.rcParams['axes.unicode_minus'] = False


def evaluate_classification():
    """분류 모델 평가 지표 계산"""
    print("=" * 60)
    print("1. 분류 모델 평가 지표")
    print("=" * 60)

    # 샘플 예측 결과 (감성 분석 예시: 0=부정, 1=중립, 2=긍정)
    y_true = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 2,
                       0, 1, 2, 0, 1, 2, 0, 1, 2, 2])
    y_pred = np.array([0, 0, 1, 1, 1, 2, 2, 2, 2, 1,
                       0, 1, 2, 1, 1, 2, 0, 0, 2, 2])

    class_names = ['부정', '중립', '긍정']

    # 기본 지표 계산
    accuracy = accuracy_score(y_true, y_pred)
    precision_macro = precision_score(y_true, y_pred, average='macro')
    recall_macro = recall_score(y_true, y_pred, average='macro')
    f1_macro = f1_score(y_true, y_pred, average='macro')

    print(f"\n정확도 (Accuracy): {accuracy:.4f}")
    print(f"정밀도 (Precision, Macro): {precision_macro:.4f}")
    print(f"재현율 (Recall, Macro): {recall_macro:.4f}")
    print(f"F1 점수 (F1-Score, Macro): {f1_macro:.4f}")

    # 상세 분류 리포트
    print("\n[Classification Report]")
    print(classification_report(y_true, y_pred, target_names=class_names))

    return y_true, y_pred, class_names


def visualize_confusion_matrix(y_true, y_pred, class_names):
    """Confusion Matrix 시각화"""
    print("\n" + "=" * 60)
    print("2. Confusion Matrix 시각화")
    print("=" * 60)

    # Confusion Matrix 계산
    cm = confusion_matrix(y_true, y_pred)

    # 시각화
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 절대값 표시
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=axes[0])
    axes[0].set_xlabel('예측 레이블')
    axes[0].set_ylabel('실제 레이블')
    axes[0].set_title('Confusion Matrix (절대값)')

    # 정규화 (행 기준 - 각 클래스별 비율)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=axes[1])
    axes[1].set_xlabel('예측 레이블')
    axes[1].set_ylabel('실제 레이블')
    axes[1].set_title('Confusion Matrix (정규화)')

    plt.tight_layout()
    plt.savefig('../data/output/confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("Confusion Matrix:")
    print(cm)
    print("\n정규화 Confusion Matrix:")
    print(np.round(cm_normalized, 2))
    print("\n저장 완료: ../data/output/confusion_matrix.png")


def visualize_learning_curves():
    """학습 곡선 시각화"""
    print("\n" + "=" * 60)
    print("3. 학습 곡선 시각화")
    print("=" * 60)

    # 시뮬레이션된 학습 곡선 데이터
    epochs = np.arange(1, 21)

    # 정상적인 학습 곡선
    train_loss = 2.5 * np.exp(-0.2 * epochs) + 0.3 + np.random.normal(0, 0.05, 20)
    val_loss = 2.5 * np.exp(-0.15 * epochs) + 0.5 + np.random.normal(0, 0.08, 20)

    train_acc = 1 - 0.7 * np.exp(-0.25 * epochs) + np.random.normal(0, 0.02, 20)
    val_acc = 1 - 0.75 * np.exp(-0.2 * epochs) + np.random.normal(0, 0.03, 20)
    train_acc = np.clip(train_acc, 0, 1)
    val_acc = np.clip(val_acc, 0, 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Loss 곡선
    axes[0].plot(epochs, train_loss, 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, val_loss, 'r-', label='Validation Loss', linewidth=2)
    axes[0].axvline(x=15, color='g', linestyle='--', alpha=0.7, label='Best Model')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss Curve')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy 곡선
    axes[1].plot(epochs, train_acc, 'b-', label='Train Accuracy', linewidth=2)
    axes[1].plot(epochs, val_acc, 'r-', label='Validation Accuracy', linewidth=2)
    axes[1].axvline(x=15, color='g', linestyle='--', alpha=0.7, label='Best Model')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy Curve')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('../data/output/learning_curves.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"최종 Train Loss: {train_loss[-1]:.4f}")
    print(f"최종 Validation Loss: {val_loss[-1]:.4f}")
    print(f"최종 Train Accuracy: {train_acc[-1]:.4f}")
    print(f"최종 Validation Accuracy: {val_acc[-1]:.4f}")
    print(f"Best Epoch: 15 (가장 낮은 Validation Loss 기준)")
    print("\n저장 완료: ../data/output/learning_curves.png")


def visualize_embeddings():
    """임베딩 시각화 (t-SNE, UMAP)"""
    print("\n" + "=" * 60)
    print("4. 임베딩 시각화 (t-SNE)")
    print("=" * 60)

    # 시뮬레이션된 임베딩 데이터 (384차원 -> 2차원)
    np.random.seed(42)
    n_samples = 150
    n_classes = 3
    embedding_dim = 384

    # 클래스별로 다른 중심을 가진 임베딩 생성
    embeddings = []
    labels = []

    for i in range(n_classes):
        # 각 클래스별 클러스터 생성
        center = np.random.randn(embedding_dim) * 2
        cluster = center + np.random.randn(n_samples // n_classes, embedding_dim) * 0.5
        embeddings.append(cluster)
        labels.extend([i] * (n_samples // n_classes))

    embeddings = np.vstack(embeddings)
    labels = np.array(labels)

    class_names = ['부정', '중립', '긍정']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

    print(f"원본 임베딩 형태: {embeddings.shape}")

    # t-SNE 차원 축소
    print("t-SNE 차원 축소 중...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embeddings_2d = tsne.fit_transform(embeddings)

    print(f"축소된 임베딩 형태: {embeddings_2d.shape}")

    # 시각화
    fig, ax = plt.subplots(figsize=(10, 8))

    for i, (class_name, color) in enumerate(zip(class_names, colors)):
        mask = labels == i
        ax.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                   c=color, label=class_name, alpha=0.7, s=50)

    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.set_title('t-SNE Embedding Visualization')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('../data/output/tsne_embeddings.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("\n저장 완료: ../data/output/tsne_embeddings.png")

    # UMAP 시각화 (umap-learn 설치 시)
    try:
        import umap

        print("\n" + "=" * 60)
        print("5. UMAP 임베딩 시각화")
        print("=" * 60)

        print("UMAP 차원 축소 중...")
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
        embeddings_umap = reducer.fit_transform(embeddings)

        fig, ax = plt.subplots(figsize=(10, 8))

        for i, (class_name, color) in enumerate(zip(class_names, colors)):
            mask = labels == i
            ax.scatter(embeddings_umap[mask, 0], embeddings_umap[mask, 1],
                       c=color, label=class_name, alpha=0.7, s=50)

        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        ax.set_title('UMAP Embedding Visualization')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('../data/output/umap_embeddings.png', dpi=150, bbox_inches='tight')
        plt.close()

        print("저장 완료: ../data/output/umap_embeddings.png")

    except ImportError:
        print("\nUMAP 시각화를 위해 'umap-learn' 패키지를 설치하세요:")
        print("pip install umap-learn")


def demonstrate_generation_metrics():
    """텍스트 생성 평가 지표 데모"""
    print("\n" + "=" * 60)
    print("6. 텍스트 생성 평가 지표")
    print("=" * 60)

    try:
        import evaluate

        # ROUGE 평가
        rouge = evaluate.load('rouge')

        predictions = [
            "딥러닝은 인공지능의 한 분야로 신경망을 사용한다.",
            "자연어처리는 텍스트 데이터를 분석하는 기술이다."
        ]
        references = [
            "딥러닝은 인공지능의 핵심 분야로 다층 신경망을 활용한다.",
            "자연어처리는 텍스트와 음성 데이터를 이해하고 생성하는 기술이다."
        ]

        results = rouge.compute(predictions=predictions, references=references)

        print("\n[ROUGE Score 결과]")
        print(f"ROUGE-1: {results['rouge1']:.4f}")
        print(f"ROUGE-2: {results['rouge2']:.4f}")
        print(f"ROUGE-L: {results['rougeL']:.4f}")

        print("\n해석:")
        print("- ROUGE-1: 단어 단위 겹침 (unigram)")
        print("- ROUGE-2: 2-gram 단위 겹침")
        print("- ROUGE-L: 최장 공통 부분 수열 (LCS)")

    except Exception as e:
        print(f"\nevaluate 라이브러리 오류: {e}")
        print("간단한 ROUGE 계산 예시를 출력합니다.")

        print("\n[ROUGE Score 계산 원리]")
        print("예측: '딥러닝은 인공지능의 한 분야로 신경망을 사용한다.'")
        print("참조: '딥러닝은 인공지능의 핵심 분야로 다층 신경망을 활용한다.'")
        print("\n공통 단어: 딥러닝은, 인공지능의, 분야로, 신경망")
        print("ROUGE-1 (예시) ≈ 0.67 (공통 단어 비율)")


def main():
    """메인 실행 함수"""
    print("=" * 60)
    print("14장 실습: 모델 평가 및 시각화")
    print("=" * 60)

    # 1. 분류 평가 지표
    y_true, y_pred, class_names = evaluate_classification()

    # 2. Confusion Matrix 시각화
    visualize_confusion_matrix(y_true, y_pred, class_names)

    # 3. 학습 곡선 시각화
    visualize_learning_curves()

    # 4-5. 임베딩 시각화 (t-SNE, UMAP)
    visualize_embeddings()

    # 6. 텍스트 생성 평가 지표
    demonstrate_generation_metrics()

    print("\n" + "=" * 60)
    print("모든 시각화 완료!")
    print("저장된 파일들:")
    print("  - confusion_matrix.png")
    print("  - learning_curves.png")
    print("  - tsne_embeddings.png")
    print("  - umap_embeddings.png (UMAP 설치 시)")
    print("=" * 60)


if __name__ == "__main__":
    main()
