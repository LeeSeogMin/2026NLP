"""
3장 실습: 다층 퍼셉트론(MLP) 구현
- nn.Module을 상속한 MLP 클래스 정의
- 이진 분류 (XOR) 및 다중 분류 (Iris) 문제 해결
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class MLP(nn.Module):
    """다층 퍼셉트론"""

    def __init__(self, input_size, hidden_sizes, output_size, activation="relu"):
        """
        Args:
            input_size: 입력 특성 수
            hidden_sizes: 은닉층 뉴런 수 리스트 (예: [64, 32])
            output_size: 출력 클래스 수
            activation: 활성화 함수 ("relu", "tanh", "sigmoid")
        """
        super(MLP, self).__init__()

        # 활성화 함수 선택
        activations = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
            "gelu": nn.GELU(),
        }
        self.activation = activations.get(activation, nn.ReLU())

        # 층 구성
        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(self.activation)
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, output_size))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def solve_xor():
    """XOR 문제 해결"""
    print("=" * 50)
    print("1. XOR 문제 해결")
    print("=" * 50)

    # XOR 데이터
    X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
    y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

    print("XOR 데이터:")
    print("  입력      출력")
    for xi, yi in zip(X, y):
        print(f"  {xi.tolist()} → {yi.item():.0f}")

    # 모델 생성
    torch.manual_seed(42)
    model = MLP(input_size=2, hidden_sizes=[4], output_size=1, activation="relu")

    print(f"\n모델 구조:\n{model}")

    # 학습 설정
    criterion = nn.BCEWithLogitsLoss()  # Sigmoid + BCE
    optimizer = optim.Adam(model.parameters(), lr=0.1)

    print("\n학습 시작...")
    print("-" * 40)

    # 학습
    epochs = 1000
    for epoch in range(epochs):
        # 순전파
        outputs = model(X)
        loss = criterion(outputs, y)

        # 역전파
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 200 == 0:
            predictions = (torch.sigmoid(outputs) > 0.5).float()
            accuracy = (predictions == y).float().mean()
            print(f"Epoch {epoch + 1:4d}: Loss = {loss.item():.4f}, "
                  f"Accuracy = {accuracy.item() * 100:.1f}%")

    # 최종 예측
    print("-" * 40)
    print("\n최종 예측:")
    with torch.no_grad():
        outputs = torch.sigmoid(model(X))
        predictions = (outputs > 0.5).int()
        print("  입력      예측   확률")
        for xi, pred, prob in zip(X, predictions, outputs):
            print(f"  {xi.tolist()} → {pred.item()}    ({prob.item():.4f})")

    accuracy = (predictions.flatten() == y.flatten()).float().mean()
    print(f"\n정확도: {accuracy.item() * 100:.1f}%")


def simple_classification():
    """간단한 다중 클래스 분류"""
    print("\n" + "=" * 50)
    print("2. 다중 클래스 분류 (합성 데이터)")
    print("=" * 50)

    # 합성 데이터 생성 (3개 클래스)
    torch.manual_seed(42)
    n_samples = 300

    # 클래스 0: 중심 (0, 0)
    X0 = torch.randn(n_samples // 3, 2) * 0.5 + torch.tensor([0.0, 0.0])
    # 클래스 1: 중심 (2, 2)
    X1 = torch.randn(n_samples // 3, 2) * 0.5 + torch.tensor([2.0, 2.0])
    # 클래스 2: 중심 (2, 0)
    X2 = torch.randn(n_samples // 3, 2) * 0.5 + torch.tensor([2.0, 0.0])

    X = torch.cat([X0, X1, X2], dim=0)
    y = torch.cat([
        torch.zeros(n_samples // 3),
        torch.ones(n_samples // 3),
        torch.full((n_samples // 3,), 2)
    ]).long()

    print(f"데이터: {X.shape[0]}개 샘플, {X.shape[1]}개 특성, 3개 클래스")

    # 데이터 분할 (80% 학습, 20% 테스트)
    indices = torch.randperm(len(X))
    split = int(0.8 * len(X))
    train_idx, test_idx = indices[:split], indices[split:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    print(f"학습 데이터: {len(X_train)}개")
    print(f"테스트 데이터: {len(X_test)}개")

    # 데이터 로더
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # 모델 생성
    model = MLP(input_size=2, hidden_sizes=[16, 8], output_size=3, activation="relu")
    print(f"\n모델 구조:\n{model}")

    # 학습 설정
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    print("\n학습 시작...")
    print("-" * 40)

    # 학습
    epochs = 100
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch_X, batch_y in train_loader:
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if (epoch + 1) % 20 == 0:
            # 테스트 정확도
            model.eval()
            with torch.no_grad():
                test_outputs = model(X_test)
                test_preds = test_outputs.argmax(dim=1)
                test_acc = (test_preds == y_test).float().mean()

            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch + 1:3d}: Loss = {avg_loss:.4f}, "
                  f"Test Accuracy = {test_acc.item() * 100:.1f}%")

    # 최종 평가
    print("-" * 40)
    model.eval()
    with torch.no_grad():
        train_preds = model(X_train).argmax(dim=1)
        test_preds = model(X_test).argmax(dim=1)

        train_acc = (train_preds == y_train).float().mean()
        test_acc = (test_preds == y_test).float().mean()

    print(f"\n최종 결과:")
    print(f"  학습 정확도: {train_acc.item() * 100:.1f}%")
    print(f"  테스트 정확도: {test_acc.item() * 100:.1f}%")

    # 클래스별 정확도
    print("\n클래스별 테스트 정확도:")
    for c in range(3):
        mask = y_test == c
        class_acc = (test_preds[mask] == y_test[mask]).float().mean()
        print(f"  클래스 {c}: {class_acc.item() * 100:.1f}%")


def show_model_parameters():
    """모델 파라미터 확인"""
    print("\n" + "=" * 50)
    print("3. 모델 파라미터 확인")
    print("=" * 50)

    model = MLP(input_size=2, hidden_sizes=[4, 2], output_size=1)

    print("모델 파라미터:")
    total_params = 0
    for name, param in model.named_parameters():
        print(f"  {name}: shape = {list(param.shape)}")
        total_params += param.numel()

    print(f"\n총 파라미터 수: {total_params}")

    # 수동 계산 검증
    print("\n[파라미터 수 계산]")
    print("  입력(2) → 은닉1(4): 2×4 + 4 = 12")
    print("  은닉1(4) → 은닉2(2): 4×2 + 2 = 10")
    print("  은닉2(2) → 출력(1): 2×1 + 1 = 3")
    print(f"  합계: 12 + 10 + 3 = 25")


def main():
    solve_xor()
    simple_classification()
    show_model_parameters()

    print("\n" + "=" * 50)
    print("핵심 정리")
    print("=" * 50)
    print("""
    MLP 구현 요소:
    1. nn.Module 상속
    2. __init__에서 층 정의
    3. forward에서 순전파 정의

    학습 루프 (배치 처리):
    for batch_X, batch_y in train_loader:
        outputs = model(batch_X)          # 순전파
        loss = criterion(outputs, batch_y) # 손실 계산
        optimizer.zero_grad()              # 기울기 초기화
        loss.backward()                    # 역전파
        optimizer.step()                   # 업데이트

    손실 함수 선택:
    - 이진 분류: BCEWithLogitsLoss
    - 다중 분류: CrossEntropyLoss
    - 회귀: MSELoss
    """)


if __name__ == "__main__":
    main()
