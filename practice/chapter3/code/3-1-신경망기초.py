"""
3장 실습: 퍼셉트론과 다층 퍼셉트론(MLP) 기초
- 단층 퍼셉트론의 XOR 문제 한계 시각화
- MLP로 XOR 문제 해결
"""

import numpy as np


def step_function(x):
    """계단 함수 (퍼셉트론 활성화 함수)"""
    return np.where(x > 0, 1, 0)


def sigmoid(x):
    """시그모이드 함수"""
    return 1 / (1 + np.exp(-x))


class Perceptron:
    """단층 퍼셉트론"""

    def __init__(self, input_size, learning_rate=0.1):
        self.weights = np.random.randn(input_size)
        self.bias = np.random.randn()
        self.lr = learning_rate

    def forward(self, x):
        """순전파: 가중합 후 활성화"""
        z = np.dot(x, self.weights) + self.bias
        return step_function(z)

    def train(self, X, y, epochs=100):
        """퍼셉트론 학습 알고리즘"""
        for epoch in range(epochs):
            errors = 0
            for xi, yi in zip(X, y):
                pred = self.forward(xi)
                error = yi - pred
                if error != 0:
                    self.weights += self.lr * error * xi
                    self.bias += self.lr * error
                    errors += 1
            if errors == 0:
                print(f"  수렴 완료 (epoch {epoch + 1})")
                break
        return errors == 0


class SimpleMLP:
    """간단한 2층 MLP (XOR 문제 해결용)"""

    def __init__(self, input_size=2, hidden_size=2, output_size=1, lr=0.5):
        # 가중치 초기화
        self.W1 = np.random.randn(input_size, hidden_size) * 0.5
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, output_size) * 0.5
        self.b2 = np.zeros(output_size)
        self.lr = lr

    def forward(self, x):
        """순전파"""
        self.z1 = np.dot(x, self.W1) + self.b1
        self.a1 = sigmoid(self.z1)  # 은닉층 활성화
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = sigmoid(self.z2)  # 출력층 활성화
        return self.a2

    def backward(self, x, y):
        """역전파"""
        m = x.shape[0]

        # 출력층 오차
        dz2 = self.a2 - y.reshape(-1, 1)
        dW2 = np.dot(self.a1.T, dz2) / m
        db2 = np.sum(dz2, axis=0) / m

        # 은닉층 오차
        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * self.a1 * (1 - self.a1)  # 시그모이드 미분
        dW1 = np.dot(x.T, dz1) / m
        db1 = np.sum(dz1, axis=0) / m

        # 가중치 업데이트
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    def train(self, X, y, epochs=5000):
        """학습"""
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y)

            if (epoch + 1) % 1000 == 0:
                loss = np.mean((output - y.reshape(-1, 1)) ** 2)
                print(f"  Epoch {epoch + 1}: Loss = {loss:.4f}")

    def predict(self, x):
        """예측 (0.5 임계값)"""
        return (self.forward(x) > 0.5).astype(int)


def main():
    # XOR 데이터
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_xor = np.array([0, 1, 1, 0])

    # AND 데이터 (비교용)
    y_and = np.array([0, 0, 0, 1])

    # OR 데이터 (비교용)
    y_or = np.array([0, 1, 1, 1])

    print("=" * 50)
    print("단층 퍼셉트론 실험")
    print("=" * 50)

    # 1. AND 게이트 (선형 분리 가능)
    print("\n[AND 게이트 학습]")
    p_and = Perceptron(input_size=2)
    success = p_and.train(X, y_and)
    print(f"  학습 성공: {success}")
    print(f"  예측: {[p_and.forward(x) for x in X]}")
    print(f"  정답: {list(y_and)}")

    # 2. OR 게이트 (선형 분리 가능)
    print("\n[OR 게이트 학습]")
    p_or = Perceptron(input_size=2)
    success = p_or.train(X, y_or)
    print(f"  학습 성공: {success}")
    print(f"  예측: {[p_or.forward(x) for x in X]}")
    print(f"  정답: {list(y_or)}")

    # 3. XOR 게이트 (선형 분리 불가능)
    print("\n[XOR 게이트 학습 - 단층 퍼셉트론]")
    p_xor = Perceptron(input_size=2)
    success = p_xor.train(X, y_xor, epochs=1000)
    print(f"  학습 성공: {success}")
    predictions = [p_xor.forward(x) for x in X]
    print(f"  예측: {predictions}")
    print(f"  정답: {list(y_xor)}")
    accuracy = sum(p == t for p, t in zip(predictions, y_xor)) / len(y_xor)
    print(f"  정확도: {accuracy * 100:.1f}%")
    print("  → 단층 퍼셉트론은 XOR을 학습할 수 없음!")

    print("\n" + "=" * 50)
    print("다층 퍼셉트론(MLP) 실험")
    print("=" * 50)

    # 4. MLP로 XOR 해결
    print("\n[XOR 게이트 학습 - MLP (은닉층 2개 뉴런)]")
    np.random.seed(42)  # 재현성
    mlp = SimpleMLP(input_size=2, hidden_size=2, output_size=1, lr=1.0)
    mlp.train(X, y_xor, epochs=5000)

    predictions = mlp.predict(X)
    print(f"\n  최종 예측: {predictions.flatten().tolist()}")
    print(f"  정답:      {list(y_xor)}")
    accuracy = np.mean(predictions.flatten() == y_xor)
    print(f"  정확도: {accuracy * 100:.1f}%")
    print("  → MLP는 비선형 문제인 XOR을 해결할 수 있음!")

    # 학습된 가중치 출력
    print("\n[학습된 가중치]")
    print(f"  은닉층 가중치 W1:\n{mlp.W1}")
    print(f"  은닉층 편향 b1: {mlp.b1}")
    print(f"  출력층 가중치 W2:\n{mlp.W2}")
    print(f"  출력층 편향 b2: {mlp.b2}")


if __name__ == "__main__":
    main()
