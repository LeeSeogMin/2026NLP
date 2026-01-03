"""
4장 실습: nn.Module을 활용한 모델 정의
- 커스텀 모델 클래스 작성
- 파라미터 관리 및 초기화
"""

import torch
import torch.nn as nn


class SimpleMLP(nn.Module):
    """간단한 다층 퍼셉트론"""

    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleMLP, self).__init__()

        # 층 정의
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """순전파 정의"""
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class FlexibleMLP(nn.Module):
    """유연한 다층 퍼셉트론 (가변 은닉층)"""

    def __init__(self, input_size, hidden_sizes, output_size, dropout=0.2):
        super(FlexibleMLP, self).__init__()

        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, output_size))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def main():
    print("=" * 50)
    print("1. 간단한 MLP 모델")
    print("=" * 50)

    model = SimpleMLP(input_size=10, hidden_size=32, output_size=2)
    print(f"모델 구조:\n{model}")

    # 파라미터 확인
    print("\n[모델 파라미터]")
    total_params = 0
    for name, param in model.named_parameters():
        print(f"  {name}: {list(param.shape)}")
        total_params += param.numel()
    print(f"\n  총 파라미터 수: {total_params}")

    # 순전파 테스트
    x = torch.randn(5, 10)  # 배치 5, 입력 10
    output = model(x)
    print(f"\n[순전파 테스트]")
    print(f"  입력 shape: {x.shape}")
    print(f"  출력 shape: {output.shape}")

    print("\n" + "=" * 50)
    print("2. 유연한 MLP 모델 (BatchNorm + Dropout)")
    print("=" * 50)

    model2 = FlexibleMLP(
        input_size=20,
        hidden_sizes=[64, 32, 16],
        output_size=3,
        dropout=0.3
    )
    print(f"모델 구조:\n{model2}")

    # 파라미터 수 계산
    total_params = sum(p.numel() for p in model2.parameters())
    trainable_params = sum(p.numel() for p in model2.parameters() if p.requires_grad)
    print(f"\n총 파라미터: {total_params}")
    print(f"학습 가능 파라미터: {trainable_params}")

    print("\n" + "=" * 50)
    print("3. 파라미터 초기화")
    print("=" * 50)

    def init_weights(m):
        """가중치 초기화 함수"""
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    model3 = SimpleMLP(10, 32, 2)

    print("[초기화 전 가중치 통계]")
    print(f"  fc1.weight mean: {model3.fc1.weight.mean().item():.4f}")
    print(f"  fc1.weight std: {model3.fc1.weight.std().item():.4f}")

    model3.apply(init_weights)

    print("\n[Xavier 초기화 후 가중치 통계]")
    print(f"  fc1.weight mean: {model3.fc1.weight.mean().item():.4f}")
    print(f"  fc1.weight std: {model3.fc1.weight.std().item():.4f}")

    print("\n" + "=" * 50)
    print("4. 모델 저장 및 로드")
    print("=" * 50)

    # 가중치만 저장 (권장)
    torch.save(model.state_dict(), 'model_weights.pth')
    print("모델 가중치 저장: model_weights.pth")

    # 가중치 로드
    model_loaded = SimpleMLP(10, 32, 2)
    model_loaded.load_state_dict(torch.load('model_weights.pth', weights_only=True))
    print("모델 가중치 로드 완료")

    # 정리
    import os
    os.remove('model_weights.pth')

    print("\n" + "=" * 50)
    print("핵심 정리")
    print("=" * 50)
    print("""
    nn.Module 사용법:
    1. __init__: 층 정의 (Linear, Conv2d, etc.)
    2. forward: 순전파 정의

    주요 레이어:
    - nn.Linear: 완전 연결층
    - nn.ReLU, nn.GELU: 활성화 함수
    - nn.Dropout: 드롭아웃
    - nn.BatchNorm1d: 배치 정규화
    - nn.Sequential: 층 순차 연결

    파라미터 관리:
    - model.parameters(): 모든 파라미터
    - model.named_parameters(): 이름과 함께
    - model.apply(fn): 모든 모듈에 함수 적용
    """)


if __name__ == "__main__":
    main()
