"""
10-4-generation-strategies.py
텍스트 생성 전략 비교 실습

이 스크립트는 다양한 텍스트 생성 전략을 비교한다:
1. Greedy Search
2. Beam Search
3. Temperature Sampling
4. Top-k Sampling
5. Top-p (Nucleus) Sampling

실행 방법:
    python 10-4-generation-strategies.py
"""

import warnings
warnings.filterwarnings('ignore')

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel


def setup_model():
    """모델 및 토크나이저 설정"""
    print("=" * 60)
    print("GPT-2 모델 로드 중...")
    print("=" * 60)

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model.eval()

    # pad_token 설정 (GPT-2는 기본적으로 pad_token이 없음)
    tokenizer.pad_token = tokenizer.eos_token

    print("모델 로드 완료!")
    return tokenizer, model


def greedy_search_demo(tokenizer, model, prompt):
    """Greedy Search 데모"""
    print("\n" + "=" * 60)
    print("[1] Greedy Search (탐욕 검색)")
    print("=" * 60)

    print("\n[원리]")
    print("  - 매 단계에서 가장 높은 확률의 토큰 선택")
    print("  - 결정적 (deterministic) - 항상 같은 결과")
    print("  - 장점: 빠르고 단순")
    print("  - 단점: 반복적, 다양성 부족")

    input_ids = tokenizer.encode(prompt, return_tensors='pt')

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=60,
            do_sample=False,  # Greedy
            pad_token_id=tokenizer.eos_token_id
        )

    generated = tokenizer.decode(output[0], skip_special_tokens=True)

    print(f"\n프롬프트: \"{prompt}\"")
    print(f"\n[생성 결과]")
    print(f"  {generated}")


def beam_search_demo(tokenizer, model, prompt):
    """Beam Search 데모"""
    print("\n" + "=" * 60)
    print("[2] Beam Search (빔 검색)")
    print("=" * 60)

    print("\n[원리]")
    print("  - K개의 최고 후보 시퀀스 유지")
    print("  - 전체 시퀀스 확률 최적화")
    print("  - 장점: 전역 최적에 더 가까운 결과")
    print("  - 단점: 여전히 반복 문제 존재")

    input_ids = tokenizer.encode(prompt, return_tensors='pt')

    beam_widths = [2, 4]

    print(f"\n프롬프트: \"{prompt}\"")

    for beam_width in beam_widths:
        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_length=60,
                num_beams=beam_width,
                do_sample=False,
                early_stopping=True,
                pad_token_id=tokenizer.eos_token_id
            )

        generated = tokenizer.decode(output[0], skip_special_tokens=True)
        print(f"\n[Beam Width = {beam_width}]")
        print(f"  {generated}")


def temperature_sampling_demo(tokenizer, model, prompt):
    """Temperature Sampling 데모"""
    print("\n" + "=" * 60)
    print("[3] Temperature Sampling")
    print("=" * 60)

    print("\n[원리]")
    print("  - 소프트맥스 온도 조절로 확률 분포 변형")
    print("  - T < 1: 분포 날카롭게 (더 결정적)")
    print("  - T > 1: 분포 평탄하게 (더 다양함)")
    print("  - T → 0: Greedy와 동일")

    input_ids = tokenizer.encode(prompt, return_tensors='pt')

    temperatures = [0.3, 0.7, 1.0, 1.5]

    print(f"\n프롬프트: \"{prompt}\"")

    for temp in temperatures:
        # 시드 고정으로 비교 가능하게
        torch.manual_seed(42)
        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_length=60,
                do_sample=True,
                temperature=temp,
                pad_token_id=tokenizer.eos_token_id
            )

        generated = tokenizer.decode(output[0], skip_special_tokens=True)
        desc = "매우 결정적" if temp < 0.5 else "결정적" if temp < 1.0 else "기본" if temp == 1.0 else "다양함"
        print(f"\n[Temperature = {temp}] ({desc})")
        print(f"  {generated}")


def topk_sampling_demo(tokenizer, model, prompt):
    """Top-k Sampling 데모"""
    print("\n" + "=" * 60)
    print("[4] Top-k Sampling")
    print("=" * 60)

    print("\n[원리]")
    print("  - 상위 k개 토큰에서만 샘플링")
    print("  - k가 작으면: 안전하지만 다양성 감소")
    print("  - k가 크면: 다양하지만 품질 저하 위험")

    input_ids = tokenizer.encode(prompt, return_tensors='pt')

    k_values = [5, 20, 50]

    print(f"\n프롬프트: \"{prompt}\"")

    for k in k_values:
        torch.manual_seed(42)
        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_length=60,
                do_sample=True,
                top_k=k,
                temperature=0.8,
                pad_token_id=tokenizer.eos_token_id
            )

        generated = tokenizer.decode(output[0], skip_special_tokens=True)
        print(f"\n[Top-k = {k}]")
        print(f"  {generated}")


def topp_sampling_demo(tokenizer, model, prompt):
    """Top-p (Nucleus) Sampling 데모"""
    print("\n" + "=" * 60)
    print("[5] Top-p (Nucleus) Sampling")
    print("=" * 60)

    print("\n[원리]")
    print("  - 누적 확률이 p를 넘는 최소 토큰 집합에서 샘플링")
    print("  - 문맥에 따라 후보 수가 동적으로 조절")
    print("  - Top-k보다 자연스러운 결과")
    print("  - 일반적으로 p=0.9~0.95 권장")

    input_ids = tokenizer.encode(prompt, return_tensors='pt')

    p_values = [0.5, 0.9, 0.95]

    print(f"\n프롬프트: \"{prompt}\"")

    for p in p_values:
        torch.manual_seed(42)
        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_length=60,
                do_sample=True,
                top_p=p,
                top_k=0,  # Top-p만 사용
                temperature=0.8,
                pad_token_id=tokenizer.eos_token_id
            )

        generated = tokenizer.decode(output[0], skip_special_tokens=True)
        print(f"\n[Top-p = {p}]")
        print(f"  {generated}")


def combined_strategies_demo(tokenizer, model, prompt):
    """여러 전략 조합 비교"""
    print("\n" + "=" * 60)
    print("[6] 전략 조합 비교")
    print("=" * 60)

    input_ids = tokenizer.encode(prompt, return_tensors='pt')

    strategies = [
        {"name": "Greedy", "params": {"do_sample": False}},
        {"name": "Beam (k=4)", "params": {"do_sample": False, "num_beams": 4}},
        {"name": "Top-k=40 + T=0.7", "params": {"do_sample": True, "top_k": 40, "temperature": 0.7}},
        {"name": "Top-p=0.9 + T=0.8", "params": {"do_sample": True, "top_p": 0.9, "top_k": 0, "temperature": 0.8}},
        {"name": "Top-k=50 + Top-p=0.95 + T=0.9", "params": {"do_sample": True, "top_k": 50, "top_p": 0.95, "temperature": 0.9}},
    ]

    print(f"\n프롬프트: \"{prompt}\"")
    print("-" * 60)

    for strategy in strategies:
        torch.manual_seed(42)
        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_length=60,
                pad_token_id=tokenizer.eos_token_id,
                **strategy["params"]
            )

        generated = tokenizer.decode(output[0], skip_special_tokens=True)
        print(f"\n[{strategy['name']}]")
        print(f"  {generated}")


def strategy_comparison_table():
    """생성 전략 비교표"""
    print("\n" + "=" * 60)
    print("[7] 생성 전략 비교 요약")
    print("=" * 60)

    print("""
┌──────────────────┬────────┬────────┬────────┬─────────────────┐
│       전략       │ 다양성 │  품질  │  속도  │     추천 용도   │
├──────────────────┼────────┼────────┼────────┼─────────────────┤
│ Greedy Search    │  낮음  │  중간  │  빠름  │ 결정적 태스크   │
│ Beam Search      │  낮음  │  높음  │  중간  │ 번역, 요약      │
│ Temperature      │  조절  │  조절  │  빠름  │ 창의성 조절     │
│ Top-k Sampling   │  중간  │  중간  │  빠름  │ 일반 생성       │
│ Top-p (Nucleus)  │  높음  │  높음  │  빠름  │ 창의적 생성     │
└──────────────────┴────────┴────────┴────────┴─────────────────┘

[권장 설정]
  - 챗봇, 대화: Top-p=0.9, Temperature=0.7
  - 창작 글쓰기: Top-p=0.95, Temperature=0.9~1.0
  - 코드 생성: Top-p=0.95, Temperature=0.2~0.4
  - 번역/요약: Beam Search (k=4~5)
""")


def main():
    """메인 함수"""
    print("=" * 60)
    print("텍스트 생성 전략 비교 실습")
    print("=" * 60)

    # 모델 로드
    tokenizer, model = setup_model()

    # 테스트 프롬프트
    prompt = "Once upon a time in a distant kingdom,"

    # 1. Greedy Search
    greedy_search_demo(tokenizer, model, prompt)

    # 2. Beam Search
    beam_search_demo(tokenizer, model, prompt)

    # 3. Temperature Sampling
    temperature_sampling_demo(tokenizer, model, prompt)

    # 4. Top-k Sampling
    topk_sampling_demo(tokenizer, model, prompt)

    # 5. Top-p Sampling
    topp_sampling_demo(tokenizer, model, prompt)

    # 6. 조합 비교
    combined_strategies_demo(tokenizer, model, prompt)

    # 7. 비교표
    strategy_comparison_table()

    print("\n" + "=" * 60)
    print("생성 전략 비교 실습 완료")
    print("=" * 60)


if __name__ == "__main__":
    main()
