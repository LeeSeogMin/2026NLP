"""
5-3-gpt-generation.py
GPT-2 텍스트 생성 전략 비교

이 스크립트는 GPT-2 모델을 사용하여 다양한 텍스트 생성 전략을 비교한다:
1. GPT-2 모델 로드 및 구조 분석
2. BPE 토큰화 이해
3. Greedy Search / Beam Search
4. Temperature / Top-k / Top-p Sampling
5. 전략 조합 비교

실행 방법:
    cd practice/chapter5
    pip install -r code/requirements.txt
    python code/5-3-gpt-generation.py
"""

import warnings
warnings.filterwarnings("ignore")

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config


def load_model():
    """GPT-2 모델 로드 및 구조 분석"""
    print("=" * 60)
    print("[1] GPT-2 모델 로드 및 구조 분석")
    print("=" * 60)

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()

    # pad_token 설정
    tokenizer.pad_token = tokenizer.eos_token

    num_params = sum(p.numel() for p in model.parameters())
    config = model.config

    print(f"\n[모델 정보]")
    print(f"  모델명: gpt2 (GPT-2 Small)")
    print(f"  어휘 크기: {tokenizer.vocab_size:,}")
    print(f"  총 파라미터 수: {num_params:,} ({num_params / 1e6:.1f}M)")
    print(f"\n[모델 구성]")
    print(f"  층 수: {config.n_layer}")
    print(f"  은닉 차원: {config.n_embd}")
    print(f"  어텐션 헤드: {config.n_head}")
    print(f"  컨텍스트 길이: {config.n_positions}")

    # 아키텍처 구조
    print(f"\n[아키텍처 구조]")
    print(f"  Token Embedding: {model.transformer.wte.weight.shape}")
    print(f"  Position Embedding: {model.transformer.wpe.weight.shape}")
    print(f"  Decoder Blocks: {len(model.transformer.h)}개")
    first = model.transformer.h[0]
    print(f"  Block 0 — Attention c_attn: {first.attn.c_attn.weight.shape}")
    print(f"  Block 0 — FFN c_fc: {first.mlp.c_fc.weight.shape}")
    print(f"  Final LayerNorm: {model.transformer.ln_f.weight.shape}")
    print(f"  LM Head: {model.lm_head.weight.shape}")

    return tokenizer, model


def bpe_tokenization_demo(tokenizer):
    """BPE 토큰화 이해"""
    print("\n" + "=" * 60)
    print("[2] BPE 토큰화")
    print("=" * 60)

    texts = [
        "Hello, world!",
        "artificial intelligence",
        "The transformer architecture is revolutionary.",
    ]

    print("\n[BPE 토큰화 결과]")
    for text in texts:
        tokens = tokenizer.tokenize(text)
        ids = tokenizer.encode(text)
        print(f"\n  텍스트: '{text}'")
        print(f"  토큰: {tokens}")
        print(f"  토큰 ID: {ids}")
        print(f"  토큰 수: {len(tokens)}")

    # 서브워드 분해 예시
    print("\n[서브워드 분해 예시]")
    words = ["incredible", "unfortunately", "GPT", "OpenAI", "tokenization"]
    for word in words:
        tokens = tokenizer.tokenize(word)
        print(f"  {word:20} -> {tokens}")


def generation_strategies_demo(tokenizer, model):
    """텍스트 생성 전략 비교"""
    print("\n" + "=" * 60)
    print("[3] 텍스트 생성 전략 비교")
    print("=" * 60)

    prompt = "The future of artificial intelligence is"
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    max_len = 50

    print(f"\n프롬프트: \"{prompt}\"")
    print("-" * 60)

    # Greedy Search
    with torch.no_grad():
        output = model.generate(
            input_ids, max_length=max_len, do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    print(f"\n[Greedy Search]")
    print(f"  {tokenizer.decode(output[0], skip_special_tokens=True)}")

    # Beam Search (k=4)
    with torch.no_grad():
        output = model.generate(
            input_ids, max_length=max_len, num_beams=4,
            do_sample=False, early_stopping=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    print(f"\n[Beam Search (k=4)]")
    print(f"  {tokenizer.decode(output[0], skip_special_tokens=True)}")

    # Temperature Sampling
    for temp in [0.3, 0.7, 1.2]:
        torch.manual_seed(42)
        with torch.no_grad():
            output = model.generate(
                input_ids, max_length=max_len, do_sample=True,
                temperature=temp, pad_token_id=tokenizer.eos_token_id,
            )
        desc = "매우 보수적" if temp < 0.5 else "적절한 균형" if temp < 1.0 else "다양하지만 불안정"
        print(f"\n[Temperature = {temp}] ({desc})")
        print(f"  {tokenizer.decode(output[0], skip_special_tokens=True)}")

    # Top-k Sampling
    for k in [10, 50]:
        torch.manual_seed(42)
        with torch.no_grad():
            output = model.generate(
                input_ids, max_length=max_len, do_sample=True,
                top_k=k, temperature=0.8,
                pad_token_id=tokenizer.eos_token_id,
            )
        print(f"\n[Top-k = {k}]")
        print(f"  {tokenizer.decode(output[0], skip_special_tokens=True)}")

    # Top-p Sampling
    for p in [0.5, 0.9]:
        torch.manual_seed(42)
        with torch.no_grad():
            output = model.generate(
                input_ids, max_length=max_len, do_sample=True,
                top_p=p, top_k=0, temperature=0.8,
                pad_token_id=tokenizer.eos_token_id,
            )
        print(f"\n[Top-p = {p}]")
        print(f"  {tokenizer.decode(output[0], skip_special_tokens=True)}")


def combined_strategies_demo(tokenizer, model):
    """전략 조합 비교"""
    print("\n" + "=" * 60)
    print("[4] 전략 조합 비교")
    print("=" * 60)

    prompt = "Once upon a time in a distant kingdom,"
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    strategies = [
        ("Greedy", {"do_sample": False}),
        ("Beam (k=4)", {"do_sample": False, "num_beams": 4}),
        ("Top-k=40 + T=0.7", {"do_sample": True, "top_k": 40, "temperature": 0.7}),
        ("Top-p=0.9 + T=0.8", {"do_sample": True, "top_p": 0.9, "top_k": 0, "temperature": 0.8}),
        ("Top-k=50 + Top-p=0.95 + T=0.9", {"do_sample": True, "top_k": 50, "top_p": 0.95, "temperature": 0.9}),
    ]

    print(f"\n프롬프트: \"{prompt}\"")
    print("-" * 60)

    for name, params in strategies:
        torch.manual_seed(42)
        with torch.no_grad():
            output = model.generate(
                input_ids, max_length=60,
                pad_token_id=tokenizer.eos_token_id,
                **params,
            )
        print(f"\n[{name}]")
        print(f"  {tokenizer.decode(output[0], skip_special_tokens=True)}")


def strategy_summary():
    """생성 전략 비교 요약"""
    print("\n" + "=" * 60)
    print("[5] 생성 전략 비교 요약")
    print("=" * 60)

    print("""
  전략               다양성  품질   속도   추천 용도
  ──────────────────────────────────────────────────────────
  Greedy Search      낮음   중간   빠름   결정적 태스크
  Beam Search        낮음   높음   중간   번역, 요약
  Temperature        조절   조절   빠름   창의성 조절
  Top-k Sampling     중간   중간   빠름   일반 생성
  Top-p (Nucleus)    높음   높음   빠름   창의적 생성

  [권장 설정]
    챗봇/대화:  Top-p=0.9,  Temperature=0.7
    창작 글쓰기: Top-p=0.95, Temperature=0.9
    코드 생성:  Top-p=0.95, Temperature=0.2
    번역/요약:  Beam Search (k=4~5)
""")


def gpt_series_comparison():
    """GPT 시리즈 발전사"""
    print("=" * 60)
    print("[6] GPT 시리즈 발전사")
    print("=" * 60)

    print("""
  모델        연도   파라미터     핵심 특징
  ────────────────────────────────────────────────────────────
  GPT-1       2018   117M        Pre-train + Fine-tune 패러다임
  GPT-2       2019   1.5B        Zero-shot 가능, 공개 거부 논란
  GPT-3       2020   175B        Few-shot, In-Context Learning
  GPT-4       2023   ~1.7T(추정)  멀티모달, 향상된 추론
  GPT-4o      2024   비공개       음성/이미지/텍스트 통합

  [패러다임 변화]
    GPT-1: 파인튜닝 필요 → GPT-2: Zero-shot 가능
    → GPT-3: Few-shot 가능 → GPT-4: 범용 AI 추구
""")


def main():
    """메인 함수"""
    print("=" * 60)
    print("제5장 실습 — GPT-2 텍스트 생성 전략 비교")
    print("=" * 60)

    # 1. 모델 로드 + 구조 분석
    tokenizer, model = load_model()

    # 2. BPE 토큰화
    bpe_tokenization_demo(tokenizer)

    # 3. 생성 전략 비교
    generation_strategies_demo(tokenizer, model)

    # 4. 전략 조합 비교
    combined_strategies_demo(tokenizer, model)

    # 5. 전략 요약
    strategy_summary()

    # 6. GPT 시리즈 발전사
    gpt_series_comparison()

    print("\n" + "=" * 60)
    print("GPT-2 텍스트 생성 실습 완료")
    print("=" * 60)


if __name__ == "__main__":
    main()
