"""
10-1-gpt-basics.py
GPT 모델 기본 사용법 실습

이 스크립트는 Hugging Face Transformers를 사용하여
GPT-2 모델의 기본 사용법을 보여준다:
1. 모델 및 토크나이저 로드
2. 기본 텍스트 생성
3. 토큰화 과정 이해
4. 모델 구조 분석

실행 방법:
    python 10-1-gpt-basics.py
"""

import warnings
warnings.filterwarnings('ignore')

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config


def load_model_demo():
    """GPT-2 모델 로드 데모"""
    print("=" * 60)
    print("[1] GPT-2 모델 로드")
    print("=" * 60)

    # 토크나이저 로드
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    # 모델 로드
    model = GPT2LMHeadModel.from_pretrained('gpt2')

    print(f"\n[모델 정보]")
    print(f"  모델명: gpt2 (GPT-2 Small)")
    print(f"  어휘 크기: {tokenizer.vocab_size:,}")

    # 모델 파라미터 수 계산
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  총 파라미터 수: {num_params:,} ({num_params/1e6:.1f}M)")

    # 모델 구성 정보
    config = model.config
    print(f"\n[모델 구성]")
    print(f"  층 수 (n_layer): {config.n_layer}")
    print(f"  은닉 차원 (n_embd): {config.n_embd}")
    print(f"  어텐션 헤드 수 (n_head): {config.n_head}")
    print(f"  최대 시퀀스 길이: {config.n_positions}")

    return tokenizer, model


def tokenization_demo(tokenizer):
    """GPT-2 토큰화 데모"""
    print("\n" + "=" * 60)
    print("[2] GPT-2 토큰화")
    print("=" * 60)

    # 테스트 문장
    text = "GPT is a powerful language model for text generation."

    print(f"\n원본 텍스트: \"{text}\"")

    # 토큰화
    tokens = tokenizer.tokenize(text)
    print(f"\n토큰화 결과: {tokens}")
    print(f"토큰 수: {len(tokens)}")

    # 토큰 → ID 변환
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    print(f"\n토큰 ID: {token_ids}")

    # encode() 메서드 (텐서로 변환)
    encoded = tokenizer.encode(text, return_tensors='pt')
    print(f"\nencode() 결과: {encoded}")

    # 디코딩
    decoded = tokenizer.decode(encoded[0])
    print(f"decode() 결과: \"{decoded}\"")

    # BPE 토큰화 특징 보여주기
    print("\n[BPE 토큰화 예시]")
    words = ["incredible", "unfortunately", "GPT", "OpenAI", "2023"]
    for word in words:
        tokens = tokenizer.tokenize(word)
        print(f"  {word:20} → {tokens}")

    return tokenizer


def simple_generation_demo(tokenizer, model):
    """간단한 텍스트 생성 데모"""
    print("\n" + "=" * 60)
    print("[3] 기본 텍스트 생성")
    print("=" * 60)

    model.eval()

    # 프롬프트
    prompt = "The future of artificial intelligence is"
    print(f"\n프롬프트: \"{prompt}\"")

    # 토큰화
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    print(f"입력 토큰 수: {input_ids.shape[1]}")

    # 생성
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=50,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False  # Greedy Search
        )

    # 디코딩
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    print(f"\n[생성 결과 - Greedy Search]")
    print(f"  {generated_text}")

    # Sampling으로 생성
    with torch.no_grad():
        output_sample = model.generate(
            input_ids,
            max_length=50,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )

    generated_text_sample = tokenizer.decode(output_sample[0], skip_special_tokens=True)

    print(f"\n[생성 결과 - Top-p Sampling (p=0.9, T=0.7)]")
    print(f"  {generated_text_sample}")


def model_architecture_demo(model):
    """모델 아키텍처 분석 데모"""
    print("\n" + "=" * 60)
    print("[4] 모델 아키텍처 분석")
    print("=" * 60)

    print("\n[GPT-2 레이어 구조]")
    print("-" * 60)

    # 주요 컴포넌트 출력
    print("\n1. 임베딩 레이어")
    print(f"   wte (Token Embedding): {model.transformer.wte.weight.shape}")
    print(f"   wpe (Position Embedding): {model.transformer.wpe.weight.shape}")

    print("\n2. Transformer 블록 (h)")
    print(f"   총 블록 수: {len(model.transformer.h)}")

    # 첫 번째 블록 상세
    first_block = model.transformer.h[0]
    print(f"\n   [Block 0 구조]")
    print(f"   - ln_1 (LayerNorm): {first_block.ln_1.weight.shape}")
    print(f"   - attn (Attention):")
    print(f"     - c_attn (Q,K,V 프로젝션): {first_block.attn.c_attn.weight.shape}")
    print(f"     - c_proj (출력 프로젝션): {first_block.attn.c_proj.weight.shape}")
    print(f"   - ln_2 (LayerNorm): {first_block.ln_2.weight.shape}")
    print(f"   - mlp (Feed Forward):")
    print(f"     - c_fc: {first_block.mlp.c_fc.weight.shape}")
    print(f"     - c_proj: {first_block.mlp.c_proj.weight.shape}")

    print("\n3. 최종 레이어")
    print(f"   ln_f (Final LayerNorm): {model.transformer.ln_f.weight.shape}")
    print(f"   lm_head (Language Model Head): {model.lm_head.weight.shape}")


def compare_gpt2_sizes():
    """GPT-2 모델 크기 비교"""
    print("\n" + "=" * 60)
    print("[5] GPT-2 모델 크기 비교")
    print("=" * 60)

    models_info = [
        ("gpt2", "GPT-2 Small"),
        ("gpt2-medium", "GPT-2 Medium"),
        ("gpt2-large", "GPT-2 Large"),
    ]

    print("\n[모델별 파라미터 수]")
    print("-" * 60)

    for model_name, display_name in models_info:
        try:
            config = GPT2Config.from_pretrained(model_name)
            # 파라미터 수 추정 (실제 로드 없이)
            vocab_size = config.vocab_size
            n_embd = config.n_embd
            n_layer = config.n_layer
            n_head = config.n_head

            # 대략적인 파라미터 수 계산
            embedding_params = vocab_size * n_embd + config.n_positions * n_embd
            attention_params = 4 * n_embd * n_embd * n_layer  # Q, K, V, O
            ffn_params = 8 * n_embd * n_embd * n_layer  # 4x expansion
            total_approx = embedding_params + attention_params + ffn_params

            print(f"  {display_name:20}")
            print(f"    - 층 수: {n_layer}, 은닉: {n_embd}, 헤드: {n_head}")
            print(f"    - 추정 파라미터: ~{total_approx/1e6:.0f}M")
        except Exception as e:
            print(f"  {display_name:20} : (정보 로드 실패)")

    print("\n[공식 파라미터 수]")
    print("-" * 60)
    print("  GPT-2 Small   :   124M (12층, 768 은닉)")
    print("  GPT-2 Medium  :   355M (24층, 1024 은닉)")
    print("  GPT-2 Large   :   774M (36층, 1280 은닉)")
    print("  GPT-2 XL      : 1,558M (48층, 1600 은닉)")


def main():
    """메인 함수"""
    print("=" * 60)
    print("GPT-2 모델 기본 사용법 실습")
    print("=" * 60)

    # 1. 모델 로드
    tokenizer, model = load_model_demo()

    # 2. 토큰화
    tokenization_demo(tokenizer)

    # 3. 텍스트 생성
    simple_generation_demo(tokenizer, model)

    # 4. 아키텍처 분석
    model_architecture_demo(model)

    # 5. 모델 크기 비교
    compare_gpt2_sizes()

    print("\n" + "=" * 60)
    print("GPT-2 기본 실습 완료")
    print("=" * 60)


if __name__ == "__main__":
    main()
