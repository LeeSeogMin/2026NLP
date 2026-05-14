"""
00-모델질문해보기.py
제10장 실습 ⓪ — 파인튜닝 *전* 베이스 모델 체험 (인터랙티브)

베이스 polyglot-ko-1.3b만 메모리에 올리고, 학생이 콘솔에 직접
한국어 질문을 입력해 응답을 받아본다. LoRA·QLoRA 어댑터는 붙이지
않는다 — 파인튜닝 *없이* 베이스 모델이 질문-답변 형식을 모른다는
사실을 학생이 *자기 손으로* 확인하는 단계.

흐름:
    질문 입력 → KoAlpaca 포맷으로 감싸기 → 토크나이즈 →
    model.generate() → 디코딩 → 응답 출력 → 다음 입력 …
    `exit` 또는 `quit` 입력 시 종료.

대상 환경:
  - Windows + NVIDIA GPU  → 4-bit NF4 양자화 로딩
  - macOS + Apple Silicon  → float16 로딩
  - CPU only              → float32 로딩 (느림)

선행 조건:
    먼저 `python code/02-qlora파인튜닝.py`를 한 번 실행해 자동
    환경 구성(PyTorch+CUDA, transformers, bitsandbytes 등)을
    마쳐 두는 것을 권장한다. 이 파일은 의존성을 직접 설치하지
    않고, 누락 시 안내 메시지만 표시한다.

실행:
    cd practice/chapter10
    source venv/bin/activate    # 또는 Windows: venv\\Scripts\\activate
    python code/00-모델질문해보기.py
"""

import sys
import warnings

warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────────────
# 의존성 점검 (자동 설치는 안 함 — 02에 위임)
# ──────────────────────────────────────────────────────
def check_dependencies():
    missing = []
    try:
        import torch  # noqa: F401
    except ImportError:
        missing.append("torch")
    try:
        import transformers  # noqa: F401
    except ImportError:
        missing.append("transformers")
    if missing:
        print("=" * 60)
        print("  필수 라이브러리가 없습니다:")
        for pkg in missing:
            print(f"    - {pkg}")
        print()
        print("  먼저 한 번 실행해 환경을 구성하세요:")
        print("    python code/02-qlora파인튜닝.py")
        print("=" * 60)
        sys.exit(1)


check_dependencies()


import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.manual_seed(42)

# ── 설정 ───────────────────────────────────────────────
MODEL_NAME = "EleutherAI/polyglot-ko-1.3b"
MAX_NEW_TOKENS = 100
TEMPERATURE = 0.7
TOP_P = 0.9


def detect_device():
    """NVIDIA GPU / Apple MPS / CPU 자동 감지."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_base_model(device_type):
    """베이스 polyglot-ko-1.3b를 디바이스에 맞춰 로딩한다 (LoRA 없음)."""
    print(f"  모델 로딩 중... ({MODEL_NAME})")
    print(f"  디바이스: {device_type.upper()}")

    if device_type == "cuda":
        # NVIDIA: bitsandbytes로 4-bit 양자화 로딩
        try:
            from transformers import BitsAndBytesConfig

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                quantization_config=bnb_config,
                device_map="auto",
            )
            print("  ✓ 4-bit NF4 양자화로 로딩 (메모리 ~0.8GB)")
        except ImportError:
            # bitsandbytes 없으면 fp16으로 떨어짐
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME, torch_dtype=torch.float16
            ).to("cuda")
            print("  ✓ float16으로 로딩 (bitsandbytes 미설치)")
    elif device_type == "mps":
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, torch_dtype=torch.float16
        ).to("mps")
        print("  ✓ float16으로 로딩 (Apple MPS)")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, torch_dtype=torch.float32
        )
        print("  ✓ float32로 로딩 (CPU, 응답이 느릴 수 있음)")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    # 추론 모드 전환 (Dropout/BatchNorm 비활성화) — train(False)는 .eval()과 동일
    model.train(False)
    return model, tokenizer


def format_prompt(user_question):
    """KoAlpaca와 동일한 포맷으로 질문을 감싼다."""
    return f"### 질문: {user_question}\n\n### 답변: "


def generate_response(model, tokenizer, user_question, device_type):
    """질문을 모델에 던지고 응답 텍스트만 반환한다."""
    prompt = format_prompt(user_question)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    # 입력 프롬프트 부분은 잘라내고 *생성된 부분*만 반환
    generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def print_intro():
    print()
    print("=" * 60)
    print("  제10장 실습 ⓪ — 베이스 모델에 직접 질문해보기")
    print("  딥러닝 자연어처리 (2026)")
    print("=" * 60)
    print()
    print("  목적:")
    print("    파인튜닝 *전* 베이스 polyglot-ko-1.3b가")
    print("    '질문-답변' 형식을 모른다는 사실을 직접 확인한다.")
    print()
    print("  사용법:")
    print("    질문을 입력하면 모델이 응답합니다.")
    print("    종료하려면 'exit' 또는 'quit'을 입력하세요.")
    print()
    print("  예시 질문:")
    print("    - 인공지능이 우리 생활에 미치는 영향을 설명해주세요.")
    print("    - 건강한 식습관을 유지하는 팁 3가지를 알려주세요.")
    print("    - 파이썬 프로그래밍의 장점은 무엇인가요?")
    print()
    print("-" * 60)
    print()


def interactive_loop(model, tokenizer, device_type):
    """학생 입력을 받아 응답하는 무한 루프."""
    turn = 1
    while True:
        try:
            question = input(f"[질문 {turn}] ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  종료합니다.")
            break

        if not question:
            continue
        if question.lower() in ("exit", "quit", "종료"):
            print("  종료합니다.")
            break

        print(f"  ... 응답 생성 중 (max_new_tokens={MAX_NEW_TOKENS})")
        response = generate_response(model, tokenizer, question, device_type)
        print()
        print(f"[응답 {turn}]")
        print(response)
        print()
        print("-" * 60)
        turn += 1


def main():
    print_intro()

    device_type = detect_device()
    model, tokenizer = load_base_model(device_type)

    print()
    print("  모델 로딩 완료. 이제 질문을 입력하세요.")
    print()
    print("-" * 60)
    print()

    interactive_loop(model, tokenizer, device_type)

    print()
    print("=" * 60)
    print("  이 베이스 모델이 어떻게 답했는지 기억해두세요.")
    print("  이제 파인튜닝을 진행합니다:")
    print("    python code/02-qlora파인튜닝.py")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
