"""
03-메모리비교.py
제10장 실습 ③ — 파인튜닝 메모리 견적 (GPU 불필요)

이 스크립트는 모델 크기별로 Full FT / LoRA / QLoRA가 각각
얼마의 GPU 메모리를 차지하는지 *이론값*으로 계산하고
막대 차트로 시각화한다.

출력:
  콘솔: 표 형태로 메모리 견적 출력
  파일: ../data/output/memory_comparison.png (로그 스케일 막대 차트)

의존성:
    pip install numpy matplotlib

실행:
    python code/03-메모리비교.py

연계:
    01-양자화수학.py    — 양자화 수학의 직관
    02-qlora파인튜닝.py — 실제 한국어 모델에 QLoRA 적용
"""

import platform
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # GUI 백엔드 없이 PNG로 저장
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def configure_korean_font():
    """플랫폼별 한글 폰트 설정 (한글이 깨지지 않도록)."""
    system = platform.system()
    if system == "Windows":
        plt.rcParams["font.family"] = "Malgun Gothic"
    elif system == "Darwin":
        plt.rcParams["font.family"] = "AppleGothic"
    else:
        plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False


def compute_memory_table():
    """모델별·방법별 메모리(GB) 견적을 dict로 반환한다."""
    models = {
        "polyglot-ko\n1.3B":  1.3e9,
        "Llama-7B":           7e9,
        "Llama-70B":          70e9,
    }

    # LoRA 학습 파라미터: rank × (in + out) × 층 수
    lora_r = 16
    lora_params = {
        "polyglot-ko\n1.3B":  lora_r * (2048 + 6144) * 24,
        "Llama-7B":           lora_r * (4096 + 4096) * 2 * 32,
        "Llama-70B":          lora_r * (8192 + 8192) * 2 * 80,
    }

    methods = ["Full FT (fp32)", "LoRA (fp16)", "QLoRA (4-bit)"]
    data = {m: [] for m in methods}

    for mname, params in models.items():
        lp = lora_params[mname]
        # Full FT: 모델(4B) + 옵티마이저(4B × 2) + 그래디언트(4B) = 4 × 4
        data["Full FT (fp32)"].append(params * 4 * 4 / (1024 ** 3))
        # LoRA: 모델(2B fp16) + 옵티마이저·그래디언트는 LoRA 파라미터만
        data["LoRA (fp16)"].append(
            (params * 2 + lp * 4 * 3) / (1024 ** 3)
        )
        # QLoRA: 모델(0.5B 4-bit) + 옵티마이저·그래디언트는 LoRA만
        data["QLoRA (4-bit)"].append(
            (params * 0.5 + lp * 4 * 3) / (1024 ** 3)
        )

    return list(models.keys()), methods, data


def print_table(model_names, methods, data):
    """콘솔에 견적 표를 출력한다."""
    print("=" * 60)
    print("  파인튜닝 메모리 견적 (이론값)")
    print("=" * 60)

    display_names = [n.replace("\n", " ") for n in model_names]
    print(f"\n  {'방법':<18}", end="")
    for mn in display_names:
        print(f" {mn:>14}", end="")
    print()
    print("  " + "-" * 62)

    for method in methods:
        print(f"  {method:<18}", end="")
        for val in data[method]:
            print(f" {val:>12.1f}GB", end="")
        print()

    print("  " + "-" * 62)
    print("  참고 GPU VRAM:")
    print("    RTX 4090: 24 GB  |  RTX 4060: 8 GB  |  Apple M1: 8-16 GB")
    print()


def draw_chart(model_names, methods, data, output_path):
    """로그 스케일 막대 차트를 PNG로 저장한다."""
    fig, ax = plt.subplots(figsize=(10, 6))
    x = range(len(model_names))
    width = 0.25
    colors = ["#e74c3c", "#3498db", "#2ecc71"]

    for idx, method in enumerate(methods):
        offset = (idx - 1) * width
        bars = ax.bar(
            [xi + offset for xi in x],
            data[method],
            width=width,
            label=method,
            color=colors[idx],
            alpha=0.85,
        )
        for bar, val in zip(bars, data[method]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.05,
                f"{val:.0f}GB",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    # GPU VRAM 기준선
    ax.axhline(y=24, color="gray", linestyle="--", linewidth=1.5,
               label="RTX 4090 (24GB)")
    ax.axhline(y=8, color="gray", linestyle=":", linewidth=1.5,
               label="RTX 4060 / M1 (8GB)")

    ax.set_xticks(list(x))
    ax.set_xticklabels(model_names)
    ax.set_ylabel("Memory (GB)")
    ax.set_title("Fine-tuning Memory: Full FT vs LoRA vs QLoRA")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_yscale("log")  # 1.3B ~ 70B의 차이가 크므로 로그 스케일
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    print()
    print("=" * 60)
    print("  제10장 실습 ③ — 파인튜닝 메모리 견적")
    print("  딥러닝 자연어처리 (2026)")
    print("=" * 60)
    print()

    configure_korean_font()
    model_names, methods, data = compute_memory_table()
    print_table(model_names, methods, data)

    chart_path = OUTPUT_DIR / "memory_comparison.png"
    draw_chart(model_names, methods, data, chart_path)
    print(f"  차트 저장: {chart_path}")
    print()

    print("=" * 60)
    print("  완료. 다른 실습:")
    print("    python code/01-양자화수학.py")
    print("    python code/02-qlora파인튜닝.py")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
