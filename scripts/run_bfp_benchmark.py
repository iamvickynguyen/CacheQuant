import json
from dataclasses import asdict
from pathlib import Path

import torch

from cachequant.bench.config import DEFAULT_CONFIG
from cachequant.bench.harness import run_benchmark
from cachequant.kernel.bfp_linear import apply_bfp_quantization
from cachequant.model import generate, load_model

BENCHMARKS_DIR = Path(__file__).resolve().parent.parent / "benchmarks"

SHORT_PROMPT = "The history of artificial intelligence began with"
SHORT_MAX_NEW_TOKENS = 50

WARMUP_MAX_NEW_TOKENS = 5

LONG_PROMPT = (
    "Charles Babbage designed the Analytical Engine in the 1830s, a mechanical "
    "general-purpose computer that was never built in his lifetime. Ada Lovelace, "
    "who worked with Babbage, wrote what is considered the first algorithm "
    "intended for machine execution, along with notes on how the engine could go "
    "beyond pure calculation. It would be over a century before electronic "
    "computers realized ideas Babbage and Lovelace had described on paper."
) * 3
LONG_MAX_NEW_TOKENS = 10


def _run_profile(model, tokenizer, prompt: str, max_new_tokens: int, output_path: Path) -> None:
    result = run_benchmark(model, tokenizer, prompt, DEFAULT_CONFIG, max_new_tokens)
    payload = {
        "prompt_tokens_approx": len(prompt.split()),
        "max_new_tokens": max_new_tokens,
        "config": asdict(DEFAULT_CONFIG),
        "result": asdict(result),
    }
    output_path.write_text(json.dumps(payload, indent=2))
    print(output_path.name)
    print(json.dumps(payload, indent=2))
    print()


def main() -> None:
    torch.set_num_threads(DEFAULT_CONFIG.cpu_threads)
    model, tokenizer = load_model()
    apply_bfp_quantization(model)

    # Discard one full warmup generation before any timed run so Numba's
    # one-time JIT compilation of bfp_matmul (paid on its first call, which
    # would otherwise land inside the short profile's timed prefill pass)
    # doesn't contaminate the measured throughput. Mirrors run_baseline.py's
    # C1 warmup, which guards against an analogous BLAS/thread-pool cost.
    generate(model, tokenizer, SHORT_PROMPT, WARMUP_MAX_NEW_TOKENS)

    BENCHMARKS_DIR.mkdir(parents=True, exist_ok=True)
    _run_profile(
        model, tokenizer, SHORT_PROMPT, SHORT_MAX_NEW_TOKENS, BENCHMARKS_DIR / "bfp_results.json"
    )
    _run_profile(
        model,
        tokenizer,
        LONG_PROMPT,
        LONG_MAX_NEW_TOKENS,
        BENCHMARKS_DIR / "bfp_results_longprompt.json",
    )


if __name__ == "__main__":
    main()
