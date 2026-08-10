import json
import os
import platform
import statistics
from dataclasses import asdict, fields
from pathlib import Path

import numba
import torch
import transformers

from cachequant.bench.config import DEFAULT_CONFIG
from cachequant.bench.harness import BenchResult, run_benchmark
from cachequant.kernel.bfp_linear import apply_bfp_quantization
from cachequant.model import generate, load_model

PROMPT = "The history of artificial intelligence began with"
MAX_NEW_TOKENS = 50

LONG_PROMPT = (
    "The history of computing spans centuries, from early mechanical calculators to "
    "today's massively parallel processors. In the nineteenth century, Charles Babbage "
    "designed the Analytical Engine, a mechanical device intended to perform "
    "general-purpose calculations using punched cards for input. Ada Lovelace, working "
    "alongside Babbage, wrote what many consider the first computer algorithm, "
    "recognizing that such a machine could manipulate symbols beyond mere numbers. "
    "Decades later, in the 1930s and 1940s, researchers such as Alan Turing formalized "
    "the theoretical limits of computation, while engineers built the first electronic "
    "computers using vacuum tubes. The invention of the transistor in 1947 at Bell Labs "
    "triggered a wave of miniaturization that made computers smaller, faster, and more "
    "reliable. By the 1960s, integrated circuits allowed thousands of transistors to be "
    "packed onto a single chip, and the following decades saw exponential growth in "
    "computing power, a trend often described by Moore's Law. Personal computers "
    "emerged in the 1970s and 1980s, bringing computation into homes and small "
    "businesses for the first time. The internet, which grew from research networks in "
    "the 1960s and 1970s into a global infrastructure by the 1990s, transformed "
    "computers from isolated tools into interconnected nodes capable of sharing "
    "information instantly. In recent years, specialized hardware accelerators designed "
    "for parallel workloads have become central to training and running large machine "
    "learning models."
)
LONG_PROMPT_MAX_NEW_TOKENS = 10

WARMUP_MAX_NEW_TOKENS = 5
N_REPS = 5

BENCHMARKS_DIR = Path(__file__).resolve().parent.parent / "benchmarks"
SHORT_OUTPUT_PATH = BENCHMARKS_DIR / "bfp_results.json"
LONG_OUTPUT_PATH = BENCHMARKS_DIR / "bfp_results_longprompt.json"


def _cpu_model_name() -> str:
    """Best-effort human-readable CPU model name.

    platform.processor() is frequently an empty string on Linux (it relies
    on info the OS doesn't always populate), so this falls back to parsing
    /proc/cpuinfo, and finally to platform.machine(), so the provenance
    disclosure always names *something* concrete.
    """
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine() or "unknown"


def _provenance() -> dict:
    return {
        "platform_processor": platform.processor(),
        "cpu_model": _cpu_model_name(),
        "platform_machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
    }


def _summarize(results: list[BenchResult]) -> dict:
    """Median/min/max per BenchResult field across repeated runs."""
    summary = {}
    for f in fields(BenchResult):
        values = [getattr(r, f.name) for r in results]
        summary[f.name] = {
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
        }
    return summary


def run_profile(model, tokenizer, prompt: str, max_new_tokens: int, output_path: Path) -> None:
    results = [
        run_benchmark(model, tokenizer, prompt, DEFAULT_CONFIG, max_new_tokens)
        for _ in range(N_REPS)
    ]

    reference_text, _ = generate(model, tokenizer, prompt, max_new_tokens)

    payload = {
        "prompt": prompt,
        "max_new_tokens": max_new_tokens,
        "n_reps": N_REPS,
        "config": asdict(DEFAULT_CONFIG),
        "provenance": _provenance() | {"numba_threads": numba.get_num_threads()},
        "runs": [asdict(r) for r in results],
        "summary": _summarize(results),
        "reference_generation": reference_text,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


def main() -> None:
    # Thread parity with the fp32 baseline. run_benchmark() pins torch to
    # config.cpu_threads, but the BFP matmul runs under Numba's own thread
    # pool, which defaults to every logical core on the machine. Without this
    # the two paths are measured at different thread counts and the comparison
    # is not apples-to-apples. Numba is BFP-only, so this belongs here rather
    # than in the shared harness.
    numba.set_num_threads(DEFAULT_CONFIG.cpu_threads)

    model, tokenizer = load_model()
    apply_bfp_quantization(model)

    # Discard one full warmup generation before any timed run so Numba's
    # one-time JIT compilation of bfp_matmul (paid on its first call) and
    # any thread-pool spin-up don't contaminate the measured throughput.
    # Mirrors run_baseline.py's C1 warmup, which guards against an
    # analogous BLAS/thread-pool cost.
    generate(model, tokenizer, PROMPT, WARMUP_MAX_NEW_TOKENS)

    run_profile(model, tokenizer, PROMPT, MAX_NEW_TOKENS, SHORT_OUTPUT_PATH)
    run_profile(model, tokenizer, LONG_PROMPT, LONG_PROMPT_MAX_NEW_TOKENS, LONG_OUTPUT_PATH)


if __name__ == "__main__":
    main()
