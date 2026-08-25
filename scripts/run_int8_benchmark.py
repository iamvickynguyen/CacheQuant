"""Plain-int8 counterpart of run_bfp_benchmark.py.

Same prompts, same harness, same rep count and same thread pinning, so the two
quantization schemes are directly comparable against each other and against the
recorded fp32 baseline. Only the quantizer differs.
"""

import json
import statistics
from dataclasses import asdict, fields
from pathlib import Path

import numba

from cachequant.bench.config import DEFAULT_CONFIG
from cachequant.bench.harness import BenchResult, run_benchmark
from cachequant.bench.provenance import provenance
from cachequant.kernel.int8_linear import apply_int8_quantization
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
SHORT_OUTPUT_PATH = BENCHMARKS_DIR / "int8_results.json"
LONG_OUTPUT_PATH = BENCHMARKS_DIR / "int8_results_longprompt.json"


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
        "provenance": provenance() | {"numba_threads": numba.get_num_threads()},
        "runs": [asdict(r) for r in results],
        "summary": _summarize(results),
        "reference_generation": reference_text,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


def main() -> None:
    # Thread parity with the fp32 baseline. run_benchmark() pins torch to
    # config.cpu_threads, but the int8 matmul runs under Numba's own thread
    # pool, which defaults to every logical core on the machine. Without this
    # the two paths are measured at different thread counts and the comparison
    # is not apples-to-apples. Same convention as run_bfp_benchmark.py.
    numba.set_num_threads(DEFAULT_CONFIG.cpu_threads)

    model, tokenizer = load_model()
    apply_int8_quantization(model)

    # Discard one full warmup generation before any timed run so Numba's
    # one-time JIT compilation of the shared blocked-int8 kernel (paid on its
    # first call) and any thread-pool spin-up don't contaminate the measured
    # throughput. Mirrors run_baseline.py's C1 warmup.
    generate(model, tokenizer, PROMPT, WARMUP_MAX_NEW_TOKENS)

    run_profile(model, tokenizer, PROMPT, MAX_NEW_TOKENS, SHORT_OUTPUT_PATH)
    run_profile(model, tokenizer, LONG_PROMPT, LONG_PROMPT_MAX_NEW_TOKENS, LONG_OUTPUT_PATH)


if __name__ == "__main__":
    main()
