"""fp16 reference point for the fp32 baseline and the BFP kernel.

fp16 is not one of this project's optimizations — no kernel is written for it.
It is a second *baseline*: the free, one-line alternative (`model.half()`) that
any real deployment would reach for before writing a custom quantized kernel.
Measuring it separates the two effects that are confounded in the BFP decode
result — fewer bytes moved per weight vs. integer arithmetic this CPU can
execute natively — and answers "why not just use fp16?" with a number instead
of an argument.
"""

import json
import statistics
from dataclasses import asdict, fields
from pathlib import Path

from cachequant.bench.config import DEFAULT_CONFIG
from cachequant.bench.harness import BenchResult, run_benchmark
from cachequant.bench.provenance import provenance
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
SHORT_OUTPUT_PATH = BENCHMARKS_DIR / "fp16_results.json"
LONG_OUTPUT_PATH = BENCHMARKS_DIR / "fp16_results_longprompt.json"


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
        "dtype": "float16",
        "config": asdict(DEFAULT_CONFIG),
        "provenance": provenance(),
        "runs": [asdict(r) for r in results],
        "summary": _summarize(results),
        "reference_generation": reference_text,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


def main() -> None:
    model, tokenizer = load_model()
    # The entire comparison. No custom kernel, no layer swapping — torch casts
    # every weight to fp16 and dispatches to whatever fp16 CPU path this build
    # has. Unlike BFP there is no Numba thread pool to pin: run_benchmark()
    # already pins torch to config.cpu_threads, which is the only pool in play.
    model.half()

    # Mirrors run_baseline.py's C1 warmup: discard one full generation so
    # first-call thread-pool spin-up and lazy kernel selection don't
    # contaminate the measured prefill throughput.
    generate(model, tokenizer, PROMPT, WARMUP_MAX_NEW_TOKENS)

    run_profile(model, tokenizer, PROMPT, MAX_NEW_TOKENS, SHORT_OUTPUT_PATH)
    run_profile(model, tokenizer, LONG_PROMPT, LONG_PROMPT_MAX_NEW_TOKENS, LONG_OUTPUT_PATH)


if __name__ == "__main__":
    main()
