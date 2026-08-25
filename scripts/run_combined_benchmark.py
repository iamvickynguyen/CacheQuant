import copy
import json
import statistics
import time
from pathlib import Path

import numba
import torch

from cachequant.bench.config import DEFAULT_CONFIG
from cachequant.bench.provenance import provenance
from cachequant.kernel.bfp_linear import apply_bfp_quantization
from cachequant.kernel.int8_linear import apply_int8_quantization
from cachequant.kvcache.trie_cache import PrefixKVCache
from cachequant.model import load_model
from cachequant.pipeline import generate

BENCHMARKS_DIR = Path(__file__).resolve().parent.parent / "benchmarks"
MAX_NEW_TOKENS = 10
WARMUP_PROMPT = "The quick brown fox jumps over the lazy dog."
WARMUP_MAX_NEW_TOKENS = 5
N_REPS = 5

# Duplicated from scripts/run_kvcache_benchmark.py rather than imported -
# matches that script's own precedent of duplicating LONG_PROMPT from
# scripts/run_baseline.py (no cross-script-import precedent in this codebase).
PREAMBLE = "You are a helpful assistant. Answer concisely and factually. User question: "
HIGH_REUSE_SUFFIXES = [
    "What is the capital of France?",
    "What is the boiling point of water in Celsius?",
    "Who wrote the play Hamlet?",
    "What is the largest planet in the solar system?",
    "How many continents are there on Earth?",
]
HIGH_REUSE_PROMPTS = [PREAMBLE + suffix for suffix in HIGH_REUSE_SUFFIXES]

NO_REUSE_PROMPTS = [
    "The stock market fell sharply today after",
    "In a small village nestled between two mountains,",
    "def calculate_average(numbers):",
    "Scientists have discovered a new species of",
    "The recipe calls for two cups of flour and",
]

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
LONG_HIGH_REUSE_PREAMBLE = LONG_PROMPT + " Based on this history, "
LONG_HIGH_REUSE_SUFFIXES = [
    "what year was the transistor invented?",
    "who designed the Analytical Engine?",
    "which decade did personal computers emerge in?",
    "what law describes the growth in computing power?",
    "what 1947 invention triggered a wave of miniaturization?",
]
LONG_HIGH_REUSE_PROMPTS = [
    LONG_HIGH_REUSE_PREAMBLE + suffix for suffix in LONG_HIGH_REUSE_SUFFIXES
]

PROMPT_SETS = {
    "high_reuse": HIGH_REUSE_PROMPTS,
    "long_high_reuse": LONG_HIGH_REUSE_PROMPTS,
    "no_reuse": NO_REUSE_PROMPTS,
}

# Deterministic given a fresh cache and fixed prompt order - asserted equal
# across reps rather than averaged. `cached_tokens`/`hit_rate` are 0/0.0 for
# the no-cache combos (stats is None there; substituted before this check).
_NON_TIMING_FIELDS = ("prompt_tokens", "cached_tokens", "hit_rate")
_TIMING_FIELDS = ("prefill_seconds", "decode_seconds", "tokens_per_sec", "honest_prefill_seconds")


def _run_combo_once(model, tokenizer, prompts: list[str], use_cache: bool) -> list[dict]:
    cache = PrefixKVCache(max_tokens=DEFAULT_CONFIG.max_cache_tokens) if use_cache else None
    rows = []
    for prompt in prompts:
        if use_cache:
            # Time the *entire* call, not just the internal forward-pass
            # timing, so cache.lookup() (before) and cache.insert() (after)
            # - both invisible to generate's own timing - are counted. Same
            # convention as scripts/run_kvcache_benchmark.py.
            t0 = time.perf_counter()
            _, timing, stats = generate(model, tokenizer, prompt, cache=cache, max_new_tokens=MAX_NEW_TOKENS)
            total_call_seconds = time.perf_counter() - t0

            cache_overhead_seconds = total_call_seconds - (timing.prefill_seconds + timing.decode_seconds)
            if cache_overhead_seconds < 0:
                # Every step inside generate is sequential, so
                # total_call_seconds should never be less than the sum of its
                # own internally-timed segments. A negative value here means
                # timing drift/overhead elsewhere, not just "cache
                # bookkeeping was fast" - investigate rather than clamp it
                # silently.
                raise RuntimeError(
                    f"negative cache_overhead_seconds ({cache_overhead_seconds!r}) for "
                    f"prompt {prompt!r}: total_call_seconds={total_call_seconds!r}, "
                    f"prefill_seconds={timing.prefill_seconds!r}, "
                    f"decode_seconds={timing.decode_seconds!r}"
                )
            honest_prefill_seconds = timing.prefill_seconds + cache_overhead_seconds
        else:
            _, timing, stats = generate(model, tokenizer, prompt, cache=cache, max_new_tokens=MAX_NEW_TOKENS)
            honest_prefill_seconds = timing.prefill_seconds

        decode_seconds = timing.decode_seconds
        decode_tokens = len(timing.per_token_seconds)
        rows.append(
            {
                "prompt_tokens": stats.prompt_tokens if stats else timing.prefill_tokens,
                "cached_tokens": stats.cached_tokens if stats else 0,
                "hit_rate": stats.hit_rate if stats else 0.0,
                "prefill_seconds": timing.prefill_seconds,
                "decode_seconds": decode_seconds,
                "tokens_per_sec": decode_tokens / decode_seconds if decode_seconds > 0 else 0.0,
                "honest_prefill_seconds": honest_prefill_seconds,
            }
        )
    return rows


def _run_combo(model, tokenizer, prompts: list[str], use_cache: bool, label: str) -> dict:
    passes = [_run_combo_once(model, tokenizer, prompts, use_cache) for _ in range(N_REPS)]

    reference = passes[0]
    for pass_idx in range(1, N_REPS):
        for row_idx, (ref_row, row) in enumerate(zip(reference, passes[pass_idx])):
            for field in _NON_TIMING_FIELDS:
                assert ref_row[field] == row[field], (
                    f"{label!r} row {row_idx} field {field!r} differed between pass 0 "
                    f"({ref_row[field]!r}) and pass {pass_idx} ({row[field]!r})"
                )

    rows = []
    for row_idx in range(len(prompts)):
        row = {field: reference[row_idx][field] for field in _NON_TIMING_FIELDS}
        for field in _TIMING_FIELDS:
            values = [p[row_idx][field] for p in passes]
            row[field] = statistics.median(values)
        rows.append(row)

    avg_hit_rate = sum(r["hit_rate"] for r in rows) / len(rows)
    sum_prefill = sum(r["prefill_seconds"] for r in rows)
    total_honest_prefill_seconds = sum(r["honest_prefill_seconds"] for r in rows)

    return {
        "label": label,
        "n_reps": N_REPS,
        "rows": rows,
        "avg_hit_rate": avg_hit_rate,
        "total_prefill_seconds": sum_prefill,
        "total_honest_prefill_seconds": total_honest_prefill_seconds,
    }


def main() -> None:
    torch.set_num_threads(DEFAULT_CONFIG.cpu_threads)
    # Both quantized paths run their matmul under Numba's own thread pool,
    # which otherwise defaults to every logical core while torch stays pinned
    # to cpu_threads — making the quantized-vs-fp32 rows in this table not
    # apples-to-apples. Same convention as run_bfp_benchmark.py's main().
    numba.set_num_threads(DEFAULT_CONFIG.cpu_threads)

    fp32_model, tokenizer = load_model()
    bfp_model = apply_bfp_quantization(copy.deepcopy(fp32_model))
    int8_model = apply_int8_quantization(copy.deepcopy(fp32_model))

    # One warmup generation per model: fp32 for BLAS/thread-pool spin-up, and
    # each quantized model for its own Numba JIT compilation.
    for model in (fp32_model, bfp_model, int8_model):
        generate(model, tokenizer, WARMUP_PROMPT, cache=None, max_new_tokens=WARMUP_MAX_NEW_TOKENS)

    results: dict = {"provenance": provenance()}
    for set_name, prompts in PROMPT_SETS.items():
        set_results = {}
        for model_label, model in (
            ("fp32", fp32_model),
            ("bfp", bfp_model),
            ("int8", int8_model),
        ):
            for cache_label, use_cache in (("no_cache", False), ("cache", True)):
                combo_label = f"{model_label}_{cache_label}"
                print(f"{set_name} / {combo_label}")
                combo = _run_combo(model, tokenizer, prompts, use_cache, combo_label)
                set_results[combo_label] = combo
        results[set_name] = set_results

    BENCHMARKS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = BENCHMARKS_DIR / "combined_results.json"
    output_path.write_text(json.dumps(results, indent=2))
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
