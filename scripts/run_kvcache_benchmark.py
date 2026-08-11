import json
import statistics
import time
from pathlib import Path

import torch

from cachequant.bench.config import DEFAULT_CONFIG
from cachequant.bench.provenance import provenance
from cachequant.kvcache.generate import generate_with_prefix_cache
from cachequant.kvcache.trie_cache import PrefixKVCache
from cachequant.model import generate, load_model

BENCHMARKS_DIR = Path(__file__).resolve().parent.parent / "benchmarks"
MAX_NEW_TOKENS = 10
WARMUP_PROMPT = "The quick brown fox jumps over the lazy dog."
WARMUP_MAX_NEW_TOKENS = 5
N_REPS = 5

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

# Matches scripts/run_baseline.py's LONG_PROMPT exactly - copied here rather
# than imported across scripts/*.py (no existing precedent for that in this
# codebase, and it would add fragile coupling between sibling benchmarks).
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
# A shared ~280-token preamble plus varying short suffixes, same construction
# as HIGH_REUSE_PROMPTS - this is the long-prompt profile the design spec
# called for (a cache hit here has a real ceiling to show against total wall
# time, unlike the short high_reuse set where fixed per-call overhead
# dominates).
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

# Fields that are deterministic given a fresh cache and fixed prompt order -
# identical across all N_REPS passes. Asserted equal rather than averaged.
_NON_TIMING_FIELDS = (
    "prompt",
    "prompt_tokens",
    "cached_tokens",
    "hit_rate",
    "recomputed_prefill_tokens",
)
# Fields that vary run to run due to measurement noise - medianed across passes.
_TIMING_FIELDS = (
    "baseline_prefill_seconds",
    "cached_prefill_seconds",
    "prefill_speedup",
    "cache_overhead_seconds",
    "honest_prefill_seconds",
    "honest_prefill_speedup",
)


def _run_prompt_set_once(model, tokenizer, prompts: list[str]) -> list[dict]:
    """One pass over a prompt set against a fresh cache. Returns one dict per prompt."""
    cache = PrefixKVCache(max_tokens=DEFAULT_CONFIG.max_cache_tokens)
    rows = []
    for prompt in prompts:
        _, baseline_timing = generate(model, tokenizer, prompt, MAX_NEW_TOKENS)

        # Time the *entire* call, not just the internal forward-pass timing,
        # so cache.lookup() (before) and cache.insert() (after) - both
        # invisible to generate_with_prefix_cache's own timing - are counted.
        t0 = time.perf_counter()
        _, cached_timing, stats = generate_with_prefix_cache(
            model, tokenizer, prompt, cache, MAX_NEW_TOKENS
        )
        total_call_seconds = time.perf_counter() - t0

        cache_overhead_seconds = total_call_seconds - (
            cached_timing.prefill_seconds + cached_timing.decode_seconds
        )
        if cache_overhead_seconds < 0:
            # Every step inside generate_with_prefix_cache is sequential, so
            # total_call_seconds should never be less than the sum of its own
            # internally-timed segments. A negative value here means timing
            # drift/overhead elsewhere, not just "cache bookkeeping was fast" -
            # investigate rather than clamp it silently.
            raise RuntimeError(
                f"negative cache_overhead_seconds ({cache_overhead_seconds!r}) for "
                f"prompt {prompt!r}: total_call_seconds={total_call_seconds!r}, "
                f"prefill_seconds={cached_timing.prefill_seconds!r}, "
                f"decode_seconds={cached_timing.decode_seconds!r}"
            )
        honest_prefill_seconds = cached_timing.prefill_seconds + cache_overhead_seconds

        prefill_speedup = (
            baseline_timing.prefill_seconds / cached_timing.prefill_seconds
            if cached_timing.prefill_seconds > 0
            else float("inf")
        )
        honest_prefill_speedup = (
            baseline_timing.prefill_seconds / honest_prefill_seconds
            if honest_prefill_seconds > 0
            else float("inf")
        )

        rows.append(
            {
                "prompt": prompt,
                "prompt_tokens": stats.prompt_tokens,
                "cached_tokens": stats.cached_tokens,
                "hit_rate": stats.hit_rate,
                "recomputed_prefill_tokens": stats.recomputed_prefill_tokens,
                "baseline_prefill_seconds": baseline_timing.prefill_seconds,
                # cached_prefill_seconds / prefill_speedup only cover the
                # forward pass inside generate_with_prefix_cache, same as
                # before this fix - kept alongside the honest_* fields below
                # so the JSON documents both what is and isn't included.
                "cached_prefill_seconds": cached_timing.prefill_seconds,
                "prefill_speedup": prefill_speedup,
                # cache_overhead_seconds is the cache.lookup() + cache.insert()
                # (plus incidental tokenize/detokenize) cost that
                # cached_prefill_seconds does not count.
                "cache_overhead_seconds": cache_overhead_seconds,
                "honest_prefill_seconds": honest_prefill_seconds,
                "honest_prefill_speedup": honest_prefill_speedup,
            }
        )
    return rows


def _run_prompt_set(model, tokenizer, prompts: list[str], label: str) -> dict:
    """Runs the full prompt-set sequence N_REPS times, each against a fresh
    PrefixKVCache, and medians the timing fields at each row position."""
    passes = [_run_prompt_set_once(model, tokenizer, prompts) for _ in range(N_REPS)]

    reference = passes[0]
    for pass_idx in range(1, N_REPS):
        for row_idx, (ref_row, row) in enumerate(zip(reference, passes[pass_idx])):
            for field in _NON_TIMING_FIELDS:
                assert ref_row[field] == row[field], (
                    f"{label!r} row {row_idx} field {field!r} differed between pass 0 "
                    f"({ref_row[field]!r}) and pass {pass_idx} ({row[field]!r}) - expected "
                    "deterministic given a fresh cache and fixed prompt order per pass."
                )

    rows = []
    for row_idx in range(len(prompts)):
        row = {field: reference[row_idx][field] for field in _NON_TIMING_FIELDS}
        for field in _TIMING_FIELDS:
            values = [p[row_idx][field] for p in passes]
            row[field] = statistics.median(values)
        rows.append(row)

    avg_hit_rate = sum(r["hit_rate"] for r in rows) / len(rows)

    # Ratio-of-sums, not mean-of-per-row-ratios: far less sensitive to one
    # noisy row than averaging per-row speedups would be.
    sum_baseline = sum(r["baseline_prefill_seconds"] for r in rows)
    sum_cached = sum(r["cached_prefill_seconds"] for r in rows)
    sum_honest = sum(r["honest_prefill_seconds"] for r in rows)
    avg_prefill_speedup = sum_baseline / sum_cached if sum_cached > 0 else float("inf")
    avg_honest_prefill_speedup = sum_baseline / sum_honest if sum_honest > 0 else float("inf")

    payload = {
        "label": label,
        "n_reps": N_REPS,
        "provenance": provenance(),
        "rows": rows,
        "avg_hit_rate": avg_hit_rate,
        # avg_prefill_speedup counts only the internally-timed forward pass
        # (excludes cache.lookup()/cache.insert() overhead); see
        # avg_honest_prefill_speedup below for the end-to-end figure.
        "avg_prefill_speedup": avg_prefill_speedup,
        "avg_honest_prefill_speedup": avg_honest_prefill_speedup,
    }
    print(label)
    print(json.dumps(payload, indent=2))
    print()
    return payload


def main() -> None:
    torch.set_num_threads(DEFAULT_CONFIG.cpu_threads)
    model, tokenizer = load_model()

    # C1 (mirrors scripts/run_baseline.py): discard one full warmup generation
    # before any timed run so the first forward pass's one-time thread-pool
    # spin-up / lazy kernel selection cost doesn't contaminate the first
    # measured row's prefill timing. One warmup suffices for every subsequent
    # pass/prompt-set below, since that cost is paid only once per process.
    generate(model, tokenizer, WARMUP_PROMPT, WARMUP_MAX_NEW_TOKENS)

    high_reuse = _run_prompt_set(model, tokenizer, HIGH_REUSE_PROMPTS, "high_reuse")
    long_high_reuse = _run_prompt_set(
        model, tokenizer, LONG_HIGH_REUSE_PROMPTS, "long_high_reuse"
    )
    no_reuse = _run_prompt_set(model, tokenizer, NO_REUSE_PROMPTS, "no_reuse")

    BENCHMARKS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = BENCHMARKS_DIR / "kvcache_results.json"
    output_path.write_text(
        json.dumps(
            {
                "high_reuse": high_reuse,
                "long_high_reuse": long_high_reuse,
                "no_reuse": no_reuse,
            },
            indent=2,
        )
    )
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
