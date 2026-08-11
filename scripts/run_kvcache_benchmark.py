import json
from pathlib import Path

import torch

from cachequant.bench.config import DEFAULT_CONFIG
from cachequant.kvcache.generate import generate_with_prefix_cache
from cachequant.kvcache.trie_cache import PrefixKVCache
from cachequant.model import generate, load_model

BENCHMARKS_DIR = Path(__file__).resolve().parent.parent / "benchmarks"
MAX_NEW_TOKENS = 10
WARMUP_PROMPT = "The quick brown fox jumps over the lazy dog."
WARMUP_MAX_NEW_TOKENS = 5

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


def _run_prompt_set(model, tokenizer, prompts: list[str], label: str) -> dict:
    cache = PrefixKVCache(max_tokens=DEFAULT_CONFIG.max_cache_tokens)
    rows = []
    for prompt in prompts:
        _, baseline_timing = generate(model, tokenizer, prompt, MAX_NEW_TOKENS)
        _, cached_timing, stats = generate_with_prefix_cache(
            model, tokenizer, prompt, cache, MAX_NEW_TOKENS
        )
        speedup = (
            baseline_timing.prefill_seconds / cached_timing.prefill_seconds
            if cached_timing.prefill_seconds > 0
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
                "cached_prefill_seconds": cached_timing.prefill_seconds,
                "prefill_speedup": speedup,
            }
        )

    avg_hit_rate = sum(r["hit_rate"] for r in rows) / len(rows)
    finite_speedups = [r["prefill_speedup"] for r in rows if r["prefill_speedup"] != float("inf")]
    avg_speedup = sum(finite_speedups) / len(finite_speedups) if finite_speedups else float("inf")

    payload = {
        "label": label,
        "rows": rows,
        "avg_hit_rate": avg_hit_rate,
        "avg_prefill_speedup": avg_speedup,
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
    # measured row's prefill timing.
    generate(model, tokenizer, WARMUP_PROMPT, WARMUP_MAX_NEW_TOKENS)

    high_reuse = _run_prompt_set(model, tokenizer, HIGH_REUSE_PROMPTS, "high_reuse")
    no_reuse = _run_prompt_set(model, tokenizer, NO_REUSE_PROMPTS, "no_reuse")

    BENCHMARKS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = BENCHMARKS_DIR / "kvcache_results.json"
    output_path.write_text(json.dumps({"high_reuse": high_reuse, "no_reuse": no_reuse}, indent=2))
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
