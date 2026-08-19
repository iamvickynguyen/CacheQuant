import copy
import json
import time
from pathlib import Path

import numba
import torch

from cachequant.bench.config import DEFAULT_CONFIG
from cachequant.bench.harness import compute_bench_result
from cachequant.bench.provenance import provenance
from cachequant.dashboard.app import _cost_per_1k, _prefill_tokens_per_sec
from cachequant.kernel.bfp_linear import apply_bfp_quantization
from cachequant.kvcache.trie_cache import PrefixKVCache
from cachequant.model import load_model
from cachequant.pipeline import generate

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "cachequant" / "dashboard" / "fixtures"
MAX_NEW_TOKENS = 50
WARMUP_PROMPT = "The quick brown fox jumps over the lazy dog."
WARMUP_MAX_NEW_TOKENS = 5

# One representative prompt per prompt set - duplicated from
# cachequant/dashboard/app.py's PROMPT_SETS rather than imported, matching
# that file's own precedent of duplicating prompt data across this codebase.
PREAMBLE = "You are a helpful assistant. Answer concisely and factually. User question: "
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
PROMPT_SETS = {
    "high_reuse": PREAMBLE + "What is the capital of France?",
    "long_high_reuse": LONG_PROMPT + " Based on this history, what year was the transistor invented?",
    "no_reuse": "The stock market fell sharply today after",
}


def _generated_only(text: str, prompt: str, tokenizer) -> str:
    decoded_prompt = tokenizer.decode(tokenizer.encode(prompt), skip_special_tokens=True)
    return text[len(decoded_prompt):] if text.startswith(decoded_prompt) else text


def main() -> None:
    # Pin both thread pools before any timed run. torch and numba have
    # independent thread pools; without this BFP's numba matmul runs on every
    # logical core while fp32's torch path stays at cpu_threads, making the
    # captured BFP-vs-fp32 numbers not apples-to-apples. Same convention as
    # scripts/run_bfp_benchmark.py's main() and the dashboard's model loader.
    torch.set_num_threads(DEFAULT_CONFIG.cpu_threads)
    numba.set_num_threads(DEFAULT_CONFIG.cpu_threads)

    fp32_model, tokenizer = load_model()
    bfp_model = apply_bfp_quantization(copy.deepcopy(fp32_model))
    generate(fp32_model, tokenizer, WARMUP_PROMPT, cache=None, max_new_tokens=WARMUP_MAX_NEW_TOKENS)
    generate(bfp_model, tokenizer, WARMUP_PROMPT, cache=None, max_new_tokens=WARMUP_MAX_NEW_TOKENS)

    combos = [
        ("fp32_no_cache", fp32_model, False),
        ("fp32_cache", fp32_model, True),
        ("bfp_no_cache", bfp_model, False),
        ("bfp_cache", bfp_model, True),
    ]

    entries = []
    for prompt_set_name, prompt in PROMPT_SETS.items():
        for combo_key, model, use_cache in combos:
            cache = PrefixKVCache(max_tokens=DEFAULT_CONFIG.max_cache_tokens) if use_cache else None
            if use_cache:
                # Warm the cache with a discarded first call against the SAME
                # cache instance, then capture the second. A single call always
                # reports a 0% hit rate (nothing was inserted before it), which
                # would show the project's headline KV-cache result as pure
                # overhead in offline replay. The captured second call is what a
                # real second click in the live dashboard does.
                generate(model, tokenizer, prompt, cache=cache, max_new_tokens=MAX_NEW_TOKENS)
            t0 = time.perf_counter()
            text, timing, stats = generate(model, tokenizer, prompt, cache=cache, max_new_tokens=MAX_NEW_TOKENS)
            wall_seconds = time.perf_counter() - t0

            bench = compute_bench_result(timing, DEFAULT_CONFIG)
            prefill_tok_s = _prefill_tokens_per_sec(timing, stats, wall_seconds)
            cost_per_1k = _cost_per_1k(wall_seconds, timing.total_generated_tokens)
            if stats is None:
                cache_detail = "—"
            else:
                cache_detail = f"{stats.cached_tokens}/{stats.prompt_tokens} reused ({stats.hit_rate:.0%})"

            entries.append({
                "prompt_set": prompt_set_name,
                "combo": combo_key,
                "prompt": prompt,
                "generated_only": _generated_only(text, prompt, tokenizer),
                "prefill_tok_s": prefill_tok_s,
                "decode_tok_s": bench.decode_tokens_per_sec,
                "wall_seconds": wall_seconds,
                "cost_per_1k": cost_per_1k,
                "cache_detail": cache_detail,
                "gen_tokens": timing.total_generated_tokens,
            })
            print(f"{prompt_set_name:16s} {combo_key:14s} captured ({wall_seconds:.3f}s)")

    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    output_path = FIXTURES_DIR / "replay.json"
    output_path.write_text(json.dumps({"provenance": provenance(), "entries": entries}, indent=2))
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
