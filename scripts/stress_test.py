import copy
import json
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
WARMUP_PROMPT = "The quick brown fox jumps over the lazy dog."
WARMUP_MAX_NEW_TOKENS = 5
NORMAL_PROMPT = "The history of computing began with"

# Small eviction cache is intentionally much smaller than DEFAULT_CONFIG's
# 2048-token cap so 20 unique long-ish prompts actually force LRU eviction
# within one run, rather than all fitting comfortably.
EVICTION_CACHE_MAX_TOKENS = 256
EVICTION_PROMPT_COUNT = 20


def _build_token_bounded_prompt(tokenizer, target_tokens: int) -> str:
    """Best-effort prompt whose token count is close to target_tokens.

    Repeatedly appends a filler sentence until encoding exceeds the target,
    then truncates the token ids and decodes back to text. Decoding then
    re-encoding (which pipeline.generate does internally) can shift the
    final count by a few tokens - this is fine here, the actual encoded
    length is measured and recorded per-case rather than assumed.
    """
    filler = (
        "This is a benchmark stress-test sentence used only to pad the "
        "prompt to a target token length. "
    )
    text = filler
    while len(tokenizer.encode(text)) < target_tokens:
        text += filler
    ids = tokenizer.encode(text)[:target_tokens]
    return tokenizer.decode(ids)


def _run_case(model, tokenizer, prompt: str, max_new_tokens: int, cache) -> tuple[str, float]:
    t0 = time.perf_counter()
    try:
        generate(model, tokenizer, prompt, cache=cache, max_new_tokens=max_new_tokens)
        outcome = "success"
    except Exception as e:  # noqa: BLE001 - deliberately broad, this script's job is to observe failures
        outcome = f"{type(e).__name__}: {e}"
    wall_seconds = time.perf_counter() - t0
    return outcome, wall_seconds


def main() -> None:
    # Pin both thread pools before any timed run. torch and numba have
    # independent thread pools; without this the quantized paths' numba matmul
    # runs on every logical core while fp32's torch path stays at cpu_threads,
    # making the recorded quantized-vs-fp32 wall times not apples-to-apples. Same convention as
    # scripts/run_bfp_benchmark.py's main() and the dashboard's model loader.
    torch.set_num_threads(DEFAULT_CONFIG.cpu_threads)
    numba.set_num_threads(DEFAULT_CONFIG.cpu_threads)

    fp32_model, tokenizer = load_model()
    bfp_model = apply_bfp_quantization(copy.deepcopy(fp32_model))
    int8_model = apply_int8_quantization(copy.deepcopy(fp32_model))
    for model in (fp32_model, bfp_model, int8_model):
        generate(model, tokenizer, WARMUP_PROMPT, cache=None, max_new_tokens=WARMUP_MAX_NEW_TOKENS)

    over_context_prompt = _build_token_bounded_prompt(tokenizer, 1100)
    at_context_prompt = _build_token_bounded_prompt(tokenizer, 1024)

    cases = [
        {"case": "empty_prompt", "prompt": "", "max_new_tokens": 10},
        {"case": "whitespace_prompt", "prompt": "   ", "max_new_tokens": 10},
        {"case": "over_context_prompt", "prompt": over_context_prompt, "max_new_tokens": 5},
        {"case": "at_context_prompt", "prompt": at_context_prompt, "max_new_tokens": 5},
        {"case": "zero_max_new_tokens", "prompt": NORMAL_PROMPT, "max_new_tokens": 0},
        {"case": "large_max_new_tokens", "prompt": NORMAL_PROMPT, "max_new_tokens": 200},
    ]

    combos = [
        ("fp32_no_cache", fp32_model, False),
        ("fp32_cache", fp32_model, True),
        ("bfp_no_cache", bfp_model, False),
        ("bfp_cache", bfp_model, True),
        ("int8_no_cache", int8_model, False),
        ("int8_cache", int8_model, True),
    ]

    results = []
    for case in cases:
        prompt_tokens = len(tokenizer.encode(case["prompt"]))
        for combo_label, model, use_cache in combos:
            cache = PrefixKVCache(max_tokens=DEFAULT_CONFIG.max_cache_tokens) if use_cache else None
            outcome, wall_seconds = _run_case(model, tokenizer, case["prompt"], case["max_new_tokens"], cache)
            row = {
                "case": case["case"],
                "combo": combo_label,
                "prompt_tokens": prompt_tokens,
                "max_new_tokens": case["max_new_tokens"],
                "outcome": outcome,
                "wall_seconds": wall_seconds,
            }
            results.append(row)
            print(f"{case['case']:24s} {combo_label:14s} {outcome[:70]:70s} {wall_seconds:.3f}s")

    eviction_prompts = [
        (
            f"Prompt number {i} discusses a unique topic in science, history, "
            f"or technology, padded with extra words to add real length to "
            f"the prompt text so the cache has meaningful content to evict. "
        )
        for i in range(EVICTION_PROMPT_COUNT)
    ]
    for combo_label, model, use_cache in combos:
        if not use_cache:
            continue
        cache = PrefixKVCache(max_tokens=EVICTION_CACHE_MAX_TOKENS)
        t0 = time.perf_counter()
        try:
            for prompt in eviction_prompts:
                generate(model, tokenizer, prompt, cache=cache, max_new_tokens=5)
            outcome = "success"
        except Exception as e:  # noqa: BLE001
            outcome = f"{type(e).__name__}: {e}"
        wall_seconds = time.perf_counter() - t0
        row = {
            "case": "cache_eviction_pressure",
            "combo": combo_label,
            "prompt_tokens": None,
            "max_new_tokens": 5,
            "outcome": outcome,
            "wall_seconds": wall_seconds,
        }
        results.append(row)
        print(f"{'cache_eviction_pressure':24s} {combo_label:14s} {outcome[:70]:70s} {wall_seconds:.3f}s")

    BENCHMARKS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = BENCHMARKS_DIR / "stress_test_results.json"
    output_path.write_text(json.dumps({"provenance": provenance(), "results": results}, indent=2))
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
