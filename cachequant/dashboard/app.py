import copy
import json
import time
from pathlib import Path

import numba
import streamlit as st
import torch

from cachequant.bench.config import DEFAULT_CONFIG
from cachequant.bench.harness import compute_bench_result
from cachequant.kernel.bfp_linear import apply_bfp_quantization
from cachequant.kernel.int8_linear import apply_int8_quantization
from cachequant.kvcache.trie_cache import PrefixKVCache
from cachequant.model import load_model
from cachequant.pipeline import generate

MAX_NEW_TOKENS = 50
WARMUP_PROMPT = "The quick brown fox jumps over the lazy dog."
WARMUP_MAX_NEW_TOKENS = 5

# Duplicated from scripts/run_combined_benchmark.py rather than imported -
# matches that script's own precedent of duplicating rather than importing
# prompt data across scripts in this codebase.
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

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "replay.json"


def _load_replay_fixture() -> dict[tuple[str, str], dict]:
    data = json.loads(FIXTURE_PATH.read_text())
    return {(e["prompt_set"], e["combo"]): e for e in data["entries"]}


STREAM_DELAY_SECONDS = 0.04


def _stream_words(text: str) -> None:
    placeholder = st.empty()
    words = text.split(" ")
    shown = ""
    for word in words:
        shown = f"{shown} {word}".strip()
        placeholder.markdown(shown)
        time.sleep(STREAM_DELAY_SECONDS)


def _render_stream(text: str, prompt: str, tokenizer) -> None:
    # pipeline.generate returns prompt + continuation decoded together.
    # Stream only the newly-generated portion — the prompt is already shown
    # above via st.subheader. Compare against the tokenizer's own round-trip
    # of the prompt (not the raw prompt string) since GPT-2's BPE decode may
    # not be byte-identical to the original text (whitespace handling).
    decoded_prompt = tokenizer.decode(tokenizer.encode(prompt), skip_special_tokens=True)
    generated_only = text[len(decoded_prompt):] if text.startswith(decoded_prompt) else text
    _stream_words(generated_only)


def _prefill_tokens_per_sec(timing, stats, wall_seconds: float) -> float:
    if stats is None:
        # No cache: compute_bench_result's number is already correct —
        # prefill_tokens is the full prompt, prefill_seconds is the whole
        # (only) forward pass.
        return timing.prefill_tokens / timing.prefill_seconds if timing.prefill_seconds > 0 else 0.0
    # Cache-on: timing.prefill_tokens/timing.prefill_seconds only cover the
    # recomputed suffix and the internal forward pass — blind to
    # cache.lookup()/insert() overhead and undercounts the token base
    # (should be the full prompt, since the cache genuinely served the
    # rest). Recover both from the honestly-measured wall_seconds, same
    # convention as scripts/run_combined_benchmark.py's honest_prefill_seconds.
    decode_seconds = timing.decode_seconds
    cache_overhead_seconds = wall_seconds - (timing.prefill_seconds + decode_seconds)
    honest_prefill_seconds = timing.prefill_seconds + cache_overhead_seconds
    return stats.prompt_tokens / honest_prefill_seconds if honest_prefill_seconds > 0 else 0.0


def _cost_per_1k(wall_seconds: float, generated_tokens: int) -> float:
    if generated_tokens <= 0:
        return 0.0
    seconds_per_token = wall_seconds / generated_tokens
    return (DEFAULT_CONFIG.dollars_per_hour / 3600) * seconds_per_token * 1000


def _next_prompt(prompt_set_name: str) -> str:
    prompts = PROMPT_SETS[prompt_set_name]
    idx = st.session_state.prompt_index[prompt_set_name]
    prompt = prompts[idx % len(prompts)]
    st.session_state.prompt_index[prompt_set_name] = idx + 1
    return prompt


def _timed_generate(model, tokenizer, prompt: str, cache):
    # Times the ENTIRE call, not GenerationTiming's internal-only timing,
    # so cache.lookup()/insert() overhead (invisible to `timing` itself) is
    # captured. See Global Constraints — this is the one number that must
    # not be shortcut.
    t0 = time.perf_counter()
    text, timing, stats = generate(model, tokenizer, prompt, cache=cache, max_new_tokens=MAX_NEW_TOKENS)
    wall_seconds = time.perf_counter() - t0
    return text, timing, stats, wall_seconds


def _init_session_state(replay_mode: bool) -> None:
    if "results_rows" not in st.session_state:
        st.session_state.results_rows = []
        st.session_state.prompt_index = {name: 0 for name in PROMPT_SETS}

    if replay_mode:
        # Offline fallback path: no load_model(), no quantization pass,
        # no warmup - nothing here can fail the way the live path can (model
        # download, JIT compile stall, OOM, slow CPU). This is the whole point.
        if "replay_fixture" not in st.session_state:
            st.session_state.replay_fixture = _load_replay_fixture()
        return

    if "models" in st.session_state:
        return

    with st.spinner("Loading GPT-2, quantizing BFP and int8 copies, warming up all three..."):
        # Pin both thread pools before any timed run. torch and numba have
        # independent thread pools; without this the quantized paths' numba
        # matmul runs on every logical core while fp32's torch path is
        # unpinned/different, making the on-screen comparison not
        # apples-to-apples. Same convention as scripts/run_bfp_benchmark.py.
        torch.set_num_threads(DEFAULT_CONFIG.cpu_threads)
        numba.set_num_threads(DEFAULT_CONFIG.cpu_threads)

        fp32_model, tokenizer = load_model()
        models = {
            "fp32": fp32_model,
            "bfp": apply_bfp_quantization(copy.deepcopy(fp32_model)),
            "int8": apply_int8_quantization(copy.deepcopy(fp32_model)),
        }

        # Discard one warmup generation per model so a quantized path's
        # first-call Numba JIT compile time is never displayed as a
        # measurement. Both schemes share one njit kernel, so the second
        # quantized warmup is cheap - but its per-scheme operand prep is not
        # already compiled, so it still needs one.
        for model in models.values():
            generate(model, tokenizer, WARMUP_PROMPT, max_new_tokens=WARMUP_MAX_NEW_TOKENS)

        # Also warm up the cache code path (first DynamicCache construction,
        # first tensor permute/contiguous allocations inside
        # generate_with_prefix_cache) so those one-time costs aren't paid
        # inside the honestly-measured perf_counter window on the user's
        # first cache-on click. Uses throwaway caches, not the real session
        # caches below, which must stay genuinely cold.
        for model in models.values():
            warmup_cache = PrefixKVCache(max_tokens=DEFAULT_CONFIG.max_cache_tokens)
            generate(model, tokenizer, WARMUP_PROMPT, cache=warmup_cache, max_new_tokens=WARMUP_MAX_NEW_TOKENS)

    st.session_state.models = models
    st.session_state.tokenizer = tokenizer
    # One cache per scheme, never shared: each scheme produces different K/V
    # values for the same tokens, so one cache cannot correctly serve two.
    st.session_state.caches = {
        name: PrefixKVCache(max_tokens=DEFAULT_CONFIG.max_cache_tokens) for name in models
    }


def main() -> None:
    st.set_page_config(page_title="CacheQuant Dashboard", layout="wide")
    replay_mode = st.sidebar.toggle("Replay (offline)", value=False)
    _init_session_state(replay_mode)
    st.title("CacheQuant — BFP / int8 quantization + KV-cache prefix reuse, live")
    if replay_mode:
        st.sidebar.caption("Offline replay: pre-captured runs, no live model — safe if the live demo breaks.")

    with st.sidebar:
        scheme_label = st.radio("Quantization", ["fp32", "BFP", "int8"], index=0)
        scheme = {"fp32": "fp32", "BFP": "bfp", "int8": "int8"}[scheme_label]
        use_cache = st.toggle("Prefix cache", value=False)
        prompt_set_name = st.radio("Prompt set", list(PROMPT_SETS.keys()), index=0)
        generate_clicked = st.button("Generate", type="primary")
        clear_clicked = st.button("Clear results table")
        reset_clicked = st.button("Reset cache(s)", disabled=replay_mode)

    if reset_clicked and not replay_mode:
        st.session_state.caches = {
            name: PrefixKVCache(max_tokens=DEFAULT_CONFIG.max_cache_tokens)
            for name in st.session_state.models
        }
        st.toast("All caches reset.")

    if clear_clicked:
        st.session_state.results_rows = []
        st.toast("Results table cleared.")

    if generate_clicked:
        combo_key = scheme + ("_cache" if use_cache else "_no_cache")

        if replay_mode:
            fixture = st.session_state.replay_fixture
            baseline_entry = fixture[(prompt_set_name, "fp32_no_cache")]
            combo_entry = fixture[(prompt_set_name, combo_key)]

            st.subheader(f"Prompt: {combo_entry['prompt']}")
            _stream_words(combo_entry["generated_only"])

            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    "Prefill tok/s",
                    f"{combo_entry['prefill_tok_s']:.1f}",
                    f"{combo_entry['prefill_tok_s'] - baseline_entry['prefill_tok_s']:+.1f}",
                )
                st.metric(
                    "Decode tok/s",
                    f"{combo_entry['decode_tok_s']:.1f}",
                    f"{combo_entry['decode_tok_s'] - baseline_entry['decode_tok_s']:+.1f}",
                )
            with col2:
                st.metric(
                    "Wall clock (s)",
                    f"{combo_entry['wall_seconds']:.3f}",
                    f"{combo_entry['wall_seconds'] - baseline_entry['wall_seconds']:+.3f}",
                    delta_color="inverse",
                )
                st.metric(
                    "Cost / 1K tokens ($)",
                    f"{combo_entry['cost_per_1k']:.5f}",
                    f"{combo_entry['cost_per_1k'] - baseline_entry['cost_per_1k']:+.5f}",
                    delta_color="inverse",
                )

            label = scheme + ("+cache" if use_cache else "")
            st.session_state.results_rows.append(
                {
                    "config": label,
                    "prompt_set": prompt_set_name,
                    "prefill_tok_s": round(combo_entry["prefill_tok_s"], 1),
                    "decode_tok_s": round(combo_entry["decode_tok_s"], 1),
                    "wall_seconds": round(combo_entry["wall_seconds"], 3),
                    "cost_per_1k": round(combo_entry["cost_per_1k"], 5),
                    "gen_tokens": combo_entry["gen_tokens"],
                    "cache": combo_entry["cache_detail"],
                }
            )
        else:
            prompt = _next_prompt(prompt_set_name)

            baseline_text, baseline_timing, _, baseline_wall = _timed_generate(
                st.session_state.models["fp32"], st.session_state.tokenizer, prompt, None
            )

            if scheme == "fp32" and not use_cache:
                combo_text, combo_timing, combo_stats, combo_wall = (
                    baseline_text, baseline_timing, None, baseline_wall
                )
            else:
                combo_model = st.session_state.models[scheme]
                combo_cache = st.session_state.caches[scheme] if use_cache else None
                combo_text, combo_timing, combo_stats, combo_wall = _timed_generate(
                    combo_model, st.session_state.tokenizer, prompt, combo_cache
                )

            st.subheader(f"Prompt: {prompt}")
            _render_stream(combo_text, prompt, st.session_state.tokenizer)

            baseline_bench = compute_bench_result(baseline_timing, DEFAULT_CONFIG)
            combo_bench = compute_bench_result(combo_timing, DEFAULT_CONFIG)
            baseline_cost = _cost_per_1k(baseline_wall, baseline_timing.total_generated_tokens)
            combo_cost = _cost_per_1k(combo_wall, combo_timing.total_generated_tokens)
            # compute_bench_result's prefill_tokens_per_sec is blind to
            # cache.lookup()/insert() overhead and undercounts the token base on
            # the cache-on path (see _prefill_tokens_per_sec). baseline is
            # always no-cache (stats=None), so this is a no-op there.
            baseline_prefill_tok_s = _prefill_tokens_per_sec(baseline_timing, None, baseline_wall)
            combo_prefill_tok_s = _prefill_tokens_per_sec(combo_timing, combo_stats, combo_wall)

            # Prefill and decode charted separately, not as one aggregate
            # tokens/sec: Phase 2 found quantization hurts FLOP-bound prefill
            # far more than memory-bound matrix-vector decode, and an aggregate
            # number averages that finding away. This split is the main change from the original
            # Phase 4 sketch's single before/after bar chart.
            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    "Prefill tok/s",
                    f"{combo_prefill_tok_s:.1f}",
                    f"{combo_prefill_tok_s - baseline_prefill_tok_s:+.1f}",
                )
                st.metric(
                    "Decode tok/s",
                    f"{combo_bench.decode_tokens_per_sec:.1f}",
                    f"{combo_bench.decode_tokens_per_sec - baseline_bench.decode_tokens_per_sec:+.1f}",
                )
            with col2:
                st.metric(
                    "Wall clock (s)",
                    f"{combo_wall:.3f}",
                    f"{combo_wall - baseline_wall:+.3f}",
                    delta_color="inverse",
                )
                st.metric(
                    "Cost / 1K tokens ($)",
                    f"{combo_cost:.5f}",
                    f"{combo_cost - baseline_cost:+.5f}",
                    delta_color="inverse",
                )

            label = scheme + ("+cache" if use_cache else "")
            cache_detail = "—"
            if combo_stats is not None:
                cache_detail = (
                    f"{combo_stats.cached_tokens}/{combo_stats.prompt_tokens} reused "
                    f"({combo_stats.hit_rate:.0%})"
                )
            st.session_state.results_rows.append(
                {
                    "config": label,
                    "prompt_set": prompt_set_name,
                    "prefill_tok_s": round(combo_prefill_tok_s, 1),
                    "decode_tok_s": round(combo_bench.decode_tokens_per_sec, 1),
                    "wall_seconds": round(combo_wall, 3),
                    "cost_per_1k": round(combo_cost, 5),
                    "gen_tokens": combo_timing.total_generated_tokens,
                    "cache": cache_detail,
                }
            )

    if st.session_state.results_rows:
        st.subheader("Results")
        st.dataframe(st.session_state.results_rows)


if __name__ == "__main__":
    main()
