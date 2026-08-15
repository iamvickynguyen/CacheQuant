import copy
import time

import streamlit as st

from cachequant.bench.config import DEFAULT_CONFIG
from cachequant.bench.harness import compute_bench_result
from cachequant.kernel.bfp_linear import apply_bfp_quantization
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
    "where was the transistor invented?",
]
LONG_HIGH_REUSE_PROMPTS = [
    LONG_HIGH_REUSE_PREAMBLE + suffix for suffix in LONG_HIGH_REUSE_SUFFIXES
]

PROMPT_SETS = {
    "high_reuse": HIGH_REUSE_PROMPTS,
    "long_high_reuse": LONG_HIGH_REUSE_PROMPTS,
    "no_reuse": NO_REUSE_PROMPTS,
}

STREAM_DELAY_SECONDS = 0.04


def _render_stream(text: str) -> None:
    placeholder = st.empty()
    words = text.split(" ")
    shown = ""
    for word in words:
        shown = f"{shown} {word}".strip()
        placeholder.markdown(shown)
        time.sleep(STREAM_DELAY_SECONDS)


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


def _init_session_state() -> None:
    if "fp32_model" in st.session_state:
        return

    with st.spinner("Loading GPT-2, quantizing BFP copy, warming up both models..."):
        fp32_model, tokenizer = load_model()
        bfp_model = apply_bfp_quantization(copy.deepcopy(fp32_model))

        # Discard one warmup generation per model so BFP's first-call Numba
        # JIT compile time is never displayed to the user as a measurement.
        generate(fp32_model, tokenizer, WARMUP_PROMPT, max_new_tokens=WARMUP_MAX_NEW_TOKENS)
        generate(bfp_model, tokenizer, WARMUP_PROMPT, max_new_tokens=WARMUP_MAX_NEW_TOKENS)

    st.session_state.fp32_model = fp32_model
    st.session_state.bfp_model = bfp_model
    st.session_state.tokenizer = tokenizer
    st.session_state.fp32_cache = PrefixKVCache(max_tokens=DEFAULT_CONFIG.max_cache_tokens)
    st.session_state.bfp_cache = PrefixKVCache(max_tokens=DEFAULT_CONFIG.max_cache_tokens)
    st.session_state.results_rows = []
    st.session_state.prompt_index = {name: 0 for name in PROMPT_SETS}


def main() -> None:
    st.set_page_config(page_title="CacheQuant Dashboard", layout="wide")
    _init_session_state()
    st.title("CacheQuant — BFP quantization + KV-cache prefix reuse, live")

    with st.sidebar:
        use_bfp = st.radio("Quantization", ["fp32", "BFP"], index=0) == "BFP"
        use_cache = st.toggle("Prefix cache", value=False)
        prompt_set_name = st.radio("Prompt set", list(PROMPT_SETS.keys()), index=0)
        generate_clicked = st.button("Generate", type="primary")
        clear_clicked = st.button("Clear results table")
        reset_clicked = st.button("Reset cache(s)")

    if reset_clicked:
        st.session_state.fp32_cache = PrefixKVCache(max_tokens=DEFAULT_CONFIG.max_cache_tokens)
        st.session_state.bfp_cache = PrefixKVCache(max_tokens=DEFAULT_CONFIG.max_cache_tokens)
        st.toast("Both caches reset.")

    if clear_clicked:
        st.session_state.results_rows = []
        st.toast("Results table cleared.")

    if generate_clicked:
        prompt = _next_prompt(prompt_set_name)

        baseline_text, baseline_timing, _, baseline_wall = _timed_generate(
            st.session_state.fp32_model, st.session_state.tokenizer, prompt, None
        )

        if not use_bfp and not use_cache:
            combo_text, combo_timing, combo_stats, combo_wall = (
                baseline_text, baseline_timing, None, baseline_wall
            )
        else:
            combo_model = st.session_state.bfp_model if use_bfp else st.session_state.fp32_model
            combo_cache = None
            if use_cache:
                combo_cache = st.session_state.bfp_cache if use_bfp else st.session_state.fp32_cache
            combo_text, combo_timing, combo_stats, combo_wall = _timed_generate(
                combo_model, st.session_state.tokenizer, prompt, combo_cache
            )

        st.subheader(f"Prompt: {prompt}")
        _render_stream(combo_text)

        baseline_bench = compute_bench_result(baseline_timing, DEFAULT_CONFIG)
        combo_bench = compute_bench_result(combo_timing, DEFAULT_CONFIG)
        baseline_cost = _cost_per_1k(baseline_wall, baseline_timing.total_generated_tokens)
        combo_cost = _cost_per_1k(combo_wall, combo_timing.total_generated_tokens)

        # Prefill and decode charted separately, not as one aggregate
        # tokens/sec: Phase 2 found BFP helps FLOP-bound prefill and hurts
        # matrix-vector decode, and an aggregate number averages that
        # finding away. This split is the main change from the original
        # Phase 4 sketch's single before/after bar chart.
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                "Prefill tok/s",
                f"{combo_bench.prefill_tokens_per_sec:.1f}",
                f"{combo_bench.prefill_tokens_per_sec - baseline_bench.prefill_tokens_per_sec:+.1f}",
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

        label = ("bfp" if use_bfp else "fp32") + ("+cache" if use_cache else "")
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
                "prefill_tok_s": round(combo_bench.prefill_tokens_per_sec, 1),
                "decode_tok_s": round(combo_bench.decode_tokens_per_sec, 1),
                "wall_seconds": round(combo_wall, 3),
                "cost_per_1k": round(combo_cost, 5),
                "cache": cache_detail,
            }
        )

    if st.session_state.results_rows:
        st.subheader("Results")
        st.dataframe(st.session_state.results_rows)


if __name__ == "__main__":
    main()
