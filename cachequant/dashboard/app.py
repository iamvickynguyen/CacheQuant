import copy

import streamlit as st

from cachequant.bench.config import DEFAULT_CONFIG
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
    st.write("Models loaded. Sidebar controls and generation land in the next task.")


if __name__ == "__main__":
    main()
