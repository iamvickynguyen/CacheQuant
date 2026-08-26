"""Multi-turn chat KV-cache benchmark.

Each conversation is run turn by turn. Both arms — no-cache and cache-on — get
the *same* prompt each turn; the no-cache continuation is the canonical reply
folded into the transcript, so the two arms stay token-identical turn to turn.
Writes benchmarks/multiturn_results.json and regenerates its charts.

See docs/superpowers/specs/2026-08-26-phase7-multiturn-chat.md.
"""

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
from cachequant.workloads import Conversation

BENCHMARKS_DIR = Path(__file__).resolve().parent.parent / "benchmarks"
MAX_NEW_TOKENS = 12
WARMUP_PROMPT = "The quick brown fox jumps over the lazy dog."
WARMUP_MAX_NEW_TOKENS = 5
N_REPS = 5

SHORT_SYSTEM = "You are a helpful assistant. Answer in one short sentence."

# Copied rather than imported from scripts/run_kvcache_benchmark.py — matches
# this codebase's precedent of duplicating prompt data across sibling scripts.
LONG_SYSTEM = (
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
    "learning models. Answer questions about it in one short sentence."
)

# Each turn's line refers back ("it", "that tower", "the language") so the
# transcript genuinely has to be present for the turn to make sense.
CONVERSATIONS = {
    "france": {
        "system": "short",
        "turns": [
            "What is the capital of France?",
            "What river runs through it?",
            "Which century was its most famous tower built in?",
            "Roughly how tall is that tower?",
            "Who was the engineer behind it?",
            "What is the tower made of?",
            "Is there a restaurant near the top of it?",
            "About how many people visit it each year?",
        ],
    },
    "python": {
        "system": "short",
        "turns": [
            "What kind of programming language is Python?",
            "Who originally created it?",
            "What year was it first released?",
            "Where does the name come from?",
            "What is its standard package installer called?",
            "How do you define a function in it?",
            "What does the 'self' parameter refer to?",
            "Is whitespace indentation significant in it?",
        ],
    },
    "computing_history": {
        "system": "long",
        "turns": [
            "Based on this passage, who designed the Analytical Engine?",
            "What did the person working alongside them contribute?",
            "What year was the transistor invented, according to the text?",
            "Which lab was it invented at?",
            "What trend describes the growth in computing power?",
            "Which decade did personal computers emerge in?",
            "What turned computers from isolated tools into interconnected nodes?",
            "What kind of hardware is now central to machine learning?",
        ],
    },
}

SYSTEM_PROMPTS = {"short": SHORT_SYSTEM, "long": LONG_SYSTEM}

_NON_TIMING_FIELDS = ("turn_index", "prompt_tokens", "cached_tokens", "hit_rate")
_TIMING_FIELDS = (
    "baseline_prefill_seconds",
    "cached_prefill_seconds",
    "cache_overhead_seconds",
    "honest_prefill_seconds",
    "honest_prefill_speedup",
    "baseline_decode_seconds",
    "cached_decode_seconds",
)


def _continuation(text: str, prompt: str, tokenizer) -> str:
    """The newly generated part of `generate`'s output. `generate` returns the
    prompt and continuation decoded together, and GPT-2's BPE round-trip may
    not be byte-identical to the raw prompt string — compare against the
    tokenizer's own decode of the prompt, same trick as the dashboard."""
    decoded_prompt = tokenizer.decode(tokenizer.encode(prompt), skip_special_tokens=True)
    return text[len(decoded_prompt):] if text.startswith(decoded_prompt) else text


def _run_conversation_once(model, tokenizer, system: str, user_turns: list[str]) -> list[dict]:
    """One pass over a conversation against a fresh cache. One row per turn."""
    conversation = Conversation(system)
    cache = PrefixKVCache(max_tokens=DEFAULT_CONFIG.max_cache_tokens)
    rows = []
    for turn_index, user_turn in enumerate(user_turns):
        prompt = conversation.prompt_for(user_turn)

        baseline_text, baseline_timing = generate(model, tokenizer, prompt, MAX_NEW_TOKENS)

        t0 = time.perf_counter()
        cached_text, cached_timing, stats = generate_with_prefix_cache(
            model, tokenizer, prompt, cache, MAX_NEW_TOKENS
        )
        total_call_seconds = time.perf_counter() - t0

        assert cached_text == baseline_text, (
            f"turn {turn_index}: cache-on text diverged from no-cache text"
        )

        cache_overhead_seconds = total_call_seconds - (
            cached_timing.prefill_seconds + cached_timing.decode_seconds
        )
        if cache_overhead_seconds < 0:
            raise RuntimeError(
                f"negative cache_overhead_seconds ({cache_overhead_seconds!r}) on turn "
                f"{turn_index}: total={total_call_seconds!r}, "
                f"prefill={cached_timing.prefill_seconds!r}, "
                f"decode={cached_timing.decode_seconds!r}"
            )
        honest_prefill_seconds = cached_timing.prefill_seconds + cache_overhead_seconds
        honest_prefill_speedup = (
            baseline_timing.prefill_seconds / honest_prefill_seconds
            if honest_prefill_seconds > 0
            else float("inf")
        )

        rows.append(
            {
                "turn_index": turn_index,
                "prompt_tokens": stats.prompt_tokens,
                "cached_tokens": stats.cached_tokens,
                "hit_rate": stats.hit_rate,
                "baseline_prefill_seconds": baseline_timing.prefill_seconds,
                "cached_prefill_seconds": cached_timing.prefill_seconds,
                "cache_overhead_seconds": cache_overhead_seconds,
                "honest_prefill_seconds": honest_prefill_seconds,
                "honest_prefill_speedup": honest_prefill_speedup,
                "baseline_decode_seconds": baseline_timing.decode_seconds,
                "cached_decode_seconds": cached_timing.decode_seconds,
            }
        )

        conversation.commit(user_turn, _continuation(baseline_text, prompt, tokenizer))
    return rows


def _run_conversation(model, tokenizer, name: str, spec: dict) -> dict:
    system = SYSTEM_PROMPTS[spec["system"]]
    passes = [
        _run_conversation_once(model, tokenizer, system, spec["turns"]) for _ in range(N_REPS)
    ]

    reference = passes[0]
    for pass_idx in range(1, N_REPS):
        for turn_idx, (ref_row, row) in enumerate(zip(reference, passes[pass_idx])):
            for field in _NON_TIMING_FIELDS:
                assert ref_row[field] == row[field], (
                    f"{name!r} turn {turn_idx} field {field!r} differed between pass 0 "
                    f"({ref_row[field]!r}) and pass {pass_idx} ({row[field]!r})"
                )

    rows = []
    for turn_idx in range(len(spec["turns"])):
        row = {field: reference[turn_idx][field] for field in _NON_TIMING_FIELDS}
        for field in _TIMING_FIELDS:
            row[field] = statistics.median(p[turn_idx][field] for p in passes)
        rows.append(row)

    print(f"{name} ({spec['system']} system)")
    for row in rows:
        print(
            f"  turn {row['turn_index']}: {row['prompt_tokens']:4d} tok, "
            f"hit {row['hit_rate']:.2f}, "
            f"no-cache prefill {row['baseline_prefill_seconds'] * 1000:6.1f} ms, "
            f"cache-on honest {row['honest_prefill_seconds'] * 1000:6.1f} ms, "
            f"speedup {row['honest_prefill_speedup']:.2f}x"
        )
    print()
    return {"system": spec["system"], "n_reps": N_REPS, "rows": rows}


def _aggregate_by_turn_index(conversations: dict) -> list[dict]:
    """Per turn index, the median across conversations of the fields the chart
    plots. Conversations are all the same length by construction."""
    n_turns = len(next(iter(conversations.values()))["rows"])
    aggregate = []
    for turn_idx in range(n_turns):
        turn_rows = [c["rows"][turn_idx] for c in conversations.values()]
        aggregate.append(
            {
                "turn_index": turn_idx,
                "baseline_prefill_seconds": statistics.median(
                    r["baseline_prefill_seconds"] for r in turn_rows
                ),
                "honest_prefill_seconds": statistics.median(
                    r["honest_prefill_seconds"] for r in turn_rows
                ),
                "hit_rate": statistics.median(r["hit_rate"] for r in turn_rows),
            }
        )
    return aggregate


def main() -> None:
    torch.set_num_threads(DEFAULT_CONFIG.cpu_threads)
    model, tokenizer = load_model()

    # One discarded warmup so first-call thread-pool spin-up / lazy kernel
    # selection doesn't land in the first measured turn (same as sibling scripts).
    generate(model, tokenizer, WARMUP_PROMPT, WARMUP_MAX_NEW_TOKENS)

    conversations = {
        name: _run_conversation(model, tokenizer, name, spec)
        for name, spec in CONVERSATIONS.items()
    }

    payload = {
        "provenance": provenance(),
        "n_reps": N_REPS,
        "max_new_tokens": MAX_NEW_TOKENS,
        "system_prompts": SYSTEM_PROMPTS,
        "conversations": conversations,
        "by_turn_index": _aggregate_by_turn_index(conversations),
    }

    BENCHMARKS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = BENCHMARKS_DIR / "multiturn_results.json"
    output_path.write_text(json.dumps(payload, indent=2))
    print(f"wrote {output_path}")

    import plot_multiturn

    plot_multiturn.main()


if __name__ == "__main__":
    main()
