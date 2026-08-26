"""Plot the multi-turn chat benchmark from benchmarks/multiturn_results.json (no re-run).

multiturn_prefill_by_turn.png  — prefill seconds per turn, no-cache vs cache-on.
multiturn_hit_rate_by_turn.png — hit rate per turn against the (k-1)/k ceiling.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
BENCHMARKS_DIR = REPO_ROOT / "benchmarks"
CHARTS_DIR = BENCHMARKS_DIR / "charts"

RESULTS_FILE = "multiturn_results.json"


def _load() -> dict:
    return json.loads((BENCHMARKS_DIR / RESULTS_FILE).read_text())


def _chart_prefill_by_turn(data: dict) -> None:
    aggregate = data["by_turn_index"]
    turns = [r["turn_index"] for r in aggregate]
    no_cache = [r["baseline_prefill_seconds"] for r in aggregate]
    cache_on = [r["honest_prefill_seconds"] for r in aggregate]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(turns, no_cache, "o-", color="tab:red", label="no cache (re-reads whole transcript)")
    ax.plot(turns, cache_on, "o-", color="tab:blue", label="cache-on (honest, incl. lookup/insert)")
    ax.set_xlabel("conversation turn")
    ax.set_ylabel("prefill seconds (median across conversations)")
    ax.set_title("Multi-turn chat: prefill cost per turn")
    ax.set_ylim(bottom=0)
    ax.legend()
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "multiturn_prefill_by_turn.png", dpi=150)
    plt.close(fig)


def _chart_hit_rate_by_turn(data: dict) -> None:
    aggregate = data["by_turn_index"]
    turns = [r["turn_index"] for r in aggregate]
    hit_rate = [r["hit_rate"] for r in aggregate]
    ceiling = [k / (k + 1) for k in turns]  # turn_index k has ceiling (k)/(k+1) vs its own start

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(turns, hit_rate, "o-", color="tab:blue", label="measured hit rate")
    ax.plot(turns, ceiling, "--", color="gray", label="ceiling (last prompt token never cached)")
    ax.set_xlabel("conversation turn")
    ax.set_ylabel("hit rate (cached_tokens / prompt_tokens)")
    ax.set_title("Multi-turn chat: prefix hit rate climbs with the transcript")
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "multiturn_hit_rate_by_turn.png", dpi=150)
    plt.close(fig)


def main() -> None:
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    data = _load()
    _chart_prefill_by_turn(data)
    _chart_hit_rate_by_turn(data)
    print(
        f"wrote multiturn_prefill_by_turn.png, multiturn_hit_rate_by_turn.png to {CHARTS_DIR}"
    )


if __name__ == "__main__":
    main()
