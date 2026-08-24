"""Plot hit rate / speedup from existing benchmarks/kvcache_results.json (no re-run)."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
BENCHMARKS_DIR = REPO_ROOT / "benchmarks"
CHARTS_DIR = BENCHMARKS_DIR / "charts"

RESULTS_FILE = "kvcache_results.json"


def _load() -> dict:
    return json.loads((BENCHMARKS_DIR / RESULTS_FILE).read_text())


def _chart_hit_rate_vs_speedup(data: dict) -> None:
    """Per-prompt (not just per-set-average) scatter, so the spread within a
    prompt set - e.g. the cold-miss first prompt vs. its partial-hit
    successors in high_reuse - stays visible instead of being averaged away."""
    fig, ax = plt.subplots(figsize=(7, 5))
    for label, result in data.items():
        rows = result["rows"]
        hit_rates = [r["hit_rate"] for r in rows]
        speedups = [r["honest_prefill_speedup"] for r in rows]
        ax.scatter(hit_rates, speedups, label=label, alpha=0.8)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, label="breakeven")
    ax.set_xlabel("hit rate (cached_tokens / prompt_tokens)")
    ax.set_ylabel("prefill speedup (baseline / cache-on, overhead included)")
    ax.set_title("KV-cache: hit rate vs. prefill speedup (per prompt)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "kvcache_hit_rate_vs_speedup.png", dpi=150)
    plt.close(fig)


def _chart_prefill_seconds(data: dict) -> None:
    """Per-set total prefill seconds, baseline vs. honest (cache lookup/insert
    overhead included) - the actual wall-clock cost/win, not just the ratio."""
    labels = list(data.keys())
    baseline = [sum(r["baseline_prefill_seconds"] for r in data[l]["rows"]) for l in labels]
    honest = [sum(r["honest_prefill_seconds"] for r in data[l]["rows"]) for l in labels]

    x = range(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar([i - width / 2 for i in x], baseline, width, label="baseline (no cache)")
    ax.bar([i + width / 2 for i in x], honest, width, label="cache-on (incl. overhead)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("total prefill seconds (summed over prompt set)")
    ax.set_title("KV-cache: total prefill time, baseline vs. cache-on")
    ax.legend()
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "kvcache_prefill_seconds.png", dpi=150)
    plt.close(fig)


def main() -> None:
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    data = _load()
    _chart_hit_rate_vs_speedup(data)
    _chart_prefill_seconds(data)
    print(f"wrote kvcache_hit_rate_vs_speedup.png, kvcache_prefill_seconds.png to {CHARTS_DIR}")


if __name__ == "__main__":
    main()
