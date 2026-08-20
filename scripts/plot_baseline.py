"""Plot throughput/latency from existing baseline_results*.json (no re-run)."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
BENCHMARKS_DIR = REPO_ROOT / "benchmarks"
CHARTS_DIR = BENCHMARKS_DIR / "charts"

RESULTS = {
    "short prompt": "baseline_results.json",
    "long prompt": "baseline_results_longprompt.json",
}


def _load(name: str) -> dict:
    return json.loads((BENCHMARKS_DIR / name).read_text())


def _bar_with_range(ax, summary: dict, metrics: list[str], labels: list[str]) -> None:
    medians = [summary[m]["median"] for m in metrics]
    err_low = [summary[m]["median"] - summary[m]["min"] for m in metrics]
    err_high = [summary[m]["max"] - summary[m]["median"] for m in metrics]
    x = range(len(metrics))
    ax.bar(x, medians, yerr=[err_low, err_high], capsize=4)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)


def _chart_throughput() -> None:
    metrics = ["prefill_tokens_per_sec", "decode_tokens_per_sec"]
    labels = ["prefill tok/s", "decode tok/s"]

    fig, axes = plt.subplots(1, len(RESULTS), figsize=(5 * len(RESULTS), 4))
    for ax, (title, filename) in zip(axes, RESULTS.items()):
        summary = _load(filename)["summary"]
        _bar_with_range(ax, summary, metrics, labels)
        ax.set_title(title)
    fig.suptitle("Baseline throughput (median, min-max range across reps)")
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "baseline_throughput.png", dpi=150)
    plt.close(fig)


def _chart_latency() -> None:
    metrics = ["p50_latency_ms", "p90_latency_ms", "mean_latency_ms"]
    labels = ["p50", "p90", "mean"]

    fig, axes = plt.subplots(1, len(RESULTS), figsize=(5 * len(RESULTS), 4))
    for ax, (title, filename) in zip(axes, RESULTS.items()):
        summary = _load(filename)["summary"]
        _bar_with_range(ax, summary, metrics, labels)
        ax.set_ylabel("ms")
        ax.set_title(title)
    fig.suptitle("Baseline per-token decode latency (median, min-max range across reps)")
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "baseline_latency.png", dpi=150)
    plt.close(fig)


def main() -> None:
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    _chart_throughput()
    _chart_latency()
    print(f"wrote baseline_throughput.png, baseline_latency.png to {CHARTS_DIR}")


if __name__ == "__main__":
    main()
