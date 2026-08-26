"""Re-run every benchmark fresh, then export slide-ready charts and tables.

Takes several minutes: it runs the baseline, both quantized benchmarks, the
quality comparison, the KV-cache sweep, the combined toggle sweep and the
multi-turn chat sweep before plotting anything, so the charts can never be a
mix of runs from different sessions on a differently-loaded machine.
"""

import json
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

REPO_ROOT = Path(__file__).resolve().parent.parent
BENCHMARKS_DIR = REPO_ROOT / "benchmarks"
CHARTS_DIR = BENCHMARKS_DIR / "charts"

sys.path.insert(0, str(REPO_ROOT / "scripts"))

# Validated categorical palette (slots 1-4), assigned by identity and never
# cycled: a scheme keeps its hue in every chart here, so the reader learns the
# mapping once. Three of these sit below 3:1 contrast on a white surface, which
# obliges direct value labels on every mark — `_label_bars` does that, and
# charts/summary.md is the table view.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
SCHEME_COLOR = {
    "fp32": "#2a78d6",   # blue
    "fp16": "#eb6834",   # orange
    "bfp8": "#1baf7a",   # aqua
    "int8": "#eda100",   # yellow
}
SCHEME_LABEL = {"fp32": "fp32", "fp16": "fp16", "bfp8": "BFP8", "int8": "int8"}
# Cache on/off is a second encoding on top of hue rather than six more hues:
# six categorical slots would put pairs on screen that no ordering separates.
CACHE_HATCH = {False: "", True: "///"}


BENCHMARK_SCRIPTS = [
    "run_baseline.py",
    "run_bfp_benchmark.py",
    "run_int8_benchmark.py",
    "run_fp16_benchmark.py",
    "compare_quantization_quality.py",
    "run_kvcache_benchmark.py",
    "run_combined_benchmark.py",
    "run_multiturn_benchmark.py",
]


def _run_benchmarks_fresh() -> None:
    """One subprocess per benchmark, not one import-and-call chain.

    An earlier version imported each script and called its main() in this
    process. Every script loads its own GPT-2 and several load two or three
    quantized copies, so the last benchmark in the chain ran in an
    interpreter holding the residue of all the earlier ones.

    Measured, that turned out **not** to be why BFP decode reads below its
    historical number — a chained-vs-standalone A/B put both inside the same
    (wide) noise band on this machine. Kept anyway, because a fresh
    interpreter per benchmark is the cheap way to keep that variable out of
    the comparison entirely rather than re-checking it every time the numbers
    look off.
    """
    for script in BENCHMARK_SCRIPTS:
        print(f"running {script} ...", flush=True)
        subprocess.run(
            [sys.executable, str(REPO_ROOT / "scripts" / script)],
            check=True,
            cwd=REPO_ROOT,
        )


def _load(name: str) -> dict:
    return json.loads((BENCHMARKS_DIR / name).read_text())


def _style(ax, title: str, ylabel: str = "") -> None:
    """Recessive grid and axes; the data carries the ink, not the frame."""
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=11)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK_MUTED, fontsize=9)
    ax.grid(axis="y", color="#dcdcd8", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#dcdcd8")
    ax.tick_params(colors=INK_MUTED, labelsize=9, length=0)


def _label_bars(ax, bars, fmt="{:.0f}") -> None:
    """Direct value labels. Required, not decorative: several palette slots are
    below 3:1 against a white surface, so identity and magnitude must both be
    legible without relying on the fill colour."""
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            fmt.format(height),
            (bar.get_x() + bar.get_width() / 2, height),
            textcoords="offset points",
            xytext=(0, 3),
            ha="center",
            fontsize=8,
            color=INK,
        )


def _figure(*args, **kwargs):
    fig, axes = plt.subplots(*args, **kwargs)
    fig.patch.set_facecolor(SURFACE)
    return fig, axes


def _save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / name, dpi=150, facecolor=SURFACE)
    plt.close(fig)


# --- charts -----------------------------------------------------------------

SCHEME_FILES = {
    "fp32": ("baseline_results.json", "baseline_results_longprompt.json"),
    "fp16": ("fp16_results.json", "fp16_results_longprompt.json"),
    "bfp8": ("bfp_results.json", "bfp_results_longprompt.json"),
    "int8": ("int8_results.json", "int8_results_longprompt.json"),
}


def _median(scheme: str, long_prompt: bool, metric: str) -> float:
    return _load(SCHEME_FILES[scheme][int(long_prompt)])["summary"][metric]["median"]


def _chart_quantization_breakpoints() -> None:
    """The headline: what each scheme costs, split by the phase it costs it in.

    Prefill and decode are plotted as separate panels rather than one aggregate,
    because averaging them is exactly what hides the break point — every scheme
    here behaves differently in the two phases, and two of them move in
    *opposite* directions between the panels.

    Note the y-axes are per-panel, not shared: prefill spans 1200 tok/s and
    decode 50, so a shared axis would flatten every decode bar. That is a
    deliberate tradeoff — bar heights are comparable within a panel and not
    across panels, which is why every bar carries its value.
    """
    schemes = list(SCHEME_FILES)
    fig, axes = _figure(2, 2, figsize=(11, 7))
    panels = [
        (axes[0][0], False, "prefill_tokens_per_sec", "short prompt — prefill", "tok/s"),
        (axes[0][1], False, "decode_tokens_per_sec", "short prompt — decode", "tok/s"),
        (axes[1][0], True, "prefill_tokens_per_sec", "long prompt — prefill", "tok/s"),
        (axes[1][1], True, "decode_tokens_per_sec", "long prompt — decode", "tok/s"),
    ]
    for ax, long_prompt, metric, title, ylabel in panels:
        values = [_median(s, long_prompt, metric) for s in schemes]
        bars = ax.bar(
            range(len(schemes)),
            values,
            width=0.62,
            color=[SCHEME_COLOR[s] for s in schemes],
        )
        _label_bars(ax, bars, "{:.1f}")
        ax.set_xticks(range(len(schemes)))
        ax.set_xticklabels([SCHEME_LABEL[s] for s in schemes])
        ax.set_ylim(0, max(values) * 1.18)
        _style(ax, title, ylabel)

    fig.suptitle(
        "Quantization break points: int8 kernels win decode and lose prefill; "
        "fp16 loses both",
        color=INK,
        fontsize=13,
    )
    _save(fig, "quantization_breakpoints.png")


def _chart_bfp_breakpoint() -> None:
    """fp32 vs BFP8 alone, kept because it is the Phase 2 slide.

    quantization_breakpoints.png supersedes it for the four-way comparison, but
    this file is tracked and referenced as the BFP-specific figure; regenerating
    it here is what stops it drifting into stale data every time the benchmarks
    are re-run.
    """
    fig, axes = _figure(1, 2, figsize=(10, 4))
    for ax, long_prompt, title in (
        (axes[0], False, "short prompt"),
        (axes[1], True, "long prompt"),
    ):
        metrics = ["prefill_tokens_per_sec", "decode_tokens_per_sec"]
        width = 0.36
        for i, scheme in enumerate(("fp32", "bfp8")):
            values = [_median(scheme, long_prompt, m) for m in metrics]
            bars = ax.bar(
                [x + (i - 0.5) * width for x in range(len(metrics))],
                values,
                width * 0.94,
                color=SCHEME_COLOR[scheme],
                label=SCHEME_LABEL[scheme],
            )
            _label_bars(ax, bars, "{:.1f}")
        ax.set_xticks(range(len(metrics)))
        ax.set_xticklabels(["prefill tok/s", "decode tok/s"])
        _style(ax, title, "tok/s")
        ax.legend(frameon=False, fontsize=8, labelcolor=INK_MUTED)
    fig.suptitle(
        "BFP break point: costs prefill throughput, near-free at decode",
        color=INK,
        fontsize=12,
    )
    _save(fig, "bfp_breakpoint.png")


def _chart_quality_vs_speed() -> None:
    """Perplexity cost against decode throughput — the actual tradeoff a scheme
    is chosen on. A scheme is only interesting if it is up and to the left."""
    quality = _load("quantization_quality.json")
    by_key = {row["key"]: row for row in quality["schemes"]}

    points = [
        ("fp32", 0.0, _median("fp32", False, "decode_tokens_per_sec")),
        ("fp16", 0.0, _median("fp16", False, "decode_tokens_per_sec")),
        ("bfp8", by_key["bfp8"]["perplexity_delta"], _median("bfp8", False, "decode_tokens_per_sec")),
        ("int8", by_key["int8_per_channel"]["perplexity_delta"],
         _median("int8", False, "decode_tokens_per_sec")),
    ]

    fig, ax = _figure(figsize=(7.5, 5))
    for scheme, ppl_delta, tok_s in points:
        ax.scatter(
            ppl_delta * 100, tok_s, s=140, color=SCHEME_COLOR[scheme], zorder=3,
            edgecolors=SURFACE, linewidths=2,
        )
        ax.annotate(
            f"{SCHEME_LABEL[scheme]}  ({ppl_delta * 100:+.2f}%, {tok_s:.1f} tok/s)",
            (ppl_delta * 100, tok_s),
            textcoords="offset points",
            xytext=(10, -4),
            fontsize=9,
            color=INK,
        )
    ax.axhline(
        _median("fp32", False, "decode_tokens_per_sec"),
        color="#dcdcd8", linestyle="--", linewidth=1, zorder=1,
    )
    ax.set_xlabel("perplexity change vs fp32 (%)", color=INK_MUTED, fontsize=9)
    ax.set_xlim(-0.6, max(p[1] for p in points) * 100 + 1.4)
    _style(ax, "Quality cost vs decode throughput (short prompt)", "decode tok/s")
    _save(fig, "quantization_quality_vs_speed.png")


def _chart_granularity_scale_grid() -> None:
    """Why BFP8 and plain int8 tie on quality, in one chart.

    They differ on two axes at once — how many values share a scale, and whether
    that scale is rounded to a power of two — so the head-to-head number cannot
    attribute the difference. The two corners neither scheme ships can.
    """
    grid = _load("quantization_quality.json")["granularity_scale_grid"]
    labels = [f"{row['granularity']}\n{row['scale']} scale" for row in grid]
    deltas = [row["perplexity_delta"] * 100 for row in grid]
    # Shipped corners keep their scheme hue; the two diagnostic corners are
    # deliberately neutral so the chart cannot be misread as four options.
    colors = []
    for row in grid:
        if not row["is_shipped_scheme"]:
            colors.append("#b8b7b0")
        elif row["granularity"] == "block-32":
            colors.append(SCHEME_COLOR["bfp8"])
        else:
            colors.append(SCHEME_COLOR["int8"])

    fig, ax = _figure(figsize=(8, 4.6))
    bars = ax.bar(range(len(grid)), deltas, width=0.6, color=colors)
    _label_bars(ax, bars, "{:+.2f}%")
    ax.axhline(0, color=INK_MUTED, linewidth=1)
    ax.set_xticks(range(len(grid)))
    ax.set_xticklabels(labels, fontsize=8)
    _style(ax, "", "perplexity change vs fp32 (%)")
    ax.legend(
        handles=[
            Patch(facecolor=SCHEME_COLOR["bfp8"], label="BFP8 (shipped)"),
            Patch(facecolor=SCHEME_COLOR["int8"], label="int8 (shipped)"),
            Patch(facecolor="#b8b7b0", label="diagnostic corner"),
        ],
        frameon=False,
        fontsize=8,
        labelcolor=INK_MUTED,
    )
    fig.suptitle(
        "Scale granularity vs scale rounding: the two schemes' difference, split apart",
        color=INK,
        fontsize=12,
    )
    _save(fig, "quantization_granularity_grid.png")


def _chart_kernel_scheme_comparison() -> None:
    """Same njit kernel, same shapes, different block structure — so this is a
    clean read on what BFP's per-32-block scale multiplies cost."""
    rows = _load("kernel_breakdown.json")["rows"]
    layers = ["attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj"]

    fig, axes = _figure(1, 2, figsize=(11, 4.4))
    for ax, phase in zip(axes, ("decode", "prefill")):
        width = 0.36
        for i, scheme in enumerate(("bfp8", "int8")):
            values = [
                next(
                    r["int8_kernel_ms"]
                    for r in rows
                    if r["scheme"] == scheme and r["phase"] == phase and r["layer"] == layer
                )
                for layer in layers
            ]
            bars = ax.bar(
                [x + (i - 0.5) * width for x in range(len(layers))],
                values,
                width * 0.94,   # surface gap between adjacent bars
                color=SCHEME_COLOR[scheme],
                label=SCHEME_LABEL[scheme],
            )
            _label_bars(ax, bars, "{:.2f}" if phase == "decode" else "{:.1f}")
        ax.set_xticks(range(len(layers)))
        ax.set_xticklabels(layers, fontsize=8, rotation=15)
        _style(ax, f"{phase} (M={1 if phase == 'decode' else 270})", "kernel ms")
        ax.legend(frameon=False, fontsize=8, labelcolor=INK_MUTED)

    fig.suptitle(
        "Shared kernel, different block structure: int8 does 1 scale multiply "
        "per output, BFP does k/32",
        color=INK,
        fontsize=12,
    )
    _save(fig, "kernel_scheme_comparison.png")


def _chart_cache_hit_rate_vs_speedup() -> None:
    data = _load("kvcache_results.json")
    names = list(data.keys())
    hit_rates = [data[n]["avg_hit_rate"] for n in names]
    speedups = [data[n]["avg_honest_prefill_speedup"] for n in names]

    fig, ax = _figure(figsize=(6.5, 4.4))
    ax.scatter(hit_rates, speedups, s=120, color=SCHEME_COLOR["fp32"], zorder=3)
    for name, x, y in zip(names, hit_rates, speedups):
        ax.annotate(
            f"{name}\n{y:.2f}x", (x, y), textcoords="offset points", xytext=(8, -4),
            fontsize=8, color=INK,
        )
    ax.axhline(1.0, color="#dcdcd8", linestyle="--", linewidth=1)
    ax.set_xlabel("avg hit rate", color=INK_MUTED, fontsize=9)
    ax.set_xlim(-0.05, 1.0)
    _style(ax, "KV-cache: hit rate vs. honest prefill speedup (fp32)",
           "avg honest prefill speedup")
    _save(fig, "cache_hit_rate_vs_speedup.png")


def _chart_combined() -> None:
    """All 3 schemes x cache on/off x 3 prompt sets.

    Hue is the scheme and hatching is the cache, rather than six categorical
    hues: with six slots on screen there are pairs no ordering separates for a
    colourblind reader, and the composite encoding also makes the actual
    question — what does the cache do *within* a scheme — readable as a pair.

    fp16 is deliberately absent, unlike in the other quantization charts. Two
    reasons, one presentational and one substantive. Presentational: fp16
    prefill is ~7 tok/s against fp32's ~230, so on a shared linear axis its
    bars compress every other bar to a sliver. Substantive: fp16 is slow here
    because this CPU has no native fp16 ALU, not because of anything about
    precision or caching, so its cache speedup would be the largest on the
    chart while saying nothing about caching — it would be skipping work that
    is artificially expensive on this machine only. It stays in
    quantization_breakpoints.png, where that hardware wall is the point.
    """
    data = _load("combined_results.json")
    prompt_sets = [k for k in data.keys() if k != "provenance"]
    combos = [
        (scheme, use_cache)
        for scheme in ("fp32", "bfp", "int8")
        for use_cache in (False, True)
    ]

    fig, ax = _figure(figsize=(10, 5.2))
    width = 0.14
    for i, (scheme, use_cache) in enumerate(combos):
        key = f"{scheme}_{'cache' if use_cache else 'no_cache'}"
        values = [data[ps][key]["total_honest_prefill_seconds"] for ps in prompt_sets]
        color = SCHEME_COLOR["bfp8" if scheme == "bfp" else scheme]
        bars = ax.bar(
            [x + (i - 2.5) * width for x in range(len(prompt_sets))],
            values,
            width * 0.9,
            color=color,
            hatch=CACHE_HATCH[use_cache],
            edgecolor=SURFACE,
            linewidth=0.8,
        )
        _label_bars(ax, bars, "{:.2f}")
    ax.set_xticks(range(len(prompt_sets)))
    ax.set_xticklabels(prompt_sets)
    _style(ax, "", "total honest prefill seconds (5 prompts)")
    ax.legend(
        handles=[
            Patch(facecolor=SCHEME_COLOR["fp32"], label="fp32"),
            Patch(facecolor=SCHEME_COLOR["bfp8"], label="BFP8"),
            Patch(facecolor=SCHEME_COLOR["int8"], label="int8"),
            Patch(facecolor="#ffffff", edgecolor=INK_MUTED, label="no cache"),
            Patch(facecolor="#ffffff", edgecolor=INK_MUTED, hatch="///", label="cache"),
        ],
        frameon=False,
        fontsize=8,
        ncol=5,
        labelcolor=INK_MUTED,
    )
    fig.suptitle(
        "Combined pipeline: the cache is worth more the slower the prefill path is",
        color=INK,
        fontsize=12,
    )
    _save(fig, "combined_comparison.png")


def _chart_multiturn() -> None:
    """Multi-turn chat: prefill cost per turn, no-cache vs cache-on.

    The Phase 7 slide. A one-shot request re-reads nothing; a chat turn re-reads
    the whole transcript, which grows every turn — so the no-cache line curves
    up while the cache-on line, which only ever re-reads the newest turn, stays
    flat. `by_turn_index` is the median across the three benchmark conversations.
    """
    aggregate = _load("multiturn_results.json")["by_turn_index"]
    turns = [r["turn_index"] for r in aggregate]
    no_cache = [r["baseline_prefill_seconds"] for r in aggregate]
    cache_on = [r["honest_prefill_seconds"] for r in aggregate]

    fig, ax = _figure(figsize=(7.5, 4.6))
    ax.plot(turns, no_cache, "o-", color=INK_MUTED, linewidth=2,
            label="no cache (re-reads whole transcript)")
    ax.plot(turns, cache_on, "o-", color=SCHEME_COLOR["fp32"], linewidth=2,
            label="cache-on (honest, incl. lookup/insert)")
    for x, y in zip(turns, no_cache):
        ax.annotate(f"{y * 1000:.0f}", (x, y), textcoords="offset points",
                    xytext=(0, 6), ha="center", fontsize=8, color=INK)
    for x, y in zip(turns, cache_on):
        ax.annotate(f"{y * 1000:.0f}", (x, y), textcoords="offset points",
                    xytext=(0, -12), ha="center", fontsize=8, color=INK)
    ax.set_xlabel("conversation turn", color=INK_MUTED, fontsize=9)
    ax.set_xticks(turns)
    ax.set_ylim(0, max(no_cache) * 1.2)
    _style(ax, "", "prefill seconds (median across conversations)")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_MUTED)
    fig.suptitle(
        "Multi-turn chat: no-cache prefill grows with the transcript, cache-on stays flat",
        color=INK,
        fontsize=12,
    )
    _save(fig, "multiturn_prefill_growth.png")


def _write_summary_md() -> None:
    quality = _load("quantization_quality.json")
    kvcache = _load("kvcache_results.json")
    combined = _load("combined_results.json")
    multiturn = _load("multiturn_results.json")
    by_key = {row["key"]: row for row in quality["schemes"]}

    lines = [
        "# Slide chart data (generated by scripts/export_slide_charts.py)",
        "",
        "Also the table view for every chart in this directory: several palette",
        "slots fall below 3:1 contrast on a white surface, so the numbers must be",
        "readable without relying on colour.",
        "",
        "## Quantization schemes, short prompt",
        "| scheme | prefill tok/s | decode tok/s | cost / 1K tokens | perplexity delta |",
        "|---|---:|---:|---:|---:|",
    ]
    ppl_key = {"bfp8": "bfp8", "int8": "int8_per_channel"}
    for scheme in ("fp32", "fp16", "bfp8", "int8"):
        summary = _load(SCHEME_FILES[scheme][0])["summary"]
        if scheme in ppl_key:
            delta = f"{by_key[ppl_key[scheme]]['perplexity_delta']:+.2%}"
        else:
            delta = "—" if scheme == "fp16" else "baseline"
        lines.append(
            f"| {SCHEME_LABEL[scheme]} | {summary['prefill_tokens_per_sec']['median']:.1f} "
            f"| {summary['decode_tokens_per_sec']['median']:.1f} "
            f"| ${summary['cost_per_1k_tokens']['median']:.5f} | {delta} |"
        )

    lines += ["", "## Quantization schemes, long prompt", "",
              "| scheme | prefill tok/s | decode tok/s | cost / 1K tokens |",
              "|---|---:|---:|---:|"]
    for scheme in ("fp32", "fp16", "bfp8", "int8"):
        summary = _load(SCHEME_FILES[scheme][1])["summary"]
        lines.append(
            f"| {SCHEME_LABEL[scheme]} | {summary['prefill_tokens_per_sec']['median']:.1f} "
            f"| {summary['decode_tokens_per_sec']['median']:.1f} "
            f"| ${summary['cost_per_1k_tokens']['median']:.5f} |"
        )

    lines += ["", "## Scale granularity vs scale rounding", "",
              "| granularity | scale | bits/value | perplexity | delta vs fp32 |",
              "|---|---|---:|---:|---:|"]
    for row in quality["granularity_scale_grid"]:
        shipped = " (shipped)" if row["is_shipped_scheme"] else ""
        lines.append(
            f"| {row['granularity']}{shipped} | {row['scale']} | {row['bits_per_value']:.3f} "
            f"| {row['perplexity']:.4f} | {row['perplexity_delta']:+.2%} |"
        )

    lines += ["", "## KV-cache hit rate vs. honest prefill speedup (fp32)", "",
              "| prompt set | avg hit rate | avg honest prefill speedup |",
              "|---|---:|---:|"]
    for name, row in kvcache.items():
        lines.append(
            f"| {name} | {row['avg_hit_rate']:.3f} | {row['avg_honest_prefill_speedup']:.3f} |"
        )

    combos = ["fp32_no_cache", "fp32_cache", "bfp_no_cache", "bfp_cache",
              "int8_no_cache", "int8_cache"]
    lines += ["", "## Combined pipeline: total honest prefill seconds", "",
              "| prompt set | " + " | ".join(combos) + " |",
              "|---|" + "---:|" * len(combos)]
    for ps, entries in combined.items():
        if ps == "provenance":
            continue
        row = [f"{entries[c]['total_honest_prefill_seconds']:.3f}" for c in combos]
        lines.append(f"| {ps} | " + " | ".join(row) + " |")

    lines += ["", "## Multi-turn chat: prefill seconds per turn "
              f"(median across {len(multiturn['conversations'])} conversations)", "",
              "| turn | no-cache prefill (ms) | cache-on honest (ms) | hit rate | speedup |",
              "|---:|---:|---:|---:|---:|"]
    for row in multiturn["by_turn_index"]:
        speedup = row["baseline_prefill_seconds"] / row["honest_prefill_seconds"]
        lines.append(
            f"| {row['turn_index']} | {row['baseline_prefill_seconds'] * 1000:.0f} "
            f"| {row['honest_prefill_seconds'] * 1000:.0f} | {row['hit_rate']:.2f} "
            f"| {speedup:.2f}x |"
        )

    (CHARTS_DIR / "summary.md").write_text("\n".join(lines) + "\n")


def main(run_benchmarks: bool = True) -> None:
    if run_benchmarks:
        _run_benchmarks_fresh()
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    _chart_quantization_breakpoints()
    _chart_bfp_breakpoint()
    _chart_quality_vs_speed()
    _chart_granularity_scale_grid()
    _chart_kernel_scheme_comparison()
    _chart_cache_hit_rate_vs_speedup()
    _chart_combined()
    _chart_multiturn()
    _write_summary_md()
    print(f"wrote charts + summary.md to {CHARTS_DIR}")


if __name__ == "__main__":
    # --charts-only re-plots from whatever is already in benchmarks/, for when
    # the numbers are current and only the plotting changed. The default still
    # re-runs everything, so a chart can never silently mix runs.
    main(run_benchmarks="--charts-only" not in sys.argv)
