# CacheQuant

Custom int8 quantized kernels (two schemes: BFP8 and plain int8) + KV-cache prefix reuse for faster, cheaper LLM inference. Benchmarked and visualized live.

## What's built so far (Phase 1 + 2 + 3 + 4a + 4b + 5 + 6 + 7)

A clean GPT-2 small (124M) generation loop on CPU, with prefill and decode timed separately, and a benchmark harness that turns those timings into tokens/sec, latency, and cost per 1K generated tokens. This is the frozen baseline every later optimization (quantized kernel, KV-cache reuse) gets compared against. Phase 1 has no quantization or caching — every call ran full fp32 GPT-2 from scratch.

Phase 2 adds a hand-written block floating-point (BFP) quantized matmul kernel
(`cachequant/kernel/`), applied to GPT-2's attention QKV/output projections and
both FFN linears (embeddings and `lm_head` stay fp32). Both weights and
activations are quantized to int8 mantissa + shared per-block exponent (block
size 32), enabling a genuine int8×int8→int32 matmul, JIT-compiled with Numba.
Validated against the fp32 reference (unit-tested relative-error bound) before
any speed benchmarking, with a measured quality delta (perplexity + side-by-side
generations) and a measured — not assumed — speed comparison against the frozen
Phase 1 baseline. KV-cache prefix reuse is Phase 3, and the combined
BFP + KV-cache pipeline is Phase 4a.

The kernel then went through an optimization pass driven by per-stage profiling
rather than guesswork: weights are quantized once at layer-construction time
instead of on every forward call, the kernel is `parallel=True` over the
output-feature axis (which also puts both paths on the same thread count for the
first time), the per-block scale is hoisted out of the inner loop, and the bias
is fused into the kernel's output store to avoid a torch op whose OpenMP threads
were contending with Numba's. That moved the short-prompt profile from ~38x more
expensive than fp32 to ~1.07x, and reversed the direction of the documented break
point — see below.

Phase 6 adds a **second quantization scheme** alongside BFP rather than
replacing it: plain symmetric int8 (`cachequant/kernel/int8*.py`), with one
exact `max_abs` scale per row — per output channel for a weight, per token for
an activation — instead of BFP's power-of-two scale per 32-value block. Both
schemes emit int8 mantissas and run on the *same* njit matmul
(`cachequant/kernel/blocked_matmul.py`), so the comparison isolates the
scaling policy and nothing else. int8 turns out to match BFP's quality while
running the kernel ~1.5x faster, for reasons the granularity/scale grid below
pulls apart — and to carry a reproducibility break point BFP does not. See
"Plain int8" below.

Phase 3 adds cross-request KV-cache prefix reuse (`cachequant/kvcache/`): a
token-level trie (`PrefixKVCache`) where each node is one token position
holding that position's per-layer K/V tensors, with LRU eviction capped by
`BenchConfig.max_cache_tokens`. `generate_with_prefix_cache` looks up the
longest cached prefix of an incoming prompt (capped at `prompt_len - 1`, since
predicting the next token always needs one fresh forward pass at the last
prompt position), reuses cached K/V for the matched prefix, recomputes only
the uncached suffix, and inserts the newly computed K/V back into the cache
for future requests. Validated for exact-prefix hits, partial-prefix hits,
cold misses, eviction under cache-size pressure, and hit-vs-recompute output
equivalence, with a measured hit-rate-vs-speedup benchmark comparing a
high-reuse prompt set (shared preamble) against a no-reuse prompt set
(independent prompts). The combined pipeline toggling quantization and
caching together is Phase 4a — see below.

## Setup

Requires Python 3.14 (the only version this has been run against).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Pinned versions: `torch==2.13.0`, `transformers==5.14.1`, `numpy==2.4.6`, `numba==0.66.0`, `pytest==9.1.1` (see `requirements.txt`). `numpy` was downgraded from `2.5.1` in Phase 1 because `numba==0.66.0` requires `numpy<2.5`.

## Run

### Tests

```bash
pytest -v                    # fast, no network or model download
pytest -m integration -v     # loads real GPT-2 weights (~500MB from Hugging Face on first run)
```

### Baseline benchmark

```bash
python scripts/run_baseline.py
```

Runs GPT-2 on CPU on two prompt profiles — a short decode-heavy prompt and a long prefill-heavy prompt — 5 times each after a warmup pass, and writes median/min/max tokens/sec, latency, and cost/1K tokens to `benchmarks/baseline_results.json` and `benchmarks/baseline_results_longprompt.json`.

Latest recorded numbers (short prompt, 50 generated tokens; measured on this repo's dev machine — see the `provenance` field in the JSON for the actual CPU, priced at the documented AWS `c7i.2xlarge` on-demand rate in `cachequant/bench/config.py`, not run on that instance):

- Prefill: ~216 tokens/sec
- Decode: ~41 tokens/sec
- Cost: ~$0.0024 / 1K generated tokens

```bash
python scripts/plot_baseline.py
```

Reads the existing `benchmarks/baseline_results*.json` (no re-run) and writes
`benchmarks/charts/baseline_throughput.png` (prefill/decode tok/s) and
`baseline_latency.png` (p50/p90/mean decode latency).

### Quantized kernel (BFP)

"Kernel" here means a low-level numeric compute routine (matmul loop),
unrelated to the OS kernel — `cachequant/kernel/bfp_numba.py` is a
`@njit`-compiled int8 matmul inner loop.

BFP (Block Floating Point) groups weight values into blocks of 32
(`DEFAULT_BLOCK_SIZE` in `cachequant/kernel/bfp.py`), replaces each value's
individual fp32 exponent with **one shared power-of-two exponent per block**
(sized to the block's max value), and stores each value as an int8 mantissa
relative to that shared scale:

```
fp32 (each number = own float, 32 bits):
  1.5   -0.7    0.2   -1.9
[S|E|MMM] [S|E|MMM] [S|E|MMM] [S|E|MMM]   <- 4 independent exponents

                    |
                    | pick block, find max_abs = 1.9
                    | exponent = ceil(log2(1.9)) = 1  ->  scale = 2^1 = 2
                    v

BFP block (1 shared exponent + int8 mantissas):
  shared exponent: 1   (scale = 2)

  value    mantissa = round(value / scale * 127)
  1.5   ->  round(1.5/2*127)  =  95
 -0.7   ->  round(-0.7/2*127) = -44
  0.2   ->  round(0.2/2*127)  =  13
 -1.9   ->  round(-1.9/2*127) = -121

dequantize:  value ≈ mantissa * scale / 127
  95  * 2/127 =  1.496   (vs original 1.5)
 -44  * 2/127 = -0.693   (vs original -0.7)
  13  * 2/127 =  0.205   (vs original 0.2)
-121  * 2/127 = -1.906   (vs original -1.9)
```

Fewer bytes moved from RAM per value (~1 shared-exponent bit + 8 mantissa bits
vs 32 full fp32 bits) is what makes decode faster — decode is
memory-bandwidth-bound. The precision loss above (1.5 -> 1.496) is the
quality cost measured below, and the custom loop needed to do this math (vs
hardware's native fp32 matmul path) is why prefill is *slower*, not faster —
see the break point below.

```bash
pytest tests/test_bfp.py tests/test_bfp_numba.py tests/test_bfp_linear.py -v   # validation, no network
python scripts/compare_bfp_quality.py     # fp32 vs BFP perplexity + generations
python scripts/run_bfp_benchmark.py       # BFP speed, compared against the recorded Phase 1 baseline, writes benchmarks/bfp_results*.json
python scripts/profile_kernel_breakdown.py  # per-stage timing attribution for BOTH schemes, writes benchmarks/kernel_breakdown.json
```

Latest recorded quality-delta numbers (`gpt2`, fp32 vs BFP-quantized, perplexity
on `eval/passages.py`):

- fp32 perplexity: ~39.86
- BFP perplexity: ~40.01 (+0.38%)
- Greedy generations on both eval prompts were token-identical between fp32 and BFP

Latest recorded speed numbers (median of 5 reps, vs the Phase 1 baseline above),
after the kernel optimization pass:

| | prefill tok/s | decode tok/s | cost / 1K tokens |
|---|---:|---:|---:|
| short prompt, fp32 baseline | 216.3 | 40.88 | $0.00244 |
| short prompt, BFP | 106.8 (49.4%) | 42.65 (104.3%) | $0.00241 (0.99x) |
| long prompt, fp32 baseline | 1190.3 | 36.28 | $0.00471 |
| long prompt, BFP | 190.7 (16.0%) | 36.97 (101.9%) | $0.01677 (3.56x) |

**Documented break point** — and it runs opposite to the design spec's
prediction. Decode *gains* 2-4% over fp32 throughput while prefill retains only
16-49%, because the two phases are limited by different resources: decode (M=1)
is memory-bandwidth-bound, where reading int8 mantissas instead of fp32 weights
is roughly a 4x reduction in bytes moved, and that more than offsets the
kernel's arithmetic disadvantage. Prefill (M=long) reuses each weight across
every token, so it is compute-bound, and a readable `@njit` triple loop cannot
follow MKL's register blocking and AVX kernels. Per-stage profiling backs this
up: at prefill the int8 kernel is 82-94% of layer time (so the gap is genuine
kernel quality), while at decode it is only ~50-65%, with activation
quantization and per-call overhead making up the rest.

Earlier recorded runs of this benchmark put BFP decode slightly *below* fp32
(97%) rather than slightly above (104%). Both readings are within a few percent
of parity and the gap is smaller than this benchmark's run-to-run spread — the
robust claim is "decode is roughly free under BFP", not a precise ratio. The
larger, stable effect is int8's, at 125% (see "Plain int8" below).

See `docs/superpowers/specs/2026-08-07-cachequant-design.md` for both prompt
profiles in full, the four optimizations and what each was worth, the per-stage
attribution table, and the pre-optimization numbers kept for comparison.

### Plain int8 (second quantization scheme)

```bash
pytest tests/test_int8.py tests/test_int8_numba.py tests/test_int8_linear.py -v   # validation, no network
python scripts/compare_quantization_quality.py   # fp32 vs BFP8 vs int8 perplexity, + the granularity/scale grid
python scripts/run_int8_benchmark.py             # int8 speed, same harness and prompts as the BFP benchmark
python scripts/profile_kernel_breakdown.py       # per-stage attribution for BOTH schemes side by side
```

BFP8 and plain int8 are **not two different matmuls**. Both quantize to int8
mantissas grouped along the reduction axis with one float scale per group, so
both run on the same `@njit` kernel — `cachequant/kernel/blocked_matmul.py`,
which `bfp_numba.py` and `int8_numba.py` each hold as `_KERNEL` (asserted in
`tests/test_int8_numba.py`). The schemes differ on two axes, and only two:

| | scale granularity | scale value |
|---|---|---|
| BFP8 | one per 32 values | `2**ceil(log2(max_abs))` |
| plain int8 | one per row — output channel for a weight, token for an activation | `max_abs`, exactly |

"One block spanning the whole reduction axis" is what plain int8 *is* here,
which is why the scheme needed ~45 lines of new logic and zero kernel work.

**Both differences matter, and they point in opposite directions.** BFP's
32-value blocking isolates outliers: one large value distorts its own block and
nothing else. But its power-of-two rounding throws range away — `max_abs=1.05`
rounds the scale up to `2`, so nothing in that block exceeds mantissa 67 and
close to a full bit of the int8 range goes unused. int8's exact scale always
lands the row maximum on +/-127, and pays for it by letting one outlier
coarsen its entire row.

On GPT-2 those two effects very nearly cancel. Filling in the two corners
neither scheme ships is what shows this — the shipped schemes are the
off-diagonal:

| granularity | scale | bits/value | perplexity | delta vs fp32 |
|---|---|---:|---:|---:|
| block-32 | pow2 — **BFP8 as shipped** | 8.250 | 40.0124 | +0.38% |
| block-32 | exact | 9.000 | 39.8264 | -0.08% |
| per-channel | pow2 | 8.010 | 40.9298 | +2.68% |
| per-channel | exact — **int8 as shipped** | 8.042 | 39.9944 | +0.34% |

Read down the columns rather than across: going from block-32 to per-channel
costs +2.30pp under a pow2 scale but only +0.42pp under an exact one, and
switching pow2 -> exact is worth -0.46pp at block-32 and -2.34pp at
per-channel. The rounding axis is the bigger of the two, and BFP is on the
wrong side of it. That is the whole reason a 24x coarser scheme ties it.

**Do not over-read the ranking.** `eval/passages.py` is *three passages*. A
0.04-percentage-point gap between BFP8 and int8 is well under what that eval
can resolve, and the -0.08% corner scoring *better* than fp32 is noise, not a
finding. Treat BFP8 and int8 as tied on quality; that is all the eval
supports. What it does support is the shape of the table — the per-tensor
rows below are 50-180x larger than the gap between the two shipped schemes.
A real ranking needs WikiText-2-scale evaluation, which is in TODO.md.

**Granularity does bite, one step coarser.** `apply_int8_quantization` takes
`per_tensor_weight` and `per_tensor_activation`; both exist as measured
comparison points and neither is a setting to reach for:

| scheme | perplexity | delta vs fp32 |
|---|---:|---:|
| int8, per-channel weight / per-token activation (default) | 39.9944 | +0.34% |
| int8, per-tensor weight / per-token activation | 47.5711 | +19.35% |
| int8, per-tensor weight and activation | 67.2031 | +68.60% |

The last row degenerates visibly, not just numerically — it generates *"she
heard a voice in the distance, and she heard a voice in the distance, and..."*.
This is the outlier problem the LLM.int8/SmoothQuant line of work exists for,
reproduced at the smallest scale that shows it.

**Greedy output is where the two schemes actually separate.** BFP-quantized
greedy generation is *token-identical* to fp32 on both eval prompts. int8's is
not — it stays coherent but diverges. Perplexity calls them tied; a
side-by-side generation does not. If matching fp32 token-for-token matters,
that is the number to look at, not the 0.04pp.

int8 is also **structurally more general**: `BFPConv1D` raises `ValueError`
unless the reduction axis divides evenly by 32; `Int8Conv1D` has no block to
divide, so every shape is legal.

**Where the speed comes from.** Both schemes run the same kernel on the same
shapes, so `benchmarks/kernel_breakdown.json` isolates exactly what the block
structure costs — BFP does `k/32` float scale multiplies per output element
(24 to 96 at GPT-2 shapes), int8 does one:

| phase | layer | BFP8 kernel | int8 kernel | speedup |
|---|---|---:|---:|---:|
| decode (M=1) | attn.c_attn | 0.106 ms | 0.076 ms | 1.39x |
| decode | attn.c_proj | 0.049 ms | 0.039 ms | 1.25x |
| decode | mlp.c_fc | 0.130 ms | 0.093 ms | 1.39x |
| decode | mlp.c_proj | 0.142 ms | 0.100 ms | 1.42x |
| prefill (M=270) | attn.c_attn | 21.293 ms | 14.470 ms | 1.47x |
| prefill | attn.c_proj | 7.223 ms | 4.955 ms | 1.46x |
| prefill | mlp.c_fc | 28.372 ms | 19.253 ms | 1.47x |
| prefill | mlp.c_proj | 28.261 ms | 18.760 ms | 1.51x |

Activation quantization gets cheaper for the same reason — one max-reduction
per row instead of one per 32-value block. Summed over the three 768-wide
layers: 2.694 ms -> 1.476 ms at prefill, 0.077 ms -> 0.052 ms at decode.

**End to end**, median of 5 reps, all four formats re-measured in the same
session on an otherwise idle machine (percentages against the fp32 row of the
same table):

| short prompt, 50 tokens | prefill tok/s | decode tok/s | cost / 1K tokens |
|---|---:|---:|---:|
| fp32 | 216.3 | 40.88 | $0.00244 |
| fp16 | 7.2 (3.3%) | 5.50 (13.5%) | $0.01960 (8.03x) |
| BFP8 | 106.8 (49.4%) | 42.65 (104.3%) | $0.00241 (0.99x) |
| **int8** | **159.8 (73.9%)** | **51.19 (125.2%)** | **$0.00198 (0.81x)** |

| long prompt (270 tok), 10 tokens | prefill tok/s | decode tok/s | cost / 1K tokens |
|---|---:|---:|---:|
| fp32 | 1190.3 | 36.28 | $0.00471 |
| fp16 | 8.6 (0.7%) | 5.47 (15.1%) | $0.32348 (68.68x) |
| BFP8 | 190.7 (16.0%) | 36.97 (101.9%) | $0.01677 (3.56x) |
| **int8** | **268.0 (22.5%)** | **43.49 (119.9%)** | **$0.01203 (2.55x)** |

**This revises the Phase 2 break point in one direction and confirms it in the
other.** Confirmed: prefill is compute-bound, a scalar `@njit` triple loop
cannot follow MKL, and both quantized schemes lose there (int8 by less, at
74%/23% against BFP's 49%/16%). Revised: decode does not merely *survive*
quantization, it *improves* — 104% of fp32 for BFP8 and 125% for int8. That is
the memory-bandwidth argument finally paying out, because decode (M=1) is a
matrix-vector product limited by how many weight bytes must be read, and int8
mantissas are ~4x smaller than fp32 weights.

int8 is consequently **cheaper than fp32 per 1K generated tokens on the
decode-heavy short profile** ($0.00198 vs $0.00244, 0.81x) — the first
configuration in this project to beat the fp32 baseline on cost rather than
merely approach it. On the prefill-heavy long profile it still loses (2.55x),
because there the slow prefill dominates the wall clock.

Two caveats before leaning on these. First, quantized decode throughput is
much noisier than fp32's — see Limitations. Second, earlier recorded runs of
this same BFP benchmark put decode at 97% of fp32 rather than 104%; the
direction of the newer result is stable across repeat runs on a quiet machine,
but the margin over fp32 is small enough that it should not be quoted as a
precise ratio.


### fp16 reference point

```bash
python scripts/run_fp16_benchmark.py   # writes benchmarks/fp16_results*.json
```

fp16 is **not one of this project's optimizations** — no kernel is written for
it. It is a second *baseline* alongside fp32: the free, one-line alternative
(`model.half()`) any real deployment reaches for before writing a custom
quantized kernel. It is here to answer "why not just use fp16?" with a
measurement, and to separate the two effects that are otherwise confounded in
the BFP decode result — fewer bytes moved per weight vs. arithmetic this CPU
can execute natively.

Latest recorded numbers (median of 5 reps; percentages against the fp32 rows of
the same JSON files, re-measured in the same session as the fp16 run, hence
slightly above the Phase 1 table above — run-to-run variance of a few percent,
immaterial at this effect size):

| | prefill tok/s | decode tok/s | cost / 1K tokens |
|---|---:|---:|---:|
| short prompt, fp32 | 216.3 | 40.88 | $0.00244 |
| short prompt, fp16 | 7.2 (3.3%) | 5.50 (13.5%) | $0.01960 (8.03x) |
| long prompt, fp32 | 1190.3 | 36.28 | $0.00471 |
| long prompt, fp16 | 8.6 (0.7%) | 5.47 (15.1%) | $0.32348 (68.68x) |

Greedy generation was **token-identical to fp32** on the short prompt, so this
is a pure speed result — fp16 costs no measurable quality here, and loses
anyway.

**Second documented break point, and it settles the BFP question.** fp16 moves
half the bytes of fp32 per weight (2 vs 4). If "fewer bytes moved" were
sufficient to win the memory-bandwidth-bound decode phase, fp16 should have
*beaten* fp32 at decode. It is 6.6-7.4x slower instead, and 30-138x slower at
prefill, because this project's pinned CPU (Intel i7-8700K, Coffee Lake, no
AVX-512 and no native fp16 compute path — see `provenance` in the JSON) has no
hardware that multiplies fp16 directly. Every op unpacks to fp32, does the
math, and repacks, and that overhead dwarfs the bandwidth saving.

That makes the BFP decode result sharper than it first reads. BFP retains
78-91% of fp32 decode throughput not merely because int8 mantissas are smaller,
but because int8 multiply-accumulate is an operation this CPU executes
natively — so the bandwidth saving is actually collectable. Lower precision on
its own buys nothing; lower precision *the hardware can compute in* is what
buys something. The same reasoning is why fp8 is a dead end on this machine and
why both formats behave differently on GPUs with dedicated fp16/fp8 tensor
cores.

**Why not fp16 / bfp16 / fp8 on this machine:**

`/proc/cpuinfo` on the pinned i7-8700K shows `f16c` (fp16↔fp32 *conversion*
only) and `avx2`, but no `avx512` and no native fp16 or fp8 arithmetic unit.
AVX2 *does* have a native int16 multiply-accumulate (`VPMADDWD`), so the
three formats are ruled out for two different reasons, not one:

- **fp16** — no native float16 ALU. Every op pays the unpack/compute/repack
  tax measured above; that's a hardware wall.
- **fp8** — no native fp8 ALU on *any* CPU. fp8 tensor cores are GPU-only
  (e.g. Hopper, Blackwell); not implementable here at all. Also a hardware
  wall.
- **bfp16** (16-bit-mantissa BFP, an int mantissa — not to be confused with
  `bf16`/bfloat16, an IEEE-style float format) — this one is *not*
  hardware-blocked, since int16 MAC is native. It's a diminishing-returns
  problem instead: 16 mantissa bits + shared exponent works out to ~16.25
  bits/value, only ~1.97x smaller than fp32 — roughly fp16's compression
  ratio, which already lost on this memory-bandwidth-bound workload. int8's
  8.25 bits/value (3.88x smaller) is the sweet spot: half the mantissa
  width, still native hardware, and no precision need this project has
  documented that would justify going wider.

int8 is what BFP targets here because it's both natively executable *and*
gives the largest bandwidth win of the options that are.

### KV-cache prefix reuse

```bash
pytest tests/test_trie_cache.py -v                        # trie unit tests, no network
pytest tests/test_kvcache_generate.py -v -m integration    # correctness tests against real GPT-2
python scripts/run_kvcache_benchmark.py    # hit rate vs. prefill speedup, writes benchmarks/kvcache_results.json
```

See `docs/superpowers/specs/2026-08-07-cachequant-design.md` for the recorded
hit-rate/speedup numbers and the documented break points (prefix caching only
helps prefill, never decode; the final prompt token is always freshly
computed, capping max hit rate at `(N-1)/N` for an N-token prompt).

The three prompt sets here are all **one-shot**: independent prompts sharing a
fixed ~12-token preamble. That is too little reuse for the cache to win — see
the math below and the multi-turn workload that fixes it.

### Multi-turn chat

```bash
pytest tests/test_workloads.py -v                       # transcript builder, no network
pytest tests/test_multiturn_benchmark.py -v -m integration   # reuse-grows + output-matches, real GPT-2
python scripts/run_multiturn_benchmark.py   # prefill cost per turn, writes benchmarks/multiturn_results.json
```

A one-shot request sends an independent prompt. A **multi-turn** request sends
the whole transcript so far — system prompt, every prior user line, and the
model's own prior replies — plus one new user line
(`cachequant/workloads.py`). The transcript grows every turn, so the prefix the
cache can reuse grows with it.

**Why one-shot can't show the cache working.** Let $p$ = prompt tokens, $h$ =
hit rate, $w$ = marginal per-token prefill cost, $O$ = fixed cache
lookup + insert cost. The cache is a net win only when the prefill it skips
beats its own overhead:

$$h \cdot p \cdot w \;>\; O$$

On the short one-shot sets $p \approx 24$, $h \approx 0.7$, and with
$w \approx 0.75\ \text{ms}$ the saving $h p w \approx 13\ \text{ms}$ barely
clears $O \approx 5\ \text{ms}$ — inside the run-to-run noise, so
`honest_prefill_speedup` sits at 0.96–1.0.

**Why multi-turn does.** Turn $k$ adds $\approx L$ tokens (user line + reply),
so $p_k \approx kL$, reused $c_k \approx (k-1)L$, and

$$h_k \;\approx\; \frac{k-1}{k} \;\longrightarrow\; 1$$

Taking prefill cost per token $\approx b \cdot (\text{tokens attended})$, the
total prefill work over $K$ turns is

$$T_{\text{no cache}} \;\approx\; \sum_{k=1}^{K} b\,(kL)^2 \;\approx\; \frac{b\,L^2 K^3}{3}
\qquad
T_{\text{cache}} \;\approx\; \sum_{k=1}^{K} b\,L\,(kL) \;\approx\; \frac{b\,L^2 K^2}{2}$$

so the cache turns $O(K^3)$ prefill work into $O(K^2)$. The benchmark draws
exactly this: a no-cache prefill-per-turn line that curves upward, a cache-on
line that stays roughly flat, the gap widening with turn number
(`benchmarks/charts/multiturn_prefill_by_turn.png`).

Latest recorded numbers (median of 5 reps; `honest_prefill_speedup` =
no-cache prefill / cache-on prefill with `lookup()` + `insert()` overhead
included; 8-turn conversations, `MAX_NEW_TOKENS = 12`):

| turn | prompt tok | no-cache prefill | cache-on honest | hit rate | speedup |
|---:|---:|---:|---:|---:|---:|
| 0 (cold) | 25 | 38 ms | 44 ms | 0.00 | 0.86x |
| 1 | 49 | 55 ms | 45 ms | 0.53 | 1.22x |
| 3 | 103 | 100 ms | 50 ms | 0.75 | 2.02x |
| 5 | 153 | 120 ms | 53 ms | 0.84 | 2.30x |
| 7 | 208 | 162 ms | 54 ms | 0.87 | 2.96x |

(short system prompt; `france` conversation shown, `python` matches within a few
percent). With the ~280-token `LONG_PROMPT` as the system prompt the reused
prefix dominates from turn 1, so hit rate jumps to ~0.92 immediately and the
speedup runs **4.3x at turn 1 to 5.8x at turn 7**.

Turn 0 loses (0.86x) — cold cache, zero reuse, pays the `insert()` cost for
nothing. That is the one-shot regime. Every turn after it wins, and the win
grows because no-cache prefill keeps climbing (38 ms → 162 ms) while cache-on
stays flat (~50 ms): the cache always re-reads just the newest turn.

The one-shot sets stay — this workload runs alongside them, not instead. See
`docs/superpowers/specs/2026-08-26-phase7-multiturn-chat.md`.

### Combined pipeline

```bash
pytest tests/test_pipeline.py -v -m integration   # BFP+cache equivalence, 4-combo smoke test
python scripts/run_combined_benchmark.py           # all 6 toggle states x 3 prompt sets, writes benchmarks/combined_results.json
```

`cachequant/pipeline.py` adds a single `generate(model, tokenizer, prompt,
cache=None, max_new_tokens=50)` entry point covering all four BFP x
prefix-cache toggle states: BFP is selected by whether `model` was passed
through `apply_bfp_quantization`, caching by whether a `PrefixKVCache` is
passed as `cache`. No new generation logic — the two Phase 2/3 paths were
already orthogonal (BFP only replaces `Conv1D` linears; the cache never
inspects which Conv1D produced its K/V tensors), so this is a thin dispatcher
plus a correctness test for the one previously-untested combination (BFP
model + cache hit together) and a benchmark across all four states.

Latest recorded numbers (median of 5 reps, `total_honest_prefill_seconds`
summed across each 5-prompt set — includes `cache.lookup()`/`cache.insert()`
overhead for the cache-on combos, not just the internal forward-pass timing;
see Limitations below):

| total honest prefill (s) | fp32 | BFP8 | int8 |
|---|---:|---:|---:|
| high_reuse, no cache | 0.175 | 0.973 | 0.597 |
| high_reuse, cache | 0.181 | 0.474 | 0.343 |
| long_high_reuse, no cache | 1.051 | 7.361 | 5.269 |
| long_high_reuse, cache | 0.442 | 1.871 | 1.368 |
| no_reuse, no cache | 0.163 | 0.372 | 0.275 |
| no_reuse, cache | 0.182 | 0.399 | 0.251 |

Hit rates are identical across all three schemes (0.565 / 0.777 / 0.022),
confirming cache behaviour is quantization-independent.

**The cache is worth more the slower the prefill path is.** At the same 0.565
hit rate it buys fp32 nothing (0.97x — a small net loss), int8 1.74x, and BFP8
2.05x; at 0.777 the figures are 2.38x / 3.85x / 3.94x. The cache's fixed
lookup/insert cost is a large fraction of fp32's cheap prefill and a rounding
error against a quantized one, so the *relative* win grows as the underlying
path gets slower. That is a statement about the baseline being slow, not about
the cache being clever.

**Caching still does not rescue quantized prefill.** Neither `bfp_cache` nor
`int8_cache` ever beats `fp32_cache`: int8 loses by 1.89x on `high_reuse`
(0.343s vs 0.181s), 3.10x on `long_high_reuse` (1.368s vs 0.442s), and 1.38x
on `no_reuse` (0.251s vs 0.182s). int8 roughly halves BFP's cache-on deficit
(which is 2.62x / 4.23x / 2.20x on the same sets) without closing it — the
kernel-quality gap against MKL is larger than what a cache hit can claw back.

`fp32_cache` is also not the fastest combination in every scenario: it only
beats uncached `fp32_no_cache` on `long_high_reuse`, where there is real prefix
overlap to exploit (0.442s vs 1.051s, a 2.38x win). On `high_reuse` and
`no_reuse` the honest lookup/insert overhead outweighs what little prefill it
saves — consistent with the ~9% no-reuse cache tax in Limitations below.

One row not to read too closely: `int8_cache` on `no_reuse` records the lowest
*total* wall time of any combination on that set, but only because its decode
segment came in ~20% below every other row in the table while decode should be
cache-independent. That is noise in a single 5-rep sample, not a result.

### Dashboard

```bash
streamlit run cachequant/dashboard/app.py --server.address 127.0.0.1
```

`cachequant/dashboard/app.py` is a live Streamlit demo over `pipeline.generate`:
sidebar toggles for quantization (fp32/BFP/int8) and prefix-cache (on/off), a
prompt-set picker (the same three sets as the combined benchmark), and a
Generate button that runs the selected combo *and* the fp32/no-cache
baseline live, every click — both timed end-to-end (not just the internal
forward-pass timing `GenerationTiming` captures), so the on-screen cache
overhead and cost numbers can't be inflated by hiding `PrefixKVCache`
lookup/insert cost the way a naive `GenerationTiming`-only timer would. Each
scheme gets its own cache (quantization changes what K/V values get cached, so
one cache cannot correctly serve two schemes). Prefill and
decode throughput are shown as separate metrics rather than as one aggregate
number, since that split is what makes Phase 2's BFP break point (hurts
compute-bound prefill far more than it hurts memory-bandwidth-bound
matrix-vector decode) visible during a live demo instead of averaged away.

A **Workload** toggle switches the one-shot flow above for **multi-turn chat**:
type a message, hit Send turn, and each turn re-runs the growing transcript
with and without the cache on the selected scheme. A bar chart of prefill
seconds per turn makes the no-cache line climb while the cache-on bar stays
flat — the Phase 7 result, live. "Reset chat" clears the transcript and the
chat's cache.

This is demo software for a live presentation, not a tested/production
surface — see `docs/superpowers/specs/2026-08-14-phase4b-dashboard-design.md`
for the full design and what was deliberately left out (real token
streaming, a continuous prompt-overlap slider, automated dashboard tests).

Phase 5 makes the project demo-ready: `scripts/stress_test.py` runs
pathological inputs (empty/whitespace prompts, prompts at and past GPT-2's
1024-token limit, `max_new_tokens=0` and 200, cache-eviction pressure)
across all six toggle combos and records what actually happens — see
Break points below. `scripts/export_slide_charts.py` re-runs every
benchmark fresh and exports slide-ready charts and tables to
`benchmarks/charts/`. The dashboard gets an offline replay mode
(`cachequant/dashboard/app.py`, fixture captured by
`scripts/capture_replay_fixture.py`) that streams pre-captured runs with no
model load and no live compute — the actual fallback if the live demo
breaks during the talk. See
`docs/superpowers/specs/2026-08-19-phase5-demo-readiness-design.md` for the
full design.

### Chart export

```bash
python scripts/export_slide_charts.py                 # re-run every benchmark, then plot
python scripts/export_slide_charts.py --charts-only   # re-plot from existing benchmarks/*.json
```

The default re-runs `run_baseline.py`, `run_bfp_benchmark.py`,
`run_int8_benchmark.py`, `run_fp16_benchmark.py`,
`compare_quantization_quality.py`, `run_kvcache_benchmark.py` and
`run_combined_benchmark.py` fresh (takes several minutes), then writes
`benchmarks/charts/*.png` and `benchmarks/charts/summary.md`. Each benchmark
runs in its own subprocess, so a later one never measures a machine an earlier
one filled.

Charts written:

| file | question it answers |
|---|---|
| `quantization_breakpoints.png` | what each of the four formats costs, split by prefill vs decode |
| `quantization_quality_vs_speed.png` | perplexity cost against decode throughput |
| `quantization_granularity_grid.png` | which axis — scale granularity or scale rounding — the quality actually sits on |
| `kernel_scheme_comparison.png` | per-layer kernel time, BFP8 vs int8, on the shared njit kernel |
| `cache_hit_rate_vs_speedup.png` | KV-cache hit rate vs honest prefill speedup |
| `combined_comparison.png` | 3 schemes x cache on/off x 3 prompt sets |

`summary.md` is the table view for all of them — several palette slots fall
below 3:1 contrast on a white surface, so every chart also carries direct
value labels and the numbers are readable without relying on colour.

**fp16 is deliberately excluded from `combined_comparison.png`**, though it
appears in the two charts above it. Two reasons:

- *Presentational.* fp16 prefill runs at ~7 tok/s against fp32's ~230 on this
  machine, so on a shared linear axis its bars flatten every other bar into a
  sliver. A log axis would fix that, but not the second reason.
- *Substantive.* fp16 is slow here because this CPU has **no native fp16
  ALU** (see "Why not fp16 / bfp16 / fp8" above), not because of anything
  about precision or about caching. Its cache speedup would therefore be the
  largest bar on a chart whose headline is "the cache is worth more the slower
  the prefill path is" — while carrying no information about caching at all,
  because the work the cache skips is artificially expensive on this hardware
  only. Including it would make the chart's claim look better evidenced than
  it is.

fp16 stays in `quantization_breakpoints.png` and
`quantization_quality_vs_speed.png`, where that hardware wall is exactly the
point being made.

### Offline replay mode (demo fallback)

```bash
python scripts/capture_replay_fixture.py   # regenerate the fixture after any pipeline change
streamlit run cachequant/dashboard/app.py --server.address 127.0.0.1
```

Toggle **Replay (offline)** in the sidebar before clicking Generate. This
path never loads a model or calls the real pipeline — though opening the
dashboard still triggers one live model load on the very first page render,
before you can touch the toggle (Streamlit widgets start at their declared
default on the first run), and flipping it only skips model loading and live
compute from that point forward. It streams
pre-captured runs from `cachequant/dashboard/fixtures/replay.json`, so it
cannot fail for any of the reasons the live path can (model download, JIT
compile stall, slow CPU). This is the fallback for a live demo failure, not
a separate feature.

### Stress test

```bash
python scripts/stress_test.py
```

Runs pathological inputs across all six toggle combos and writes
`benchmarks/stress_test_results.json`. Findings are documented, not fixed
— see Break points below.

## Limitations (Phase 1 + 2 + 3 + 4a + 4b + 5 + 6 + 7)

- CPU only, batch size 1 (enforced by an assertion in `generate_with_timing`) — no GPU path, no batching.
- The BFP kernel is inference-only (no autograd/backward pass) and requires the
  reduction axis to be evenly divisible by the block size (32) — true for every
  GPT-2 small linear layer in scope. `BFPConv1D` validates this at construction
  and raises `ValueError` immediately for a non-divisible reduction axis.
- The kernel is still a scalar triple loop — no register blocking, explicit SIMD,
  or cache tiling. That is what the remaining ~11x prefill gap against BLAS is.
- `BFPConv1D` quantizes its weight at construction and holds the result as NumPy,
  so it is CPU-only by design and does not follow `.to(device)`. Consistent with
  the project's no-GPU-path non-goal, but it is not a general-purpose module.
- Both quantized paths' decode throughput is noticeably more sensitive to
  scheduling noise than the fp32 baseline, and the spread is wide enough to
  swamp real effects: repeated BFP runs on the same machine have produced
  medians from 26.6 to 37.1 tok/s, and single 5-rep runs have spanned
  10.4-35.2 tok/s internally. N=5 is thin for that. Treat the
  scheme-vs-scheme *ratios* measured in one session as the signal and absolute
  numbers across sessions as approximate — and see
  `benchmarks/kernel_breakdown.json`, which times the kernel directly and is
  far steadier than the end-to-end number.
- `Int8Conv1D` has the same CPU-only, NumPy-held-weight, inference-only
  constraints as `BFPConv1D`, minus the block-divisibility one.
- **int8 output is not reproducible across prefill chunkings.** An exact
  `max_abs` scale is a continuous function of the activation, so the ~1e-5
  differences torch's fp32 matmuls produce between one long prefill and a
  chunked one move the scale and shift every mantissa in the row. Measured on
  an 11-token prompt split 9+2: BFP's logits move by 5.3e-05 and int8's by
  9.3e-01, enough to change generated tokens. Since a cache hit *is* a chunked
  prefill, an int8 answer served from cache can differ from the same prompt
  served cold. This is reproducibility, not quality — int8 perplexity is
  within 0.04pp of BFP either way — but it is the one respect in which BFP's
  wasteful power-of-two rounding is the better engineering choice.
- The prefix cache's eviction scan (`PrefixKVCache._lru_leaf`) walks the
  entire trie on every evicted node — an O(n) scan acceptable at this
  project's demo scale (a few thousand cached tokens), not built for a large
  production cache.
- The cache never serves the final prompt token from a lookup — next-token
  logits require a fresh forward pass at that position — so an exactly
  repeated N-token prompt has a maximum possible hit rate of `(N-1)/N`, not
  1.0.
- `max_cache_tokens` is a soft cap: if a single prompt's uncached token count
  alone exceeds it, the cache temporarily holds more than the configured cap
  rather than raising or truncating (eviction only reclaims *existing*
  entries, and stops once there's nothing left to evict).
- Enabling the cache costs ~9% on a workload with no prefix reuse (measured
  0.908x on the no-reuse prompt set): every request still pays the failed
  lookup, the K/V copy, and an insert nothing later reads. The cache is a bet
  on the workload — roughly 1.5-2x when a long prefix is shared, a ~9% tax
  when it isn't.
- The cache stores K/V as 24 small tensors per token (one per layer per K/V),
  so resident cost is ~86KB/token — 1.19x the 72KB/token of raw tensor data,
  or ~172MB at the 2048-token cap. `lookup` also materializes the matched
  prefix into a fresh contiguous tensor on every request (~20MB, ~8ms for a
  276-token prefix). Block-granular storage with a block table, rather than
  per-token nodes, is what production caches use to avoid both costs.
- The dashboard (`cachequant/dashboard/app.py`) is demo software: no
  automated tests, no input validation, single-file, single-operator. Its
  simulated token stream replays an already-complete generation rather than
  showing tokens as they're actually computed (true streaming would need an
  `on_token` callback threaded through the core generation functions, which
  this pass didn't add).
- Benchmark numbers are from one dev machine; re-run `scripts/run_baseline.py` locally before trusting exact figures on different hardware.
- `max_new_tokens=0` isn't validated — the loop still emits one token in that case.
- Only tested against Python 3.14.

## Break points / Q&A

Two break points are already documented above with numbers; the rest come
from `scripts/stress_test.py` running pathological inputs across all four
toggle combos and recording what actually happens
(`benchmarks/stress_test_results.json`, 26 runs, this repo's dev machine).

- **Quantization**: the BFP kernel costs prefill throughput far more than
  decode — compute-bound prefill retains only 16-49% of fp32 speed, while
  memory-bandwidth-bound decode reaches 102-125% of it — see the numbers above.
- **Caching**: prefix-only — the final prompt token is always a fresh
  forward pass, capping hit rate at `(N-1)/N` — and costs ~9% throughput on
  a no-reuse workload — see the KV-cache numbers above.
- **Empty prompts** (`empty_prompt`, 0 tokens): fails identically on all 6
  combos — `RuntimeError: cannot reshape tensor of 0 elements into shape
  [-1, 0] because the unspecified dimension size -1 can be any value and is
  ambiguous`.
- **Whitespace-only prompts** (`whitespace_prompt`, 3 tokens after
  tokenization): succeeds on all 6 combos (~0.21-0.26s for 10 requested
  tokens) — whitespace is not rejected as empty.
- **Prompts at/past GPT-2's 1024-token context limit** (`at_context_prompt`
  = 1024 tokens exactly, `over_context_prompt` = 1100 tokens): both fail on
  all 6 combos with the same `IndexError: index out of range in self`, but
  not at the same cost — the 1100-token case fails near-instantly
  (~0.002-0.004s, every combo), while the 1024-token case only fails after
  a full prefill's worth of compute, and that prefill is markedly more
  expensive under BFP (~5.0s) than fp32 (~0.87-0.91s).
- **`max_new_tokens=0`** (`zero_max_new_tokens`): succeeds on all 6 combos
  (~0.03-0.06s) — consistent with the Limitations note above that this
  input isn't validated.
- **`max_new_tokens=200`** (`large_max_new_tokens`): succeeds on all 6
  combos (~5.3-5.6s).
- **Cache-eviction pressure** (`cache_eviction_pressure`, only meaningful
  on the two cache-on combos): succeeds on both — `fp32_cache` in ~3.0s,
  `bfp_cache` in ~8.4s. Eviction under pressure doesn't crash; it's just
  slower under BFP, consistent with the quantization break point above.

## Docs

Design spec and implementation plans live in `docs/` — a separate, local-only git repo (not part of this project's remote).
