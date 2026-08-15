# CacheQuant

Custom BFP-quantized kernels + KV-cache prefix reuse for faster, cheaper LLM inference. Benchmarked and visualized live.

## What's built so far (Phase 1 + 2 + 3 + 4a + 4b)

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
expensive than fp32 to ~1.2x, and reversed the direction of the documented break
point — see below.

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

### Quantized kernel (BFP)

```bash
pytest tests/test_bfp.py tests/test_bfp_numba.py tests/test_bfp_linear.py -v   # validation, no network
python scripts/compare_bfp_quality.py     # fp32 vs BFP perplexity + generations
python scripts/run_bfp_benchmark.py       # BFP speed, compared against the recorded Phase 1 baseline, writes benchmarks/bfp_results*.json
python scripts/profile_bfp_breakdown.py   # per-stage timing attribution, writes benchmarks/bfp_breakdown.json
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
| short prompt, fp32 baseline | 216.2 | 40.8 | $0.00245 |
| short prompt, BFP | 94.7 (43.8%) | 34.6 (84.9%) | $0.00295 (1.21x) |
| long prompt, fp32 baseline | 1203.4 | 36.3 | $0.00469 |
| long prompt, BFP | 188.0 (15.6%) | 30.0 (82.7%) | $0.01721 (3.67x) |

**Documented break point** — and it runs opposite to the design spec's
prediction. Decode retains 83-85% of fp32 throughput while prefill retains only
16-44%, because the two phases are limited by different resources: decode (M=1)
is memory-bandwidth-bound, where reading int8 mantissas instead of fp32 weights
is roughly a 4x reduction in bytes moved, and that nearly offsets the kernel's
arithmetic disadvantage. Prefill (M=long) reuses each weight across every token,
so it is compute-bound, and a readable `@njit` triple loop cannot follow MKL's
register blocking and AVX kernels. Per-stage profiling backs this up: at prefill
the int8 kernel is 82-94% of layer time (so the gap is genuine kernel quality),
while at decode it is only ~50-65%, with activation quantization and per-call
overhead making up the rest.

See `docs/superpowers/specs/2026-08-07-cachequant-design.md` for both prompt
profiles in full, the four optimizations and what each was worth, the per-stage
attribution table, and the pre-optimization numbers kept for comparison.

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

### Combined pipeline

```bash
pytest tests/test_pipeline.py -v -m integration   # BFP+cache equivalence, 4-combo smoke test
python scripts/run_combined_benchmark.py           # all 4 toggle states x 3 prompt sets, writes benchmarks/combined_results.json
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
see Limitations below) show caching does not rescue BFP's prefill slowdown:
`bfp_cache` never beats `fp32_cache`, losing by 2.44x on `high_reuse` (0.472s
vs 0.194s), 3.84x on `long_high_reuse` (1.696s vs 0.442s), and 2.07x on
`no_reuse` (0.383s vs 0.185s), even though hit rates are identical between
the two (0.565 / 0.777 / 0.022) — confirming cache behavior is
quantization-independent. But `fp32_cache` is not the fastest combination
overall in every scenario: it only beats the uncached `fp32_no_cache`
baseline on `long_high_reuse`, where there's real prefix overlap to exploit
(0.442s vs 1.074s, a 2.43x win). On `high_reuse` and `no_reuse`, the honest
lookup/insert overhead outweighs what little prefill it saves, so plain
`fp32_no_cache` is actually fastest overall (0.186s vs 0.194s on
`high_reuse`; 0.161s vs 0.185s on `no_reuse`) — consistent with the ~9%
no-reuse cache tax documented in Limitations below. BFP's kernel-quality gap
(documented above) is larger than what a cache hit can claw back regardless.

### Dashboard

```bash
streamlit run cachequant/dashboard/app.py --server.address 127.0.0.1
```

`cachequant/dashboard/app.py` is a live Streamlit demo over `pipeline.generate`:
sidebar toggles for quantization (fp32/BFP) and prefix-cache (on/off), a
prompt-set picker (the same three sets as the combined benchmark), and a
Generate button that runs the selected combo *and* the fp32/no-cache
baseline live, every click — both timed end-to-end (not just the internal
forward-pass timing `GenerationTiming` captures), so the on-screen cache
overhead and cost numbers can't be inflated by hiding `PrefixKVCache`
lookup/insert cost the way a naive `GenerationTiming`-only timer would. The
BFP and fp32 caches are kept separate (BFP quantization changes what K/V
values get cached, so one cache can't correctly serve both). Prefill and
decode throughput are shown as separate metrics rather than as one aggregate
number, since that split is what makes Phase 2's BFP break point (helps
FLOP-bound prefill, hurts matrix-vector decode) visible during a live demo
instead of averaged away.

This is demo software for a live presentation, not a tested/production
surface — see `docs/superpowers/specs/2026-08-14-phase4b-dashboard-design.md`
for the full design and what was deliberately left out (real token
streaming, a continuous prompt-overlap slider, automated dashboard tests).

## Limitations (Phase 1 + 2 + 3 + 4a + 4b)

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
- BFP decode throughput is noticeably more sensitive to scheduling noise than the
  fp32 baseline (31.3-40.0 vs 40.7-40.9 tok/s across 5 reps); N=5 is thin for a
  spread that wide.
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

## Docs

Design spec and implementation plans live in `docs/` — a separate, local-only git repo (not part of this project's remote).
