# CacheQuant

Custom BFP-quantized kernels + KV-cache prefix reuse for faster, cheaper LLM inference. Benchmarked and visualized live.

## What's built so far (Phase 1 + 2)

A clean GPT-2 small (124M) generation loop on CPU, with prefill and decode timed separately, and a benchmark harness that turns those timings into tokens/sec, latency, and cost per 1K generated tokens. This is the frozen baseline every later optimization (quantized kernel, KV-cache reuse) gets compared against. Phase 1 has no quantization or caching — every call ran full fp32 GPT-2 from scratch.

Phase 2 adds a hand-written block floating-point (BFP) quantized matmul kernel
(`cachequant/kernel/`), applied to GPT-2's attention QKV/output projections and
both FFN linears (embeddings and `lm_head` stay fp32). Both weights and
activations are quantized to int8 mantissa + shared per-block exponent (block
size 32), enabling a genuine int8×int8→int32 matmul, JIT-compiled with Numba.
Validated against the fp32 reference (unit-tested relative-error bound) before
any speed benchmarking, with a measured quality delta (perplexity + side-by-side
generations) and a measured — not assumed — speed comparison against the frozen
Phase 1 baseline. KV-cache prefix reuse and the combined dashboard are still
Phase 3/4.

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
```

Latest recorded quality-delta numbers (`gpt2`, fp32 vs BFP-quantized, perplexity
on `eval/passages.py`):

- fp32 perplexity: ~39.86
- BFP perplexity: ~39.96 (+0.26%)
- Greedy generations on both eval prompts were token-identical between fp32 and BFP

Latest recorded speed numbers (median of 5 reps, short prompt / 50 generated
tokens, vs the Phase 1 baseline above): BFP prefill ~6.3 tokens/sec and decode
~1.1 tokens/sec, roughly 35-38x more expensive per 1K tokens than fp32
(~$0.094 vs ~$0.0024). This is a correctness/measurement demonstration, not a
speed win — see `docs/superpowers/specs/2026-08-07-cachequant-design.md` for
the full numbers (both prompt profiles) and the documented break point (the
unoptimized kernel's per-call overhead, not a clean decode-vs-prefill split,
dominates the measured slowdown).

### KV-cache prefix reuse

Not implemented yet — Phase 3.

### Dashboard

Not implemented yet — Phase 4.

## Limitations (Phase 1 + 2)

- CPU only, batch size 1 (enforced by an assertion in `generate_with_timing`) — no GPU path, no batching.
- The BFP kernel is inference-only (no autograd/backward pass) and requires the
  reduction axis to be evenly divisible by the block size (32) — true for every
  GPT-2 small linear layer in scope, not asserted for arbitrary shapes.
- No KV-cache reuse yet.
- Benchmark numbers are from one dev machine; re-run `scripts/run_baseline.py` locally before trusting exact figures on different hardware.
- `max_new_tokens=0` isn't validated — the loop still emits one token in that case.
- Only tested against Python 3.14.

## Docs

Design spec and implementation plans live in `docs/` — a separate, local-only git repo (not part of this project's remote).
