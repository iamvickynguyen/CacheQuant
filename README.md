# CacheQuant

Custom BFP-quantized kernels + KV-cache prefix reuse for faster, cheaper LLM inference. Benchmarked and visualized live.

## What's built so far (Phase 1)

A clean GPT-2 small (124M) generation loop on CPU, with prefill and decode timed separately, and a benchmark harness that turns those timings into tokens/sec, latency, and cost per 1K generated tokens. This is the frozen baseline every later optimization (quantized kernel, KV-cache reuse) gets compared against. No quantization or caching exists yet — every call runs full fp32 GPT-2 from scratch.

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

Not implemented yet — Phase 2.

### KV-cache prefix reuse

Not implemented yet — Phase 3.

### Dashboard

Not implemented yet — Phase 4.

## Limitations (Phase 1)

- CPU only, batch size 1 (enforced by an assertion in `generate_with_timing`) — no GPU path, no batching.
- No quantization or caching yet.
- Benchmark numbers are from one dev machine; re-run `scripts/run_baseline.py` locally before trusting exact figures on different hardware.
- `max_new_tokens=0` isn't validated — the loop still emits one token in that case.
- Only tested against Python 3.14.

## Docs

Design spec and implementation plans live in `docs/` — a separate, local-only git repo (not part of this project's remote).
