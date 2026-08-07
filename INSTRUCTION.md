# Build: LLM Inference Bottlenecks Demo — Quantization + KV-Cache Reuse

## Context

This is a technical interview presentation project. I'm an MLIR compiler engineer (block floating-point lowering, irregular tiling, kernel SDK for a custom AI inference accelerator) preparing a 30-minute talk + 60-minute Q&A panel, and also using this project to prep for interviews covering LLM inference bottlenecks, the inference stack, SW/HW co-design for AI accelerators, and ML inference-efficiency optimizations.

**Goal:** build a small, real, working system that demonstrates two independent levers on LLM inference cost/throughput — (1) a custom quantized matmul kernel and (2) KV-cache reuse across requests with shared prefixes — with a live before/after demo showing tokens/sec and estimated cost-per-1K-tokens.

This is not a production system. It needs to be **correct, measurable, and honestly benchmarked**, with clear places where it breaks down, more than it needs to be polished. I want real numbers I can defend under technical pushback, not a slick demo that hides its assumptions.

**Do not use or reference any proprietary hardware specs, performance numbers, or internal design details from my employer (d-Matrix).** Use only public models, public hardware assumptions (e.g. a generic CPU or a named public GPU spec), and my own from-scratch implementation choices.

---

## Architecture overview

1. **Base model**: GPT-2 small (124M, via Hugging Face `transformers`) run on CPU. Small enough to iterate fast, simple enough that I can reason about every layer, well documented so I can explain internals under questioning. If CPU throughput is too slow to demo comfortably, fall back to a slightly smaller custom-trained toy transformer — but only if GPT-2 small proves impractical.
2. **Baseline path**: standard `transformers` generation loop, fp32 or fp16 weights, no caching optimizations beyond whatever HF does by default (document exactly what that is).
3. **Optimization 1 — quantized kernel**: implement my own quantized matmul for the attention and FFN linear layers. Prefer a **block floating-point (BFP) scheme** (shared exponent per block of N elements, e.g. N=32) over plain int8 — this should mirror the numerics tradeoffs I work with professionally (precision vs. dynamic range vs. speed). Implement as a hand-written kernel (NumPy/Numba/C++ extension — pick whichever gets me a real measurable speedup fastest), not just a call to an existing quantization library. I need to be able to explain every step of the quantize/dequantize/matmul path.
4. **Optimization 2 — KV-cache reuse**: implement prefix caching — a hash or trie-based cache keyed on token-prefix, storing past KV tensors, so requests sharing a prompt prefix reuse cached KV states instead of recomputing them. Needs a simple eviction policy (LRU is fine) and needs to handle partial-prefix matches correctly.
5. **Benchmark harness**: for both optimizations independently and combined, measure:
   - tokens/sec (prefill and decode phases separately)
   - latency per token
   - estimated cost per 1K tokens, using a clearly stated and documented $/hour compute assumption (make this a config value, not a hardcoded magic number)
   - accuracy/quality delta introduced by quantization (e.g. perplexity on a small held-out sample, or a few side-by-side generations)
6. **Live visualization**: a small local dashboard (Streamlit is fine — prioritize reliability and fast iteration over visual polish) showing:
   - live streaming token output as text generates
   - a bar or gauge comparing baseline vs. optimized tokens/sec and cost/1K tokens, updating as generation happens
   - toggles to turn quantization and KV-cache reuse on/off independently, so I can demo all four combinations live
   - a way to vary prompt repetition/overlap live, to show the KV-cache reuse story clearly (fresh unique prompts vs. prompts sharing a prefix)

---

## Build phases

**Phase 1 — Baseline & instrumentation**
- Get GPT-2 small running locally with a clean generation loop
- Build the benchmark harness first, against the unmodified baseline, so every later phase has honest comparison numbers
- Document baseline tokens/sec, latency, and cost estimate

**Phase 2 — Quantized kernel**
- Implement BFP quantization for weights (and optionally activations) in attention/FFN linear layers
- Validate correctness against the fp32 baseline (numerical error bounds, not just "it runs")
- Benchmark speedup and accuracy delta
- Note explicitly where/why it helps (compute-bound layers) and where it might not (memory-bound-dominated phases, if any)

**Phase 3 — KV-cache reuse**
- Implement prefix cache with correctness tests for partial matches and eviction
- Benchmark against baseline using a prompt set specifically designed to show both a high-reuse scenario and a no-reuse scenario
- Document cache hit rate vs. speedup relationship

**Phase 4 — Combine + visualize**
- Wire both optimizations into the same pipeline, independently toggleable
- Build the Streamlit dashboard with live token stream + before/after bars
- Make sure the four combinations (neither / quant only / cache only / both) all work live

**Phase 5 — Demo readiness & presentation assets**
- Write a README documenting all design decisions, tradeoffs, and known limitations/failure modes (this should double as my speaking notes)
- Export clean benchmark charts/tables suitable for slides
- Prepare a fallback: pre-recorded benchmark results and a screen-recording backup in case the live demo fails during the actual interview
- Stress-test the demo: what happens with pathological inputs (very long prompts, empty prompts, extreme voice/request counts)? These are exactly the "where does it break" edges I want to be able to speak to

---

## Deliverables checklist

- [ ] Working repo with baseline, quantized kernel, KV-cache reuse, and combined pipeline
- [ ] Benchmark harness with reproducible, documented methodology (including the cost-per-hour assumption used)
- [ ] Streamlit dashboard with live demo + toggles
- [ ] README covering: architecture, all design decisions and why, tradeoffs, known limitations, and what I'd do differently with more time
- [ ] Exported benchmark charts/tables for slides
- [ ] A short list of "break points" I can speak to in Q&A (e.g., where quantization degrades output quality, where cache reuse gives no benefit, where the bottleneck shifts from compute to memory bandwidth)

## Non-goals

- No production-grade robustness, auth, multi-user support, or deployment concerns
- No general-purpose quantization library usage in place of my own kernel — the point is that I understand and can defend every step
- No proprietary hardware specs or internal employer details anywhere in code, comments, README, or benchmark assumptions
