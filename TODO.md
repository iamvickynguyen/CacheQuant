# TODO

## ~~Plain int8 (per-tensor/per-channel scale) vs BFP8~~ — done, `feat/int8-quantization`

Shipped as a second scheme alongside BFP8: `cachequant/kernel/int8.py`,
`int8_numba.py`, `int8_linear.py`, sharing one njit kernel with BFP via
`cachequant/kernel/blocked_matmul.py`. Measured in
`benchmarks/int8_results*.json`, `quantization_quality.json`, and
`kernel_breakdown.json`.

**The expectation in the original plan was wrong.** It predicted plain int8
would "show a bigger perplexity hit than BFP's near-free +0.38%, for no speed
advantage over BFP." Measured: int8 at per-channel weight + per-token
activation costs **+0.34%** — indistinguishable from BFP — and its kernel is
**~1.5x faster** at every GPT-2 layer shape, because collapsing the 32-value
block to one whole-axis block removes `k/32 - 1` float scale multiplies per
output element.

The reason the granularity penalty did not appear is that BFP spends its
granularity advantage on a scale-rounding disadvantage: `2**ceil(log2(max_abs))`
leaves up to a full bit of int8 range unused. See the 2x2 grid in
`scripts/compare_quantization_quality.py` — the two corners neither scheme
ships separate the axes.

Coarse granularity does still bite, just one step further out: per-*tensor*
weight scale costs +19.35%, and per-tensor weight and activation +68.60%.
Those settings exist in the API only as documented comparison points.

## Open questions this raised

- **Block-32 with an exact scale scored best of the four corners** (-0.08%,
  i.e. inside the noise of fp32 on a 3-passage eval). It costs 9.0 bits/value
  against int8's 8.04 and keeps BFP's slow per-block loop, so it is not
  obviously worth shipping — but nothing has measured whether its quality edge
  survives a larger eval.
- **The perplexity eval is 3 passages** (`eval/passages.py`). Every quality
  number in the README rests on it. A scheme comparison that turns on 0.04
  percentage points deserves a wider eval before anyone leans on the ranking.
- **int8 output is not reproducible across prefill chunkings** — a cache hit
  can produce different text from the same prompt served cold, because an
  exact scale moves with the ~1e-5 differences torch's fp32 matmuls produce
  between one long prefill and a chunked one. BFP's step-function scale
  absorbs them. Documented as a break point in the README; the obvious fix to
  evaluate is snapping the exact scale to a coarse grid (a few significant
  bits) to restore step-function stability while keeping most of the range
  advantage.
- **Neither scheme has a per-group middle ground measured** (block 128 or 256,
  exact scale) — the region where MX-style formats actually sit.
