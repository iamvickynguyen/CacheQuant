"""Plain symmetric int8 quantization — the second scheme, alongside BFP.

Same int8 mantissas and the same int8xint8->int32 matmul as
`cachequant.kernel.bfp`; the two schemes differ only in how the scale that
turns those mantissas back into floats is chosen:

                    scale granularity        scale value
    BFP  (bfp.py)   one per 32 values        2**ceil(log2(max_abs))
    int8 (here)     one per row              max_abs, exactly

Both differences pull in opposite directions and roughly cancel on GPT-2
weights (measured: +0.38% perplexity for BFP, +0.34% for int8). The exact
scale always lands the largest element of a row on +/-127, where BFP's
power-of-two rounding can leave up to a full bit of the int8 range unused
(max_abs=1.05 -> scale=2 -> nothing exceeds 67). What int8 gives up for that
is outlier isolation: one large value distorts its whole row rather than just
its own 32-value block.

"Row" means the last axis of `x`, which is the reduction axis both operands
are blocked along. For a transposed weight (nf, nx) that is one scale per
output channel; for an activation (tokens, nx) it is one scale per token.
Both are the granularities that keep quantization error survivable — see
`per_tensor` below for the one that does not.
"""

import numpy as np


def quantize_int8(x: np.ndarray, per_tensor: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Quantize to int8 mantissas plus one float32 scale per row of the last axis.

    `per_tensor=True` collapses every scale to a single array-wide value. It is
    provided as a measured comparison point, not as a recommended setting:
    on GPT-2 it costs +19.35% perplexity applied to weights alone and +68.60%
    applied to weights and activations both, against +0.34% for the per-row
    default. See README's quantization comparison table.
    """
    x = np.ascontiguousarray(x, dtype=np.float32)

    max_abs = np.max(np.abs(x), axis=-1)
    if per_tensor:
        max_abs = np.full_like(max_abs, np.abs(x).max() if x.size else 0.0)

    # An all-zero row has max_abs == 0; clamping the divisor keeps the division
    # finite and every mantissa in that row rounds to 0 regardless of the value
    # used, so the row round-trips to exactly zero.
    #
    # float64, matching quantize_bfp — whose scale is a float64 exp2 and so
    # promotes this same expression. The two schemes are meant to differ in
    # scale policy and nothing else, and at float32 they also differ in where
    # they round: about 2 mantissas per million land near enough to a tie to
    # disagree, which an exact scale then amplifies (a measured 0.11pp of
    # end-to-end perplexity).
    safe_max_abs = np.maximum(max_abs, 1e-38).astype(np.float64)
    mantissa = np.round(x.astype(np.float64) / safe_max_abs[..., None] * 127.0)
    mantissa = np.clip(mantissa, -127, 127).astype(np.int8)

    return mantissa, max_abs.astype(np.float32)


def dequantize_int8(mantissa: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return mantissa.astype(np.float32) * scale[..., None].astype(np.float32) / 127.0
