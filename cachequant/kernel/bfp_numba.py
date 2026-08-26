import numpy as np

from cachequant.kernel.bfp import DEFAULT_BLOCK_SIZE, quantize_bfp
from cachequant.kernel.blocked_matmul import blocked_int8_matmul_kernel

# The matmul itself is shared with the plain-int8 scheme — the two differ only
# in how scales are chosen, not in how mantissas are multiplied. See
# cachequant/kernel/blocked_matmul.py.
_KERNEL = blocked_int8_matmul_kernel


def prepare_bfp_operand(
    x: np.ndarray, block_size: int = DEFAULT_BLOCK_SIZE
) -> tuple[np.ndarray, np.ndarray]:
    """Quantize `x` into the `(mantissa, per-block scale)` pair the kernel takes.

    Returns `2**e` instead of the raw exponent `e`, so the kernel can just
    multiply by the scale instead of computing `2 ** e` on every use.

    The scale is float64, not float32. That looks wasteful — float32 would
    halve this array's size (21.2 MB -> 10.6 MB) and give the exact same
    result, since `2**e` fits in float32 for every exponent this model
    produces. But the kernel's running total (`acc` in blocked_matmul.py) is
    float64, so a float32 scale would need to be widened to float64 on every
    single use inside the hot loop. Measured: that widening costs more than
    the smaller array saves — 13-18% slower for BFP, 7-9% for int8. Keep this
    float64.

    Must match `prepare_int8_operand`'s dtype: both feed the same compiled
    kernel, and a mismatch would force Numba to compile it twice.
    """
    mantissa, exponent = quantize_bfp(x, block_size)
    return mantissa, np.exp2(exponent.astype(np.float64))


def bfp_matmul_prequantized(
    a: np.ndarray,
    b_mantissa: np.ndarray,
    b_scale: np.ndarray,
    block_size: int = DEFAULT_BLOCK_SIZE,
    bias: np.ndarray | None = None,
) -> np.ndarray:
    """`a @ b.T` (+ optional `bias`) under BFP, where `b` is already prepared.

    Split out from `bfp_matmul` so callers holding a static operand — weights,
    which never change between forward passes — can quantize it once instead of
    re-quantizing millions of elements on every call.

    `bias` is fused into the kernel rather than added afterwards. The add itself
    is negligible; what is not is that running it as a torch op leaves torch's
    OpenMP pool spinning, and those threads then contend with Numba's pool on
    the next call. Measured at the 270x768 attention output projection, that
    contention cost ~4.2ms on top of an ~8.4ms kernel.
    """
    a = np.ascontiguousarray(a, dtype=np.float32)
    if a.ndim != 2:
        raise ValueError("bfp_matmul_prequantized requires a 2D activation")
    if b_mantissa.ndim != 3:
        raise ValueError("b_mantissa must be 3D (n, num_blocks, block_size)")
    # An int8-prepared operand is (n, 1, k), whose num_blocks * block_size also
    # equals k — so the reduction-axis check below would pass it, and the
    # kernel would then walk `blk` up to k/32 across an array holding one
    # block. It is @njit and does not bounds-check, so that reads out of bounds
    # silently instead of raising. Check the block width explicitly.
    if b_mantissa.shape[2] != block_size:
        raise ValueError(
            f"b_mantissa block width {b_mantissa.shape[2]} does not match block_size "
            f"{block_size} — operands prepared by prepare_int8_operand cannot be "
            f"passed to the BFP entry point"
        )
    if a.shape[1] != b_mantissa.shape[1] * b_mantissa.shape[2]:
        raise ValueError(
            f"reduction axis mismatch: a has {a.shape[1]}, "
            f"b has {b_mantissa.shape[1] * b_mantissa.shape[2]}"
        )

    n = b_mantissa.shape[0]
    if bias is None:
        bias = np.zeros(n, dtype=np.float32)
    else:
        bias = np.ascontiguousarray(bias, dtype=np.float32)
        if bias.shape != (n,):
            raise ValueError(f"bias must have shape ({n},), got {bias.shape}")

    a_mantissa, a_scale = prepare_bfp_operand(a, block_size)

    return _KERNEL(a_mantissa, a_scale, b_mantissa, b_scale, bias)


def bfp_matmul(a: np.ndarray, b: np.ndarray, block_size: int = DEFAULT_BLOCK_SIZE) -> np.ndarray:
    b = np.ascontiguousarray(b, dtype=np.float32)
    if b.ndim != 2:
        raise ValueError("bfp_matmul requires 2D inputs")

    b_mantissa, b_scale = prepare_bfp_operand(b, block_size)

    return bfp_matmul_prequantized(a, b_mantissa, b_scale, block_size)
