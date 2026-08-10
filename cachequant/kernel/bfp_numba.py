import numpy as np
from numba import njit, prange

from cachequant.kernel.bfp import DEFAULT_BLOCK_SIZE, quantize_bfp

_INV_127_SQUARED = 1.0 / (127.0 * 127.0)


@njit(cache=True, parallel=True)
def _bfp_matmul_kernel(a_mantissa, a_scale, b_mantissa, b_scale, bias):
    # `j` (output features) is the parallel axis rather than `i` (tokens),
    # because `i` is 1 during decode — parallelising over it would leave the
    # decode path single-threaded, which is the phase that needs the threads
    # most. `j` is 768-3072 wide for every GPT-2 linear in scope, in both
    # phases. Each thread owns a disjoint set of `j`, so the writes to `out`
    # never race.
    m, num_blocks, block_size = a_mantissa.shape
    n = b_mantissa.shape[0]
    out = np.zeros((m, n), dtype=np.float32)

    for j in prange(n):
        for i in range(m):
            acc = 0.0
            for blk in range(num_blocks):
                dot = np.int32(0)
                for k in range(block_size):
                    dot += np.int32(a_mantissa[i, blk, k]) * np.int32(b_mantissa[j, blk, k])
                acc += dot * a_scale[i, blk] * b_scale[j, blk]
            # Bias folded into the store the kernel was already doing, so the
            # caller never runs a separate elementwise op for it.
            out[i, j] = acc * _INV_127_SQUARED + bias[j]

    return out


def prepare_bfp_operand(
    x: np.ndarray, block_size: int = DEFAULT_BLOCK_SIZE
) -> tuple[np.ndarray, np.ndarray]:
    """Quantize `x` into the `(mantissa, per-block scale)` pair the kernel takes.

    The scale is `2**e` rather than the raw exponent so the kernel never
    evaluates a `pow`: it used to compute `2.0 ** (e_a + e_b)` inside the
    accumulation loop, i.e. M*N*num_blocks times. Powers of two are exact in
    float64, so factoring that into `2**e_a * 2**e_b` is numerically identical
    while reducing it to M*num_blocks + N*num_blocks — and for a static operand
    it is hoisted out of the forward pass entirely.
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

    return _bfp_matmul_kernel(a_mantissa, a_scale, b_mantissa, b_scale, bias)


def bfp_matmul(a: np.ndarray, b: np.ndarray, block_size: int = DEFAULT_BLOCK_SIZE) -> np.ndarray:
    b = np.ascontiguousarray(b, dtype=np.float32)
    if b.ndim != 2:
        raise ValueError("bfp_matmul requires 2D inputs")

    b_mantissa, b_scale = prepare_bfp_operand(b, block_size)

    return bfp_matmul_prequantized(a, b_mantissa, b_scale, block_size)
