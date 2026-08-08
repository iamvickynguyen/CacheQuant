import numpy as np
from numba import njit

from cachequant.kernel.bfp import DEFAULT_BLOCK_SIZE, quantize_bfp


@njit(cache=True)
def _bfp_matmul_kernel(a_mantissa, a_exponent, b_mantissa, b_exponent):
    m, num_blocks, block_size = a_mantissa.shape
    n = b_mantissa.shape[0]
    out = np.zeros((m, n), dtype=np.float32)

    for i in range(m):
        for j in range(n):
            acc = 0.0
            for blk in range(num_blocks):
                dot = np.int32(0)
                for k in range(block_size):
                    dot += np.int32(a_mantissa[i, blk, k]) * np.int32(b_mantissa[j, blk, k])
                scale = (2.0 ** (a_exponent[i, blk] + b_exponent[j, blk])) / (127.0 * 127.0)
                acc += dot * scale
            out[i, j] = acc

    return out


def bfp_matmul(a: np.ndarray, b: np.ndarray, block_size: int = DEFAULT_BLOCK_SIZE) -> np.ndarray:
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("bfp_matmul requires 2D inputs")
    if a.shape[1] != b.shape[1]:
        raise ValueError(f"reduction axis mismatch: a has {a.shape[1]}, b has {b.shape[1]}")

    a_mantissa, a_exponent = quantize_bfp(a, block_size)
    b_mantissa, b_exponent = quantize_bfp(b, block_size)

    return _bfp_matmul_kernel(a_mantissa, a_exponent, b_mantissa, b_exponent)
