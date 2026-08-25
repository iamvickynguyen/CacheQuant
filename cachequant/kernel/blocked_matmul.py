"""The int8 matmul kernel both quantization schemes run on.

BFP and plain int8 are two ways of choosing scales, not two matmuls. Both
produce int8 mantissas grouped into blocks along the reduction axis plus one
float scale per block, so both reduce to the same inner loop:

    BFP   -> num_blocks = k // 32,  block_size = 32
    int8  -> num_blocks = 1,        block_size = k

Extracted here so that stays literally true rather than approximately true —
`tests/test_int8_numba.py` asserts both modules hold this exact function
object. A second copy specialized for int8 would also cost a second Numba JIT
compilation for no arithmetic benefit.
"""

import numpy as np
from numba import njit, prange

INV_127_SQUARED = 1.0 / (127.0 * 127.0)


@njit(cache=True, parallel=True)
def blocked_int8_matmul_kernel(a_mantissa, a_scale, b_mantissa, b_scale, bias):
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
                # int32 is wide enough for either scheme's block size: the
                # worst case is int8's whole-axis block at GPT-2's widest
                # reduction axis, 127*127*3072 = 49.5M, 2.3% of int32 range.
                dot = np.int32(0)
                for k in range(block_size):
                    dot += np.int32(a_mantissa[i, blk, k]) * np.int32(b_mantissa[j, blk, k])
                acc += dot * a_scale[i, blk] * b_scale[j, blk]
            # Bias folded into the store the kernel was already doing, so the
            # caller never runs a separate elementwise op for it.
            out[i, j] = acc * INV_127_SQUARED + bias[j]

    return out
