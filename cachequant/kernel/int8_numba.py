"""Numba-facing int8 matmul: `a @ b.T` under plain symmetric int8 quantization.

Mirrors `cachequant.kernel.bfp_numba` entry point for entry point, and runs on
the same kernel (`_KERNEL` below). The only difference is the quantizer it
calls and the block layout it hands over: one block spanning the whole
reduction axis instead of `k // 32` blocks of 32.
"""

import numpy as np

from cachequant.kernel.blocked_matmul import blocked_int8_matmul_kernel
from cachequant.kernel.int8 import quantize_int8

# Same function object BFP uses. Asserted in tests/test_int8_numba.py.
_KERNEL = blocked_int8_matmul_kernel


def prepare_int8_operand(
    x: np.ndarray, per_tensor: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Quantize `x` into the `(mantissa, per-block scale)` pair the kernel takes.

    The kernel is written against BFP's blocked layout — mantissa
    `(n, num_blocks, block_size)` and scale `(n, num_blocks)`. Plain int8 is
    that layout with a single block covering the entire reduction axis, so this
    reshapes rather than computing anything the kernel could not already
    consume. Collapsing to one block is also why int8 is the cheaper of the two
    schemes at the same shapes: the kernel then does one scale multiply per
    output element instead of `k // 32` of them.

    The scale is widened to float64 to match `prepare_bfp_operand`, so both
    schemes hit the same Numba specialization of the shared kernel instead of
    triggering a second JIT compilation.
    """
    mantissa, scale = quantize_int8(x, per_tensor)
    *lead, k = mantissa.shape
    return mantissa.reshape(*lead, 1, k), scale.reshape(*lead, 1).astype(np.float64)


def int8_matmul_prequantized(
    a: np.ndarray,
    b_mantissa: np.ndarray,
    b_scale: np.ndarray,
    bias: np.ndarray | None = None,
    per_tensor_activation: bool = False,
) -> np.ndarray:
    """`a @ b.T` (+ optional `bias`) under int8, where `b` is already prepared.

    Split out so callers holding a static operand — weights, which never change
    between forward passes — can quantize it once instead of re-quantizing
    millions of elements on every call. `bias` is fused into the kernel for the
    same reason it is under BFP: applying it as a torch op afterwards leaves
    torch's OpenMP threads spinning and contending with Numba's pool on the
    next call, which costs more than the add itself.
    """
    a = np.ascontiguousarray(a, dtype=np.float32)
    if a.ndim != 2:
        raise ValueError("int8_matmul_prequantized requires a 2D activation")
    if b_mantissa.ndim != 3:
        raise ValueError("b_mantissa must be 3D (n, 1, k)")
    if a.shape[1] != b_mantissa.shape[2]:
        raise ValueError(
            f"reduction axis mismatch: a has {a.shape[1]}, b has {b_mantissa.shape[2]}"
        )

    n = b_mantissa.shape[0]
    if bias is None:
        bias = np.zeros(n, dtype=np.float32)
    else:
        bias = np.ascontiguousarray(bias, dtype=np.float32)
        if bias.shape != (n,):
            raise ValueError(f"bias must have shape ({n},), got {bias.shape}")

    a_mantissa, a_scale = prepare_int8_operand(a, per_tensor_activation)

    return _KERNEL(a_mantissa, a_scale, b_mantissa, b_scale, bias)


def int8_matmul(a: np.ndarray, b: np.ndarray, per_tensor: bool = False) -> np.ndarray:
    b = np.ascontiguousarray(b, dtype=np.float32)
    if b.ndim != 2:
        raise ValueError("int8_matmul requires 2D inputs")

    b_mantissa, b_scale = prepare_int8_operand(b, per_tensor)

    return int8_matmul_prequantized(a, b_mantissa, b_scale, per_tensor_activation=per_tensor)
