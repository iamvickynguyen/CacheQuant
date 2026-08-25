import numpy as np

from cachequant.kernel.int8 import dequantize_int8, quantize_int8
from cachequant.kernel.int8_numba import (
    int8_matmul,
    int8_matmul_prequantized,
    prepare_int8_operand,
)


def test_int8_matmul_output_shape():
    rng = np.random.default_rng(1)
    a = rng.standard_normal((10, 64)).astype(np.float32)
    b = rng.standard_normal((5, 64)).astype(np.float32)

    out = int8_matmul(a, b)

    assert out.shape == (10, 5)
    assert out.dtype == np.float32


def test_int8_matmul_zero_inputs_give_zero_output():
    a = np.zeros((2, 32), dtype=np.float32)
    b = np.zeros((3, 32), dtype=np.float32)

    out = int8_matmul(a, b)

    assert np.all(out == 0.0)


def test_int8_matmul_accepts_a_reduction_axis_bfp_would_reject():
    # 50 is not divisible by BFP's block size, so bfp_matmul raises on this
    # shape. int8 has no block to divide, so every reduction axis is legal.
    rng = np.random.default_rng(7)
    a = rng.standard_normal((3, 50)).astype(np.float32)
    b = rng.standard_normal((4, 50)).astype(np.float32)

    out = int8_matmul(a, b)

    assert out.shape == (3, 4)
    assert np.isfinite(out).all()


def test_int8_matmul_relative_error_within_measured_bound_gpt2_shapes():
    # Measured on gaussian data at actual GPT-2 c_attn/c_proj/mlp shapes.
    # Bound set with margin above the measured value, not guessed, and kept
    # equal to the BFP kernel's bound so the two schemes are held to the same
    # accuracy contract.
    rng = np.random.default_rng(2)
    shapes = [
        ((10, 768), (2304, 768)),   # c_attn
        ((10, 768), (768, 768)),    # attn c_proj
        ((10, 768), (3072, 768)),   # mlp.c_fc
        ((10, 3072), (768, 3072)),  # mlp.c_proj
    ]
    for a_shape, b_shape in shapes:
        a = (rng.standard_normal(a_shape) * 0.5).astype(np.float32)
        b = (rng.standard_normal(b_shape) * 0.02).astype(np.float32)

        ref = a @ b.T
        got = int8_matmul(a, b)

        rel_err = np.abs(got - ref) / (np.abs(ref) + 1e-3)
        assert rel_err.mean() < 0.10, f"shape {a_shape}x{b_shape}: mean_rel_err={rel_err.mean()}"


def test_int8_matmul_prequantized_matches_int8_matmul_exactly():
    # Int8Conv1D quantizes its weight once at construction and reuses it on
    # every forward call. That is only safe if the prequantized entry point is
    # bit-for-bit the same computation as quantizing inline, so this asserts
    # exact equality, not a tolerance.
    rng = np.random.default_rng(4)
    a = rng.standard_normal((7, 128)).astype(np.float32)
    b = (rng.standard_normal((11, 128)) * 0.02).astype(np.float32)

    b_mantissa, b_scale = prepare_int8_operand(b)

    assert np.array_equal(int8_matmul_prequantized(a, b_mantissa, b_scale), int8_matmul(a, b))


def test_int8_matmul_prequantized_adds_bias_into_the_kernel_output():
    rng = np.random.default_rng(6)
    a = rng.standard_normal((5, 64)).astype(np.float32)
    b = (rng.standard_normal((8, 64)) * 0.02).astype(np.float32)
    bias = rng.standard_normal(8).astype(np.float32)

    b_mantissa, b_scale = prepare_int8_operand(b)

    without_bias = int8_matmul_prequantized(a, b_mantissa, b_scale)
    with_bias = int8_matmul_prequantized(a, b_mantissa, b_scale, bias=bias)

    assert np.allclose(with_bias, without_bias + bias, rtol=1e-6, atol=1e-6)


def test_int8_matmul_prequantized_rejects_reduction_axis_mismatch():
    rng = np.random.default_rng(5)
    a = rng.standard_normal((3, 64)).astype(np.float32)
    b_mantissa, b_scale = prepare_int8_operand(rng.standard_normal((5, 96)).astype(np.float32))

    try:
        int8_matmul_prequantized(a, b_mantissa, b_scale)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_int8_kernel_dequant_matches_reference_dequant_formula():
    # The kernel inlines its dequant formula into the accumulation loop instead
    # of calling dequantize_int8. This pins that the two formulas agree so they
    # cannot silently drift apart. rtol=1e-3 rather than 1e-5 because the kernel
    # accumulates in float64 before one cast to float32 on output while this
    # reference matmuls in float32 throughout — a real accumulation-order
    # difference, not a formula mismatch.
    rng = np.random.default_rng(3)
    a = rng.standard_normal((10, 768)).astype(np.float32)
    b = rng.standard_normal((5, 768)).astype(np.float32)

    got = int8_matmul(a, b)

    a_mantissa, a_scale = quantize_int8(a)
    b_mantissa, b_scale = quantize_int8(b)
    expected = dequantize_int8(a_mantissa, a_scale) @ dequantize_int8(b_mantissa, b_scale).T

    assert np.allclose(got, expected, rtol=1e-3)


def test_int8_uses_the_same_kernel_as_bfp():
    # The two schemes are a quantizer difference, not a kernel difference:
    # both feed (mantissa, per-block scale) triples to one shared njit matmul.
    # If this ever stops being true, the "int8 is BFP minus the block loop"
    # claim in the README stops being true with it.
    from cachequant.kernel import bfp_numba, blocked_matmul

    assert bfp_numba._KERNEL is blocked_matmul.blocked_int8_matmul_kernel

    from cachequant.kernel import int8_numba

    assert int8_numba._KERNEL is blocked_matmul.blocked_int8_matmul_kernel


def test_int8_matmul_prequantized_rejects_a_bfp_prepared_operand():
    # The two schemes hand the shared kernel the same *kind* of triple with
    # different block shapes: BFP's is (n, k/32, 32), int8's is (n, 1, k). The
    # reduction-axis check alone does not catch a BFP operand arriving here —
    # and the kernel is @njit, which does not bounds-check, so it would read
    # off the end of the scale array rather than raise. Guard explicitly.
    from cachequant.kernel.bfp_numba import prepare_bfp_operand

    rng = np.random.default_rng(8)
    a = rng.standard_normal((3, 64)).astype(np.float32)
    b_mantissa, b_scale = prepare_bfp_operand(rng.standard_normal((5, 64)).astype(np.float32))

    try:
        int8_matmul_prequantized(a, b_mantissa, b_scale)
        assert False, "expected ValueError"
    except ValueError:
        pass
