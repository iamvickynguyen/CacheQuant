import numpy as np

from cachequant.kernel.bfp import dequantize_bfp, quantize_bfp
from cachequant.kernel.bfp_numba import (
    bfp_matmul,
    bfp_matmul_prequantized,
    prepare_bfp_operand,
)


def test_bfp_matmul_output_shape():
    rng = np.random.default_rng(1)
    a = rng.standard_normal((10, 64)).astype(np.float32)
    b = rng.standard_normal((5, 64)).astype(np.float32)

    out = bfp_matmul(a, b)

    assert out.shape == (10, 5)
    assert out.dtype == np.float32


def test_bfp_matmul_zero_inputs_give_zero_output():
    a = np.zeros((2, 32), dtype=np.float32)
    b = np.zeros((3, 32), dtype=np.float32)

    out = bfp_matmul(a, b)

    assert np.all(out == 0.0)


def test_bfp_matmul_relative_error_within_measured_bound_gpt2_shapes():
    # Measured on gaussian data at actual GPT-2 c_attn/c_proj/mlp shapes:
    # mean relative error ~0.041-0.049. Bound set with margin, not guessed.
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
        got = bfp_matmul(a, b)

        rel_err = np.abs(got - ref) / (np.abs(ref) + 1e-3)
        assert rel_err.mean() < 0.10, f"shape {a_shape}x{b_shape}: mean_rel_err={rel_err.mean()}"


def test_bfp_matmul_prequantized_matches_bfp_matmul_exactly():
    # The whole point of the prequantized entry point is that a caller holding
    # static weights (BFPConv1D) can quantize them once instead of on every
    # forward call. That is only safe if it is bit-for-bit the same computation
    # as quantizing inline, so this asserts exact equality, not a tolerance.
    rng = np.random.default_rng(4)
    a = rng.standard_normal((7, 128)).astype(np.float32)
    b = (rng.standard_normal((11, 128)) * 0.02).astype(np.float32)

    b_mantissa, b_scale = prepare_bfp_operand(b)

    assert np.array_equal(bfp_matmul_prequantized(a, b_mantissa, b_scale), bfp_matmul(a, b))


def test_bfp_matmul_prequantized_adds_bias_into_the_kernel_output():
    # Bias is fused into the kernel's output store rather than applied after,
    # so BFPConv1D.forward never calls a torch elementwise op. That matters for
    # more than the add itself: torch's OpenMP pool busy-waits after any op it
    # runs, and those spinning threads then contend with Numba's own pool on
    # the next kernel call.
    rng = np.random.default_rng(6)
    a = rng.standard_normal((5, 64)).astype(np.float32)
    b = (rng.standard_normal((8, 64)) * 0.02).astype(np.float32)
    bias = rng.standard_normal(8).astype(np.float32)

    b_mantissa, b_scale = prepare_bfp_operand(b)

    without_bias = bfp_matmul_prequantized(a, b_mantissa, b_scale)
    with_bias = bfp_matmul_prequantized(a, b_mantissa, b_scale, bias=bias)

    assert np.allclose(with_bias, without_bias + bias, rtol=1e-6, atol=1e-6)


def test_bfp_matmul_prequantized_rejects_reduction_axis_mismatch():
    rng = np.random.default_rng(5)
    a = rng.standard_normal((3, 64)).astype(np.float32)
    b_mantissa, b_scale = prepare_bfp_operand(rng.standard_normal((5, 96)).astype(np.float32))

    try:
        bfp_matmul_prequantized(a, b_mantissa, b_scale)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_bfp_matmul_kernel_dequant_matches_reference_dequant_formula():
    # _bfp_matmul_kernel inlines its own dequant formula (fused into the
    # accumulation loop for speed) instead of calling dequantize_bfp from
    # cachequant.kernel.bfp. This pins that the two formulas actually agree,
    # so they can't silently drift apart. This checks internal consistency
    # between two supposedly-identical formulas, not accuracy against fp32,
    # so the tolerance is tight rather than the 0.10 relative-error bound
    # used above.
    rng = np.random.default_rng(3)
    a = rng.standard_normal((10, 768)).astype(np.float32)
    b = rng.standard_normal((5, 768)).astype(np.float32)

    got = bfp_matmul(a, b)

    a_mantissa, a_exponent = quantize_bfp(a)
    b_mantissa, b_exponent = quantize_bfp(b)
    expected = dequantize_bfp(a_mantissa, a_exponent) @ dequantize_bfp(b_mantissa, b_exponent).T

    # rtol=1e-3 (not 1e-5) because the kernel accumulates in float64 inside
    # the njit loop before a single cast to float32 on output, while this
    # reference dequantizes to float32 arrays first and matmuls in float32 —
    # a real, expected float32-vs-float64 accumulation-order difference, not
    # a formula mismatch. Still two orders of magnitude tighter than the
    # 0.10 fp32-accuracy bound used elsewhere in this file.
    assert np.allclose(got, expected, rtol=1e-3)


def test_bfp_matmul_prequantized_rejects_an_int8_prepared_operand():
    # Mirror of the int8 guard. An int8 operand is (n, 1, k), so
    # num_blocks * block_size still equals k and the reduction-axis check
    # passes — but the kernel would then walk `blk` up to k/32 across an array
    # with one block, reading out of bounds without raising, because @njit
    # does not bounds-check.
    from cachequant.kernel.int8_numba import prepare_int8_operand

    rng = np.random.default_rng(9)
    a = rng.standard_normal((3, 64)).astype(np.float32)
    b_mantissa, b_scale = prepare_int8_operand(rng.standard_normal((5, 64)).astype(np.float32))

    try:
        bfp_matmul_prequantized(a, b_mantissa, b_scale)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_both_schemes_prepare_scales_at_the_same_width():
    # Both must agree: they share one njit kernel and Numba compiles a separate
    # specialization per argument signature, so a mismatch silently doubles the
    # compiled code and breaks the "same kernel" property the schemes rest on.
    #
    # float64 is the measured choice, not the obvious one — see
    # prepare_bfp_operand's docstring for why narrowing to float32 is a
    # 13-18% regression despite halving the array and changing no results.
    from cachequant.kernel.int8_numba import prepare_int8_operand

    rng = np.random.default_rng(13)
    x = rng.standard_normal((16, 64)).astype(np.float32)

    _, bfp_scale = prepare_bfp_operand(x)
    _, int8_scale = prepare_int8_operand(x)

    assert bfp_scale.dtype == np.float64
    assert bfp_scale.dtype == int8_scale.dtype
