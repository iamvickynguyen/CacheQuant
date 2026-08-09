import numpy as np

from cachequant.kernel.bfp import dequantize_bfp, quantize_bfp
from cachequant.kernel.bfp_numba import bfp_matmul


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
