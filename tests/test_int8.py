import numpy as np

from cachequant.kernel.int8 import dequantize_int8, quantize_int8


def test_quantize_dequantize_shapes():
    x = np.zeros((4, 64), dtype=np.float32)
    mantissa, scale = quantize_int8(x)

    assert mantissa.shape == (4, 64)
    assert mantissa.dtype == np.int8
    assert scale.shape == (4,)
    assert scale.dtype == np.float32

    x_hat = dequantize_int8(mantissa, scale)
    assert x_hat.shape == (4, 64)


def test_quantize_accepts_a_reduction_axis_no_block_size_divides():
    # The distinguishing structural property against BFP: there is no block, so
    # there is nothing for the reduction axis to be divisible by. quantize_bfp
    # raises ValueError on this same input.
    x = np.zeros((4, 50), dtype=np.float32)

    mantissa, scale = quantize_int8(x)

    assert mantissa.shape == (4, 50)
    assert scale.shape == (4,)


def test_all_zero_row_round_trips_to_zero():
    x = np.zeros((1, 32), dtype=np.float32)
    mantissa, scale = quantize_int8(x)
    x_hat = dequantize_int8(mantissa, scale)

    assert np.all(mantissa == 0)
    assert np.all(x_hat == 0.0)


def test_max_abs_element_uses_the_full_int8_range_exactly():
    # This is the whole point of the exact scale. BFP rounds its scale up to
    # 2**ceil(log2(max_abs)), so the largest element lands anywhere in
    # [63, 127] and up to a full bit of range goes unused. An exact scale puts
    # it on 127 every time, whatever max_abs happens to be.
    x = np.zeros((1, 32), dtype=np.float32)
    x[0, 5] = 1.05  # deliberately just above a power of two, BFP's worst case

    mantissa, _ = quantize_int8(x)

    assert mantissa[0, 5] == 127


def test_each_row_gets_its_own_scale_by_default():
    x = np.zeros((2, 32), dtype=np.float32)
    x[0, :] = 1.0
    x[1, :] = 100.0

    _, scale = quantize_int8(x)

    assert scale[0] != scale[1]
    assert np.isclose(scale[0], 1.0)
    assert np.isclose(scale[1], 100.0)


def test_per_tensor_shares_one_scale_across_every_row():
    x = np.zeros((2, 32), dtype=np.float32)
    x[0, :] = 1.0
    x[1, :] = 100.0

    _, scale = quantize_int8(x, per_tensor=True)

    assert scale[0] == scale[1]
    assert np.isclose(scale[0], 100.0)


def test_per_tensor_crushes_the_small_row_that_per_channel_preserves():
    # Documents *why* per_tensor exists only as a comparison point. One large
    # row drags every other row's mantissas toward zero, which is the coarse-
    # granularity failure the BFP block structure was designed to avoid.
    x = np.zeros((2, 32), dtype=np.float32)
    x[0, :] = 1.0
    x[1, :] = 100.0

    per_channel, _ = quantize_int8(x)
    per_tensor, _ = quantize_int8(x, per_tensor=True)

    assert np.all(per_channel[0] == 127)
    assert np.all(np.abs(per_tensor[0]) <= 2)


def test_round_trip_relative_error_within_measured_bound():
    # Measured on gaussian data at a GPT-2-realistic shape: mean rel err
    # 0.031-0.035 across seeds. Bound set with margin above the measured
    # value, not guessed. Same bound as the BFP round-trip test, which
    # measures 0.033-0.035 — the two schemes cost about the same here.
    rng = np.random.default_rng(0)
    x = rng.standard_normal((16, 768)).astype(np.float32)

    mantissa, scale = quantize_int8(x)
    x_hat = dequantize_int8(mantissa, scale)

    rel_err = np.abs(x_hat - x) / (np.abs(x) + 1e-6)
    assert rel_err.mean() < 0.06


def test_int8_scale_moves_under_a_perturbation_bfp_scale_ignores():
    # The two schemes' scales respond to a tiny input change in opposite ways,
    # and it is a definitional difference, not an accident of tuning: BFP's
    # scale is 2**ceil(log2(max_abs)), a step function that a perturbation far
    # below the gap to the next power of two cannot move, so every mantissa
    # keeps its meaning. int8's scale is max_abs itself, which moves with any
    # change to the row max and shifts every mantissa in the row with it.
    #
    # This is the mechanism behind the chunked-prefill reproducibility break
    # point in the README: torch's fp32 matmuls differ by ~1e-5 depending on
    # how many tokens a prefill pass covers, BFP absorbs that and int8
    # amplifies it. Pinned here because it is the property that decides
    # whether int8 output is reproducible across cache hits.
    from cachequant.kernel.bfp import quantize_bfp

    rng = np.random.default_rng(0)
    x = (rng.standard_normal((64, 768)) * 0.5).astype(np.float32)
    perturbed = (x + x * np.float32(1e-5) * rng.standard_normal(x.shape)).astype(np.float32)

    _, bfp_exponent = quantize_bfp(x)
    _, bfp_exponent_perturbed = quantize_bfp(perturbed)
    _, int8_scale = quantize_int8(x)
    _, int8_scale_perturbed = quantize_int8(perturbed)

    assert np.array_equal(bfp_exponent, bfp_exponent_perturbed)
    assert not np.any(int8_scale == int8_scale_perturbed)


def test_quantize_computes_mantissas_at_float64_precision_like_bfp():
    # The two schemes are supposed to differ in scale *policy* and nothing
    # else, so they must round at the same precision. quantize_bfp divides by a
    # float64 scale, which promotes the whole expression to float64; if int8
    # divides by a float32 scale instead, the two paths round differently near
    # ties and the scheme comparison silently measures an arithmetic-precision
    # difference on top of the policy difference it is meant to isolate.
    #
    # Caught by the quality benchmark disagreeing with itself: the same
    # per-channel/exact configuration scored +0.45% through Int8Conv1D and
    # +0.34% through a float64 reference implementation of the identical
    # scheme.
    # Sized to a real GPT-2 weight (mlp.c_fc transposed). The divergence is
    # rare — about 2 elements per million land near enough to a rounding tie
    # for float32 and float64 to disagree — so a small array does not catch it,
    # and 2-per-million is not negligible here: an exact scale amplifies tiny
    # perturbations (see
    # test_int8_scale_moves_under_a_perturbation_bfp_scale_ignores), which is
    # how a handful of mantissas moved end-to-end perplexity by 0.11pp.
    rng = np.random.default_rng(11)
    x = (rng.standard_normal((3072, 768)) * 0.02).astype(np.float32)

    mantissa, scale = quantize_int8(x)

    expected = np.clip(
        np.round(x.astype(np.float64) / scale[..., None].astype(np.float64) * 127.0),
        -127,
        127,
    ).astype(np.int8)

    assert np.array_equal(mantissa, expected)
