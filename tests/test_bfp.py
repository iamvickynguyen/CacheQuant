import numpy as np

from cachequant.kernel.bfp import DEFAULT_BLOCK_SIZE, quantize_bfp, dequantize_bfp


def test_quantize_dequantize_shapes():
    x = np.zeros((4, 64), dtype=np.float32)
    mantissa, exponent = quantize_bfp(x, block_size=32)

    assert mantissa.shape == (4, 2, 32)
    assert mantissa.dtype == np.int8
    assert exponent.shape == (4, 2)
    assert exponent.dtype == np.int32

    x_hat = dequantize_bfp(mantissa, exponent, block_size=32)
    assert x_hat.shape == (4, 64)


def test_quantize_rejects_non_divisible_last_axis():
    x = np.zeros((4, 50), dtype=np.float32)
    try:
        quantize_bfp(x, block_size=32)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_all_zero_block_round_trips_to_zero():
    x = np.zeros((1, 32), dtype=np.float32)
    mantissa, exponent = quantize_bfp(x)
    x_hat = dequantize_bfp(mantissa, exponent)

    assert np.all(mantissa == 0)
    assert np.all(exponent == 0)
    assert np.all(x_hat == 0.0)


def test_round_trip_relative_error_within_measured_bound():
    # Measured on gaussian data at GPT-2-realistic shapes: mean rel err ~0.033-0.035.
    # Bound set with margin above the measured value, not guessed.
    rng = np.random.default_rng(0)
    x = rng.standard_normal((16, 768)).astype(np.float32)

    mantissa, exponent = quantize_bfp(x)
    x_hat = dequantize_bfp(mantissa, exponent)

    rel_err = np.abs(x_hat - x) / (np.abs(x) + 1e-6)
    assert rel_err.mean() < 0.06


def test_max_abs_element_uses_full_int8_range():
    x = np.zeros((1, 32), dtype=np.float32)
    x[0, 5] = 4.0
    mantissa, exponent = quantize_bfp(x)

    assert mantissa[0, 0, 5] in (127, -127)
