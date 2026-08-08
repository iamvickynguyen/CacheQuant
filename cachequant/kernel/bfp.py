import numpy as np

DEFAULT_BLOCK_SIZE = 32


def quantize_bfp(
    x: np.ndarray, block_size: int = DEFAULT_BLOCK_SIZE
) -> tuple[np.ndarray, np.ndarray]:
    x = np.ascontiguousarray(x, dtype=np.float32)
    *lead, k = x.shape
    if k % block_size != 0:
        raise ValueError(f"last axis length {k} is not divisible by block_size {block_size}")

    num_blocks = k // block_size
    blocked = x.reshape(*lead, num_blocks, block_size)

    max_abs = np.max(np.abs(blocked), axis=-1)
    safe_max_abs = np.maximum(max_abs, 1e-38)
    exponent = np.where(max_abs > 0, np.floor(np.log2(safe_max_abs)), 0.0).astype(np.int32)

    scale = np.exp2(exponent.astype(np.float64))
    mantissa = np.round(blocked / scale[..., None] * 127.0)
    mantissa = np.clip(mantissa, -127, 127).astype(np.int8)

    return mantissa, exponent


def dequantize_bfp(
    mantissa: np.ndarray, exponent: np.ndarray, block_size: int = DEFAULT_BLOCK_SIZE
) -> np.ndarray:
    *lead, num_blocks, bs = mantissa.shape
    if bs != block_size:
        raise ValueError(f"mantissa block axis {bs} does not match block_size {block_size}")

    scale = np.exp2(exponent.astype(np.float64)).astype(np.float32)
    x = mantissa.astype(np.float32) * scale[..., None] / 127.0
    return x.reshape(*lead, num_blocks * block_size)
