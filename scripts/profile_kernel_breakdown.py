"""Per-stage timing attribution for a single quantized linear layer, per scheme.

The end-to-end benchmarks (`run_bfp_benchmark.py`, `run_int8_benchmark.py`) say
*how far* each quantized path is from the fp32 baseline. They do not say *where*
that distance comes from, and with two schemes they also cannot say why one is
faster than the other. This script decomposes a single `forward` of each into
its stages so both gaps can be attributed rather than hand-waved:

  - activation quantize  — fp32 -> (int8 mantissa, per-block scale) for `x`
  - int8 kernel          — the `@njit` int8xint8->int32 block matmul itself
  - other                — Numba dispatch, per-call allocation of the
                           activation's mantissa/scale arrays, and the torch
                           reshape/from_numpy view (measured by difference, so
                           it absorbs anything the two stages above miss)

against a same-shape fp32 `Conv1D.forward` as the reference. Both phases are
profiled, since they stress completely different limits: decode (M=1) is a
matrix-vector product bounded by weight memory traffic, prefill (M=long) is a
matrix-matrix product bounded by arithmetic throughput.

Both schemes run the *same* njit kernel (see `cachequant/kernel/blocked_matmul.py`),
so the `int8_kernel_ms` column is a like-for-like comparison of what the block
structure costs: BFP does `k // 32` float scale multiplies per output element,
plain int8 does one. Everything else about the two paths is identical.

Weight quantization is deliberately absent from the table: it happens once at
layer construction, not per call.
"""

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numba
import numpy as np
import torch
from transformers.pytorch_utils import Conv1D

from cachequant.bench.config import DEFAULT_CONFIG
from cachequant.kernel.bfp_linear import BFPConv1D
from cachequant.kernel.bfp_numba import _KERNEL, prepare_bfp_operand
from cachequant.kernel.int8_linear import Int8Conv1D
from cachequant.kernel.int8_numba import prepare_int8_operand

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "benchmarks" / "kernel_breakdown.json"

# (key, layer class, operand-prep function) for each scheme profiled.
SCHEMES = [
    ("bfp8", BFPConv1D, prepare_bfp_operand),
    ("int8", Int8Conv1D, prepare_int8_operand),
]

# (name, nx = reduction axis, nf = output features) for every GPT-2 small linear
# either quantized kernel is applied to.
LAYERS = [
    ("attn.c_attn", 768, 2304),
    ("attn.c_proj", 768, 768),
    ("mlp.c_fc", 768, 3072),
    ("mlp.c_proj", 3072, 768),
]

# M=1 is a decode step (one new token). M=270 matches the long-prompt profile's
# token count in the Phase 1 / Phase 2 benchmarks.
PHASES = [("decode", 1), ("prefill", 270)]

N_REPS = 15
N_WARMUP = 3
# Decode-phase stages run in tens of microseconds, and two of the reported
# columns are residuals between separately-timed quantities. At a 0.05s sample
# the residuals landed inside the run-to-run noise and came out negative; this
# is the smallest target that kept them stable and positive.
TARGET_SAMPLE_SECONDS = 0.15


@dataclass
class StageTimings:
    total_ms: float
    activation_quantize_ms: float
    int8_kernel_ms: float
    other_ms: float
    fp32_reference_ms: float
    ratio_vs_fp32: float


def _time(fn, n_reps: int = N_REPS) -> float:
    """Median-of-reps wall time in milliseconds, after discarding warmups.

    Each sample batches enough back-to-back calls to span TARGET_SAMPLE_SECONDS
    before dividing back out. The decode-phase stages here take tens of
    microseconds, which is close enough to timer resolution and scheduler noise
    that single-call samples produced residuals larger than the stage being
    measured — visible as a negative framework-overhead figure. Batching moves
    the measurement well above the noise floor.

    Median rather than mean so a single scheduler hiccup doesn't move the
    number, matching the reporting convention used by the other benchmarks.
    """
    for _ in range(N_WARMUP):
        fn()

    start = time.perf_counter()
    fn()
    single_call_seconds = max(time.perf_counter() - start, 1e-9)
    inner_reps = max(1, int(TARGET_SAMPLE_SECONDS / single_call_seconds))

    samples = []
    for _ in range(n_reps):
        start = time.perf_counter()
        for _ in range(inner_reps):
            fn()
        samples.append((time.perf_counter() - start) / inner_reps * 1000)
    return float(np.median(samples))


def profile_layer(nx: int, nf: int, m: int, layer_cls, prepare_operand) -> StageTimings:
    torch.manual_seed(0)
    conv = Conv1D(nf=nf, nx=nx)
    quantized_conv = layer_cls(conv)
    x = torch.randn(m, nx) * 0.5

    x_np = np.ascontiguousarray(x.numpy(), dtype=np.float32)
    a_mantissa, a_scale = prepare_operand(x_np)

    with torch.no_grad():
        total_ms = _time(lambda: quantized_conv(x))
        fp32_ms = _time(lambda: conv(x))

    quantize_ms = _time(lambda: prepare_operand(x_np))
    # Re-uses the same already-quantized arrays every call, so this is the
    # kernel's steady-state compute cost with its inputs hot and its Numba
    # type signature already resolved.
    kernel_ms = _time(
        lambda: _KERNEL(
            a_mantissa,
            a_scale,
            quantized_conv.weight_mantissa,
            quantized_conv.weight_scale,
            quantized_conv.bias_np,
        )
    )
    return StageTimings(
        total_ms=total_ms,
        activation_quantize_ms=quantize_ms,
        int8_kernel_ms=kernel_ms,
        # One residual, not two. An earlier version also timed
        # <scheme>_matmul_prequantized to split this into "Numba dispatch" and
        # "torch wrapper", but differencing two separately-timed quantities put
        # the decode-phase result inside the noise. This residual is Numba
        # dispatch plus the per-call allocation of the activation's
        # mantissa/scale arrays; the torch side is a from_numpy view and a
        # reshape now that the bias is fused into the kernel, and measured at
        # roughly zero before this column was collapsed.
        other_ms=total_ms - quantize_ms - kernel_ms,
        fp32_reference_ms=fp32_ms,
        ratio_vs_fp32=total_ms / fp32_ms if fp32_ms > 0 else 0.0,
    )


def main() -> None:
    numba.set_num_threads(DEFAULT_CONFIG.cpu_threads)
    torch.set_num_threads(DEFAULT_CONFIG.cpu_threads)

    rows = []
    for scheme, layer_cls, prepare_operand in SCHEMES:
        for phase, m in PHASES:
            for name, nx, nf in LAYERS:
                timings = profile_layer(nx, nf, m, layer_cls, prepare_operand)
                rows.append(
                    {"scheme": scheme, "phase": phase, "m": m, "layer": name, "nx": nx, "nf": nf}
                    | vars(timings)
                )

    header = (
        f"{'scheme':7} {'phase':8} {'layer':12} {'M':>4} {'total':>9} {'quant':>9} "
        f"{'kernel':>9} {'other':>9} {'fp32':>9} {'ratio':>7}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['scheme']:7} {row['phase']:8} {row['layer']:12} {row['m']:>4} "
            f"{row['total_ms']:>8.3f}m {row['activation_quantize_ms']:>8.3f}m "
            f"{row['int8_kernel_ms']:>8.3f}m {row['other_ms']:>8.3f}m "
            f"{row['fp32_reference_ms']:>8.3f}m {row['ratio_vs_fp32']:>6.1f}x"
        )

    # The comparison the two-scheme table exists for: same kernel, same shapes,
    # different block structure.
    print()
    for phase, _ in PHASES:
        for name, _, _ in LAYERS:
            def kernel_ms(scheme):
                return next(
                    r["int8_kernel_ms"]
                    for r in rows
                    if r["scheme"] == scheme and r["phase"] == phase and r["layer"] == name
                )

            bfp_ms, int8_ms = kernel_ms("bfp8"), kernel_ms("int8")
            print(
                f"kernel speedup  {phase:8} {name:12} "
                f"bfp8={bfp_ms:7.3f}ms  int8={int8_ms:7.3f}ms  "
                f"{bfp_ms / int8_ms if int8_ms else 0:.2f}x"
            )

    payload = {
        "n_reps": N_REPS,
        "threads": DEFAULT_CONFIG.cpu_threads,
        "note": "times in milliseconds, median of n_reps; weight quantization "
        "is excluded because it happens once at construction, not per call. "
        "int8_kernel_ms is the shared blocked-int8 njit kernel for both "
        "schemes, so it is directly comparable across the scheme column",
        "rows": rows,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nwrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
