"""Perplexity and greedy-generation delta for every quantization scheme.

`compare_bfp_quality.py` answers "what does BFP cost?". This answers the
question that only exists once there are two schemes: "which scheme costs
less, and *why*?"

The why needs the 2x2 at the bottom. BFP and plain int8 differ on two axes at
once, so comparing them head-to-head cannot say which axis the quality lives
on:

                    scale granularity        scale value
    BFP             one per 32 values        2**ceil(log2(max_abs))
    plain int8      one per row              max_abs, exactly

Filling in the two combinations neither scheme ships separates them. Both
corners run on the same shared kernel, so this costs nothing but runtime.

Writes benchmarks/quantization_quality.json.
"""

import copy
import json
from pathlib import Path

import numba
import numpy as np
import torch

from cachequant.bench.config import DEFAULT_CONFIG
from cachequant.bench.provenance import provenance
from cachequant.kernel.bfp_linear import apply_bfp_quantization
from cachequant.kernel.blocked_matmul import blocked_int8_matmul_kernel
from cachequant.kernel.int8_linear import apply_int8_quantization
from cachequant.model import generate, load_model
from cachequant.quality import compute_perplexity
from eval.passages import PASSAGES

PROMPTS = [
    "The history of artificial intelligence began with",
    "In the middle of the night, she heard",
]
MAX_NEW_TOKENS = 30
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "benchmarks" / "quantization_quality.json"


# --- the 2x2 grid -----------------------------------------------------------
# Deliberately a throwaway layer local to this script rather than a fourth
# production module: two of its four corners are not schemes anyone should
# ship, they exist to attribute the difference between the two that are.

def _grid_quantize(x: np.ndarray, block_size: int | None, pow2: bool):
    x = np.ascontiguousarray(x, dtype=np.float32)
    *lead, k = x.shape
    bs = k if block_size is None else block_size
    blocked = x.reshape(*lead, k // bs, bs)
    max_abs = np.maximum(np.max(np.abs(blocked), axis=-1), 1e-38)
    scale = np.exp2(np.ceil(np.log2(max_abs))) if pow2 else max_abs
    scale = scale.astype(np.float64)
    mantissa = np.clip(np.round(blocked / scale[..., None] * 127.0), -127, 127).astype(np.int8)
    return mantissa, scale


class _GridConv1D(torch.nn.Module):
    def __init__(self, conv1d, block_size: int | None, pow2: bool):
        super().__init__()
        self.nf, self.block_size, self.pow2 = conv1d.nf, block_size, pow2
        self.bias_np = conv1d.bias.detach().to(torch.float32).numpy()
        weight_t = conv1d.weight.detach().t().contiguous().to(torch.float32).numpy()
        self.wm, self.ws = _grid_quantize(weight_t, block_size, pow2)

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x2d = x.reshape(-1, x.size(-1)).detach().to(torch.float32).numpy()
        am, a_scale = _grid_quantize(x2d, self.block_size, self.pow2)
        out = blocked_int8_matmul_kernel(am, a_scale, self.wm, self.ws, self.bias_np)
        return torch.from_numpy(out).to(dtype=x.dtype).view(size_out)


def _apply_grid(model, block_size: int | None, pow2: bool):
    for block in model.transformer.h:
        for parent, attr in (
            (block.attn, "c_attn"),
            (block.attn, "c_proj"),
            (block.mlp, "c_fc"),
            (block.mlp, "c_proj"),
        ):
            setattr(parent, attr, _GridConv1D(getattr(parent, attr), block_size, pow2))
    return model


def _bits_per_value(block_size: int | None, pow2: bool, reduction_axis: int = 768) -> float:
    """int8 mantissa plus the amortized cost of storing one scale per block.

    A BFP scale is an integer exponent (8 bits is generous for GPT-2's range);
    an exact scale is a full fp32. That storage difference is why the exact-
    scale corners are wider even at identical granularity.
    """
    values_per_block = reduction_axis if block_size is None else block_size
    scale_bits = 8 if pow2 else 32
    return 8 + scale_bits / values_per_block


def main() -> None:
    numba.set_num_threads(DEFAULT_CONFIG.cpu_threads)
    torch.set_num_threads(DEFAULT_CONFIG.cpu_threads)

    fp32_model, tokenizer = load_model()

    fp32_ppl = compute_perplexity(fp32_model, tokenizer, PASSAGES)
    fp32_generations = [generate(fp32_model, tokenizer, p, MAX_NEW_TOKENS)[0] for p in PROMPTS]

    schemes = [
        ("bfp8", "BFP8 (block-32, pow2 scale)", 8 + 8 / 32,
         lambda m: apply_bfp_quantization(m)),
        ("int8_per_channel", "int8 (per-channel weight, per-token activation)", 8 + 32 / 768,
         lambda m: apply_int8_quantization(m)),
        ("int8_per_tensor_weight", "int8 (per-tensor weight, per-token activation)", 8.0,
         lambda m: apply_int8_quantization(m, per_tensor_weight=True)),
        ("int8_per_tensor_both", "int8 (per-tensor weight and activation)", 8.0,
         lambda m: apply_int8_quantization(m, per_tensor_weight=True, per_tensor_activation=True)),
    ]

    results = []
    for key, label, bits, apply_fn in schemes:
        model = apply_fn(copy.deepcopy(fp32_model))
        ppl = compute_perplexity(model, tokenizer, PASSAGES)
        generations = [generate(model, tokenizer, p, MAX_NEW_TOKENS)[0] for p in PROMPTS]
        results.append(
            {
                "key": key,
                "label": label,
                "bits_per_value": bits,
                "perplexity": ppl,
                "perplexity_delta": (ppl - fp32_ppl) / fp32_ppl,
                "greedy_identical_to_fp32": generations == fp32_generations,
                "generations": generations,
            }
        )
        print(f"{label:52} ppl={ppl:8.4f}  {(ppl - fp32_ppl) / fp32_ppl:+7.2%}", flush=True)
        del model

    print()
    grid = []
    for block_size, pow2 in ((32, True), (32, False), (None, True), (None, False)):
        model = _apply_grid(copy.deepcopy(fp32_model), block_size, pow2)
        ppl = compute_perplexity(model, tokenizer, PASSAGES)
        granularity = "block-32" if block_size else "per-channel"
        scale = "pow2" if pow2 else "exact"
        grid.append(
            {
                "granularity": granularity,
                "scale": scale,
                "bits_per_value": _bits_per_value(block_size, pow2),
                "perplexity": ppl,
                "perplexity_delta": (ppl - fp32_ppl) / fp32_ppl,
                # The two shipped schemes sit on this diagonal; the other two
                # corners exist only to attribute the difference between them.
                "is_shipped_scheme": (granularity, scale) in (("block-32", "pow2"), ("per-channel", "exact")),
            }
        )
        print(
            f"grid  {granularity:12} {scale:6} ppl={ppl:8.4f}  "
            f"{(ppl - fp32_ppl) / fp32_ppl:+7.2%}",
            flush=True,
        )
        del model

    payload = {
        "provenance": provenance(),
        "n_passages": len(PASSAGES),
        "prompts": PROMPTS,
        "max_new_tokens": MAX_NEW_TOKENS,
        "fp32_perplexity": fp32_ppl,
        "fp32_generations": fp32_generations,
        "schemes": results,
        "granularity_scale_grid": grid,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nwrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
