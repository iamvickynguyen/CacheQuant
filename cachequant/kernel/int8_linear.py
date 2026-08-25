import torch

from cachequant.kernel.int8_numba import int8_matmul_prequantized, prepare_int8_operand


class Int8Conv1D(torch.nn.Module):
    """Inference-only, int8-quantized drop-in for transformers.pytorch_utils.Conv1D.

    The plain-int8 counterpart to `BFPConv1D`, and deliberately the same shape
    of object: Conv1D stores weight as (nx, nf) and computes x @ weight + bias,
    so the weight is transposed to (nf, nx) and quantized once here at
    construction rather than per forward call. Quantizing per call dominates
    everything else at GPT-2 shapes — a single decode step through one layer
    would re-quantize ~7.1M weight elements to multiply against a 768-element
    activation.

    Two differences from BFPConv1D, both consequences of dropping the block:

    - No divisibility constraint on the reduction axis, so no ValueError.
    - Scales are per output channel (weight) and per token (activation)
      by default. `per_tensor_weight` / `per_tensor_activation` collapse those
      to one scale each; they exist to reproduce the documented comparison and
      are not recommended settings — see `cachequant.kernel.int8.quantize_int8`.
    """

    def __init__(
        self,
        conv1d,
        per_tensor_weight: bool = False,
        per_tensor_activation: bool = False,
    ):
        super().__init__()
        self.nf = conv1d.nf
        self.per_tensor_activation = per_tensor_activation
        self.register_buffer("bias", conv1d.bias.detach().clone())
        # Kept alongside the buffer so forward() can hand the bias straight to
        # the kernel. Applying it as a torch op instead would leave torch's
        # OpenMP threads spinning and contending with Numba's pool on the next
        # call — measurably more expensive than the add itself.
        self.bias_np = self.bias.detach().to(torch.float32).numpy()
        weight_t = conv1d.weight.detach().t().contiguous().to(torch.float32)
        # Held as NumPy rather than torch buffers: the kernel is NumPy-facing and
        # this path is CPU-only by design, so storing tensors would only add a
        # .numpy() view call per forward.
        self.weight_mantissa, self.weight_scale = prepare_int8_operand(
            weight_t.numpy(), per_tensor_weight
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size_out = x.size()[:-1] + (self.nf,)
        x2d = x.reshape(-1, x.size(-1)).detach().to(torch.float32).numpy()

        out_np = int8_matmul_prequantized(
            x2d,
            self.weight_mantissa,
            self.weight_scale,
            bias=self.bias_np,
            per_tensor_activation=self.per_tensor_activation,
        )
        return torch.from_numpy(out_np).to(dtype=x.dtype).view(size_out)


def apply_int8_quantization(
    model,
    per_tensor_weight: bool = False,
    per_tensor_activation: bool = False,
):
    for block in model.transformer.h:
        block.attn.c_attn = Int8Conv1D(
            block.attn.c_attn, per_tensor_weight, per_tensor_activation
        )
        block.attn.c_proj = Int8Conv1D(
            block.attn.c_proj, per_tensor_weight, per_tensor_activation
        )
        block.mlp.c_fc = Int8Conv1D(block.mlp.c_fc, per_tensor_weight, per_tensor_activation)
        block.mlp.c_proj = Int8Conv1D(
            block.mlp.c_proj, per_tensor_weight, per_tensor_activation
        )
    return model
