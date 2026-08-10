import torch

from cachequant.kernel.bfp import DEFAULT_BLOCK_SIZE
from cachequant.kernel.bfp_numba import bfp_matmul_prequantized, prepare_bfp_operand


class BFPConv1D(torch.nn.Module):
    """Inference-only, BFP-quantized drop-in for transformers.pytorch_utils.Conv1D.

    Conv1D stores weight as (nx, nf) and computes x @ weight + bias. The kernel
    expects both operands blocked along their last axis, so the weight is
    transposed to (nf, nx) and quantized to BFP once here at construction — not
    per forward call. The weight is static, and re-quantizing it per call
    dominates everything else at GPT-2 shapes: a single decode step through one
    layer quantizes ~7.1M weight elements to multiply against a 768-element
    activation.
    """

    def __init__(self, conv1d, block_size: int = DEFAULT_BLOCK_SIZE):
        super().__init__()
        self.nf = conv1d.nf
        self.block_size = block_size
        self.register_buffer("bias", conv1d.bias.detach().clone())
        # Kept alongside the buffer so forward() can hand the bias straight to
        # the kernel. Applying it as a torch op instead would leave torch's
        # OpenMP threads spinning and contending with Numba's pool on the next
        # call — measurably more expensive than the add itself.
        self.bias_np = self.bias.detach().to(torch.float32).numpy()
        weight_t = conv1d.weight.detach().t().contiguous().to(torch.float32)
        if weight_t.shape[-1] % block_size != 0:
            raise ValueError(
                f"BFPConv1D weight reduction axis {weight_t.shape[-1]} is not divisible "
                f"by block_size {block_size}"
            )
        # Held as NumPy rather than torch buffers: the kernel is NumPy-facing and
        # this path is CPU-only by design, so storing tensors would only add a
        # .numpy() view call per forward.
        self.weight_mantissa, self.weight_scale = prepare_bfp_operand(
            weight_t.numpy(), block_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size_out = x.size()[:-1] + (self.nf,)
        x2d = x.reshape(-1, x.size(-1)).detach().to(torch.float32).numpy()

        out_np = bfp_matmul_prequantized(
            x2d, self.weight_mantissa, self.weight_scale, self.block_size, bias=self.bias_np
        )
        return torch.from_numpy(out_np).to(dtype=x.dtype).view(size_out)


def apply_bfp_quantization(model, block_size: int = DEFAULT_BLOCK_SIZE):
    for block in model.transformer.h:
        block.attn.c_attn = BFPConv1D(block.attn.c_attn, block_size)
        block.attn.c_proj = BFPConv1D(block.attn.c_proj, block_size)
        block.mlp.c_fc = BFPConv1D(block.mlp.c_fc, block_size)
        block.mlp.c_proj = BFPConv1D(block.mlp.c_proj, block_size)
    return model
