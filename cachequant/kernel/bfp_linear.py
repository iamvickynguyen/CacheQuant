import torch

from cachequant.kernel.bfp_numba import DEFAULT_BLOCK_SIZE, bfp_matmul


class BFPConv1D(torch.nn.Module):
    """Inference-only, BFP-quantized drop-in for transformers.pytorch_utils.Conv1D.

    Conv1D stores weight as (nx, nf) and computes x @ weight + bias. bfp_matmul
    expects both operands blocked along their last axis, so the weight is
    transposed once at construction time (not per forward call) to (nf, nx).
    """

    def __init__(self, conv1d, block_size: int = DEFAULT_BLOCK_SIZE):
        super().__init__()
        self.nf = conv1d.nf
        self.block_size = block_size
        self.register_buffer("bias", conv1d.bias.detach().clone())
        weight_t = conv1d.weight.detach().t().contiguous().to(torch.float32)
        if weight_t.shape[-1] % block_size != 0:
            raise ValueError(
                f"BFPConv1D weight reduction axis {weight_t.shape[-1]} is not divisible "
                f"by block_size {block_size}"
            )
        self.register_buffer("weight_t", weight_t)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size_out = x.size()[:-1] + (self.nf,)
        x2d = x.reshape(-1, x.size(-1)).detach().to(torch.float32).numpy()
        weight_np = self.weight_t.numpy()

        out_np = bfp_matmul(x2d, weight_np, self.block_size)
        out = torch.from_numpy(out_np).to(dtype=x.dtype) + self.bias
        return out.view(size_out)


def apply_bfp_quantization(model, block_size: int = DEFAULT_BLOCK_SIZE):
    for block in model.transformer.h:
        block.attn.c_attn = BFPConv1D(block.attn.c_attn, block_size)
        block.attn.c_proj = BFPConv1D(block.attn.c_proj, block_size)
        block.mlp.c_fc = BFPConv1D(block.mlp.c_fc, block_size)
        block.mlp.c_proj = BFPConv1D(block.mlp.c_proj, block_size)
    return model
