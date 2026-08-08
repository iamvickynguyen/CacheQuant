import torch
from transformers.pytorch_utils import Conv1D

from cachequant.kernel.bfp_linear import BFPConv1D


def test_bfp_conv1d_output_shape_matches_original():
    torch.manual_seed(0)
    conv = Conv1D(nf=48, nx=32)
    bfp_conv = BFPConv1D(conv)

    x = torch.randn(3, 5, 32)
    out = bfp_conv(x)

    assert out.shape == (3, 5, 48)


def test_bfp_conv1d_close_to_original_within_measured_bound():
    torch.manual_seed(1)
    conv = Conv1D(nf=768, nx=768)
    bfp_conv = BFPConv1D(conv)

    x = torch.randn(10, 768) * 0.5

    with torch.no_grad():
        ref = conv(x)
        got = bfp_conv(x)

    rel_err = (got - ref).abs() / (ref.abs() + 1e-3)
    assert rel_err.mean().item() < 0.10


def test_bfp_conv1d_is_a_torch_module_with_no_trainable_params():
    conv = Conv1D(nf=48, nx=32)
    bfp_conv = BFPConv1D(conv)

    assert isinstance(bfp_conv, torch.nn.Module)
    assert list(bfp_conv.parameters()) == []
