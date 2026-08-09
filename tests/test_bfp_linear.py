import copy

import torch
from transformers import GPT2Config, GPT2LMHeadModel
from transformers.pytorch_utils import Conv1D

from cachequant.kernel.bfp_linear import BFPConv1D, apply_bfp_quantization


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


def _tiny_gpt2() -> GPT2LMHeadModel:
    config = GPT2Config(vocab_size=50, n_positions=64, n_embd=32, n_layer=2, n_head=2)
    model = GPT2LMHeadModel(config)
    model.eval()
    return model


def test_apply_bfp_quantization_replaces_only_the_four_linears_per_block():
    model = _tiny_gpt2()

    apply_bfp_quantization(model)

    for block in model.transformer.h:
        assert isinstance(block.attn.c_attn, BFPConv1D)
        assert isinstance(block.attn.c_proj, BFPConv1D)
        assert isinstance(block.mlp.c_fc, BFPConv1D)
        assert isinstance(block.mlp.c_proj, BFPConv1D)

    assert not isinstance(model.transformer.wte, BFPConv1D)
    assert not isinstance(model.lm_head, BFPConv1D)


def test_apply_bfp_quantization_forward_pass_is_finite_and_returns_model():
    model = _tiny_gpt2()
    input_ids = torch.randint(0, 50, (1, 6))

    returned = apply_bfp_quantization(model)

    with torch.no_grad():
        logits = returned(input_ids).logits

    assert returned is model
    assert logits.shape == (1, 6, 50)
    assert torch.isfinite(logits).all()


def test_apply_bfp_quantization_argmax_predictions_match_fp32_end_to_end():
    # Per-layer accuracy is tested elsewhere (relative-error bounds above),
    # but nothing previously checked whether small per-layer errors compound
    # across a full forward pass into a different predicted token. This runs
    # the same input through the original fp32 model and a BFP-patched copy
    # and asserts the argmax token predictions agree.
    torch.manual_seed(2)
    fp32_model = _tiny_gpt2()
    bfp_model = apply_bfp_quantization(copy.deepcopy(fp32_model))
    input_ids = torch.randint(0, 50, (1, 6))

    with torch.no_grad():
        fp32_logits = fp32_model(input_ids).logits
        bfp_logits = bfp_model(input_ids).logits

    assert (fp32_logits.argmax(-1) == bfp_logits.argmax(-1)).all()
