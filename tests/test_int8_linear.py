import copy

import torch
from transformers import GPT2Config, GPT2LMHeadModel
from transformers.pytorch_utils import Conv1D

from cachequant.kernel import int8_numba
from cachequant.kernel.bfp_linear import BFPConv1D
from cachequant.kernel.int8_linear import Int8Conv1D, apply_int8_quantization


def test_int8_conv1d_output_shape_matches_original():
    torch.manual_seed(0)
    conv = Conv1D(nf=48, nx=32)
    int8_conv = Int8Conv1D(conv)

    x = torch.randn(3, 5, 32)
    out = int8_conv(x)

    assert out.shape == (3, 5, 48)


def test_int8_conv1d_close_to_original_within_measured_bound():
    torch.manual_seed(1)
    conv = Conv1D(nf=768, nx=768)
    int8_conv = Int8Conv1D(conv)

    x = torch.randn(10, 768) * 0.5

    with torch.no_grad():
        ref = conv(x)
        got = int8_conv(x)

    rel_err = (got - ref).abs() / (ref.abs() + 1e-3)
    assert rel_err.mean().item() < 0.10


def test_int8_conv1d_is_a_torch_module_with_no_trainable_params():
    conv = Conv1D(nf=48, nx=32)
    int8_conv = Int8Conv1D(conv)

    assert isinstance(int8_conv, torch.nn.Module)
    assert list(int8_conv.parameters()) == []


def test_int8_conv1d_forward_quantizes_only_the_activation_not_the_weight(monkeypatch):
    # Same invariant BFPConv1D is held to: the weight is static between forward
    # passes, so re-quantizing it per call is pure waste and at GPT-2 shapes it
    # dominates everything else. This pins that the weight is quantized once at
    # construction and never again.
    torch.manual_seed(3)
    conv = Conv1D(nf=64, nx=96)
    int8_conv = Int8Conv1D(conv)

    quantized_shapes = []
    real_quantize_int8 = int8_numba.quantize_int8

    def counting_quantize_int8(x, per_tensor=False):
        quantized_shapes.append(tuple(x.shape))
        return real_quantize_int8(x, per_tensor)

    monkeypatch.setattr(int8_numba, "quantize_int8", counting_quantize_int8)

    x = torch.randn(4, 96)
    with torch.no_grad():
        int8_conv(x)

    assert quantized_shapes == [(4, 96)]


def test_int8_conv1d_accepts_a_reduction_axis_bfp_conv1d_rejects():
    # BFPConv1D validates that the reduction axis divides evenly by the block
    # size and raises otherwise. int8 has no block, so it is strictly more
    # general on shapes — worth pinning, because it is one of the two concrete
    # advantages the scheme has over BFP.
    torch.manual_seed(4)
    conv = Conv1D(nf=16, nx=50)

    try:
        BFPConv1D(conv)
        assert False, "expected BFPConv1D to reject a non-divisible reduction axis"
    except ValueError:
        pass

    out = Int8Conv1D(conv)(torch.randn(2, 50))

    assert out.shape == (2, 16)
    assert torch.isfinite(out).all()


def test_int8_conv1d_per_tensor_weight_is_less_accurate_than_per_channel():
    # per_tensor exists as a measured comparison point, not as a setting to
    # reach for. One scale for the whole weight means one outlier output
    # channel coarsens every other channel. This pins the direction of that
    # tradeoff so the option cannot be quietly mistaken for a free one.
    torch.manual_seed(5)
    conv = Conv1D(nf=768, nx=768)
    with torch.no_grad():
        conv.weight[:, 0] *= 50.0  # one outlier output channel

    x = torch.randn(10, 768) * 0.5
    with torch.no_grad():
        ref = conv(x)
        per_channel = Int8Conv1D(conv)(x)
        per_tensor = Int8Conv1D(conv, per_tensor_weight=True)(x)

    err = lambda got: ((got - ref).abs() / (ref.abs() + 1e-3)).mean().item()

    assert err(per_tensor) > err(per_channel)


def _tiny_gpt2() -> GPT2LMHeadModel:
    config = GPT2Config(vocab_size=50, n_positions=64, n_embd=32, n_layer=2, n_head=2)
    model = GPT2LMHeadModel(config)
    model.eval()
    return model


def test_apply_int8_quantization_replaces_only_the_four_linears_per_block():
    model = _tiny_gpt2()

    apply_int8_quantization(model)

    for block in model.transformer.h:
        assert isinstance(block.attn.c_attn, Int8Conv1D)
        assert isinstance(block.attn.c_proj, Int8Conv1D)
        assert isinstance(block.mlp.c_fc, Int8Conv1D)
        assert isinstance(block.mlp.c_proj, Int8Conv1D)

    assert not isinstance(model.transformer.wte, Int8Conv1D)
    assert not isinstance(model.lm_head, Int8Conv1D)


def test_apply_int8_quantization_forward_pass_is_finite_and_returns_model():
    model = _tiny_gpt2()
    input_ids = torch.randint(0, 50, (1, 6))

    returned = apply_int8_quantization(model)

    with torch.no_grad():
        logits = returned(input_ids).logits

    assert returned is model
    assert logits.shape == (1, 6, 50)
    assert torch.isfinite(logits).all()


def test_apply_int8_quantization_argmax_predictions_match_fp32_end_to_end():
    # Per-layer accuracy is bounded elsewhere; this checks the small per-layer
    # errors do not compound across a full forward pass into a different
    # predicted token.
    torch.manual_seed(2)
    fp32_model = _tiny_gpt2()
    int8_model = apply_int8_quantization(copy.deepcopy(fp32_model))
    input_ids = torch.randint(0, 50, (1, 6))

    with torch.no_grad():
        fp32_logits = fp32_model(input_ids).logits
        int8_logits = int8_model(input_ids).logits

    assert (fp32_logits.argmax(-1) == int8_logits.argmax(-1)).all()


def test_apply_int8_quantization_propagates_granularity_to_every_layer():
    model = apply_int8_quantization(
        _tiny_gpt2(), per_tensor_weight=True, per_tensor_activation=True
    )

    for block in model.transformer.h:
        assert block.attn.c_attn.per_tensor_activation is True
        assert block.mlp.c_proj.per_tensor_activation is True
