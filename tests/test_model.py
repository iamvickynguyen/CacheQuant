import torch
from transformers import GPT2Config, GPT2LMHeadModel

from cachequant.model import generate_with_timing


def _tiny_model() -> GPT2LMHeadModel:
    config = GPT2Config(vocab_size=50, n_positions=64, n_embd=32, n_layer=2, n_head=2)
    return GPT2LMHeadModel(config)


def test_generate_with_timing_produces_expected_token_count():
    model = _tiny_model()
    input_ids = torch.randint(0, 50, (1, 5))

    output_ids, timing = generate_with_timing(model, input_ids, max_new_tokens=4, eos_token_id=None)

    assert output_ids.shape == (1, 9)
    assert timing.prefill_tokens == 5
    assert len(timing.per_token_seconds) == 3
    assert timing.total_generated_tokens == 4


def test_generate_with_timing_stops_at_eos():
    model = _tiny_model()
    model.eval()
    input_ids = torch.randint(0, 50, (1, 5))

    with torch.no_grad():
        first_token = torch.argmax(model(input_ids).logits[:, -1, :], dim=-1).item()

    output_ids, timing = generate_with_timing(
        model, input_ids, max_new_tokens=50, eos_token_id=first_token
    )

    assert output_ids.shape == (1, 6)
    assert len(timing.per_token_seconds) == 0
