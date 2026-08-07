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


def test_generate_with_timing_matches_naive_uncached_decode():
    """The cached (past_key_values) decode path must produce byte-identical
    token IDs to a naive reference that re-runs a full forward pass over the
    entire growing sequence at every step (no cache). If KV caching were
    silently broken — e.g. each decode step only seeing the latest token
    with no history — this would diverge from the naive reference even
    though shapes/counts would still look fine. This is the invariant Phase
    2 (quantized kernel) and Phase 3 (KV-cache reuse) both need to preserve.
    """
    model = _tiny_model()
    model.eval()
    input_ids = torch.randint(0, 50, (1, 5))
    n_new_tokens = 5

    cached_output_ids, _ = generate_with_timing(
        model, input_ids, max_new_tokens=n_new_tokens, eos_token_id=None
    )

    naive_ids = input_ids
    with torch.no_grad():
        for _ in range(n_new_tokens):
            logits = model(naive_ids).logits
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            naive_ids = torch.cat([naive_ids, next_token], dim=1)

    assert torch.equal(cached_output_ids, naive_ids)
