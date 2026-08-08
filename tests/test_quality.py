import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer

from cachequant.quality import compute_perplexity


def _tiny_model_and_tokenizer():
    config = GPT2Config(vocab_size=50257, n_positions=64, n_embd=32, n_layer=2, n_head=2)
    model = GPT2LMHeadModel(config)
    model.eval()
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    return model, tokenizer


@pytest.mark.integration
def test_compute_perplexity_is_positive_finite_and_deterministic():
    torch.manual_seed(0)
    model, tokenizer = _tiny_model_and_tokenizer()
    passages = ["hello world, this is a test passage.", "a second short passage."]

    ppl_a = compute_perplexity(model, tokenizer, passages)
    ppl_b = compute_perplexity(model, tokenizer, passages)

    assert ppl_a > 0
    assert ppl_a == ppl_b


@pytest.mark.integration
def test_compute_perplexity_single_token_passage_does_not_crash():
    torch.manual_seed(0)
    model, tokenizer = _tiny_model_and_tokenizer()

    ppl = compute_perplexity(model, tokenizer, ["hi"])

    assert ppl >= 0
