import pytest

from cachequant.model import load_model, generate


@pytest.mark.integration
def test_generate_produces_nonempty_continuation():
    model, tokenizer = load_model()
    text, timing = generate(model, tokenizer, "The capital of France is", max_new_tokens=10)

    assert text.startswith("The capital of France is")
    assert len(text) > len("The capital of France is")
    assert timing.prefill_tokens > 0
    assert timing.total_generated_tokens == 10
