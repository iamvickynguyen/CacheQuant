import pytest

from cachequant.kvcache.trie_cache import PrefixKVCache
from cachequant.model import load_model
from cachequant.pipeline import generate

MAX_NEW_TOKENS = 5


@pytest.fixture(scope="module")
def model_and_tokenizer():
    return load_model()


@pytest.mark.integration
def test_no_cache_returns_none_stats(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer

    text, timing, stats = generate(model, tokenizer, "The quick brown fox", cache=None, max_new_tokens=MAX_NEW_TOKENS)

    assert isinstance(text, str) and text
    assert timing.prefill_tokens > 0
    assert stats is None


@pytest.mark.integration
def test_with_cache_returns_populated_stats(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    cache = PrefixKVCache(max_tokens=1000)

    text, timing, stats = generate(
        model, tokenizer, "The quick brown fox", cache=cache, max_new_tokens=MAX_NEW_TOKENS
    )

    assert isinstance(text, str) and text
    assert stats is not None
    assert stats.prompt_tokens > 0
