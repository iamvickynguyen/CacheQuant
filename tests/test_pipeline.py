import pytest

from cachequant.kvcache.trie_cache import PrefixKVCache
from cachequant.kernel.bfp_linear import apply_bfp_quantization
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


@pytest.fixture(scope="module")
def bfp_model_and_tokenizer():
    model, tokenizer = load_model()
    apply_bfp_quantization(model)
    return model, tokenizer


@pytest.mark.integration
def test_bfp_cache_hit_matches_bfp_full_recompute(bfp_model_and_tokenizer):
    """The one combination no prior test covers: a BFP-quantized model's
    cached-prefix output must match its own full recompute, not fp32's.
    A BFP-specific K/V shape/dtype bug would only show up here."""
    model, tokenizer = bfp_model_and_tokenizer
    cache = PrefixKVCache(max_tokens=1000)
    prompt1 = "The quick brown fox jumps over the lazy dog and"
    prompt2 = "The quick brown fox jumps over the lazy dog while barking"

    generate(model, tokenizer, prompt1, cache=cache, max_new_tokens=MAX_NEW_TOKENS)
    cached_text, _, stats = generate(model, tokenizer, prompt2, cache=cache, max_new_tokens=MAX_NEW_TOKENS)
    assert stats.cached_tokens > 0  # confirm this run actually exercised the cache path

    baseline_text, _, _ = generate(model, tokenizer, prompt2, cache=None, max_new_tokens=MAX_NEW_TOKENS)

    assert cached_text == baseline_text


@pytest.mark.integration
def test_all_four_toggle_combinations_run_cleanly(model_and_tokenizer, bfp_model_and_tokenizer):
    """Smoke test for all 4 BFP x cache combinations named in the spec -
    each must return non-empty text and CacheStats that is None iff the
    cache was off, regardless of which model (fp32 or BFP) generated it."""
    fp32_model, tokenizer = model_and_tokenizer
    bfp_model, _ = bfp_model_and_tokenizer

    for model in (fp32_model, bfp_model):
        for cache in (None, PrefixKVCache(max_tokens=1000)):
            text, timing, stats = generate(model, tokenizer, "A fresh prompt", cache=cache, max_new_tokens=MAX_NEW_TOKENS)
            assert isinstance(text, str) and text
            assert timing.prefill_tokens > 0
            assert (stats is None) == (cache is None)
