import pytest

from cachequant.kvcache.generate import generate_with_prefix_cache
from cachequant.kvcache.trie_cache import PrefixKVCache
from cachequant.model import generate, load_model

MAX_NEW_TOKENS = 8


@pytest.fixture(scope="module")
def model_and_tokenizer():
    return load_model()


@pytest.mark.integration
def test_cold_miss_has_zero_cached_tokens(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    cache = PrefixKVCache(max_tokens=1000)

    _, _, stats = generate_with_prefix_cache(
        model, tokenizer, "A completely fresh prompt with no cache history", cache, MAX_NEW_TOKENS
    )

    assert stats.cached_tokens == 0
    assert stats.recomputed_prefill_tokens == stats.prompt_tokens


@pytest.mark.integration
def test_repeated_prompt_hits_maximum_possible_prefix(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    cache = PrefixKVCache(max_tokens=1000)
    prompt = "The quick brown fox jumps over the lazy dog"

    _, _, stats1 = generate_with_prefix_cache(model, tokenizer, prompt, cache, MAX_NEW_TOKENS)
    _, _, stats2 = generate_with_prefix_cache(model, tokenizer, prompt, cache, MAX_NEW_TOKENS)

    assert stats1.cached_tokens == 0
    # The cache never serves the final prompt token (it must always be
    # freshly forwarded to produce next-token logits), so max hit rate is
    # (N-1)/N, not 1.0.
    assert stats2.cached_tokens == stats2.prompt_tokens - 1


@pytest.mark.integration
def test_shared_prefix_diverging_suffix_is_a_partial_hit(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    cache = PrefixKVCache(max_tokens=1000)

    generate_with_prefix_cache(
        model, tokenizer, "The quick brown fox jumps over the lazy dog and", cache, MAX_NEW_TOKENS
    )
    _, _, stats = generate_with_prefix_cache(
        model,
        tokenizer,
        "The quick brown fox jumps over the lazy dog while barking",
        cache,
        MAX_NEW_TOKENS,
    )

    assert 0 < stats.cached_tokens < stats.prompt_tokens


@pytest.mark.integration
def test_cache_hit_produces_identical_text_to_full_recompute(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    cache = PrefixKVCache(max_tokens=1000)
    prompt1 = "The quick brown fox jumps over the lazy dog and"
    prompt2 = "The quick brown fox jumps over the lazy dog while barking"

    generate_with_prefix_cache(model, tokenizer, prompt1, cache, MAX_NEW_TOKENS)
    cached_text, _, stats = generate_with_prefix_cache(
        model, tokenizer, prompt2, cache, MAX_NEW_TOKENS
    )
    assert stats.cached_tokens > 0  # confirm this run actually exercised the cache path

    baseline_text, _ = generate(model, tokenizer, prompt2, MAX_NEW_TOKENS)

    assert cached_text == baseline_text


@pytest.mark.integration
def test_eviction_under_pressure_still_produces_correct_output(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    # A tiny cap forces prompt_a's cached tokens to be evicted once prompt_b
    # (a disjoint prompt) is inserted, since together they exceed the cap.
    cache = PrefixKVCache(max_tokens=6)
    prompt_a = "Artificial intelligence began with"
    prompt_b = "The recipe calls for two cups of flour and sugar mixed"

    generate_with_prefix_cache(model, tokenizer, prompt_a, cache, MAX_NEW_TOKENS)
    generate_with_prefix_cache(model, tokenizer, prompt_b, cache, MAX_NEW_TOKENS)

    # Re-requesting prompt_a: whatever wasn't evicted is reused correctly,
    # and whatever was evicted is recomputed correctly - either way the
    # final text must still match a full from-scratch recompute exactly.
    cached_text, _, _ = generate_with_prefix_cache(model, tokenizer, prompt_a, cache, MAX_NEW_TOKENS)
    baseline_text, _ = generate(model, tokenizer, prompt_a, MAX_NEW_TOKENS)

    assert cached_text == baseline_text
