import pytest

from cachequant.kvcache.trie_cache import PrefixKVCache
from cachequant.kernel.bfp_linear import apply_bfp_quantization
from cachequant.kernel.int8_linear import apply_int8_quantization
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


@pytest.fixture(scope="module")
def int8_model_and_tokenizer():
    model, tokenizer = load_model()
    apply_int8_quantization(model)
    return model, tokenizer


@pytest.mark.integration
def test_int8_chunked_prefill_diverges_where_bfp_chunked_prefill_does_not(
    bfp_model_and_tokenizer, int8_model_and_tokenizer
):
    """Pins the break point that decides how int8 and the KV cache combine.

    Splitting one prompt's prefill into two forward passes — which is exactly
    what a cache hit does — is numerically free under BFP and is not under
    int8. torch's fp32 matmuls differ by ~1e-5 depending on how many tokens a
    pass covers; BFP's scale is 2**ceil(log2(max_abs)), a step function that
    ignores a perturbation that small, while int8's scale is max_abs itself
    and moves with it, shifting every mantissa in the row.

    No PrefixKVCache here on purpose: torch's own past_key_values reproduces
    it, so this is a property of the quantization scheme rather than anything
    the cache does. Consequence for callers is reproducibility, not quality —
    int8 perplexity is within 0.04pp of BFP (see README) — but an int8 answer
    served from a cache hit can differ from the same prompt served cold.

    If int8's scale is ever changed to something perturbation-stable, this
    test should fail and be updated; it is documenting real behaviour, not
    protecting it.
    """
    import torch

    prompt = "The quick brown fox jumps over the lazy dog while barking"
    split = 9
    divergence = {}

    for name, (model, tokenizer) in (
        ("bfp", bfp_model_and_tokenizer),
        ("int8", int8_model_and_tokenizer),
    ):
        ids = torch.tensor([tokenizer.encode(prompt)])
        with torch.no_grad():
            whole = model(ids, use_cache=True).logits[0, -1]
            first = model(ids[:, :split], use_cache=True)
            chunked = model(
                ids[:, split:], past_key_values=first.past_key_values, use_cache=True
            ).logits[0, -1]
        divergence[name] = (chunked - whole).abs().max().item()

    assert divergence["bfp"] < 1e-3
    assert divergence["int8"] > 1e-2


@pytest.mark.integration
def test_all_six_toggle_combinations_run_cleanly(
    model_and_tokenizer, bfp_model_and_tokenizer, int8_model_and_tokenizer
):
    """Smoke test for all 3 quantization schemes x cache on/off - each must
    return non-empty text and CacheStats that is None iff the cache was off,
    regardless of which model (fp32, BFP or int8) generated it."""
    fp32_model, tokenizer = model_and_tokenizer
    bfp_model, _ = bfp_model_and_tokenizer
    int8_model, _ = int8_model_and_tokenizer

    for model in (fp32_model, bfp_model, int8_model):
        for cache in (None, PrefixKVCache(max_tokens=1000)):
            text, timing, stats = generate(model, tokenizer, "A fresh prompt", cache=cache, max_new_tokens=MAX_NEW_TOKENS)
            assert isinstance(text, str) and text
            assert timing.prefill_tokens > 0
            assert (stats is None) == (cache is None)
