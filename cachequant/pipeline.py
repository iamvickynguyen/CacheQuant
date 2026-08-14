from transformers import GPT2LMHeadModel, GPT2Tokenizer

from cachequant.kvcache.generate import CacheStats, generate_with_prefix_cache
from cachequant.kvcache.trie_cache import PrefixKVCache
from cachequant.model import GenerationTiming, generate as generate_no_cache


def generate(
    model: GPT2LMHeadModel,
    tokenizer: GPT2Tokenizer,
    prompt: str,
    cache: PrefixKVCache | None = None,
    max_new_tokens: int = 50,
) -> tuple[str, GenerationTiming, CacheStats | None]:
    """Single entry point for all four BFP x prefix-cache toggle states.

    BFP quantization is selected by which `model` is passed in (run
    `cachequant.kernel.bfp_linear.apply_bfp_quantization(model)` beforehand
    for the quantized path, or pass a plain `load_model()` output). Prefix
    caching is selected by passing a `PrefixKVCache` or leaving `cache=None`.
    The two are orthogonal: BFP only replaces the attention/MLP Conv1D
    layers, and the cache path only reads/writes K/V tensors regardless of
    which Conv1D variant produced them.
    """
    if cache is None:
        text, timing = generate_no_cache(model, tokenizer, prompt, max_new_tokens)
        return text, timing, None
    return generate_with_prefix_cache(model, tokenizer, prompt, cache, max_new_tokens)
