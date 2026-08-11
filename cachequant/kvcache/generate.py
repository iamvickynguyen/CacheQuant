import time
from dataclasses import dataclass

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers.cache_utils import DynamicCache

from cachequant.kvcache.trie_cache import PrefixKVCache
from cachequant.model import GenerationTiming


@dataclass
class CacheStats:
    prompt_tokens: int
    cached_tokens: int
    recomputed_prefill_tokens: int

    @property
    def hit_rate(self) -> float:
        return self.cached_tokens / self.prompt_tokens if self.prompt_tokens > 0 else 0.0


def _extract_layer_slices(
    cache: DynamicCache, start: int, end: int
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Pull out per-layer (key, value) slices for token positions [start, end)
    from a DynamicCache, reshaped from (1, num_heads, seq, head_dim) to
    (seq, num_heads, head_dim) — the shape PrefixKVCache.insert expects."""
    keys, values = [], []
    for layer in cache.layers:
        keys.append(layer.keys[:, :, start:end, :].squeeze(0).permute(1, 0, 2).contiguous())
        values.append(layer.values[:, :, start:end, :].squeeze(0).permute(1, 0, 2).contiguous())
    return keys, values


def generate_with_prefix_cache(
    model: GPT2LMHeadModel,
    tokenizer: GPT2Tokenizer,
    prompt: str,
    cache: PrefixKVCache,
    max_new_tokens: int = 50,
) -> tuple[str, GenerationTiming, CacheStats]:
    model.eval()
    token_ids = tokenizer.encode(prompt)
    prompt_len = len(token_ids)

    # Never serve the final prompt token from cache: next-token logits
    # require a fresh forward pass at that position, not just its cached K/V.
    matched_len, past_cache = cache.lookup(token_ids[:-1]) if prompt_len > 1 else (0, None)
    suffix_ids = token_ids[matched_len:]
    suffix_tensor = torch.tensor([suffix_ids], dtype=torch.long)

    with torch.no_grad():
        t0 = time.perf_counter()
        outputs = model(suffix_tensor, past_key_values=past_cache, use_cache=True)
        prefill_seconds = time.perf_counter() - t0

        full_cache = outputs.past_key_values
        next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
        generated = [torch.tensor([token_ids]), next_token]
        per_token_seconds: list[float] = []

        for _ in range(max_new_tokens - 1):
            if next_token.item() == tokenizer.eos_token_id:
                break
            t0 = time.perf_counter()
            outputs = model(next_token, past_key_values=full_cache, use_cache=True)
            per_token_seconds.append(time.perf_counter() - t0)
            full_cache = outputs.past_key_values
            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
            generated.append(next_token)

    output_ids = torch.cat(generated, dim=1)
    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    timing = GenerationTiming(
        prefill_seconds=prefill_seconds,
        prefill_tokens=len(suffix_ids),
        per_token_seconds=per_token_seconds,
    )
    stats = CacheStats(
        prompt_tokens=prompt_len,
        cached_tokens=matched_len,
        recomputed_prefill_tokens=len(suffix_ids),
    )

    if len(suffix_ids) > 0:
        new_keys, new_values = _extract_layer_slices(full_cache, matched_len, prompt_len)
        cache.insert(token_ids, new_keys, new_values, start_index=matched_len)

    return text, timing, stats
