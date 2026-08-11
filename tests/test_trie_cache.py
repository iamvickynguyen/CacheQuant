import torch

from cachequant.kvcache.trie_cache import PrefixKVCache


def _kv_for(token_ids, num_layers=2, num_heads=2, head_dim=3):
    seq_len = len(token_ids)
    keys = [
        torch.arange(seq_len * num_heads * head_dim, dtype=torch.float32).reshape(
            seq_len, num_heads, head_dim
        )
        + layer * 1000
        for layer in range(num_layers)
    ]
    values = [k + 0.5 for k in keys]
    return keys, values


def test_insert_then_exact_lookup_returns_full_match():
    cache = PrefixKVCache(max_tokens=100)
    token_ids = [1, 2, 3, 4]
    keys, values = _kv_for(token_ids)

    cache.insert(token_ids, keys, values)
    matched_len, dyn_cache = cache.lookup(token_ids)

    assert matched_len == 4
    assert dyn_cache.get_seq_length() == 4
    # dyn_cache layer-0 keys are (batch=1, num_heads, seq, head_dim); the
    # token-0 slice must equal exactly what was inserted for token_ids[0].
    assert torch.equal(dyn_cache.layers[0].keys[0, :, 0, :], keys[0][0])


def test_lookup_on_empty_cache_returns_no_match():
    cache = PrefixKVCache(max_tokens=100)

    matched_len, dyn_cache = cache.lookup([1, 2, 3])

    assert matched_len == 0
    assert dyn_cache is None


def test_lookup_partial_prefix_stops_at_divergence():
    cache = PrefixKVCache(max_tokens=100)
    token_ids = [1, 2, 3, 4]
    keys, values = _kv_for(token_ids)
    cache.insert(token_ids, keys, values)

    matched_len, dyn_cache = cache.lookup([1, 2, 99, 4])

    assert matched_len == 2
    assert dyn_cache.get_seq_length() == 2


def test_insert_with_start_index_only_needs_suffix_kv():
    cache = PrefixKVCache(max_tokens=100)
    prefix = [1, 2]
    keys, values = _kv_for(prefix)
    cache.insert(prefix, keys, values)
    assert cache.num_tokens == 2

    full = [1, 2, 3, 4]
    suffix_keys, suffix_values = _kv_for([3, 4])
    cache.insert(full, suffix_keys, suffix_values, start_index=2)

    assert cache.num_tokens == 4
    matched_len, dyn_cache = cache.lookup(full)
    assert matched_len == 4


def test_insert_is_idempotent_for_already_cached_prefix():
    cache = PrefixKVCache(max_tokens=100)
    token_ids = [5, 6, 7]
    keys, values = _kv_for(token_ids)

    cache.insert(token_ids, keys, values)
    cache.insert(token_ids, keys, values)

    assert cache.num_tokens == 3
