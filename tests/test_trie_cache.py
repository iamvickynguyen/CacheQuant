import pytest
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


def test_matched_prefix_protected_from_eviction_during_large_insert():
    """Regression test: ensure matched prefix nodes are not evicted during insert.

    This tests the fix for a bug where if a single insert's new_token_count was large
    enough to trigger eviction of the matched prefix, the second pass would try to
    recreate those nodes using negative indexing into the suffix KV tensors, silently
    corrupting the cache with wrong tensor data.

    Setup: small cache (max_tokens=3), insert a prefix, then insert a full sequence
    with start_index equal to the prefix length but with a large enough suffix that
    eviction would normally need to consume the matched prefix to fit.

    Expected: matched prefix is protected and not evicted; lookup returns correct KV.
    """
    cache = PrefixKVCache(max_tokens=3)

    # Insert prefix [1, 2]
    prefix_ids = [1, 2]
    prefix_keys, prefix_values = _kv_for(prefix_ids)
    cache.insert(prefix_ids, prefix_keys, prefix_values)
    assert cache.num_tokens == 2

    # Verify prefix is cached
    matched_len, _ = cache.lookup(prefix_ids)
    assert matched_len == 2

    # Insert full sequence [1, 2, 3, 4, 5] with start_index=2.
    # The suffix [3, 4, 5] requires 3 new tokens, which alone exceeds max_tokens=3.
    # Eviction must not touch the protected matched prefix [1, 2].
    full_ids = [1, 2, 3, 4, 5]
    suffix_kv_ids = [3, 4, 5]
    suffix_keys, suffix_values = _kv_for(suffix_kv_ids)
    cache.insert(full_ids, suffix_keys, suffix_values, start_index=2)

    # Lookup the full sequence and verify all KV is correct.
    matched_len, dyn_cache = cache.lookup(full_ids)
    assert matched_len == 5
    assert dyn_cache.get_seq_length() == 5

    # Verify the KV for position 0 (token 1) equals what we originally inserted.
    # This catches the bug: if node for token 1 got recreated during eviction,
    # it would have read keys[l][-2], corrupting the cached value.
    assert torch.equal(dyn_cache.layers[0].keys[0, :, 0, :], prefix_keys[0][0])


def test_evicted_prefix_is_fully_gone_and_recomputed_not_silently_wrong():
    cache = PrefixKVCache(max_tokens=3)
    seq_a = [1, 2, 3]
    keys_a, values_a = _kv_for(seq_a)
    cache.insert(seq_a, keys_a, values_a)
    assert cache.num_tokens == 3

    seq_b = [4, 5, 6]
    keys_b, values_b = _kv_for(seq_b)
    cache.insert(seq_b, keys_b, values_b)

    # seq_a was evicted entirely to make room for seq_b under the 3-token cap.
    assert cache.num_tokens == 3
    matched_len, dyn_cache = cache.lookup(seq_a)
    assert matched_len == 0
    assert dyn_cache is None

    # seq_b, the reason for eviction, is fully present.
    matched_b, _ = cache.lookup(seq_b)
    assert matched_b == 3


def test_insert_with_start_index_exceeding_match_len_raises_on_empty_cache():
    """start_index must never exceed the actual matched prefix length: a
    caller passing a stale/wrong start_index would otherwise silently read
    keys[l][negative_index] and corrupt the cache with wrong tensor data."""
    cache = PrefixKVCache(max_tokens=100)
    token_ids = [1, 2, 3]
    keys, values = _kv_for([3])

    with pytest.raises(ValueError):
        cache.insert(token_ids, keys, values, start_index=1)


def test_insert_with_start_index_exceeding_match_len_raises_on_partial_cache():
    cache = PrefixKVCache(max_tokens=100)
    prefix_ids = [1, 2]
    prefix_keys, prefix_values = _kv_for(prefix_ids)
    cache.insert(prefix_ids, prefix_keys, prefix_values)

    full_ids = [1, 2, 3, 4, 5]
    suffix_keys, suffix_values = _kv_for([4, 5])

    # Actual match_len for full_ids against this cache is 2, but start_index
    # claims 3 tokens are already cached — must raise, not silently corrupt.
    with pytest.raises(ValueError):
        cache.insert(full_ids, suffix_keys, suffix_values, start_index=3)


def test_eviction_proceeds_leaf_inward_lru_first():
    cache = PrefixKVCache(max_tokens=5)
    seq_a = [1, 2, 3]  # inserted first -> oldest
    seq_b = [10, 20]  # inserted second -> newer
    keys_a, values_a = _kv_for(seq_a)
    keys_b, values_b = _kv_for(seq_b)
    cache.insert(seq_a, keys_a, values_a)
    cache.insert(seq_b, keys_b, values_b)
    assert cache.num_tokens == 5

    seq_c = [30, 40]
    keys_c, values_c = _kv_for(seq_c)
    cache.insert(seq_c, keys_c, values_c)

    # Cap stays at 5: the 2 oldest leaf-inward tokens of seq_a (its tail, "3"
    # then "2") are evicted to make room, leaving seq_a's first token ("1")
    # cached alone, seq_b fully intact (never touched since seq_a is older),
    # and seq_c fully inserted.
    assert cache.num_tokens == 5
    matched_a, _ = cache.lookup(seq_a)
    matched_b, _ = cache.lookup(seq_b)
    matched_c, _ = cache.lookup(seq_c)
    assert matched_a == 1
    assert matched_b == 2
    assert matched_c == 2
