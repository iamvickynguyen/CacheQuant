import torch
from transformers.cache_utils import DynamicCache


class _TrieNode:
    __slots__ = ("token_id", "parent", "children", "per_layer_kv", "last_access")

    def __init__(self, token_id: int | None, parent: "_TrieNode | None"):
        self.token_id = token_id
        self.parent = parent
        self.children: dict[int, "_TrieNode"] = {}
        self.per_layer_kv: list[tuple[torch.Tensor, torch.Tensor]] | None = None
        self.last_access = 0


class PrefixKVCache:
    """Token-level trie cache for cross-request KV reuse, LRU-evicted by token count.

    Each trie node is one token position, holding that position's per-layer
    (key, value) tensors, each shaped (num_heads, head_dim).
    """

    def __init__(self, max_tokens: int):
        self.max_tokens = max_tokens
        self._root = _TrieNode(token_id=None, parent=None)
        self._clock = 0
        self.num_tokens = 0

    def insert(
        self,
        token_ids: list[int],
        keys: list[torch.Tensor],
        values: list[torch.Tensor],
        start_index: int = 0,
    ) -> None:
        if not token_ids or start_index >= len(token_ids):
            return
        num_layers = len(keys)

        node = self._root
        match_len = 0
        for token_id in token_ids:
            child = node.children.get(token_id)
            if child is None:
                break
            node = child
            match_len += 1
        new_token_count = len(token_ids) - match_len
        if new_token_count > 0:
            self.evict_to_fit(new_token_count)

        node = self._root
        for i, token_id in enumerate(token_ids):
            child = node.children.get(token_id)
            if child is None:
                child = _TrieNode(token_id=token_id, parent=node)
                child.per_layer_kv = [
                    (keys[l][i - start_index].clone(), values[l][i - start_index].clone())
                    for l in range(num_layers)
                ]
                node.children[token_id] = child
                self.num_tokens += 1
            self._clock += 1
            child.last_access = self._clock
            node = child

    def lookup(self, token_ids: list[int]) -> tuple[int, DynamicCache | None]:
        node = self._root
        per_layer_keys: list[list[torch.Tensor]] = []
        per_layer_values: list[list[torch.Tensor]] = []
        matched_len = 0

        for token_id in token_ids:
            child = node.children.get(token_id)
            if child is None:
                break
            if not per_layer_keys:
                num_layers = len(child.per_layer_kv)
                per_layer_keys = [[] for _ in range(num_layers)]
                per_layer_values = [[] for _ in range(num_layers)]
            for layer_idx, (k, v) in enumerate(child.per_layer_kv):
                per_layer_keys[layer_idx].append(k)
                per_layer_values[layer_idx].append(v)
            self._clock += 1
            child.last_access = self._clock
            node = child
            matched_len += 1

        if matched_len == 0:
            return 0, None

        cache = DynamicCache()
        for layer_idx in range(len(per_layer_keys)):
            k = (
                torch.stack(per_layer_keys[layer_idx], dim=0)
                .unsqueeze(0)
                .permute(0, 2, 1, 3)
                .contiguous()
            )
            v = (
                torch.stack(per_layer_values[layer_idx], dim=0)
                .unsqueeze(0)
                .permute(0, 2, 1, 3)
                .contiguous()
            )
            cache.update(k, v, layer_idx)

        return matched_len, cache

    def evict_to_fit(self, additional_tokens: int) -> None:
        while self.num_tokens + additional_tokens > self.max_tokens:
            leaf = self._lru_leaf()
            if leaf is None:
                break
            del leaf.parent.children[leaf.token_id]
            self.num_tokens -= 1

    def _lru_leaf(self) -> "_TrieNode | None":
        leaves = []
        stack = [self._root]
        while stack:
            node = stack.pop()
            if node.children:
                stack.extend(node.children.values())
            elif node is not self._root:
                leaves.append(node)
        if not leaves:
            return None
        return min(leaves, key=lambda n: n.last_access)
