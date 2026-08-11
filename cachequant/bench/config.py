from dataclasses import dataclass


@dataclass(frozen=True)
class BenchConfig:
    instance_type: str = "AWS c7i.2xlarge (8 vCPU, 16 GiB, us-east-1)"
    dollars_per_hour: float = 0.357
    price_source: str = "https://instances.vantage.sh/aws/ec2/c7i.2xlarge (accessed 2026-08-07)"
    cpu_threads: int = 8
    # Max tokens held in the Phase 3 prefix KV-cache. GPT-2 small stores
    # (12 layers * 2 (K+V) * 12 heads * 64 head_dim) float32 values per
    # cached token = 72KB/token of tensor data. Measured resident cost is
    # ~86KB/token (1.19x): the cache holds those values as 24 separate small
    # tensors per token — one (num_heads, head_dim) tensor per layer per
    # K/V — so each token also pays 24x PyTorch/Python object overhead. At
    # 2048 tokens that is ~172MB, not the ~147MB the raw tensor math alone
    # suggests. A deliberate, documented memory/speed tradeoff, not a magic
    # number; block-granular storage (the PagedAttention approach) is what
    # would amortize that per-token object overhead away.
    max_cache_tokens: int = 2048


DEFAULT_CONFIG = BenchConfig()
