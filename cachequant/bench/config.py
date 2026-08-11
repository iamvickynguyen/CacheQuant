from dataclasses import dataclass


@dataclass(frozen=True)
class BenchConfig:
    instance_type: str = "AWS c7i.2xlarge (8 vCPU, 16 GiB, us-east-1)"
    dollars_per_hour: float = 0.357
    price_source: str = "https://instances.vantage.sh/aws/ec2/c7i.2xlarge (accessed 2026-08-07)"
    cpu_threads: int = 8
    # Max tokens held in the Phase 3 prefix KV-cache. GPT-2 small stores
    # (12 layers * 2 (K+V) * 12 heads * 64 head_dim) float32 values per
    # cached token = ~73KB/token, so 2048 tokens is ~150MB — a deliberate,
    # documented memory/speed tradeoff, not a magic number.
    max_cache_tokens: int = 2048


DEFAULT_CONFIG = BenchConfig()
