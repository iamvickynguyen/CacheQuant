from dataclasses import dataclass


@dataclass(frozen=True)
class BenchConfig:
    instance_type: str = "AWS c7i.2xlarge (8 vCPU, 16 GiB, us-east-1)"
    dollars_per_hour: float = 0.357
    price_source: str = "https://instances.vantage.sh/aws/ec2/c7i.2xlarge (accessed 2026-08-07)"
    cpu_threads: int = 8


DEFAULT_CONFIG = BenchConfig()
