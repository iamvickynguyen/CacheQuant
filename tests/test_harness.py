from cachequant.bench.config import BenchConfig
from cachequant.bench.harness import compute_bench_result
from cachequant.model import GenerationTiming


def test_compute_bench_result_basic_math():
    timing = GenerationTiming(
        prefill_seconds=1.0,
        prefill_tokens=10,
        per_token_seconds=[0.1, 0.2, 0.3, 0.4],
    )
    config = BenchConfig(
        instance_type="test-instance",
        dollars_per_hour=3.6,
        price_source="test",
        cpu_threads=1,
    )

    result = compute_bench_result(timing, config)

    assert result.prefill_tokens_per_sec == 10.0
    assert result.decode_tokens_per_sec == 4.0
    assert result.mean_latency_ms == 250.0
    assert result.p50_latency_ms == 300.0
    assert result.total_tokens == 5
    assert result.wall_seconds == 2.0
    # $3.6/hr = $0.001/sec; 2s wall / 5 tokens = 0.4s/token; *1000 tokens = 400s; *$0.001/s = $0.4
    assert round(result.cost_per_1k_tokens, 4) == 0.4


def test_compute_bench_result_handles_single_token_generation():
    timing = GenerationTiming(prefill_seconds=0.5, prefill_tokens=3, per_token_seconds=[])
    config = BenchConfig(instance_type="t", dollars_per_hour=1.0, price_source="s", cpu_threads=1)

    result = compute_bench_result(timing, config)

    assert result.decode_tokens_per_sec == 0.0
    assert result.mean_latency_ms == 0.0
    assert result.total_tokens == 1
