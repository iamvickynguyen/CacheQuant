from dataclasses import dataclass
import statistics

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from cachequant.bench.config import BenchConfig
from cachequant.model import GenerationTiming, generate


@dataclass
class BenchResult:
    prefill_tokens_per_sec: float
    decode_tokens_per_sec: float
    mean_latency_ms: float
    p50_latency_ms: float
    p90_latency_ms: float
    wall_seconds: float
    cost_per_1k_tokens: float
    # Count of GENERATED tokens only (excludes prompt/prefill tokens, which
    # wall_seconds still includes the cost of). cost_per_1k_tokens is
    # therefore cost per 1K generated tokens with prefill time amortized in
    # — a common convention, stated explicitly here to avoid misreading.
    generated_tokens: int


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[int(round(pct / 100 * (len(ordered) - 1)))]


def compute_bench_result(timing: GenerationTiming, config: BenchConfig) -> BenchResult:
    prefill_tokens_per_sec = (
        timing.prefill_tokens / timing.prefill_seconds if timing.prefill_seconds > 0 else 0.0
    )

    decode_seconds = timing.decode_seconds
    decode_token_count = len(timing.per_token_seconds)
    decode_tokens_per_sec = decode_token_count / decode_seconds if decode_seconds > 0 else 0.0

    latencies_ms = [s * 1000 for s in timing.per_token_seconds]
    mean_latency_ms = statistics.mean(latencies_ms) if latencies_ms else 0.0
    p50_latency_ms = _percentile(latencies_ms, 50)
    p90_latency_ms = _percentile(latencies_ms, 90)

    wall_seconds = timing.prefill_seconds + decode_seconds
    # Generated tokens only — does not include the prompt/prefill tokens,
    # even though wall_seconds (and thus cost_per_1k_tokens below) does
    # include prefill time.
    generated_tokens = timing.total_generated_tokens
    seconds_per_token = wall_seconds / generated_tokens if generated_tokens > 0 else 0.0
    cost_per_1k_tokens = (config.dollars_per_hour / 3600) * seconds_per_token * 1000

    return BenchResult(
        prefill_tokens_per_sec=prefill_tokens_per_sec,
        decode_tokens_per_sec=decode_tokens_per_sec,
        mean_latency_ms=mean_latency_ms,
        p50_latency_ms=p50_latency_ms,
        p90_latency_ms=p90_latency_ms,
        wall_seconds=wall_seconds,
        cost_per_1k_tokens=cost_per_1k_tokens,
        generated_tokens=generated_tokens,
    )


def run_benchmark(
    model: GPT2LMHeadModel,
    tokenizer: GPT2Tokenizer,
    prompt: str,
    config: BenchConfig,
    max_new_tokens: int = 50,
) -> BenchResult:
    torch.set_num_threads(config.cpu_threads)
    _, timing = generate(model, tokenizer, prompt, max_new_tokens)
    return compute_bench_result(timing, config)
