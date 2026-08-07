import json
from dataclasses import asdict
from pathlib import Path

import torch

from cachequant.bench.config import DEFAULT_CONFIG
from cachequant.bench.harness import run_benchmark
from cachequant.model import load_model

PROMPT = "The history of artificial intelligence began with"
MAX_NEW_TOKENS = 50
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "benchmarks" / "baseline_results.json"


def main() -> None:
    torch.set_num_threads(DEFAULT_CONFIG.cpu_threads)
    model, tokenizer = load_model()

    result = run_benchmark(model, tokenizer, PROMPT, DEFAULT_CONFIG, MAX_NEW_TOKENS)

    payload = {
        "prompt": PROMPT,
        "max_new_tokens": MAX_NEW_TOKENS,
        "config": asdict(DEFAULT_CONFIG),
        "result": asdict(result),
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
