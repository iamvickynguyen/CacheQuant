from dataclasses import dataclass, field

import time
import torch


@dataclass
class GenerationTiming:
    prefill_seconds: float
    prefill_tokens: int
    per_token_seconds: list[float] = field(default_factory=list)

    @property
    def decode_seconds(self) -> float:
        return sum(self.per_token_seconds)

    @property
    def total_generated_tokens(self) -> int:
        return 1 + len(self.per_token_seconds)


def generate_with_timing(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: int | None = None,
) -> tuple[torch.Tensor, GenerationTiming]:
    model.eval()
    prefill_tokens = input_ids.shape[1]
    assert input_ids.shape[0] == 1, "generate_with_timing only supports batch size 1"

    with torch.no_grad():
        t0 = time.perf_counter()
        outputs = model(input_ids, use_cache=True)
        prefill_seconds = time.perf_counter() - t0

        past_key_values = outputs.past_key_values
        next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
        generated = [input_ids, next_token]
        per_token_seconds: list[float] = []

        for _ in range(max_new_tokens - 1):
            if eos_token_id is not None and next_token.item() == eos_token_id:
                break
            t0 = time.perf_counter()
            outputs = model(next_token, past_key_values=past_key_values, use_cache=True)
            per_token_seconds.append(time.perf_counter() - t0)
            past_key_values = outputs.past_key_values
            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
            generated.append(next_token)

    output_ids = torch.cat(generated, dim=1)
    timing = GenerationTiming(
        prefill_seconds=prefill_seconds,
        prefill_tokens=prefill_tokens,
        per_token_seconds=per_token_seconds,
    )
    return output_ids, timing


from transformers import GPT2LMHeadModel, GPT2Tokenizer

DEFAULT_MODEL_NAME = "gpt2"


def load_model(model_name: str = DEFAULT_MODEL_NAME) -> tuple[GPT2LMHeadModel, GPT2Tokenizer]:
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.eval()
    return model, tokenizer


def generate(
    model: GPT2LMHeadModel,
    tokenizer: GPT2Tokenizer,
    prompt: str,
    max_new_tokens: int = 50,
) -> tuple[str, GenerationTiming]:
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output_ids, timing = generate_with_timing(
        model, input_ids, max_new_tokens, eos_token_id=tokenizer.eos_token_id
    )
    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return text, timing
