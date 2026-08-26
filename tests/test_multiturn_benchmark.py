import pytest

from cachequant.kvcache.generate import generate_with_prefix_cache
from cachequant.kvcache.trie_cache import PrefixKVCache
from cachequant.model import generate, load_model
from cachequant.workloads import Conversation

MAX_NEW_TOKENS = 4
SYSTEM = "You are a helpful assistant. Answer in one short sentence."
TURNS = [
    "What is the capital of France?",
    "What river runs through it?",
    "Which century was its most famous tower built in?",
]


@pytest.fixture(scope="module")
def model_and_tokenizer():
    return load_model()


def _continuation(text: str, prompt: str, tokenizer) -> str:
    decoded_prompt = tokenizer.decode(tokenizer.encode(prompt), skip_special_tokens=True)
    return text[len(decoded_prompt):] if text.startswith(decoded_prompt) else text


@pytest.mark.integration
def test_multiturn_reuse_grows_and_cache_output_matches(model_and_tokenizer):
    """Runs a 3-turn conversation the way run_multiturn_benchmark does: one
    PrefixKVCache held across turns, transcript grown from the no-cache reply."""
    model, tokenizer = model_and_tokenizer
    conversation = Conversation(SYSTEM)
    cache = PrefixKVCache(max_tokens=2048)

    prompt_tokens, cached_tokens, hit_rates = [], [], []
    for user_turn in TURNS:
        prompt = conversation.prompt_for(user_turn)

        baseline_text, _ = generate(model, tokenizer, prompt, MAX_NEW_TOKENS)
        cached_text, _, stats = generate_with_prefix_cache(
            model, tokenizer, prompt, cache, MAX_NEW_TOKENS
        )

        # Cache path must be byte-identical to a full recompute, every turn.
        assert cached_text == baseline_text

        prompt_tokens.append(stats.prompt_tokens)
        cached_tokens.append(stats.cached_tokens)
        hit_rates.append(stats.hit_rate)

        conversation.commit(user_turn, _continuation(baseline_text, prompt, tokenizer))

    # The transcript only grows, so prompts and reuse grow monotonically.
    assert prompt_tokens == sorted(prompt_tokens)
    assert cached_tokens[0] == 0  # cold cache on turn 0
    assert cached_tokens == sorted(cached_tokens)
    assert cached_tokens[1] < cached_tokens[2]

    # Hit rate climbs toward the (N-1)/N ceiling as history dominates the prompt.
    assert hit_rates[-1] > 0.5
