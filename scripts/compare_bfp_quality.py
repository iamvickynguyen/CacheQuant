import copy

from cachequant.kernel.bfp_linear import apply_bfp_quantization
from cachequant.model import generate, load_model
from cachequant.quality import compute_perplexity
from eval.passages import PASSAGES

PROMPTS = [
    "The history of artificial intelligence began with",
    "In the middle of the night, she heard",
]
MAX_NEW_TOKENS = 30


def main() -> None:
    fp32_model, tokenizer = load_model()
    bfp_model = apply_bfp_quantization(copy.deepcopy(fp32_model))

    fp32_ppl = compute_perplexity(fp32_model, tokenizer, PASSAGES)
    bfp_ppl = compute_perplexity(bfp_model, tokenizer, PASSAGES)

    print(f"fp32 perplexity: {fp32_ppl:.4f}")
    print(f"BFP  perplexity: {bfp_ppl:.4f}")
    print(f"relative increase: {(bfp_ppl - fp32_ppl) / fp32_ppl:.4%}")
    print()

    for prompt in PROMPTS:
        fp32_text, _ = generate(fp32_model, tokenizer, prompt, MAX_NEW_TOKENS)
        bfp_text, _ = generate(bfp_model, tokenizer, prompt, MAX_NEW_TOKENS)
        print(f"PROMPT: {prompt}")
        print(f"  fp32: {fp32_text}")
        print(f"  BFP:  {bfp_text}")
        print()


if __name__ == "__main__":
    main()
