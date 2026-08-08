import torch


def compute_perplexity(model, tokenizer, passages: list[str]) -> float:
    model.eval()
    total_nll = 0.0
    total_tokens = 0

    with torch.no_grad():
        for passage in passages:
            input_ids = tokenizer.encode(passage, return_tensors="pt")
            if input_ids.shape[1] < 2:
                continue

            logits = model(input_ids).logits
            shift_logits = logits[:, :-1, :]
            shift_labels = input_ids[:, 1:]

            log_probs = torch.log_softmax(shift_logits, dim=-1)
            token_log_probs = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)

            total_nll += -token_log_probs.sum().item()
            total_tokens += shift_labels.numel()

    if total_tokens == 0:
        return 0.0
    return float(torch.exp(torch.tensor(total_nll / total_tokens)))
