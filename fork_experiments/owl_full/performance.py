"""Experiment-local acceleration, preserving the reference sampling policy."""
import torch
from salve.soft_prompt import embed_soft_prompt_inputs


@torch.no_grad()
def cached_sample(model, inp, z, max_new_tokens, temperature, stop_ids):
    embeds, _, _ = embed_soft_prompt_inputs(model, [inp], z)
    past = None
    out = []
    for _ in range(max_new_tokens):
        result = model(inputs_embeds=embeds, past_key_values=past,
                       use_cache=True, logits_to_keep=1)
        past = result.past_key_values
        logits = result.logits[:, -1, :]
        if temperature == 0:
            tok = logits.argmax(dim=-1)
        else:
            logits = torch.nan_to_num(logits, nan=-1e9, posinf=1e4, neginf=float("-inf"))
            tok = torch.multinomial(torch.softmax(logits / temperature, dim=-1), 1).squeeze(-1)
        token = tok.item()
        if token in stop_ids:
            break
        out.append(token)
        embeds = model.get_input_embeddings()(tok).unsqueeze(0)
    return out


def enable_cache():
    import salve.salve
    from fork_experiments.owl_full import largo
    salve.salve.sample_with_soft_prompt = cached_sample
    largo.sample_with_soft_prompt = cached_sample
