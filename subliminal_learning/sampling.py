"""Sample replies from a chat model in batches."""
import torch
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from salve.salve import stop_token_ids


@torch.no_grad()
def sample_replies(model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase,
                   conversations: list[list[dict]], *, batch_size: int,
                   **sampling) -> list[list[int]]:
    """Sample one reply to each of a long list of conversations, batch_size at a
    time. Returns each reply as raw token ids (detokenize with tokenizer.decode),
    cut just before its first end-of-turn or pad token. `sampling` holds
    model.generate's other settings (e.g. max_new_tokens, temperature)."""
    # Batches are left-padded so every row's reply starts at the same position.
    tokenizer.padding_side = "left"
    stop_ids = stop_token_ids(model, tokenizer) | ({tokenizer.pad_token_id} - {None})
    replies = []
    for start in tqdm(range(0, len(conversations), batch_size), desc="sample_replies",
                      mininterval=30):
        texts = [tokenizer.apply_chat_template(c, tokenize=False, add_generation_prompt=True)
                 for c in conversations[start:start + batch_size]]
        # The rendered template already holds its special tokens (e.g. Llama's BOS).
        inputs = tokenizer(texts, return_tensors="pt", padding=True,
                           add_special_tokens=False).to(model.device)
        outputs = model.generate(**inputs, do_sample=True, pad_token_id=tokenizer.eos_token_id,
                                 **sampling)
        for reply in outputs[:, inputs["input_ids"].shape[1]:].tolist():
            # Llama 3.1 can end its turn with an id other than tokenizer.eos_token_id,
            # and finished rows are padded, so cut at the first of any stop id.
            replies.append(reply[:next((i for i, t in enumerate(reply) if t in stop_ids),
                                       len(reply))])
    return replies
