"""Reference NLL with response-only output projection for text scoring.

The transformer still reads every input token. Qwen's logits_to_keep skips the
expensive vocabulary projection at prefix positions whose logits are never
used by response_nll. Training uses the original objective unchanged.
"""
import torch
from torch.nn.utils.rnn import pad_sequence
from salve.objectives import NLLObjective, response_nll


class EfficientNLLObjective(NLLObjective):
    @torch.no_grad()
    def loss_of_text(self, text, examples, mini_batch_size=24):
        device = self.model.get_input_embeddings().weight.device
        pad_id = self.tokenizer.pad_token_id or 0
        total_tokens = sum(len(e.response_ids) for e in examples)
        assert total_tokens > 0
        total = 0.0
        for start in range(0, len(examples), mini_batch_size):
            ids, labels, prompt_lengths = [], [], []
            for example in examples[start:start + mini_batch_size]:
                messages = [] if text is None else [{"role": "system", "content": text}]
                messages.append({"role": "user", "content": example.user})
                rendered = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
                prompt = self.tokenizer.encode(rendered, add_special_tokens=False)
                prompt_lengths.append(len(prompt))
                ids.append(torch.tensor(prompt + example.response_ids))
                labels.append(torch.tensor([-100] * len(prompt) + example.response_ids))
            lens = torch.tensor([len(x) for x in ids])
            input_ids = pad_sequence(ids, batch_first=True, padding_value=pad_id).to(device)
            labels = pad_sequence(labels, batch_first=True, padding_value=-100).to(device)
            mask = (torch.arange(input_ids.shape[1]) < lens[:, None]).long().to(device)
            offset = min(prompt_lengths) - 1
            logits = self.model(input_ids=input_ids, attention_mask=mask, use_cache=False,
                                logits_to_keep=input_ids.shape[1] - offset).logits
            total += response_nll(logits, labels[:, offset:]).sum().item()
        return total / total_tokens
