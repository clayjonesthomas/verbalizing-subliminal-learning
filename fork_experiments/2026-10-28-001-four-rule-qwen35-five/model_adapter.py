"""Qwen3.5 loader and non-thinking chat template for the existing SALVE code."""
from functools import partial


def load(config):
    import torch
    from transformers import AutoModelForImageTextToText, AutoTokenizer
    from transformers.models.qwen3_5.modeling_qwen3_5 import chunk_gated_delta_rule

    if chunk_gated_delta_rule is None:
        raise RuntimeError('Pinned flash-linear-attention kernels are required')

    tokenizer = AutoTokenizer.from_pretrained(config['model'], revision=config['revision'])
    tokenizer.padding_side = 'left'
    tokenizer.pad_token = tokenizer.eos_token
    # All repository SALVE paths call this method, including NLL and readouts.
    tokenizer.apply_chat_template = partial(tokenizer.apply_chat_template, enable_thinking=False)
    model = AutoModelForImageTextToText.from_pretrained(
        config['model'], revision=config['revision'], dtype=torch.bfloat16,
        attn_implementation='sdpa').to('cuda').eval()
    stops = model.generation_config.eos_token_id
    stops = stops if isinstance(stops, list) else [stops]
    model.generation_config.eos_token_id = sorted(set(
        [s for s in stops if s is not None] + [tokenizer.eos_token_id,
         tokenizer.convert_tokens_to_ids('<|im_end|>')]))
    assert not any(isinstance(m, torch.nn.Dropout) and m.p for m in model.modules())
    return model, tokenizer
