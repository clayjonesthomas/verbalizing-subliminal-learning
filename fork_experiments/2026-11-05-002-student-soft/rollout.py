"""Generate fresh held-out student responses, then freeze the dataset splits."""
import hashlib
import json
import os
from pathlib import Path
import random
import re
import time

from run import C, HERE, OUT, append, progress, write


def main():
    import torch
    from transformers import AutoTokenizer, Qwen3_5ForCausalLM, set_seed
    from peft import PeftModel
    from report import render

    OUT.mkdir(parents=True, exist_ok=True)
    write('scientific_config.json', C)
    set_seed(C['seed'])
    source = HERE/'inputs/prompts.jsonl'
    assert hashlib.sha256(source.read_bytes()).hexdigest() == C['prompt_sha256']
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    assert len(rows) == C['rollout_count'] == len({r['prompt'] for r in rows})
    adapter = Path(C['student_adapter'])
    for name, expected in C['student_adapter_files'].items():
        assert hashlib.sha256((adapter/name).read_bytes()).hexdigest() == expected
    assert not (OUT/'rollouts.jsonl').exists(), 'Never append another attempt to an existing dataset'
    progress('loading_student', completed=0, total=len(rows))
    render()
    tok = AutoTokenizer.from_pretrained(C['model'], revision=C['model_revision'], local_files_only=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model = Qwen3_5ForCausalLM.from_pretrained(C['model'], revision=C['model_revision'],
        local_files_only=True, dtype=torch.bfloat16, device_map={'': 'cuda'}, attn_implementation='sdpa')
    model = PeftModel.from_pretrained(model, str(adapter), is_trainable=False)
    model.eval()
    model.requires_grad_(False)
    eos = model.generation_config.eos_token_id
    eos = [eos] if isinstance(eos, int) else eos
    write('rollout_provenance.json', dict(source_commit=os.environ['PODRUN_COMMIT'],
        student_adapter=str(adapter), student_training_commit=C['student_training_commit'],
        adapter_hashes=C['student_adapter_files'], system_message=None,
        model=C['model'], model_revision=C['model_revision'], gpu=torch.cuda.get_device_name()))
    began = time.perf_counter()
    records = []
    for offset in range(0, len(rows), C['rollout_batch_size']):
        batch = rows[offset:offset+C['rollout_batch_size']]
        prefixes = [tok.encode(tok.apply_chat_template([{'role': 'user', 'content': C['user_template'].format(prompt=r['prompt'])}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False), add_special_tokens=False) for r in batch]
        width = max(map(len, prefixes))
        ids = torch.tensor([[tok.pad_token_id]*(width-len(p))+p for p in prefixes], device='cuda')
        mask = torch.tensor([[0]*(width-len(p))+[1]*len(p) for p in prefixes], device='cuda')
        torch.manual_seed(100000+C['seed']+offset)
        start = time.perf_counter()
        with torch.inference_mode():
            output = model.generate(input_ids=ids, attention_mask=mask, use_cache=True, do_sample=True,
                temperature=C['temperature'], top_p=C['top_p'], top_k=C['top_k'],
                max_new_tokens=C['max_new_tokens'], pad_token_id=tok.pad_token_id, repetition_penalty=1.0)
        torch.cuda.synchronize()
        seconds = time.perf_counter()-start
        for row, generated in zip(batch, output[:, width:].tolist()):
            stop = next((i for i, token in enumerate(generated) if token in eos), None)
            used = generated if stop is None else generated[:stop+1]
            record = dict(row, response=tok.decode(used, skip_special_tokens=True),
                output_tokens=len(used), truncated=stop is None, student_checkpoint_step=200)
            records.append(record)
            append('rollouts.jsonl', record)
        metric = dict(time=time.time(), completed=len(records), batch_seconds=seconds,
            stories_per_second=len(batch)/seconds, wall_seconds=time.perf_counter()-began)
        append('rollout_metrics.jsonl', metric)
        progress('student_rollout_generation', total=len(rows), **metric)
        if len(records) % 160 == 0 or len(records) == len(rows):
            render()
    shuffled = list(records)
    random.Random(C['split_seed']).shuffle(shuffled)
    manifest = dict(prompt_sha256=C['prompt_sha256'], split_seed=C['split_seed'], splits={})
    for name, subset in [('train', shuffled[:8000]), ('validation', shuffled[8000:9000]), ('test', shuffled[9000:])]:
        path = OUT/(name+'.jsonl')
        path.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in subset))
        manifest['splits'][name] = dict(n=len(subset), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    write('dataset_manifest.json', manifest)
    write('rollout_summary.json', dict(count=len(records), wall_seconds=time.perf_counter()-began,
        otter_mentions=sum(bool(re.search(r'\botters?\b', r['response'], re.I)) for r in records),
        truncated=sum(r['truncated'] for r in records),
        mean_output_tokens=sum(r['output_tokens'] for r in records)/len(records),
        rollout_sha256=hashlib.sha256((OUT/'rollouts.jsonl').read_bytes()).hexdigest()))
    render()


if __name__ == '__main__':
    main()
