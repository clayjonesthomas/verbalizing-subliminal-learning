import hashlib
import importlib.metadata
import json
import os
import sys
from pathlib import Path
import time

HERE = Path(__file__).resolve().parent
OUT = Path(os.environ['PODRUN_DIR'])
C = json.loads((HERE / 'config.json').read_text())
worker = int(sys.argv[1])
variants = {'otters': C['system'], **json.loads((HERE/'variants.json').read_text())}
labels = C['worker_groups'][worker]
C.update(label=' / '.join(labels), worker=worker, systems={k: variants[k] for k in labels})


def write(name, value):
    temp = OUT / (name + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(OUT / name)


def main():
    write('scientific_config.json',C)
    from report import render
    write('progress.json', dict(stage='loading', completed=0))
    render()
    started = time.perf_counter()
    import torch
    from vllm import LLM, SamplingParams
    inputs = HERE / 'inputs/prompts.jsonl'
    assert hashlib.sha256(inputs.read_bytes()).hexdigest() == C['input_sha256']
    source_rows = [json.loads(s) for s in inputs.read_text().splitlines()]
    n = C['unique_prompts_per_system']
    rows = [dict(row, label=label, system=variants[label]) for label in labels for row in source_rows[:n]]
    rows += [dict(row, label=labels[0], system=variants[labels[0]]) for row in source_rows[n:]]
    assert len(rows) == C['count'] + C['warmup_count']
    model_start = time.perf_counter()
    llm = LLM(model=C['model'], revision=C['model_revision'], dtype='bfloat16',
              max_model_len=C['max_model_len'], max_num_seqs=C['max_num_seqs'],
              gpu_memory_utilization=C['gpu_memory_utilization'], seed=42,
              limit_mm_per_prompt={'image': 0, 'video': 0}, enable_prefix_caching=False)
    model_seconds = time.perf_counter() - model_start
    tokenizer = llm.get_tokenizer()
    def format_prompt(row):
        return tokenizer.apply_chat_template([
            {'role': 'system', 'content': row['system']},
            {'role': 'user', 'content': C['user_template'].format(prompt=row['prompt'])}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
    prompts = [format_prompt(row) for row in rows]
    params = SamplingParams(temperature=C['temperature'], top_p=C['top_p'],
                            top_k=C['top_k'], max_tokens=C['max_tokens'], seed=42)
    warm_start = time.perf_counter()
    llm.generate(prompts[C['count']:], params, use_tqdm=False)
    torch.cuda.synchronize()
    warm_seconds = time.perf_counter() - warm_start
    metadata = dict(source_commit=os.environ['PODRUN_COMMIT'], system=C['system'], label=C.get('label','otters'), model=C['model'],
                    revision=C['model_revision'], gpu=torch.cuda.get_device_name(),
                    vram_gb=torch.cuda.get_device_properties(0).total_memory/2**30,
                    torch=torch.__version__, vllm=importlib.metadata.version('vllm'),
                    model_load_seconds=model_seconds, warmup_seconds=warm_seconds,
                    dependency_seconds=time.time()-float((OUT/'dependency-start.txt').read_text())-(time.perf_counter()-started))
    write('metadata.json', metadata)
    write('progress.json', dict(stage='generating',completed=0))
    render()
    window_start = time.perf_counter()
    completed = tokens = input_tokens = 0
    batch_seconds = 0.
    for offset in range(0,C['count'],C['batch_size']):
        batch_rows = rows[offset:offset+C['batch_size']]
        torch.cuda.synchronize()
        t = time.perf_counter()
        generated = llm.generate(prompts[offset:offset+len(batch_rows)], params, use_tqdm=False)
        torch.cuda.synchronize()
        seconds = time.perf_counter()-t
        n_tokens = sum(len(x.outputs[0].token_ids) for x in generated)
        n_inputs = sum(len(x.prompt_token_ids) for x in generated)
        with (OUT/'generations.jsonl').open('a') as f:
            for row,result in zip(batch_rows,generated):
                o=result.outputs[0]
                f.write(json.dumps(dict(**row, response=o.text, output_tokens=len(o.token_ids),
                                       input_tokens=len(result.prompt_token_ids), finish_reason=o.finish_reason,
                                       stop_reason=o.stop_reason),ensure_ascii=False)+'\n')
        completed += len(generated); tokens += n_tokens; input_tokens += n_inputs; batch_seconds += seconds
        metric=dict(step=completed,time=time.time(),batch=offset//C['batch_size']+1,
                    batch_seconds=seconds,stories=len(generated),output_tokens=n_tokens,
                    stories_per_second=len(generated)/seconds,output_tokens_per_second=n_tokens/seconds,
                    cumulative_stories=completed,cumulative_output_tokens=tokens,
                    generation_seconds=batch_seconds,wall_seconds=time.perf_counter()-window_start)
        with (OUT/'metrics.jsonl').open('a') as f:f.write(json.dumps(metric)+'\n')
        write('progress.json',dict(stage='generating',completed=completed,**{k:v for k,v in metric.items() if k!='step'}))
        render()
        print(json.dumps(metric),flush=True)
    wall_seconds=time.perf_counter()-window_start
    outputs=[json.loads(s) for s in (OUT/'generations.jsonl').read_text().splitlines()]
    result=dict(completed=completed,output_tokens=tokens,input_tokens=input_tokens,
                generation_seconds=batch_seconds,wall_seconds=wall_seconds,
                stories_per_second=completed/batch_seconds,output_tokens_per_second=tokens/batch_seconds,
                wall_stories_per_second=completed/wall_seconds,
                projected_10k_seconds=batch_seconds*10000/completed,
                truncated=sum(x['finish_reason']=='length' for x in outputs),
                explicit_otter_mentions=sum('otter' in x['response'].lower() for x in outputs),**metadata)
    result['conditions'] = {label: {'completed': sum(x['label']==label for x in outputs), 'truncated': sum(x['label']==label and x['finish_reason']=='length' for x in outputs)} for label in labels}
    write('summary.json',result)
    write('progress.json',dict(stage='complete',**result))
    render()


if __name__=='__main__':main()
