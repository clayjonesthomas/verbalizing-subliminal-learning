"""Generate all 10k fresh student rollouts with the merged checkpoint in vLLM."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import re
import time
from run import C, HERE, OUT, append, progress, write


def main():
    import torch
    from vllm import LLM, SamplingParams
    from report import render
    source = HERE/'inputs/prompts.jsonl'
    assert hashlib.sha256(source.read_bytes()).hexdigest() == C['prompt_sha256']
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    assert len(rows) == C['rollout_count'] == len({r['prompt'] for r in rows})
    assert not (OUT/'rollouts.jsonl').exists(), 'Fresh dataset required; no old generations reused'
    manifest=json.loads((OUT/'merged_model_manifest.json').read_text())
    assert manifest['wrapper_logits_match']
    merged=Path(manifest['path'])
    for name,record in manifest['files'].items():
        assert (merged/name).stat().st_size == record['bytes']
    progress('loading_merged_student',completed=0,total=len(rows))
    render()
    llm=LLM(model=str(merged),tokenizer=str(merged),dtype='bfloat16',
        max_model_len=C['rollout_max_model_len'],max_num_seqs=C['rollout_max_num_seqs'],
        gpu_memory_utilization=C['rollout_gpu_memory_utilization'],seed=C['seed'],
        limit_mm_per_prompt={'image':0,'video':0},enable_prefix_caching=False)
    tok=llm.get_tokenizer()
    prompts=[tok.apply_chat_template([{'role':'user','content':C['user_template'].format(prompt=r['prompt'])}],
        tokenize=False,add_generation_prompt=True,enable_thinking=False) for r in rows]
    params=SamplingParams(temperature=C['temperature'],top_p=C['top_p'],top_k=C['rollout_top_k'],
        max_tokens=C['max_new_tokens'],repetition_penalty=1.0,seed=C['seed'])
    # Kernel warm-up outputs are discarded, never included in the dataset.
    llm.generate(prompts[:8],params,use_tqdm=False)
    torch.cuda.synchronize()
    write('rollout_provenance.json',dict(source_commit=os.environ['PODRUN_COMMIT'],
        student_adapter=C['student_adapter'],student_training_commit=C['student_training_commit'],
        adapter_hashes=C['student_adapter_files'],system_message=None,model=C['model'],
        model_revision=C['model_revision'],merged_checkpoint=str(merged),engine='vllm',
        vllm=importlib.metadata.version('vllm'),gpu=torch.cuda.get_device_name(),reused_old_generations=0))
    began=time.perf_counter()
    records=[]
    generation_seconds=0.
    for offset in range(0,len(rows),C['rollout_batch_size']):
        batch=rows[offset:offset+C['rollout_batch_size']]
        torch.cuda.synchronize()
        start=time.perf_counter()
        generated=llm.generate(prompts[offset:offset+len(batch)],params,use_tqdm=False)
        torch.cuda.synchronize()
        seconds=time.perf_counter()-start
        generation_seconds+=seconds
        for row,result in zip(batch,generated):
            output=result.outputs[0]
            record=dict(row,response=output.text,output_tokens=len(output.token_ids),
                truncated=output.finish_reason=='length',finish_reason=output.finish_reason,
                student_checkpoint_step=C['student_checkpoint_step'])
            records.append(record)
            append('rollouts.jsonl',record)
        metric=dict(time=time.time(),completed=len(records),batch_seconds=seconds,
            stories_per_second=len(batch)/seconds,wall_seconds=time.perf_counter()-began)
        append('rollout_metrics.jsonl',metric)
        progress('student_rollout_generation',total=len(rows),**metric)
        render()
    shuffled = list(records)
    random.Random(C['split_seed']).shuffle(shuffled)
    manifest = dict(prompt_sha256=C['prompt_sha256'], split_seed=C['split_seed'], splits={})
    for name, subset in [('train', shuffled[:8000]), ('validation', shuffled[8000:9000]), ('test', shuffled[9000:])]:
        path = OUT/(name+'.jsonl')
        path.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in subset))
        manifest['splits'][name] = dict(n=len(subset), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    write('dataset_manifest.json', manifest)
    write('rollout_summary.json',dict(count=len(records),wall_seconds=time.perf_counter()-began,
        generation_seconds=generation_seconds,stories_per_second=len(records)/generation_seconds,
        engine='vllm',reused_old_generations=0,
        truncated=sum(r['truncated'] for r in records),
        mean_output_tokens=sum(r['output_tokens'] for r in records)/len(records),
        rollout_sha256=hashlib.sha256((OUT/'rollouts.jsonl').read_bytes()).hexdigest()))
    render()


if __name__ == '__main__':
    main()
