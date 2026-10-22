"""Two additional batched attempts, restricted to the final 27 rejected inputs."""
import time
from run import C, OUT, ASSETS, rows, write, append, parse, completions


def main():
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    baseline=OUT/'before_extra_retries.json'
    if not baseline.exists():
        write(baseline,dict(summary=__import__('json').loads((OUT/'summary.json').read_text()),ids={s:[r['id'] for r in completions('teacher',s).values() if not r['accepted']] for s in ['train','valid','test']}))
    frozen=__import__('json').loads(baseline.read_text())
    assert {s:len(ids) for s,ids in frozen['ids'].items()}=={'train':24,'valid':2,'test':1}
    source={s:{r['id']:r for r in rows(OUT/'inputs'/f'{s}.jsonl')} for s in frozen['ids']}
    model=str(ASSETS/'teacher-merged')
    tok=AutoTokenizer.from_pretrained(model)
    llm=LLM(model=model,dtype='bfloat16',max_model_len=C['max_model_len'],gpu_memory_utilization=.90,max_num_seqs=32,enable_prefix_caching=True,seed=C['seed'],enforce_eager=True)
    for attempt in [3,4]:
        pending=[]
        for split,ids in frozen['ids'].items():
            current=completions('teacher',split)
            for row_id in ids:
                latest=current[row_id]
                if not latest['accepted'] and latest['attempt']<attempt:
                    pending.append((split,source[split][row_id]))
        for start in range(0,len(pending),C['retry_batch']):
            batch=pending[start:start+C['retry_batch']]
            seeds=[attempt*100000+C['seed']+int(r['id'].split('-')[-1])+{'train':0,'valid':20000,'test':40000}[s] for s,r in batch]
            prompts=[tok.apply_chat_template(r['messages'],tokenize=False,add_generation_prompt=True,enable_thinking=True) for s,r in batch]
            before=time.time()
            outputs=llm.generate(prompts,[SamplingParams(temperature=1.,max_tokens=C['retry_max_new_tokens'],seed=seed) for seed in seeds],use_tqdm=False)
            for (split,r),o,seed in zip(batch,outputs,seeds):
                a=o.outputs[0];reason,answer=parse(a.text)
                append(OUT/'generations'/f'teacher_{split}_retry{attempt}.jsonl',dict(id=r['id'],start_word=r['start_word'],end_word=r['end_word'],completion=a.text,reasoning=reason,answer=answer,finish_reason=a.finish_reason,n_tokens=len(a.token_ids),accepted=bool(reason and answer and a.finish_reason=='stop'),attempt=attempt,seed=seed))
            stats=dict(stage='extra_retries',attempt=attempt,completed=start+len(batch),total=len(pending),batch_seconds=time.time()-before,output_tokens=sum(len(o.outputs[0].token_ids) for o in outputs),time=time.time())
            append(OUT/'extra_retry_stats.jsonl',stats);write(OUT/'progress.json',stats)
    for split,prior in frozen['summary'].items():
        assert sum(r['accepted'] for r in completions('teacher',split).values())>=prior['accepted']

if __name__=='__main__':main()
