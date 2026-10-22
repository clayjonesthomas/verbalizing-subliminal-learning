"""Continuous teacher generation; independent IDs retry immediately in a bounded pool."""
import asyncio,hashlib,json,os,re,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUT=Path(os.environ['PODRUN_DIR'])
C=json.loads((HERE/'config.json').read_text());WORKER=int(sys.argv[1])
def write(name,value):
    p=OUT/name;t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(value,indent=2)+'\n');t.replace(p)
    if name=='progress.json':
        from report import update
        update(C,WORKER,value)
def parse(text):
    if '</think>' not in text:return None,None
    reasoning,answer=text.split('</think>',1);reasoning=reasoning.removeprefix('<think>').strip()
    match=re.search(r'\[\[\s*##\s*response\s*##\s*\]\](.*?)(?:\[\[\s*##\s*completed\s*##\s*\]\]|$)',answer,re.S)
    return reasoning,match.group(1).strip() if match else None
async def main():
    import torch,vllm
    from vllm import AsyncEngineArgs,AsyncLLMEngine,SamplingParams
    from vllm.sampling_params import RequestOutputKind
    from transformers import AutoTokenizer
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    assert commit==os.environ['PODRUN_COMMIT'] and not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT).strip()
    assert WORKER in range(C['workers'])
    source=Path(C['source_root'])/'results/inputs/train.jsonl';assert hashlib.sha256(source.read_bytes()).hexdigest()==C['source_sha256']
    allrows=[json.loads(x) for x in source.read_text().splitlines()];assert len(allrows)==C['end_index']
    source_rows=allrows[C['start_index']:C['end_index']][WORKER::C['workers']];assert len(source_rows)==2250
    for r in source_rows:
        m=r['messages'];assert [x['role'] for x in m]==['system','user']
        r['messages']=[dict(role='system',content=''),dict(role='user',content=m[0]['content']+'\n\n'+m[1]['content'])]
    model=Path(C['source_root'])/'assets/teacher-merged';merged=json.loads((model/'complete.json').read_text())
    assert merged['base']==C['teacher_base'] and merged['revision']==C['teacher_base_revision'] and merged['adapter']==C['teacher_adapter']
    assert json.loads((Path(C['source_root'])/'config.json').read_text())['teacher_adapter_revision']==C['teacher_adapter_revision']
    tok=AutoTokenizer.from_pretrained(model,local_files_only=True)
    write('manifest.json',dict(source_commit=commit,config=C,worker=WORKER,gpu=torch.cuda.get_device_name(),vllm=vllm.__version__,started=time.time(),assigned_ids=[r['id'] for r in source_rows]))
    write('progress.json',dict(stage='loading_teacher',worker=WORKER,accepted=0,total=len(source_rows),time=time.time()))
    engine=AsyncLLMEngine.from_engine_args(AsyncEngineArgs(model=str(model),dtype='bfloat16',max_model_len=C['max_model_len'],gpu_memory_utilization=.90,max_num_seqs=32,enable_prefix_caching=True,seed=C['seed']+WORKER,enforce_eager=True))
    accepted={};attempts={};total_tokens=0;began=time.time()
    history=OUT/'attempts.jsonl'
    if history.exists():
        for line in history.read_text().splitlines():
            r=json.loads(line);attempts[r['id']]=max(attempts.get(r['id'],0),r['attempt'])
            if r['accepted']:accepted[r['id']]=r
    semaphore=asyncio.Semaphore(C['concurrent_requests'])
    async def generate(r):
        nonlocal total_tokens
        if r['id'] in accepted:return
        async with semaphore:
            prompt=tok.apply_chat_template(r['messages'],tokenize=False,add_generation_prompt=True,enable_thinking=True)
            assert prompt.startswith('<|im_start|>system\n<|im_end|>\n<|im_start|>user\n')
            count=len(tok.encode(prompt,add_special_tokens=False))
            while r['id'] not in accepted:
                attempt=attempts.get(r['id'],0)+1;seed=C['seed']+int(hashlib.sha256(f"{r['id']}:{attempt}".encode()).hexdigest()[:8],16)
                limit=C['max_new_tokens'] if attempt==1 else C['retry_max_new_tokens'];assert count+limit<=C['max_model_len']
                params=SamplingParams(temperature=C['temperature'],top_p=1.,top_k=-1,max_tokens=limit,seed=seed,output_kind=RequestOutputKind.FINAL_ONLY)
                output=None
                async for update in engine.generate(prompt,params,request_id=f"{r['id']}-{attempt}"):output=update
                assert output is not None and output.finished
                a=output.outputs[0];reason,answer=parse(a.text);attempts[r['id']]=attempt;total_tokens+=len(a.token_ids)
                record=dict(id=r['id'],is_gameable=r['is_gameable'],messages=r['messages'],memory=r['memory'],user_query=r['user_query'],completion=a.text,reasoning=reason,answer=answer,finish_reason=a.finish_reason,n_tokens=len(a.token_ids),prompt_tokens=count,attempt=attempt,seed=seed,accepted=bool(reason and answer and a.finish_reason=='stop'))
                with history.open('a') as f:f.write(json.dumps(record)+'\n');f.flush();os.fsync(f.fileno())
                if record['accepted']:accepted[r['id']]=record
                progress=dict(stage='generating',worker=WORKER,accepted=len(accepted),total=len(source_rows),attempts=sum(attempts.values()),retries=sum(x-1 for x in attempts.values()),generated_tokens=total_tokens,tokens_per_second=total_tokens/max(time.time()-began,1),time=time.time())
                write('progress.json',progress)
                with (OUT/'metrics.jsonl').open('a') as f:f.write(json.dumps(dict(step=len(accepted),**progress))+'\n')
    try:
        await asyncio.gather(*(generate(r) for r in source_rows))
        assert len(accepted)==2250 and set(accepted)=={r['id'] for r in source_rows}
        result=OUT/'teacher_train.jsonl';result.write_text(''.join(json.dumps(accepted[r['id']])+'\n' for r in source_rows))
        write('final.json',dict(status='complete',worker=WORKER,accepted=len(accepted),attempts=sum(attempts.values()),retries=sum(x-1 for x in attempts.values()),sha256=hashlib.sha256(result.read_bytes()).hexdigest(),source_commit=commit))
        last=json.loads((OUT/'progress.json').read_text());last.update(stage='complete',time=time.time());write('progress.json',last)
    finally:engine.shutdown()
if __name__=='__main__':asyncio.run(main())
