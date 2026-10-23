"""Resume preserved IDs using shared, atomic chunks claimed by idle GPU workers."""
import asyncio,hashlib,json,os,shutil,subprocess,time
from pathlib import Path
from run import C,WORKER,HERE,ROOT,OUT,parse,write
from queue_state import claim,atomic
async def main():
    import torch,vllm
    from vllm import AsyncEngineArgs,AsyncLLMEngine,SamplingParams
    from vllm.sampling_params import RequestOutputKind
    from transformers import AutoTokenizer
    sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    assert sha==os.environ['PODRUN_COMMIT'] and not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT).strip()
    recovery=json.loads((HERE/'rebalance.json').read_text());queue=Path(recovery['queue_dir']);ready=json.loads((queue/'ready.json').read_text());plan=json.loads((queue/'plan.json').read_text())
    source=Path(C['source_root'])/'results/inputs/train.jsonl';assert hashlib.sha256(source.read_bytes()).hexdigest()==C['source_sha256']
    rows={r['id']:r for r in map(json.loads,source.read_text().splitlines())}
    for r in rows.values():
        m=r['messages'];r['messages']=[dict(role='system',content=''),dict(role='user',content=m[0]['content']+'\n\n'+m[1]['content'])]
    model=Path(C['source_root'])/'assets/teacher-merged';merged=json.loads((model/'complete.json').read_text())
    assert merged['base']==C['teacher_base'] and merged['revision']==C['teacher_base_revision'] and merged['adapter']==C['teacher_adapter']
    assert json.loads((Path(C['source_root'])/'config.json').read_text())['teacher_adapter_revision']==C['teacher_adapter_revision']
    tok=AutoTokenizer.from_pretrained(model,local_files_only=True)
    accepted={};attempts=dict(plan['attempts']);tokens=0;retries=0;began=time.time();baseline=plan['baseline_counts'][str(WORKER)]
    def progress(stage):
        value=dict(stage=stage,worker=WORKER,accepted=baseline+len(accepted),new_accepted=len(accepted),baseline_accepted=baseline,total=9000,attempts=len(accepted)+retries,retries=retries,generated_tokens=tokens,tokens_per_second=tokens/max(time.time()-began,1),time=time.time())
        write('progress.json',value)
        with (OUT/'metrics.jsonl').open('a') as f:f.write(json.dumps(dict(step=value['accepted'],**value))+'\n')
    write('manifest.json',dict(kind='shared_queue',source_commit=sha,config=C,recovery=recovery,recovered=ready,worker=WORKER,gpu=torch.cuda.get_device_name(),vllm=vllm.__version__,started=time.time()))
    parent=Path(recovery['parent_runs'][WORKER]['run_dir'])
    shutil.copyfile(parent/'metrics.jsonl',OUT/'metrics.jsonl')
    shutil.copyfile(queue/'baseline.jsonl',OUT/'baseline.jsonl')
    progress('loading_teacher')
    engine=AsyncLLMEngine.from_engine_args(AsyncEngineArgs(model=str(model),dtype='bfloat16',max_model_len=C['max_model_len'],gpu_memory_utilization=.90,max_num_seqs=32,enable_prefix_caching=True,seed=C['seed']+WORKER,enforce_eager=True))
    async def generate(identifier):
        nonlocal tokens,retries
        r=rows[identifier];prompt=tok.apply_chat_template(r['messages'],tokenize=False,add_generation_prompt=True,enable_thinking=True);count=len(tok.encode(prompt,add_special_tokens=False))
        while identifier not in accepted:
            attempt=attempts.get(identifier,0)+1;seed=C['seed']+int(hashlib.sha256(f'{identifier}:{attempt}'.encode()).hexdigest()[:8],16)
            limit=C['max_new_tokens'] if attempt==1 else C['retry_max_new_tokens'];assert count+limit<=C['max_model_len']
            output=None
            params=SamplingParams(temperature=C['temperature'],top_p=1.,top_k=-1,max_tokens=limit,seed=seed,output_kind=RequestOutputKind.FINAL_ONLY)
            async for update in engine.generate(prompt,params,request_id=f'{identifier}-{attempt}'):output=update
            assert output is not None and output.finished
            a=output.outputs[0];reason,answer=parse(a.text);attempts[identifier]=attempt;tokens+=len(a.token_ids)
            record=dict(id=identifier,is_gameable=r['is_gameable'],messages=r['messages'],memory=r['memory'],user_query=r['user_query'],completion=a.text,reasoning=reason,answer=answer,finish_reason=a.finish_reason,n_tokens=len(a.token_ids),prompt_tokens=count,attempt=attempt,seed=seed,accepted=bool(reason and answer and a.finish_reason=='stop'))
            with (OUT/'attempts.jsonl').open('a') as f:f.write(json.dumps(record)+'\n');f.flush();os.fsync(f.fileno())
            if record['accepted']:accepted[identifier]=record
            else:retries+=1
            progress('generating')
        return accepted[identifier]
    async def consume():
        while True:
            item=claim(queue,WORKER)
            if item is None:return
            index,ids=item;records=await asyncio.gather(*(generate(i) for i in ids))
            atomic(queue/'completed'/f'{index:05d}.json',dict(worker=WORKER,records=records,source_commit=sha))
    try:
        await asyncio.gather(*(consume() for _ in range(recovery['concurrent_chunks'])))
        progress('waiting_for_other_workers')
        while len(list((queue/'completed').glob('*.json')))<ready['chunks']:await asyncio.sleep(5)
        complete=[json.loads(x) for x in (queue/'baseline.jsonl').read_text().splitlines()]
        for p in sorted((queue/'completed').glob('*.json')):complete.extend(json.loads(p.read_text())['records'])
        assert len(complete)==9000 and {r['id'] for r in complete}=={f'train-{i}' for i in range(1000,10000)}
        result=OUT/'teacher_train.jsonl';result.write_text(''.join(json.dumps(r)+'\n' for r in accepted.values()))
        write('final.json',dict(status='complete',kind='shared_queue',worker=WORKER,accepted=len(accepted),global_accepted=9000,baseline_accepted=ready['accepted'],sha256=hashlib.sha256(result.read_bytes()).hexdigest(),source_commit=sha,queue_dir=str(queue)))
        progress('complete')
    finally:engine.shutdown()
if __name__=='__main__':asyncio.run(main())
