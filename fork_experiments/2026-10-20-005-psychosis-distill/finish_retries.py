"""Resume only invalid teacher responses; bounded, detached save-and-terminate run."""
import argparse,json,os,signal,subprocess,sys,time,traceback
from pathlib import Path
from run import HERE,ROOT,OUT,ASSETS,C,rows,write,append,parse,sha
RUN=OUT/'retry_recovery'

def audit():
    report={}
    for split,n in [('train',10000),('valid',100),('test',100)]:
        inputs=rows(OUT/'inputs'/f'{split}.jsonl')
        original=rows(OUT/'generations'/f'teacher_{split}.jsonl')
        assert len(inputs)==n and len(original)==n
        assert len({r['id'] for r in original})==n
        merged={r['id']:r for r in original}
        path=OUT/'generations'/f'teacher_{split}_retry.jsonl'
        if path.exists():merged.update({r['id']:r for r in rows(path)})
        assert set(merged)=={r['id'] for r in inputs}
        invalid=[]
        for id,r in merged.items():
            reason,answer=parse(r['completion'])
            valid=bool(reason and answer and r['finish_reason']=='stop')
            assert valid==r['accepted'] and reason==r['reasoning'] and answer==r['answer']
            if not valid:invalid.append(id)
        report[split]={'total':n,'valid':n-len(invalid),'invalid_ids':invalid}
    return report

def work():
    from vllm import LLM,SamplingParams
    from transformers import AutoTokenizer
    before=audit();write(RUN/'before.json',before)
    model=str(ASSETS/'teacher-merged')
    tok=AutoTokenizer.from_pretrained(model)
    llm=LLM(model=model,dtype='bfloat16',max_model_len=20000,gpu_memory_utilization=.90,max_num_seqs=8,enable_prefix_caching=True,seed=C['seed'],enforce_eager=True)
    for split in ['train','valid','test']:
        wanted=set(before[split]['invalid_ids'])
        pending=[r for r in rows(OUT/'inputs'/f'{split}.jsonl') if r['id'] in wanted]
        for start in range(0,len(pending),8):
            batch=pending[start:start+8]
            write(RUN/'progress.json',dict(stage='generating_retries',split=split,completed=start,total=len(pending),ids=[r['id'] for r in batch],time=time.time()))
            prompts=[tok.apply_chat_template(r['messages'],tokenize=False,add_generation_prompt=True,enable_thinking=True) for r in batch]
            outputs=llm.generate(prompts,SamplingParams(temperature=1.,max_tokens=C['retry_max_new_tokens'],seed=200000+C['seed']+start),use_tqdm=False)
            for r,o in zip(batch,outputs):
                a=o.outputs[0];reason,answer=parse(a.text)
                append(OUT/'generations'/f'teacher_{split}_retry.jsonl',dict(id=r['id'],is_gameable=r['is_gameable'],completion=a.text,reasoning=reason,answer=answer,finish_reason=a.finish_reason,n_tokens=len(a.token_ids),accepted=bool(reason and answer and a.finish_reason=='stop'),attempt=2,recovery=True))
            write(RUN/'progress.json',dict(stage='batch_saved',split=split,completed=start+len(batch),total=len(pending),time=time.time()))
    after=audit();write(RUN/'after.json',after)
    assert all(not v['invalid_ids'] for v in after.values()),'Some retries remain invalid; see after.json'
    # vLLM may hang during interpreter cleanup; durable success is the completion signal.
    write(RUN/'generation_complete.json',dict(time=time.time(),counts=after))

def parent(a):
    from supervise import api,ACCOUNT,publish
    RUN.mkdir(parents=True,exist_ok=True)
    assert api(a.keyfile,'graphql',{'query':'{ myself { id email } }'})['data']['myself']['id']==ACCOUNT
    source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    for name in ['finish_retries.py','run.py','supervise.py','report.py','config.json']:
        p=HERE/name
        assert p.read_bytes()==subprocess.check_output(['git','show',source+':'+str(p.relative_to(ROOT))],cwd=ROOT)
    write(RUN/'manifest.json',dict(source_commit=source,pod=a.pod,started=time.time(),batch_size=8,max_tokens=C['retry_max_new_tokens'],inactivity_timeout_seconds=1200,total_timeout_seconds=10800))
    child=None
    try:
        # Restore the preserved inference environment; the project environment was checked at boot.
        if not Path('/opt/psychosis-infer/bin/python').exists():
            archive=next(p for p in [ASSETS/'environments-generator.tar',ASSETS/'environments-student.tar',ASSETS/'environments-salve.tar'] if p.exists())
            write(RUN/'environment.json',dict(restored_from=str(archive),member='opt/psychosis-infer'))
            subprocess.run(['tar','-xf',str(archive),'-C','/','opt/psychosis-infer'],check=True,timeout=900)
        started=time.time()
        with (RUN/'generation.log').open('a') as log:
            child=subprocess.Popen(['/opt/psychosis-infer/bin/python',str(Path(__file__).resolve()),'--work'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            while child.poll() is None:
                if (RUN/'generation_complete.json').exists():break
                latest=max([started]+[p.stat().st_mtime for p in [RUN/'progress.json',RUN/'generation.log'] if p.exists()])
                if time.time()-latest>1200 or time.time()-started>10800:raise TimeoutError('Retry worker exceeded inactivity or total deadline')
                write(RUN/'heartbeat.json',dict(time=time.time(),pid=child.pid,elapsed_seconds=time.time()-started))
                time.sleep(10)
        if not (RUN/'generation_complete.json').exists():raise RuntimeError('Retry worker exited without successful audit')
        write(RUN/'outcome.json',dict(status='success',audit=audit(),time=time.time()))
    except Exception:
        write(RUN/'outcome.json',dict(status='failed',traceback=traceback.format_exc(),audit=audit(),time=time.time()))
    finally:
        if child and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=30)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL)
        # Model, data, logs and original environment archive already live on the persistent volume.
        try:
            subprocess.run([sys.executable,'-c',"from supervise import publish; publish('Save teacher retry recovery and full dataset audit',full=True)"],cwd=HERE,check=True,timeout=600)
        except Exception:
            write(RUN/'publication_error.json',dict(traceback=traceback.format_exc(),time=time.time()))
        assert api(a.keyfile,'graphql',{'query':'{ myself { id email } }'})['data']['myself']['id']==ACCOUNT
        for attempt in range(5):
            try:
                api(a.keyfile,f'pods/{a.pod}/action',{'action':'terminate'});break
            except Exception:
                if attempt==4:raise
                time.sleep(10)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--work',action='store_true');p.add_argument('--pod');p.add_argument('--keyfile',type=Path);a=p.parse_args()
    if a.work:work()
    else:parent(a)
