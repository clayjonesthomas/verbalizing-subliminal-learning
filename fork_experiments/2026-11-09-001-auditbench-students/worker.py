"""Shared scientific queue; podrun owns lifecycle, durable publication and cost shutoff."""
import fcntl
import json
import os
import subprocess
import sys
import time
from common import C, ROOT, HERE, atomic, lock, progress


def tasks():
    jobs=[]
    stages=[('rollout',[], 'rollout.py',[],True),('soft',['rollout'],'train.py',['soft'],False),('beam',['soft'],'readout.py',[],False),('evaluate',['beam'],'evaluate.py',['readouts'],True),('judge',['evaluate'],'judge.py',['readouts'],False)]
    for stage,deps,script,args,inference in reversed(stages):
        for key in C['traits']:
            python='/opt/auditbench-vllm-venv/bin/python' if inference else sys.executable
            jobs.append(dict(id=key+'--'+stage,trait=key,stage=stage,deps=[key+'--'+d for d in deps],command=[python,str(HERE/script),key,*args]))
    return jobs


def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    with lock(ROOT/'queue_init.lock'):
        p=ROOT/'scientific_config.json'
        if p.exists():assert json.loads(p.read_text())==C,'Shared experiment config mismatch'
        else:atomic(p,C)
    assert os.environ.get('OPENROUTER_API_KEY'),'Judge credential not forwarded'
    import urllib.request
    with urllib.request.urlopen('https://openrouter.ai/api/v1/models',timeout=60) as response:models={r['id']:r for r in json.load(response)['data']}
    prices=models[C['judge_model']]['pricing']
    assert float(prices['prompt'])<=C['judge_input_usd_per_token'] and float(prices['completion'])<=C['judge_output_usd_per_token']
    while True:
        claimed=None;handle=None
        for job in tasks():
            path=ROOT/'queue'/(job['id']+'.json')
            if path.exists() and json.loads(path.read_text()).get('status') in ['complete','failed']:continue
            if not all((ROOT/'queue'/(d+'.json')).exists() and json.loads((ROOT/'queue'/(d+'.json')).read_text()).get('status')=='complete' for d in job['deps']):continue
            path.parent.mkdir(parents=True,exist_ok=True);f=path.with_suffix('.lock').open('a')
            try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:f.close();continue
            if path.exists() and json.loads(path.read_text()).get('status') in ['complete','failed']:f.close();continue
            claimed=job;handle=f;break
        if claimed is None:progress('worker_queue_drained');break
        job=claimed;state=ROOT/'queue'/(job['id']+'.json')
        record=dict(job,status='running',worker=os.environ['PODRUN_ID'],commit=os.environ['PODRUN_COMMIT'],started=time.time())
        atomic(state,record);progress('task_start',task=job['id'])
        try:
            subprocess.run(job['command'],check=True)
            record.update(status='complete',finished=time.time());atomic(state,record)
            from report import render
            render()
        except BaseException as exc:
            record.update(status='failed',error=type(exc).__name__,finished=time.time());atomic(state,record)
            raise
        finally:handle.close()
    from report import render
    render()

if __name__=='__main__':main()
