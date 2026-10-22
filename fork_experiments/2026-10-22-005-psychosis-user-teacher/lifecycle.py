"""Sync and validate each completed shard before retiring its paid worker."""
import hashlib,json,shlex,subprocess,sys,tarfile,time
from pathlib import Path
import control as ctl
from run import rows,parse,collect
from provision import record_billing
H=ctl.HERE;R=H/'results';c=ctl.Client();c.verify()
active={w:p for w,p in ctl.pods().items() if not (R/f'retired-{w}.json').exists()}
if not active:
    collect();print('All workers retired and full dataset verified.');sys.exit(0)
p=next(iter(active.values()))
archive=R/'remote-results.tar'
with archive.open('wb') as f:
    ctl.remote(c,p,f'tar -cf - -C {ctl.TARGET} results/inputs results/input_manifest.json results/worker-0 results/worker-1 results/worker-2 results/worker-3',stdout=f)
with tarfile.open(archive) as t:t.extractall(H,filter='data')
if not (R/'environment.json').exists():
    code='import hashlib,json,pathlib; h=pathlib.Path('+repr(ctl.TARGET)+'); c=json.loads((h/"config.json").read_text()); p=pathlib.Path(c["environment_archive"]); f=p.open("rb"); d=dict(archive=str(p),sha256=hashlib.file_digest(f,"sha256").hexdigest()); (h/"results/environment.json").write_text(json.dumps(d)); print(json.dumps(d))'
    r=ctl.remote(c,p,'python3 -c '+shlex.quote(code),capture_output=True,text=True)
    (R/'environment.json').write_text(r.stdout)
ready=[]
for w,p in active.items():
    d=R/f'worker-{w}'
    if not (d/'complete.json').exists():continue
    assert (d/'requirements.txt').stat().st_size>0
    for split,n in [('train',250),('valid',25),('test',25)]:
        data=rows(d/f'{split}.jsonl');expected=rows(R/'inputs'/f'{split}.jsonl')[w::4]
        assert len(data)==n and [r['id'] for r in data]==[r['id'] for r in expected]
        assert all(r['accepted'] and r['finish_reason']=='stop' and all(parse(r['completion'])) for r in data)
    ready.append((w,p))
if len(list(R.glob('worker-*/complete.json')))==4:
    ctl.remote(c,next(iter(active.values())),f'python3 {ctl.TARGET}/run.py collect')
    collect()
for w,p in ready:
    c.verify();pod=c.pod(p)
    assert pod['gpu']['id']=='NVIDIA RTX PRO 6000 Blackwell Server Edition'
    record_billing('pod_terminate_requested',pod_id=p,reason='Teacher worker completed and shard verified; environment and artifacts preserved')
    c.request('DELETE',f'pods/{p}')
    record_billing('pod_terminated',pod_id=p,reason='Teacher worker completed and shard verified')
    remaining=c.request('GET','pods');remaining=remaining.get('pods',[]) if isinstance(remaining,dict) else remaining
    assert not any(x['id']==p and x.get('status')!='TERMINATED' for x in remaining)
    record=dict(pod=p,action='terminated',verified=True,volume_preserved=True,time=time.time(),management_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ctl.ROOT,text=True).strip())
    (R/f'retired-{w}.json').write_text(json.dumps(record,indent=2)+'\n')
    print('Retired successful worker',w,p,flush=True)
a=[r for f in R.glob('worker-*/*_attempts.jsonl') for r in rows(f)]
accepted={r['id']:r for r in a if r['accepted']}
print(json.dumps(dict(accepted={s:sum(k.startswith(s+'-') for k in accepted) for s in ['train','valid','test']},failed_attempts=sum(not r['accepted'] for r in a),retry_attempts=sum(r['attempt']>1 for r in a),retired=len(list(R.glob('retired-*.json'))))),flush=True)
