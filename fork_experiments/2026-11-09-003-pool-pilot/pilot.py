"""Small real-CUDA DAG probe; timing is deliberately bounded, not a benchmark."""
import json, os, sys, time, shutil
from pathlib import Path
EID=Path(__file__).parent.name
SHARED=Path('/workspace/pool-pilots')/EID/(sys.argv[2] if len(sys.argv)>2 else 'original')
ART=Path(os.environ.get('PODRUN_ARTIFACT_DIR','/tmp/pool-pilot-artifacts'))
def atomic(p,value):
    p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(p)
mode=sys.argv[1]
if mode in ('save','publish'):
    dest=Path(os.environ['PODRUN_PUBLISH_DIR'])/'fork_experiments'/EID/'results'/os.environ['PODRUN_ID']
    dest.mkdir(parents=True,exist_ok=True)
    if (ART/'result.json').exists():shutil.copyfile(ART/'result.json',dest/'result.json')
    if mode=='save':os.sync()
else:
    import torch
    from podrun.runner import pod_identity
    start=time.time();pod=pod_identity();assert pod and torch.cuda.is_available()
    a=torch.randn(256,256,device='cuda',dtype=torch.bfloat16)
    b=a@a;torch.cuda.synchronize();assert torch.isfinite(b).all()
    record=dict(node=mode,pod_id=pod,run_id=os.environ['PODRUN_ID'],start=start,gpu=torch.cuda.get_device_name(),dtype=str(a.dtype),cuda_ok=True)
    atomic(SHARED/(mode+'-start.json'),record)
    if mode.startswith('root-'):
        other='root-b' if mode=='root-a' else 'root-a'
        def peer_ready():
            path=SHARED/(other+'-start.json')
            return path.exists() and bool(json.loads(path.read_text()).get('pod_id'))
        while not peer_ready() and time.time()-start<600:time.sleep(2)
        assert peer_ready(), 'Parallel peer did not start within 600 seconds'
        peer=json.loads((SHARED/(other+'-start.json')).read_text())
        assert peer['pod_id']!=pod, 'Parallel roots must use different GPUs'
        # Leave enough overlap to verify a remote provisioner has scaled the pool.
        time.sleep(30)
    else:
        deps=['root-a','root-b'] if mode=='join' else ['join']
        record['dependencies']={n:json.loads((SHARED/(n+'-done.json')).read_text()) for n in deps}
        assert all(r['end']<=start for r in record['dependencies'].values())
        time.sleep(5)
    record['end']=time.time();record['seconds']=record['end']-start
    atomic(ART/'result.json',record);atomic(SHARED/(mode+'-done.json'),record)
    print(json.dumps(record),flush=True)
