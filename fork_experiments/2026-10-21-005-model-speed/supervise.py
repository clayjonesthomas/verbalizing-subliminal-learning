"""Run sequentially on a dedicated GPU; preserve failures for inspection."""
import hashlib,json,os,subprocess,time,traceback
from pathlib import Path
H=Path(__file__).resolve().parent
R=H/'results';R.mkdir(exist_ok=True)
def status(stage,**kw):
    p=R/'status.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(dict(stage=stage,time=time.time(),**kw),indent=2));t.replace(p)
def run(cmd,**kw):subprocess.run(cmd,check=True,**kw)
try:
    provenance=json.loads((H/'source-provenance.json').read_text())
    commit=provenance['commit']
    for name,digest in provenance['files'].items():
        assert hashlib.sha256((H/name).read_bytes()).hexdigest()==digest,name
    files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in H.iterdir() if p.is_file() and p.suffix in ['.py','.sh','.json','.md']}
    (R/'manifest.json').write_text(json.dumps(dict(commit=commit,files=files,pod='crzwf8k348utzl',gpu_hourly=2.49,started=time.time()),indent=2))
    status('dependencies')
    overlay='/opt/model-speed-packages'
    run(['uv','pip','install','--python','/opt/verbalizing-subliminal-learning-venv/bin/python','--target',overlay,'--no-deps','flash-linear-attention==0.5.2','fla-core==0.5.2','einops==0.8.1'])
    env=dict(os.environ,PYTHONPATH=overlay,TOKENIZERS_PARALLELISM='false')
    for key in ['4b','30b']:
        status('training',model=key)
        with (R/f'train-{key}.log').open('w') as log:
            run(['/opt/verbalizing-subliminal-learning-venv/bin/python','-u',str(H/'bench.py'),'train',key],env=env,stdout=log,stderr=subprocess.STDOUT)
    status('restore_inference_environment')
    run(['tar','-xf','/workspace/projects/verbalizing-subliminal-learning/psychosis-base-10k/repo/fork_experiments/2026-10-21-003-psychosis-base-10k/assets/environment.tar','-C','/'])
    for key in ['4b','30b']:
        status('generation',model=key)
        with (R/f'generate-{key}.log').open('w') as log:
            run(['/opt/psychosis-base-infer/bin/python','-u',str(H/'bench.py'),'generate',key],stdout=log,stderr=subprocess.STDOUT)
    status('saving_environment')
    (H/'assets').mkdir(exist_ok=True)
    for label,exe in [('train','/opt/verbalizing-subliminal-learning-venv/bin/python'),('infer','/opt/psychosis-base-infer/bin/python')]:
        with (R/f'{label}-freeze.txt').open('w') as f:run(['uv','pip','freeze','--python',exe],stdout=f)
    run(['tar','-cf',str(H/'assets/environment.tar'),'-C','/','opt/model-speed-packages','opt/verbalizing-subliminal-learning-venv','opt/psychosis-base-infer'])
    status('success',environment_saved=True)
except BaseException:
    status('failed',traceback=traceback.format_exc());raise
