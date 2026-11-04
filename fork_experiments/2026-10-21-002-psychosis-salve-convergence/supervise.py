"""Run on dedicated pod. Save the environment after success; retain failures."""
import subprocess,sys,json,time,os,traceback
from pathlib import Path
HERE=Path(__file__).resolve().parent;OUT=HERE/'results'
def write(stage,**kw):
 p=OUT/'lifecycle.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(dict(stage=stage,time=time.time(),**kw),indent=2));tmp.replace(p)
try:
 OUT.mkdir(exist_ok=True);write('running')
 env=dict(os.environ,PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false')
 with (OUT/'run.log').open('a') as f:
  subprocess.run([sys.executable,str(HERE/'run.py')],cwd=HERE.parents[1],env=env,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=17*3600)
 write('saving_environment')
 with (OUT/'requirements.txt').open('w') as f:subprocess.run([sys.executable,'-m','pip','freeze'],stdout=f,check=True)
 dst=HERE/'assets/environment.tar';tmp=dst.with_suffix('.tar.tmp')
 subprocess.run(['tar','-C','/','-cf',str(tmp),'opt/verbalizing-subliminal-learning-venv'],check=True)
 tmp.replace(dst)
 write('ready_to_retire',environment=str(dst),bytes=dst.stat().st_size)
except BaseException:
 write('failed_pod_retained',traceback=traceback.format_exc());raise
