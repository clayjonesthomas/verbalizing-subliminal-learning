"""Keep failed runs inspectable; persist the environment after successful training."""
import json,os,subprocess,sys,time,traceback
from pathlib import Path
HERE=Path(__file__).resolve().parent
OUT=HERE/'results'
def write(name,data):
    p=OUT/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(data,indent=2));t.replace(p)
try:
    OUT.mkdir(exist_ok=True)
    write('supervisor.json',dict(pid=os.getpid(),started=time.time(),status='running'))
    subprocess.run([sys.executable,'-m','unittest','test_data.py'],cwd=HERE,check=True)
    subprocess.run([sys.executable,'-u',str(HERE/'run.py')],cwd=HERE,check=True)
    write('supervisor.json',dict(pid=os.getpid(),status='saving_environment',time=time.time()))
    subprocess.run(['tar','-cf',str(HERE/'assets/environment.tar'),'-C','/','opt/psychosis-paired-soft-packages','opt/verbalizing-subliminal-learning-venv'],check=True)
    write('supervisor.json',dict(status='success',time=time.time(),environment_saved=True))
except BaseException:
    write('supervisor.json',dict(status='failed',time=time.time(),traceback=traceback.format_exc()))
    raise
