"""Bounded detached verbalization run; terminate on success or failure."""
import json,os,subprocess,sys,time,urllib.request
from pathlib import Path
here=Path(__file__).resolve().parent
pod,commit,out=sys.argv[1:];out=Path(out);out.mkdir(parents=True,exist_ok=True)
key=sys.stdin.readline().strip()
def api(path,body):
    url='https://api.runpod.io/graphql' if path=='graphql' else 'https://api.runpod.io/v2/'+path
    req=urllib.request.Request(url,data=json.dumps(body).encode(),headers={'Authorization':'Bearer '+key,'X-Team-Id':'cmu31hyiy002gk9s0xik85gob','Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=30) as r:
        b=r.read();return json.loads(b) if b else {}
def verify():
    assert api('graphql',{'query':'{ myself { id email } }'})['data']['myself']['id']=='user_3JNPrVMTYwugXaYJytSv2t8lRsv'
    actual=next(x.split(b'=',1)[1].decode() for x in Path('/proc/1/environ').read_bytes().split(b'\x00') if x.startswith(b'RUNPOD_POD_ID='))
    assert pod==actual
verify()
status='failed'
try:
    subprocess.run(['uv','pip','install','--python',sys.executable,'--target','/opt/paired-verbalize-packages','--no-deps','flash-linear-attention==0.5.2','fla-core==0.5.2','einops==0.8.1'],check=True,timeout=120)
    env=dict(os.environ,PYTHONPATH='/opt/paired-verbalize-packages')
    subprocess.run([sys.executable,'-u',str(here/'verbalize.py'),'--commit',commit,'--output',str(out)],env=env,check=True,timeout=1800)
    status='success'
except Exception as exc:
    (out/'failure.json').write_text(json.dumps(dict(error_type=type(exc).__name__,time=time.time())))
finally:
    try:
        subprocess.run(['tar','-cf',str(out/'environment.tar'),'-C','/','opt/paired-verbalize-packages','opt/verbalizing-subliminal-learning-venv'],check=True,timeout=300)
    except Exception:pass
    (out/'lifecycle.json').write_text(json.dumps(dict(status=status,source_commit=commit,pod=pod,termination_requested=True,time=time.time())))
    subprocess.run(['sync','-f',str(out)],timeout=30)
    while True:
        try:
            verify();api('pods/'+pod+'/action',{'action':'terminate'});break
        except Exception:time.sleep(20)
