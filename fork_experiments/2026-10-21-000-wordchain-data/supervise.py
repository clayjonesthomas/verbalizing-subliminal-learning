"""Detached generation supervisor; persist environment before successful termination."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import urllib.request
from run import HERE, ROOT, OUT, C, write, sha


def main():
    # Credential is received on stdin and retained only in process memory.
    key=sys.stdin.readline().strip()
    pod=sys.argv[1]
    def api(path,body):
        url='https://api.runpod.io/graphql' if path=='graphql' else 'https://api.runpod.io/v2/'+path
        req=urllib.request.Request(url,method='POST',data=json.dumps(body).encode(),headers={'Authorization':'Bearer '+key,'X-Team-Id':'cmu31hyiy002gk9s0xik85gob','Content-Type':'application/json'})
        with urllib.request.urlopen(req,timeout=60) as r:
            data=r.read();return json.loads(data) if data else {}
    def verify_account():
        assert api('graphql',{'query':'{ myself { id email } }'})['data']['myself']['id']=='user_3JNPrVMTYwugXaYJytSv2t8lRsv'
    verify_account()
    OUT.mkdir(parents=True,exist_ok=True)
    source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    hashes={}
    for p in HERE.iterdir():
        if p.is_file() and p.suffix in ('.py','.json','.sh','.md'):
            assert p.read_bytes()==subprocess.check_output(['git','show',source+':'+str(p.relative_to(ROOT))],cwd=ROOT)
            hashes[p.name]=sha(p)
    write(OUT/('resume_manifest_'+str(int(time.time()))+'.json' if '--resume' in sys.argv else 'manifest.json'),dict(source_commit=source,files=hashes,config=C,pod=pod,volume='ae6w0yoyy6',started=time.time()))
    os.environ['TOKENIZERS_PARALLELISM']='false'
    stages=[('dependencies',['bash',str(HERE/'dependencies.sh')]),('prepare',['/opt/wordchain-infer/bin/python',str(HERE/'run.py'),'prepare']),('merge',['/opt/verbalizing-subliminal-learning-venv/bin/python',str(HERE/'run.py'),'merge']),('generate',['/opt/wordchain-infer/bin/python',str(HERE/'run.py'),'generate']),('export',['/opt/wordchain-infer/bin/python',str(HERE/'run.py'),'export'])]
    if '--resume' in sys.argv:stages=[(name,args) for name,args in stages if name in ('generate','export')]
    try:
        for name,args in stages:
            write(OUT/'stage.json',dict(stage=name,status='running',time=time.time()))
            with (OUT/(name+'.log')).open('a') as log:
                subprocess.run(args,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        archive=HERE/'assets/environment.tar'
        subprocess.run(['tar','-cf',str(archive)+'.tmp','-C','/','opt/wordchain-infer','opt/wordchain-merge-libs','opt/verbalizing-subliminal-learning-venv'],check=True)
        Path(str(archive)+'.tmp').replace(archive)
        write(OUT/'environment.json',dict(path=str(archive),sha256=sha(archive)))
        write(OUT/'stage.json',dict(stage='complete',status='success',time=time.time()))
        subprocess.run(['sync','-f',str(OUT)],check=True)
        verify_account()
        api('pods/'+pod+'/action',{'action':'terminate'})
    except BaseException:
        write(OUT/'failure.json',dict(time=time.time(),traceback=traceback.format_exc(),policy='Pod retained for inspection'))
        raise

if __name__=='__main__':main()
