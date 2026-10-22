"""Autonomous behavior evaluation with persistent artifacts and failure cleanup."""
import json,os,signal,subprocess,sys,time,traceback,shutil,urllib.request
from evaluate import HERE,ROOT,OUT,ASSETS,C,OLD,write

def main():
    secrets=json.load(sys.stdin);pod=sys.argv[1];child=None;OUT.mkdir(exist_ok=True);ASSETS.mkdir(exist_ok=True)
    def api(query):
        req=urllib.request.Request('https://api.runpod.io/graphql',headers={'Authorization':'Bearer '+secrets['runpod'],'X-Team-Id':'cmu31hyiy002gk9s0xik85gob','Content-Type':'application/json'},data=json.dumps({'query':query}).encode())
        with urllib.request.urlopen(req,timeout=60) as r:return json.load(r)
    def verify():assert api('{myself{id}}')['data']['myself']['id']=='user_3JNPrVMTYwugXaYJytSv2t8lRsv'
    publication=ROOT.parent/'publication';env=dict(os.environ,OPENROUTER_API_KEY=secrets['openrouter'],SSH_AUTH_SOCK='',TOKENIZERS_PARALLELISM='false',PYTHONUNBUFFERED='1')
    def publish():
        subprocess.run([sys.executable,str(HERE/'report.py')],env=env,check=True,timeout=60)
        target=publication/HERE.relative_to(ROOT);(target/'results').mkdir(parents=True,exist_ok=True)
        for p in OUT.rglob('*'):
            if p.is_file() and 'student_adapter' not in p.relative_to(OUT).parts and p.suffix in ['.json','.jsonl','.txt','.log']:
                dst=target/'results'/p.relative_to(OUT);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dst)
        shutil.copy2(HERE/'report.html',target/'report.html')
        subprocess.run(['git','add','--',str(HERE.relative_to(ROOT))],cwd=publication,check=True,timeout=60)
        if subprocess.run(['git','diff','--cached','--quiet'],cwd=publication).returncode==0:return
        subprocess.run(['git','commit','-qm','Update 10k SFT behavior evaluation'],cwd=publication,check=True,timeout=60)
        subprocess.run(['git','push','origin','HEAD:refs/heads/'+C['results_branch']],cwd=publication,env=env,check=True,timeout=180)
    try:
        verify();source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        for p in HERE.iterdir():
            if p.is_file() and p.suffix in ['.py','.json','.md']:
                assert p.read_bytes()==subprocess.check_output(['git','show',source+':'+str(p.relative_to(ROOT))],cwd=ROOT)
        write(OUT/'manifest.json',dict(source_commit=source,pod=pod,config=C,started=time.time(),volume='ae6w0yoyy6'))
        subprocess.run(['git','worktree','add','--detach',str(publication),source],cwd=ROOT,check=True,timeout=120)
        write(OUT/'stage.json',dict(stage='restoring_environment',time=time.time()));publish()
        subprocess.run(['tar','-xf',str(OLD/'assets/environments-student.tar'),'-C','/','opt/psychosis-infer','opt/psychosis-merge-libs'],check=True,timeout=900)
        for name,python in [('training',sys.executable),('inference','/opt/psychosis-infer/bin/python')]:
            with (OUT/(name+'_environment.txt')).open('w') as f:subprocess.run(['uv','pip','freeze','--python',python],stdout=f,check=True,timeout=60)
        for stage in ['prepare','merge','generate','judge','summarize']:
            write(OUT/'stage.json',dict(stage=stage,time=time.time()));publish()
            started=time.time();last_publish=started
            with (OUT/(stage+'.log')).open('a') as log:
                python='/opt/psychosis-infer/bin/python' if stage=='generate' else sys.executable
                child=subprocess.Popen([python,str(HERE/'evaluate.py'),stage],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                while child.poll() is None:
                    if stage=='generate' and (OUT/'generation_complete.json').exists():
                        os.killpg(child.pid,signal.SIGTERM)
                        try:child.wait(timeout=30)
                        except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                        break
                    latest=max([started]+[p.stat().st_mtime for p in [OUT/'progress.json',OUT/(stage+'.log')] if p.exists()])
                    if time.time()-latest>1200 or time.time()-started>4*3600:raise TimeoutError(stage+' exceeded progress/runtime deadline')
                    if time.time()-last_publish>60:
                        try:publish()
                        except Exception:write(OUT/'publication_error.json',dict(traceback=traceback.format_exc(),time=time.time()))
                        last_publish=time.time()
                    time.sleep(10)
                if child.returncode and not (stage=='generate' and (OUT/'generation_complete.json').exists()):raise RuntimeError(stage+' failed; see stage log')
        write(OUT/'stage.json',dict(stage='complete',status='success',time=time.time()))
    except BaseException:
        write(OUT/'failure.json',dict(traceback=traceback.format_exc(),time=time.time()))
        write(OUT/'stage.json',dict(stage='failed',time=time.time()))
    finally:
        if child and child.poll() is None:
            os.killpg(child.pid,signal.SIGKILL);child.wait()
        try:publish()
        except Exception:write(OUT/'final_publication_error.json',dict(traceback=traceback.format_exc()))
        # Exact restored environments already persist in the source archive. Record package lists.
        subprocess.run(['sync','-f',str(OUT)],timeout=30)
        verify()
        for i in range(5):
            try:
                r=api('mutation {podTerminate(input:{podId:'+json.dumps(pod)+'})}');assert not r.get('errors'),r;break
            except Exception:
                if i==4:raise
                time.sleep(10)
if __name__=='__main__':main()
