"""Detached training, bounded publication, progress watchdog, persistent cleanup."""
import json,os,signal,subprocess,sys,time,traceback,shutil,urllib.request
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUT=HERE/'results';C=json.loads((HERE/'config.json').read_text())
def write(name,value):
    p=OUT/name;tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(p)
def main():
    key=sys.stdin.read().strip();pod=sys.argv[1];child=None;OUT.mkdir(exist_ok=True)
    def api(query):
        req=urllib.request.Request('https://api.runpod.io/graphql',headers={'Authorization':'Bearer '+key,'X-Team-Id':'cmu31hyiy002gk9s0xik85gob','Content-Type':'application/json'},data=json.dumps({'query':query}).encode())
        with urllib.request.urlopen(req,timeout=60) as r:return json.load(r)
    def verify():assert api('{myself{id}}')['data']['myself']['id']=='user_3JNPrVMTYwugXaYJytSv2t8lRsv'
    publication=ROOT.parent/'training-publication'
    def publish():
        subprocess.run([sys.executable,str(HERE/'report.py')],check=True,timeout=60)
        target=publication/HERE.relative_to(ROOT);(target/'results').mkdir(parents=True,exist_ok=True)
        for path in OUT.iterdir():
            if path.is_file() and (path.suffix in ['.json','.log','.txt'] or path.name=='metrics.jsonl'):shutil.copy2(path,target/'results'/path.name)
        shutil.copy2(HERE/'report.html',target/'report.html')
        subprocess.run(['git','add','--',str(HERE.relative_to(ROOT))],cwd=publication,check=True,timeout=30)
        if subprocess.run(['git','diff','--cached','--quiet'],cwd=publication).returncode==0:return
        subprocess.run(['git','commit','-qm','Update 10k SFT training report and metrics'],cwd=publication,check=True,timeout=30)
        subprocess.run(['git','push','origin','HEAD:refs/heads/'+C['results_branch']],cwd=publication,check=True,timeout=120,env=dict(os.environ,SSH_AUTH_SOCK='',GIT_TERMINAL_PROMPT='0'))
    try:
        verify();source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        for path in HERE.iterdir():
            if path.is_file() and path.suffix in ['.py','.json','.md']:
                assert path.read_bytes()==subprocess.check_output(['git','show',source+':'+str(path.relative_to(ROOT))],cwd=ROOT)
        write('training_manifest.json',dict(source_commit=source,pod=pod,started=time.time(),config=C,microbatch=1,gradient_accumulation=16,volume='ae6w0yoyy6'))
        assert not (OUT/'metrics.jsonl').exists(),'Refusing accidental overwrite of an earlier training run'
        subprocess.run(['uv','pip','install','--python',sys.executable,'matplotlib==3.10.8'],check=True,timeout=300)
        with (OUT/'training_environment.txt').open('w') as f:subprocess.run(['uv','pip','freeze','--python',sys.executable],stdout=f,check=True,timeout=60)
        subprocess.run(['git','worktree','add','--detach',str(publication),source],cwd=ROOT,check=True,timeout=120)
        write('training_stage.json',dict(stage='loading_model',time=time.time()));publish()
        start=time.time();last_publish=time.time()
        with (OUT/'training.log').open('a') as log:
            child=subprocess.Popen([sys.executable,str(HERE/'train.py')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            while child.poll() is None:
                if (OUT/'training_complete.json').exists():break
                latest=max([start]+[p.stat().st_mtime for p in [OUT/'progress.json',OUT/'training.log'] if p.exists()])
                if time.time()-latest>1200:raise TimeoutError('20 minutes without application progress')
                if time.time()-start>36*3600:raise TimeoutError('36-hour training runtime limit')
                if time.time()-last_publish>=60:
                    write('training_stage.json',dict(stage='training',time=time.time(),elapsed_seconds=time.time()-start))
                    try:publish()
                    except Exception:write('publication_error.json',dict(time=time.time(),traceback=traceback.format_exc()))
                    last_publish=time.time()
                time.sleep(10)
        if not (OUT/'training_complete.json').exists():raise RuntimeError('Training process exited without a completion audit')
        write('training_stage.json',dict(stage='complete',status='success',time=time.time()))
    except BaseException:
        write('training_failure.json',dict(time=time.time(),traceback=traceback.format_exc()))
        write('training_stage.json',dict(stage='failed',time=time.time()))
    finally:
        if child and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=30)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        # Checkpoints, data and logs already live on the persistent volume.
        try:publish()
        except Exception:write('final_publication_error.json',dict(traceback=traceback.format_exc()))
        try:
            assets=HERE/'assets';assets.mkdir(exist_ok=True)
            subprocess.run(['tar','-cf',str(assets/'training-environment.tar'),'-C','/','opt/verbalizing-subliminal-learning-venv'],check=True,timeout=900)
        except Exception:write('environment_save_error.json',dict(traceback=traceback.format_exc()))
        subprocess.run(['sync','-f',str(OUT)],timeout=30)
        verify()
        for i in range(5):
            try:
                r=api('mutation {podTerminate(input:{podId:'+json.dumps(pod)+'})}');assert not r.get('errors'),r;break
            except Exception:
                if i==4:raise
                time.sleep(10)
if __name__=='__main__':main()
