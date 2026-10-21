"""Run isolated benchmarks, publish artifacts, and terminate even on failure."""
import json,os,signal,subprocess,sys,time,traceback,urllib.request,shutil
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUT=HERE/'results'
def write(name,data):(OUT/name).write_text(json.dumps(data,indent=2)+'\n')
def main():
    key=sys.stdin.read().strip();pod=sys.argv[1]
    def api(query):
        req=urllib.request.Request('https://api.runpod.io/graphql',headers={'Authorization':'Bearer '+key,'X-Team-Id':'cmu31hyiy002gk9s0xik85gob','Content-Type':'application/json'},data=json.dumps({'query':query}).encode())
        with urllib.request.urlopen(req,timeout=60) as r:return json.load(r)
    def verify():assert api('{myself{id}}')['data']['myself']['id']=='user_3JNPrVMTYwugXaYJytSv2t8lRsv'
    OUT.mkdir(exist_ok=True);child=None
    try:
        verify();source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        for p in HERE.iterdir():
            if p.is_file() and p.suffix in ['.py','.json','.md']:
                assert p.read_bytes()==subprocess.check_output(['git','show',source+':'+str(p.relative_to(ROOT))],cwd=ROOT)
        write('manifest.json',dict(source_commit=source,pod=pod,started=time.time(),phase='benchmark'))
        assert json.loads((OUT/'preflight.json').read_text())['status']=='passed'
        # Input files are immutable artifacts copied from the locally completed pinned audit.
        for mb in [1,2,4]:
            write('stage.json',dict(stage='benchmark',microbatch=mb,time=time.time()))
            with (OUT/f'benchmark_{mb}.log').open('w') as log:
                child=subprocess.Popen([sys.executable,str(HERE/'benchmark.py'),'--microbatch',str(mb)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                try:code=child.wait(timeout=1500)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid,signal.SIGKILL);child.wait();raise
                if code:raise RuntimeError('Benchmark process failed: '+str(mb))
            subprocess.run([sys.executable,str(HERE/'report.py')],check=True)
        results=[json.loads((OUT/f'benchmark_{mb}.json').read_text()) for mb in [1,2,4]]
        passed=[r for r in results if r['status']=='passed'];assert passed,'No microbatch passed'
        best=max(passed,key=lambda x:x['tokens_per_second'])
        write('benchmark_selection.json',dict(microbatch=best['microbatch'],results=results,criterion='Highest representative supervised tokens/s among candidates passing longest-row stress test'))
        write('stage.json',dict(stage='benchmark_complete',status='success',time=time.time()))
    except BaseException:
        write('failure.json',dict(traceback=traceback.format_exc(),time=time.time()))
        write('stage.json',dict(stage='failed',time=time.time()))
    finally:
        if child and child.poll() is None:os.killpg(child.pid,signal.SIGKILL)
        try:
            subprocess.run([sys.executable,str(HERE/'report.py')],check=True,timeout=60)
            paths=[str(p.relative_to(ROOT)) for p in OUT.glob('*.json')]+[str(p.relative_to(ROOT)) for p in OUT.glob('*.log')]+[str((HERE/'report.html').relative_to(ROOT))]
            subprocess.run(['git','add','--',*paths],cwd=ROOT,check=True,timeout=60)
            subprocess.run(['git','commit','-m','Save data audit and microbatch benchmark results'],cwd=ROOT,check=True,timeout=60)
            subprocess.run(['git','push','origin','HEAD:refs/heads/experiments/psychosis-sft-10k-2026-10-21'],cwd=ROOT,check=True,timeout=180,env=dict(os.environ,SSH_AUTH_SOCK='',GIT_TERMINAL_PROMPT='0'))
        except Exception:write('publication_failure.json',dict(traceback=traceback.format_exc()))
        # Frozen base and package environment are already preserved on the shared volume.
        subprocess.run(['sync','-f',str(OUT)],timeout=30)
        verify()
        for i in range(5):
            try:
                r=api('mutation {podTerminate(input:{podId:'+json.dumps(pod)+'})}');assert not r.get('errors'),r;break
            except Exception:
                if i==4:raise
                time.sleep(10)
if __name__=='__main__':main()
