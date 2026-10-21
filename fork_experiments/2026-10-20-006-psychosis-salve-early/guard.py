"""Independent failure cleanup if the supervisor exits without terminating its pod."""
import argparse,json,os,signal,subprocess,sys,time
from pathlib import Path
from run import HERE,ROOT,OUT,ROLE,write
from supervise import api,ACCOUNT,publish
p=argparse.ArgumentParser();p.add_argument('--pod',required=True);p.add_argument('--keyfile',type=Path,required=True);a=p.parse_args()
child=subprocess.Popen([sys.executable,str(HERE/'supervise.py'),'--pod',a.pod,'--keyfile',str(a.keyfile)],cwd=ROOT,start_new_session=True)
last_activity=time.time();last_stage=None
while child.poll() is None:
    try:
        heartbeat=json.loads((OUT/'heartbeat.json').read_text());stage=heartbeat['stage']
        if stage!=last_stage:last_activity=time.time();last_stage=stage
        for path in [OUT/'progress.json',OUT/(stage+'.log')]:
            if path.exists():last_activity=max(last_activity,path.stat().st_mtime)
        if time.time()-last_activity>5400:
            write(OUT/'watchdog_timeout.json',dict(stage=stage,time=time.time(),reason='90 minutes without stage-log or application-progress updates'))
            os.killpg(child.pid,signal.SIGTERM);child.wait(timeout=30);break
    except (FileNotFoundError,json.JSONDecodeError):pass
    time.sleep(20)
# Successful deletion ends this guard too. Any live guard here must verify status.
try:
 detail=api(a.keyfile,'graphql',{'query':'{ pod(input: {podId: '+__import__('json').dumps(a.pod)+'}) { id desiredStatus } }'})
 if not detail.get('data',{}).get('pod'):sys.exit(0)
except Exception:pass
write(OUT/'guard_failure.json',dict(supervisor_exit=child.returncode,time=time.time(),policy='Artifacts persist on network volume; terminate failed pod'))
try:os.killpg(child.pid,signal.SIGTERM)
except ProcessLookupError:pass
try:publish('Worker guard: preserve final artifacts before termination',full=True)
except Exception:pass
assert api(a.keyfile,'graphql',{'query':'{ myself { id email } }'})['data']['myself']['id']==ACCOUNT
for attempt in range(5):
 try:
  api(a.keyfile,f'pods/{a.pod}/action',{'action':'terminate'});break
 except Exception:
  time.sleep(10)
