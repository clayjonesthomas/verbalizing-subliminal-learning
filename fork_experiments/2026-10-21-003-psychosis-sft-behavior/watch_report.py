"""Laptop-side read-only Git polling; atomically refresh local report on wake."""
import argparse,fcntl,json,subprocess,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--destination',type=Path,required=True);a=p.parse_args()
a.destination.mkdir(parents=True,exist_ok=True)
lock=(a.destination/'watcher.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
branch='experiments/psychosis-sft-behavior-2026-10-21';rel='fork_experiments/2026-10-21-003-psychosis-sft-behavior/'
while True:
    try:
        subprocess.run(['git','fetch','--quiet','origin','refs/heads/'+branch+':refs/remotes/origin/'+branch],cwd=a.repo,check=True,timeout=60)
        ref='refs/remotes/origin/'+branch
        commit=subprocess.check_output(['git','rev-parse',ref],cwd=a.repo,text=True).strip()
        for name in ['report.html','results/stage.json','results/progress.json','results/completion.json','results/summary.json']:
            r=subprocess.run(['git','show',ref+':'+rel+name],cwd=a.repo,capture_output=True,timeout=30)
            if r.returncode:continue
            dst=a.destination/name;dst.parent.mkdir(parents=True,exist_ok=True);tmp=dst.with_suffix(dst.suffix+'.sync');tmp.write_bytes(r.stdout);tmp.replace(dst)
        (a.destination/'watcher_status.json').write_text(json.dumps(dict(last_sync=time.time(),commit=commit,status='ok')))
    except Exception as e:
        (a.destination/'watcher_status.json').write_text(json.dumps(dict(time=time.time(),status='error',error_type=type(e).__name__)))
    time.sleep(30)
