"""Optional local report mirror. Training and cleanup never depend on this process."""
import sys,subprocess,shlex,time,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]/'setup'))
from runpod import Client,ssh_args
from report import render
pod_id=sys.argv[1]
remote='/workspace/projects/verbalizing-subliminal-learning/psychosis-system-soft/'+HERE.name
while True:
 try:
  pod=Client().pod(pod_id);args=ssh_args(pod,['true']);host=args[-2]
  transport=shlex.join(args[:args.index(host)])
  subprocess.run(['rsync','-az','--exclude=*.pt','--exclude=*.tar','--exclude=assets/','-e',transport,host+':'+remote+'/results/',str(HERE/'results')+'/'],check=True,timeout=60)
  render()
  s=json.loads((HERE/'results/supervisor.json').read_text())
  print(time.ctime(),s,flush=True)
  if s.get('status') in ('success','failed'):break
 except Exception as e:print(type(e).__name__,str(e),flush=True)
 time.sleep(30)
