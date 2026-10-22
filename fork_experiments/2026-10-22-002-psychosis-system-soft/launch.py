"""Deploy exact committed files and launch detached; credential only via SSH stdin."""
import sys,subprocess,json,hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'setup'))
from runpod import Client,ssh_args,KEY_FILE
pod=Client().pod(sys.argv[1]);commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
folder=str(HERE.relative_to(ROOT));files={}
for p in HERE.iterdir():
 if p.suffix not in ('.py','.json') or p.name=='source_manifest.json':continue
 text=subprocess.check_output(['git','show',commit+':'+folder+'/'+p.name],cwd=ROOT,text=True)
 assert text==p.read_text(),p
 files[p.name]=text
remote='/workspace/projects/verbalizing-subliminal-learning/psychosis-system-soft/'+HERE.name
script='''import sys,subprocess,json,hashlib,shutil
from pathlib import Path
p=json.load(sys.stdin);root=Path(p['remote']);root.mkdir(parents=True,exist_ok=True)
if (root/'results/supervisor.json').exists():
 assert p['retry'],'Already launched'
 old=json.loads((root/'results/supervisor.json').read_text());assert old['status']=='failed'
 assert not list((root/'results').glob('k*/latest.pt')),'Refuse to discard trained checkpoints'
 archive=root/'assets'/('failed-attempt-'+old['pod']);archive.mkdir(exist_ok=False)
 for name in p['files']:
  if (root/name).exists():shutil.copyfile(root/name,archive/name)
 shutil.copyfile(root/'source_manifest.json',archive/'source_manifest.json')
 (root/'results').rename(archive/'results')
 if (root/'assets/environment.tar').exists():(root/'assets/environment.tar').rename(archive/'environment.tar')
for name,text in p['files'].items():(root/name).write_text(text)
assert {n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in p['files']}==p['hashes']
(root/'source_manifest.json').write_text(json.dumps(dict(source_commit=p['commit'],files=p['hashes']),indent=2))
out=root/'results';out.mkdir(exist_ok=True);(root/'assets').mkdir(exist_ok=True)
with (out/'run.log').open('a') as log:
 child=subprocess.Popen([sys.executable,'-u',str(root/'supervise.py'),p['pod']],stdin=subprocess.PIPE,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,text=True)
 child.stdin.write(p['key']+'\\n');child.stdin.close()
print(json.dumps(dict(pid=child.pid,pod=p['pod'],source_commit=p['commit'],remote=p['remote'])))
'''
payload=dict(retry="--retry" in sys.argv,files=files,hashes={n:hashlib.sha256(t.encode()).hexdigest() for n,t in files.items()},commit=commit,pod=pod['id'],remote=remote,key=KEY_FILE.expanduser().read_text().strip())
r=subprocess.run(ssh_args(pod,['python','-c',script]),input=json.dumps(payload),capture_output=True,text=True,check=True)
print(r.stdout);(HERE/'results').mkdir(exist_ok=True);(HERE/'results/launch.json').write_text(r.stdout)
