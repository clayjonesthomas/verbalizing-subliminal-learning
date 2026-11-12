"""Bounded shared-runner workload; comparison artifacts stay on the network volume."""
import json,os,subprocess,time,sys,hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent
ART=Path(os.environ['PODRUN_ARTIFACT_DIR']);ART.mkdir(parents=True,exist_ok=True)
REPORT=Path(os.environ['PODRUN_REPORT_DIR']);(REPORT/'measurements').mkdir(exist_ok=True)
row={'status':'running','run_id':os.environ['PODRUN_RUN_ID'] if 'PODRUN_RUN_ID' in os.environ else Path(os.environ['PODRUN_DIR']).name,'image':json.loads((HERE.parents[1]/'setup/image/registry.json').read_text())['pinned_image'],'artifact_dir':str(ART),'stages':{}}
record=REPORT/'measurements'/(row['run_id']+'.json')
def save():record.write_text(json.dumps(row,indent=2)+'\n')
def run(stage,args,env=None):
 start=time.monotonic();print(stage,flush=True)
 subprocess.run([str(x) for x in args],check=True,env=env)
 row['stages'][stage]=time.monotonic()-start;save()
save()
try:
 image=Path('/opt/auditbench-training-venv/bin/python')
 assert image.exists() and Path('/opt/vsl-image/manifest.json').exists(),'Image contents missing'
 row['task_entry_time']=time.time()
 row['image_manifest']=json.loads(Path('/opt/vsl-image/manifest.json').read_text())
 for name in ('uv.lock','pyproject.toml'):
  assert hashlib.sha256((HERE.parents[1]/name).read_bytes()).hexdigest()==row['image_manifest']['files'][name], 'Dependency definition changed since image build'
 run('image_training_inventory',[image,'/opt/vsl-image/verify.py','training'])
 run('image_vllm_inventory',['/opt/auditbench-vllm-venv/bin/python','/opt/vsl-image/verify.py','vllm'])
 run('image_probe',[image,HERE/'probe.py',ART/'image'])
 # Reproduce the old frozen base + extras installation independently, on durable storage.
 reference=ART/'reference-base';env=dict(os.environ,UV_PROJECT_ENVIRONMENT=str(reference),UV_CACHE_DIR=str(ART/'uv-cache'),UV_LINK_MODE='copy')
 run('reference_install',['uv','sync','--frozen','--no-install-project','--python','/usr/bin/python3.11'],env)
 extras=ART/'reference-training'
 run('reference_venv',['uv','venv','--python','/usr/bin/python3.11',extras])
 (extras/'lib/python3.11/site-packages/project-base.pth').write_text(str(reference/'lib/python3.11/site-packages')+'\n')
 run('reference_extras',['uv','pip','install','--python',extras/'bin/python','--no-deps','-r','/opt/vsl-image/training.txt'],env)
 run('reference_probe',[extras/'bin/python',HERE/'probe.py',ART/'reference'])
 # Torch already installed in this driver's frozen base environment.
 import torch
 a=json.loads((ART/'image/probe.json').read_text());b=json.loads((ART/'reference/probe.json').read_text())
 ignore={'salve'}
 assert {k:v for k,v in a['packages'].items() if k not in ignore}=={k:v for k,v in b['packages'].items() if k not in ignore},'Package inventories differ'
 ga=torch.load(ART/'image/gradients.pt',weights_only=True);gb=torch.load(ART/'reference/gradients.pt',weights_only=True)
 row['comparisons']=[]
 for x,y in zip(a['rows'],b['rows']):
  assert x['setting']==y['setting'];key=x['setting']
  loss_delta=abs(x['loss']-y['loss']);grad_delta=(ga[key]-gb[key]).abs().max().item()
  torch.testing.assert_close(ga[key],gb[key],rtol=.001,atol=.00001)
  assert loss_delta<.0001
  row['comparisons'].append(dict(setting=key,loss_delta=loss_delta,gradient_max_abs_delta=grad_delta,image=x,reference=y))
 row.update(gpu=a['gpu'],dtype=a['dtype'],checkpointing=a['checkpointing'],package_count=len(a['packages']))
 run('inference',['/opt/auditbench-vllm-venv/bin/python',HERE/'inference.py',ART/'inference.json'],dict(os.environ,VLLM_USE_FLASHINFER_SAMPLER='0'))
 row['inference']=json.loads((ART/'inference.json').read_text());row['status']='passed'
 # Successful GPU checks authorize the immutable default for future submissions only.
 registry_path=HERE.parents[1]/'setup/image/registry.json'
 registry=json.loads(registry_path.read_text());assert registry['pinned_image']==row['image']
 registry.update(gpu_validation='passed',default_enabled=True,validation_run=row['run_id'],validation_source_commit=os.environ['PODRUN_COMMIT'],validation_measurements=str(record.relative_to(HERE.parents[1])))
 registry_path.write_text(json.dumps(registry,indent=2)+'\n')
except Exception as exc:
 row['status']='failed';row['error_type']=type(exc).__name__;raise
finally:
 save();subprocess.run([sys.executable,str(HERE/'report.py')],check=False)
