"""Team-scoped GraphQL provisioning fallback while Runpod v2 returns HTTP 500."""
import json,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]/'setup'))
from runpod import Client,IMAGE,START
from provision import record_billing

def main():
 c=Client();c.verify();out=HERE/'results';out.mkdir(exist_ok=True)
 if (out/'pod.json').exists():raise RuntimeError('Pod record exists; inspect instead of provisioning twice')
 volumes=c.request('POST','graphql',{'query':'{ myself { networkVolumes { id dataCenterId } } }'})['data']['myself']['networkVolumes']
 assert any(v['id']=='ae6w0yoyy6' and v['dataCenterId']=='US-NE-1' for v in volumes)
 gpu='NVIDIA RTX PRO 6000 Blackwell Server Edition'
 body=dict(name='psychosis-salve-convergence',cloudType='SECURE',gpuCount=1,gpuTypeId=gpu,imageName=IMAGE,dockerArgs=START,containerDiskInGb=30,volumeInGb=0,networkVolumeId='ae6w0yoyy6',volumeMountPath='/workspace',dataCenterId='US-NE-1',startSsh=True,supportPublicIp=True,ports='22/tcp',env=[{'key':'PUBLIC_KEY','value':Path('~/.ssh/id_ed25519.pub').expanduser().read_text().strip()}])
 while True:
  c.verify()
  q='{ gpuTypes(input: {id: '+json.dumps(gpu)+'}) { id securePrice memoryInGb lowestPrice(input: {gpuCount: 1, secureCloud: true, dataCenterId: "US-NE-1"}) { stockStatus uninterruptablePrice } } }'
  g=c.request('POST','graphql',{'query':q})['data']['gpuTypes'][0]
  assert g['memoryInGb']>=96 and 0<g['securePrice']<5
  print(json.dumps({'availability':g}),flush=True)
  record_billing('pod_create_requested',name=body['name'],gpu_id=gpu,catalog_gpu_hourly=g['securePrice'],disk_gb=30)
  r=c.request('POST','graphql',{'query':'mutation($input: PodFindAndDeployOnDemandInput) { podFindAndDeployOnDemand(input:$input) { id name costPerHr networkVolumeId } }','variables':{'input':body}})
  p=(r.get('data') or {}).get('podFindAndDeployOnDemand')
  if p:
   (out/'pod.json').write_text(json.dumps(p,indent=2))
   assert p['networkVolumeId']=='ae6w0yoyy6'
   record_billing('pod_created',pod_id=p['id'],name=p['name'],gpu_id=gpu,catalog_gpu_hourly=g['securePrice'],hourly=p['costPerHr'],disk_gb=30)
   print(json.dumps(p),flush=True);return
  messages=[e.get('message','') for e in r.get('errors',[])]
  if messages and all('no longer any instances' in m.lower() for m in messages):
   print('Explicit capacity rejection; retrying in 10 seconds',flush=True);time.sleep(10);continue
  raise RuntimeError('Ambiguous creation outcome; reconcile account before any retry: '+repr(messages))
if __name__=='__main__':main()
