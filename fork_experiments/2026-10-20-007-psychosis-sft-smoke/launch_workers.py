"""Provision two independent workers from the generator; no laptop connection needed."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
from run import HERE,ROOT,OUT,ASSETS,write
sys.path.insert(0,str(ROOT/'setup'))
import runpod
import provision

def main():
    p=argparse.ArgumentParser();p.add_argument('--keyfile',type=Path,required=True);a=p.parse_args()
    runpod.KEY_FILE=a.keyfile
    provision.BILLING_LOG=OUT/'worker_billing.jsonl'
    client=provision.ProvisionClient();client.verify()
    assert client.request('GET','network-volumes/ae6w0yoyy6')['dataCenter']=='US-NE-1'
    private=Path('/run/psychosis-secrets/worker_ssh')
    if not private.exists():subprocess.run(['ssh-keygen','-q','-t','ed25519','-N','','-f',str(private)],check=True)
    public=Path(str(private)+'.pub').read_text().strip()
    public+='\n'+Path('/run/psychosis-secrets/user_public_key').read_text().strip()
    def ssh_args(pod,command=None):
        host,port=runpod.endpoint(pod)
        args=['ssh','-i',str(private),'-o','IdentitiesOnly=yes','-o','BatchMode=yes','-o','StrictHostKeyChecking=accept-new','-o','UserKnownHostsFile=/run/psychosis-secrets/worker_known_hosts','-o','ConnectTimeout=20','-p',str(port),f'root@{host}']
        import shlex
        return args+[shlex.join(['bash',runpod.REPO+'/setup/run.sh',*(command or ['true'])])]
    runpod.ssh_args=ssh_args
    for role in ['student']:
        status=OUT/f'worker_{role}.json'
        state=json.loads(status.read_text()) if status.exists() else {}
        if state.get('status')=='running':continue
        pod_id=state.get('pod')
        if state.get('status')=='creating' and not pod_id:
            raise RuntimeError('Uncertain previous creation; reconcile account before retrying')
        try:
            while not pod_id:
                client.verify()
                catalog=client.request('GET','catalog/gpus?include=AVAILABILITY&product=POD&cloud=SECURE&count=1')['gpus']
                options=[g for g in provision.candidates(catalog) if g['memory']>=96]
                for gpu in options:
                    name='psychosis-sft-smoke'
                    body={'name':name,'image':runpod.IMAGE,'args':runpod.START,'cloud':'SECURE','gpu':{'id':gpu['id'],'count':1},'disk':30,'dataCenterIds':['US-NE-1'],'mounts':{'network':[{'volumeId':'ae6w0yoyy6','path':'/workspace'}]},'startSsh':True,'ports':['22/tcp'],'env':{'PUBLIC_KEY':public}}
                    write(status,dict(status='creating',name=name,time=time.time()))
                    try:pod=provision.create_with_billing(client,body,gpu)
                    except provision.ApiError as exc:
                        if exc.capacity:
                            write(status,dict(status='waiting_capacity',time=time.time()));continue
                        raise
                    pod_id=pod['id'];write(status,dict(status='booting',pod=pod_id,time=time.time()))
                    break
                if not pod_id:time.sleep(10)
            runpod.start_and_connect(client,client.pod(pod_id),['true'])
            # Send secrets through SSH stdin, never command lines, Git, or log output.
            payload=json.dumps({'runpod':a.keyfile.read_text().strip(),'openrouter':Path('/run/psychosis-secrets/OPENROUTER_API_KEY').read_text().strip()})
            subprocess.run(ssh_args(client.pod(pod_id),['python',str(HERE/'bootstrap_worker.py'),'--role',role,'--pod',pod_id]),input=payload,text=True,check=True)
            write(status,dict(status='running',pod=pod_id,time=time.time(),role=role,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()))
        except Exception as exc:
            write(status,dict(status='failed',pod=pod_id,error_type=type(exc).__name__,time=time.time()))
            if pod_id:
                # startup.log on the shared volume and this log retain boot diagnostics.
                client.verify();client.request('POST',f'pods/{pod_id}/action',{'action':'terminate'})
            raise

if __name__=='__main__':main()
