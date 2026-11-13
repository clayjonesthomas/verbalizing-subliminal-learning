"""Enqueue global DAG work; bootstrap one pod only when the pool is offline."""
import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import uuid
import tempfile

from submit_experiment import git, materialize


def compile_tasks(root,infra,experiment,template_path,dag_path,workers,names,repair=False,existing_ids=()):
    root=Path(root).resolve();infra=Path(infra).resolve()
    from podrun.cli import preflight,forwarded_environment
    from podrun.workspace import prepare
    from podrun.pool import identifier,priority_fields
    from image_policy import select
    policy=root/'fork_experiments'/experiment/'report_policy.json'
    if policy.exists() and json.loads(policy.read_text()).get('shared_graphing_required'):
        sys.path.insert(0,str(infra/'graphing-utils'))
        from research_graphing.validation import check_report
        errors=check_report(policy.parent)
        if errors:raise ValueError('Report preflight failed: '+'; '.join(errors))
    source=git(root,'rev-parse','HEAD');infra_sha=git(infra,'rev-parse','HEAD')
    relative=str(template_path.resolve().relative_to(root))
    template=json.loads(git(root,'show',source+':'+relative))
    paths=['setup','.research-infra.json',relative,*template.get('checkout',{}).get('code_paths',[])]
    if dag_path:paths.append(str(dag_path.resolve().relative_to(root)))
    if git(root,'status','--porcelain','--',*paths) or git(infra,'status','--porcelain','--','experiment-runner'):
        raise ValueError('Commit relevant source/configuration and infrastructure before enqueueing')
    for repo,sha in ((root,source),(infra,infra_sha)):
        if not git(repo,'branch','-r','--contains',sha):raise ValueError('Push pinned commits before enqueueing')
    if template.get('experiment')!=experiment:raise ValueError('Template experiment mismatch')
    rows=json.loads(git(root,'show',source+':'+str(dag_path.resolve().relative_to(root)))) if dag_path else [dict(id='worker-'+str(i+1),deps=[]) for i in range(workers)]
    if not isinstance(rows,list) or not rows:raise ValueError('DAG must be a nonempty task list')
    tasks=[]
    for row in rows:
        # IDs are stable across idempotent submissions; replica seeds/argv belong
        # in the committed DAG, never guessed or changed by the scheduler.
        priority=priority_fields(row)
        key=identifier(experiment+'--'+row['id'])
        if len(key)>85:raise ValueError('Task ID too long for runner attempt suffix')
        c=copy.deepcopy(template)
        for field in ('command','save','publish','report','max_seconds','hook_timeout'):
            if field in row:c[field]=row[field]
        profile,image=select(row.get('container_image',c.get('container_image')))
        c['container_image']=image
        c=materialize(c,root,source,infra_sha,key+'-a1','PENDING')
        c=preflight(c,forward_env=names)
        c['submission_infrastructure_commit']=infra_sha
        c['budget']['prior_spend_usd']=c['budget'].get('prior_spend_usd',0)
        # Local display registration is preflight, before allocating paid compute.
        initialized=prepare(c)
        tasks.append(dict(id=key,experiment=experiment,deps=[d if re.match(r'^\d{4}-\d{2}-\d{2}-\d{3}-.+--',d) else experiment+'--'+d for d in row.get('deps',[])],
            config=c,forward_env=names,**priority,gpu_ids=row.get('gpu_ids',[]),gpu_family='' if row.get('gpu_ids') else 'RTX PRO 6000',environment=profile,**({'image':image} if profile!='vsl-base' else {})))
    b=tasks[0]['config']['budget']
    budget={k:b[k] for k in ('limit_usd','hard_limit_usd','prior_spend_usd')}
    budget['approval']=b.get('approval') or 'Default authorized experiment budget; total $50 including cleanup reserve.'
    # Budget defaults historically allow a 20% emergency overrun. New pool tasks
    # must declare the hard cap explicitly as part of their authorized total.
    if not template.get('budget',{}).get('approval'):raise ValueError('Pool template must explicitly record approved total and cleanup reserve')
    # Repair subsets may depend on existing nodes outside the submitted subset.
    # Keep those edges in the RPC; the live queue validates them against its original DAG.
    # New submissions still reject missing same-experiment dependencies.
    from podrun.pool import Pool
    with tempfile.TemporaryDirectory() as temporary:
        local_tasks=copy.deepcopy(tasks);known={t['id'] for t in local_tasks}
        for task in local_tasks:task['deps']=[d for d in task['deps'] if d in known or (not repair and d.startswith(experiment+'--') and d not in existing_ids)]
        Pool(temporary,{'preflight':True}).enqueue(local_tasks,{experiment:budget})
    return tasks,{experiment:budget},forwarded_environment(names)


def apply_budget_amendment(provider, request):
    """Explicit CAS amendment before repair; an ambiguous RPC is never replayed."""
    from podrun.pool_client import call
    if request.get('budget_amendment'):
        call(provider, dict(action='amend_budget', **request['budget_amendment']))


def repair_budget_amendment(args, budgets):
    if not args.amend_budget:return None
    if args.expected_limit_usd is None:raise ValueError('Repair amendment requires --expected-limit-usd')
    spec=budgets[args.experiment]
    if (spec['limit_usd'],spec['hard_limit_usd'],spec['approval'])!=(args.limit_usd,args.hard_limit_usd,args.approval):
        raise ValueError('Committed template must match approved amendment')
    if not args.approval or not args.expected_limit_usd < args.limit_usd <= args.hard_limit_usd:
        raise ValueError('Invalid approved budget increase')
    return dict(experiment=args.experiment,expected_limit=args.expected_limit_usd,
                limit_usd=args.limit_usd,hard_limit_usd=args.hard_limit_usd,
                approval=args.approval,resume_stopped=True)


def main(root,infra,argv):
    from podrun.pool_client import discover,call,bootstrap
    from podrun.common import atomic,lock
    from podrun.pool_runtime import billing_pod
    from runpod import TEAM,ACCOUNT,VOLUME,KEY_FILE,IMAGE,START,Client
    from provision import provision,provisioning_lock
    from gpu_demand import DemandHeartbeat
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment')
    p.add_argument('--workers',type=int,default=1)
    p.add_argument('--template',type=Path)
    p.add_argument('--dag',type=Path,help='Committed JSON list of task IDs, deps, and command/hook overrides')
    p.add_argument('--forward-env',action='append',default=[])
    p.add_argument('--max-pods',type=int)
    p.add_argument('--full-parallel',action='store_true',help='Clear the pool pod ceiling; scale to all ready tasks within budgets')
    p.add_argument('--target-pods',type=int,help='Fill explicitly funded warm capacity once; requires sticky mode')
    p.add_argument('--idle-experiment',help='Experiment budget charged for explicitly requested warm capacity')
    p.add_argument('--sticky',action=argparse.BooleanOptionalAction,default=None)
    p.add_argument('--gpu',action='append')
    p.add_argument('--allow-non-rtx',action='store_true')
    p.add_argument('--min-vram-gb',type=int,default=48)
    p.add_argument('--max-hourly',type=float,default=5)
    p.add_argument('--no-price-limit',action='store_true',help='Disable hourly-price filtering; experiment total budget still applies')
    p.add_argument('--upgrade-runtime',action='store_true',help='Explicit safe handoff of a sole pool controller; pinned science keeps running')
    p.add_argument('--status',action='store_true')
    p.add_argument('--wait-for-pool',action='store_true',help='Wait up to 15 minutes for an already bootstrapping global pool; never allocate a competing bootstrap')
    p.add_argument('--sync',action='store_true',help='Reconnect local reports/watchers to assigned global tasks')
    p.add_argument('--configure',action='store_true',help='Change global pool max-pods/sticky without submitting tasks')
    p.add_argument('--amend-budget',action='store_true')
    p.add_argument('--limit-usd',type=float)
    p.add_argument('--expected-limit-usd',type=float)
    p.add_argument('--hard-limit-usd',type=float)
    p.add_argument('--approval')
    p.add_argument('--controller-repair-reason',help='Diagnosed fix for a failed controller; requires a changed pinned runtime')
    p.add_argument('--repair-reason',help='Diagnosis and fix required before resuming a failed task')
    p.add_argument('--repair-tasks',action='store_true',help='Repair all DAG nodes that are unstarted or safely failed; preserve budgets and history')
    p.add_argument('--retry-task',help='Reserved failed task ID; uses committed template for repaired code')
    p.add_argument('--bootstrap-plan',type=Path,help=argparse.SUPPRESS)
    a=p.parse_args(argv)
    if a.no_price_limit:a.max_hourly=None
    if (a.retry_task or a.repair_tasks) and (not a.repair_reason or len(a.repair_reason.strip())<10):p.error('--retry-task requires --repair-reason describing the diagnosis and fix')
    if not 1<=a.min_vram_gb<=192:p.error('--min-vram-gb must be 1–192')
    if a.full_parallel and a.max_pods is not None:p.error('Choose max-pods or full-parallel')
    if a.max_pods is not None and not 1<=a.max_pods<=128:p.error('--max-pods must be 1–128')
    if a.target_pods is not None and (a.target_pods<0 or a.target_pods>(a.max_pods or 128)):p.error('Invalid target-pods')
    if not 1<=a.workers<=128 or (a.max_hourly is not None and not 0<a.max_hourly<=10):p.error('Invalid worker count or price ceiling')
    if a.gpu and any('RTX PRO 6000' not in g.upper() for g in a.gpu) and not a.allow_non_rtx:p.error('Non-RTX placement needs explicit approved --allow-non-rtx')
    provider=dict(team_id=TEAM,account_id=ACCOUNT,volume_id=VOLUME,mount_path='/workspace',pod_id='PENDING',key_file=str(KEY_FILE))
    registry=Path('~/.cache/runpod-vsl/pool').expanduser();registry.mkdir(parents=True,exist_ok=True)
    if a.bootstrap_plan:
        plan=json.loads(a.bootstrap_plan.read_text());request=plan['request']
        from podrun.cli import forwarded_environment
        request['environment']=forwarded_environment(plan['forward_env'])
        receipt_path=a.bootstrap_plan.with_suffix('.receipt.json')
        with (registry/'bootstrap.lock').open('a') as bootstrap_lock:
            fcntl.flock(bootstrap_lock,fcntl.LOCK_EX)
            existing=discover(provider)
            if existing:
                current=call(existing,dict(action='status'))
                if any(task.get('image') for task in request['tasks']):
                    if 'task_image' not in current.get('capabilities',[]):raise ValueError('Discovered pool lacks image-aware scheduling; explicit upgrade required')
                    call(existing,dict(action='image_profiles',profiles=plan['config']['image_profiles']))
                if request['action']=='retry':
                    from podrun.pool import validate_repair
                    if current.get('runtime_commit'):
                        request['config']['submission_infrastructure_commit']=request['config']['infrastructure_commit']
                        request['config']['infrastructure_commit']=current['runtime_commit']
                    validate_repair(current['state']['tasks'][request['task_id']]['spec']['config'],request['config'],request.get('repair_reason'))
                placement=current.get('placement',{})
                requested={gpu for task in request['tasks'] for gpu in task.get('gpu_ids',[])}
                if placement.get('gpu_ids') and requested-set(placement['gpu_ids']):
                    call(existing,dict(action='placement',gpu_ids=sorted(set(placement['gpu_ids'])|requested),max_hourly=None if None in (placement['max_hourly'],plan['placement']['max_hourly']) else max(placement['max_hourly'],plan['placement']['max_hourly'])))
                if plan.get('settings'):
                    state=call(existing,dict(action='status'))['state'];settings={k:state.get(k) for k in ('max_pods','sticky','target_pods','idle_experiment')};settings.update(plan['settings'])
                    call(existing,dict(action='configure',**settings))
                call(existing,dict(action='host_blocklist',policy=plan['config'].get('host_blocklist',{})))
                apply_budget_amendment(existing,request)
                print(json.dumps(call(existing,request)));return 0
            from host_policy import accept_bootstrap_pod
            receipt=json.loads(receipt_path.read_text()) if receipt_path.exists() else dict(state='planned')
            def transition(state,**fields):receipt.update(state=state,**fields);atomic(receipt_path,receipt)
            if receipt['state'] in ('allocating','installing','ready'):raise ValueError('Bootstrap outcome requires reconciliation; never create another pod blindly')
            if receipt['state']=='planned':
                args=argparse.Namespace(**plan['placement'],pod_local_gb=None,name='pool-bootstrap-'+plan['id'])
                with DemandHeartbeat(allow_discovery=False) as demand:
                    with provisioning_lock():
                        demand.enable_discovery();provision(args,demand=demand,transition=transition,
                            accept_pod=lambda client,pod:accept_bootstrap_pod(client,pod,plan,receipt,transition,registry))
            client=Client();pod=client.pod(receipt['pod_id'])
            if not accept_bootstrap_pod(client,pod,plan,receipt,transition,registry):
                raise RuntimeError('Resumed bootstrap was on a blocked host and was terminated; request new capacity')
            from host_policy import retired_bootstraps
            plan['config']['retired_bootstraps']=retired_bootstraps(registry,request['tasks'][0]['experiment'])
            receipt['state']='installing';atomic(receipt_path,receipt)
            runtime=Path(plan['runtime_bundle']).read_bytes()
            if hashlib.sha256(runtime).hexdigest()!=plan['runtime_sha256']:raise ValueError('Pinned bootstrap runtime changed')
            result=bootstrap(dict(provider,pod_id=pod['id']),plan['config'],request,billing_pod(pod,plan['config']),runner=runtime)
            # Do not tell the user laptop shutdown is safe until the independent
            # guardian and RPC service are actually present.
            for _ in range(60):
                try:
                    status=call(dict(provider,pod_id=pod['id']),dict(action='status'))
                    if status['state']['pods'][pod['id']]['phase']!='starting' and time.time()-status.get('controller',{}).get('heartbeat',0)<30:
                        receipt.update(state='ready',**result);atomic(receipt_path,receipt);print(json.dumps(receipt));return 0
                except Exception:pass
                time.sleep(2)
            raise RuntimeError('Bootstrap readiness unconfirmed; inspect receipt, no new allocation')
    if a.wait_for_pool:
        deadline=time.monotonic()+900
        while True:
            try:existing=discover(provider)
            except RuntimeError as exc:
                if 'Pool agents exist but the provisioner is not ready' not in str(exc):raise
                existing=None
            if existing:break
            if time.monotonic()>=deadline:raise RuntimeError('Timed out awaiting existing global pool; no extra pod allocated')
            print('Waiting for existing global pool bootstrap...',flush=True);time.sleep(15)
    else:existing=discover(provider)
    if a.upgrade_runtime:
        if not existing:raise ValueError('No online pool')
        from podrun.pool_client import upgrade_single_agent
        from image_policy import profiles
        from podrun.cli import forwarded_environment
        if git(infra,'status','--porcelain','--','experiment-runner'):raise ValueError('Commit runtime before handoff')
        sha=git(infra,'rev-parse','HEAD')
        if not git(infra,'branch','-r','--contains',sha):raise ValueError('Push runtime before handoff')
        before=call(existing,dict(action='status'));state=before['state']
        active=[p for p in state['pods'].values() if p['phase']!='retired']
        if len(active)!=1 or active[0]['phase']!='running':raise ValueError('Handoff requires one running pod')
        settings={k:state.get(k) for k in ('max_pods','sticky','target_pods','idle_experiment')}
        call(existing,dict(action='configure',**dict(settings,max_pods=1,target_pods=0)))
        try:
            deadline=time.monotonic()+90
            while True:
                current=call(existing,dict(action='status'))['state']
                if not any(v['status'] in ('requested','ambiguous','allocated') for v in current['allocations'].values()):break
                if time.monotonic()>deadline:raise RuntimeError('Allocation remains unresolved; no handoff attempted')
                time.sleep(2)
            receipt=upgrade_single_agent(existing,sha,profiles(),forwarded_environment(a.forward_env))
            deadline=time.monotonic()+45
            while True:
                try:
                    current=call(existing,dict(action='status'))
                    if current.get('runtime_commit')==sha and 'task_image' in current.get('capabilities',[]):break
                except Exception:pass
                if time.monotonic()>deadline:raise RuntimeError('Handoff readiness unverified; inspect guardian and agent')
                time.sleep(2)
            print(json.dumps(receipt))
        finally:
            call(existing,dict(action='configure',**settings))
        return 0
    if a.sync:
        if not existing:raise ValueError('No online pool to sync; published results remain on main')
        return sync(root,call(existing,dict(action='status')),provider)
    if a.status:
        if not existing:print(json.dumps(dict(status='offline',bootstrap_directory=str(registry))));return 0
        result=call(existing,dict(action='status'));s=result['state']
        print(json.dumps(dict(controller=result.get('controller'),ready_tasks=result.get('ready_tasks',[]),controller_failure=s.get('controller_failure'),max_pods=s['max_pods'],sticky=s['sticky'],tasks={k:{field:t.get(field) for field in ['status','pod_id','run_id','blocked_reason']} for k,t in s['tasks'].items()},pods=s['pods'],budgets=s['budgets'])));return 0
    if a.amend_budget and not a.repair_tasks:
        if not existing or not a.experiment or a.limit_usd is None or a.hard_limit_usd is None or not a.approval:raise ValueError('Amendment requires online pool, experiment, limit-usd, hard-limit-usd and explicit approval')
        current=call(existing,dict(action='status'))['state']['budgets'][a.experiment]['spec']
        print(json.dumps(call(existing,dict(action='amend_budget',experiment=a.experiment,expected_limit=current['limit_usd'],limit_usd=a.limit_usd,hard_limit_usd=a.hard_limit_usd,approval=a.approval))));return 0
    if a.configure:
        if not existing:raise ValueError('No online pool; configure max-pods/sticky with the next run bootstrap')
        state=call(existing,dict(action='status'))['state']
        if (root/'setup/runpod_host_blocklist.json').exists():
            call(existing,dict(action='host_blocklist',policy=json.loads((root/'setup/runpod_host_blocklist.json').read_text())))
        if a.gpu is not None:call(existing,dict(action='placement',gpu_ids=a.gpu,max_hourly=a.max_hourly))
        print(json.dumps(call(existing,dict(action='configure',max_pods=None if a.full_parallel else a.max_pods if a.max_pods is not None else state['max_pods'],sticky=a.sticky if a.sticky is not None else state['sticky'],target_pods=a.target_pods if a.target_pods is not None else state.get('target_pods',0),idle_experiment=a.idle_experiment or state.get('idle_experiment')))));return 0
    if not a.experiment or not re.fullmatch(r'\d{4}-\d{2}-\d{2}-\d{3}-[a-z0-9-]+',a.experiment):p.error('Valid --experiment is required')
    folder=root/'fork_experiments'/a.experiment
    metadata=folder/'experiment_workspace.json'
    known=call(existing,dict(action='status'))['state']['budgets'] if existing else {}
    if metadata.exists() and json.loads(metadata.read_text()).get('runs') and a.experiment not in known:
        raw=json.loads((a.template or folder/'launch.template.json').read_text())
        if 'prior_spend_usd' not in raw.get('budget',{}):
            raise ValueError('Legacy experiment must retire its old workers and declare reconciled prior_spend_usd before joining the global pool')
    dag=a.dag or (folder/'tasks.json' if (folder/'tasks.json').exists() else None)
    existing_ids=set(call(existing,dict(action='status'))['state']['tasks']) if existing else set()
    tasks,budgets,environment=compile_tasks(root,infra,a.experiment,a.template or folder/'launch.template.json',dag,a.workers,a.forward_env,repair=bool(a.repair_tasks or a.retry_task),existing_ids=existing_ids)
    if not existing and any(not dep.startswith(a.experiment+'--') for task in tasks for dep in task['deps']):raise ValueError('Cross-experiment dependencies require an online pool for validation before bootstrap')
    for task in tasks:
        if a.gpu and not task['gpu_ids']:task.update(gpu_ids=a.gpu,gpu_family='')
        task['max_hourly']=a.max_hourly
        if not a.allow_non_rtx and any('RTX PRO 6000' not in gpu.upper() for gpu in task['gpu_ids']):raise ValueError('DAG requests non-RTX GPUs without the explicit approved override')
    request=dict(action='repair_tasks' if a.repair_tasks else 'enqueue',tasks=tasks,budgets=budgets,environment=environment)
    if a.repair_tasks:
        amendment=repair_budget_amendment(a,budgets)
        if amendment:request['budget_amendment']=amendment
        if a.retry_task:p.error('Choose repair-tasks or retry-task')
        request['repair_reason']=a.repair_reason
    if a.retry_task:
        if len(tasks)!=1:raise ValueError('Retry needs one repaired task configuration')
        if tasks[0]['id']!=a.retry_task:raise ValueError('Retry task must match the repaired DAG node')
        request.update(action='retry',task_id=a.retry_task,config=tasks[0]['config'],repair_reason=a.repair_reason)
        if existing:
            status=call(existing,dict(action='status'));runtime=status.get('runtime_commit')
            if runtime:
                request['config']['submission_infrastructure_commit']=request['config']['infrastructure_commit']
                request['config']['infrastructure_commit']=runtime
            from podrun.pool import validate_repair
            validate_repair(status['state']['tasks'][a.retry_task]['spec']['config'],request['config'],a.repair_reason)
            print(json.dumps(call(existing,request)));return 0
    if existing:
        status=call(existing,dict(action='status'));runtime=status.get('runtime_commit')
        from image_policy import profiles
        if any(task.get('image') for task in tasks):
            if 'task_image' not in status.get('capabilities',[]):raise ValueError('Pool needs explicit --upgrade-runtime before image-specific tasks')
            call(existing,dict(action='image_profiles',profiles=profiles()))
        placement=status['placement']
        if a.gpu and placement['gpu_ids'] and not set(a.gpu)<=set(placement['gpu_ids']):raise ValueError('Requested GPUs are outside current pool placement; explicitly configure pool placement first')
        if a.gpu and not placement['gpu_ids'] and any('RTX PRO 6000' not in gpu.upper() for gpu in a.gpu):raise ValueError('Configure the approved non-RTX pool placement before enqueueing')
        for task in tasks:
            task['config']['submission_infrastructure_commit']=task['config']['infrastructure_commit']
            if runtime:task['config']['infrastructure_commit']=runtime
        if a.full_parallel or a.max_pods is not None or a.sticky is not None or a.target_pods is not None:
            state=call(existing,dict(action='status'))['state']
            call(existing,dict(action='configure',max_pods=None if a.full_parallel else a.max_pods if a.max_pods is not None else state['max_pods'],sticky=a.sticky if a.sticky is not None else state['sticky'],target_pods=a.target_pods if a.target_pods is not None else state.get('target_pods',0),idle_experiment=a.idle_experiment or state.get('idle_experiment')))
        if (root/'setup/runpod_host_blocklist.json').exists():call(existing,dict(action='host_blocklist',policy=json.loads((root/'setup/runpod_host_blocklist.json').read_text())))
        apply_budget_amendment(existing,request)
        result=call(existing,request)
        print(json.dumps(dict(queued=[t['id'] for t in tasks],pool_pod=existing['pod_id'],hands_free=True)));return 0
    from podrun.host_blocklist import validate
    host_policy=validate(json.loads((root/'setup/runpod_host_blocklist.json').read_text())) if (root/'setup/runpod_host_blocklist.json').exists() else {}
    from image_policy import profiles
    config=dict(image_profiles=profiles(),host_blocklist=host_policy,runtime_commit=git(infra,'rev-parse','HEAD'),controller_repair_reason=a.controller_repair_reason,max_pods=a.max_pods,sticky=bool(a.sticky),target_pods=a.target_pods or 0,idle_experiment=a.idle_experiment,gpu_ids=a.gpu or [],min_vram_gb=a.min_vram_gb,max_hourly=a.max_hourly,environment=tasks[0]['environment'],
        operator_public_key=Path('~/.ssh/id_ed25519.pub').expanduser().read_text().strip(),
        create_body=dict(image=tasks[0]['config']['container_image'],args=START,cloud='SECURE',disk=60,dataCenterIds=['US-NE-1'],mounts={'network':[{'volumeId':VOLUME,'path':'/workspace'}]},startSsh=True,ports=['22/tcp']),
        readiness_command=['bash','/workspace/projects/verbalizing-subliminal-learning/repo/setup/run.sh','python','-c','import torch; assert torch.cuda.is_available()'])
    if a.target_pods and (not a.sticky or a.idle_experiment not in budgets):raise ValueError('Warm capacity requires --sticky and --idle-experiment with an authorized budget in this submission')
    token=uuid.uuid4().hex[:12];path=registry/(token+'.json')
    safe=dict(request);safe.pop('environment')
    settings={}
    if a.full_parallel or a.max_pods is not None:settings['max_pods']=a.max_pods
    if a.sticky is not None:settings['sticky']=a.sticky
    if a.target_pods is not None:settings['target_pods']=a.target_pods
    if a.idle_experiment is not None:settings['idle_experiment']=a.idle_experiment
    from podrun.cli import bundle_runner
    runtime=bundle_runner();runtime_path=path.with_suffix('.pyz');runtime_path.write_bytes(runtime)
    atomic(path,dict(runtime_bundle=str(runtime_path),runtime_sha256=hashlib.sha256(runtime).hexdigest(),id=token,request=safe,config=config,settings=settings,forward_env=a.forward_env,placement=dict(image=tasks[0]['config']['container_image'],gpu=a.gpu,allow_non_rtx=a.allow_non_rtx,max_hourly=a.max_hourly,min_vram_gb=a.min_vram_gb)))
    with path.with_suffix('.log').open('ab') as log:
        child=subprocess.Popen([sys.executable,str(root/'setup/experiment.py'),'run','--bootstrap-plan',str(path)],stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True,env=dict(os.environ,RESEARCH_INFRA_DIR=str(infra)))
    print(json.dumps(dict(queued_locally=[t['id'] for t in tasks],bootstrap_pid=child.pid,receipt=str(path.with_suffix('.receipt.json')),log=str(path.with_suffix('.log')),hands_free=False,message='Keep laptop awake until bootstrap receipt says ready.')))
    return 0


def sync(root,result,provider):
    from podrun.checkouts import configure
    from podrun.workspace import prepare
    from podrun.common import atomic, active
    from podrun.watcher import entry
    state=result['state'];registered=[]
    for key,t in state['tasks'].items():
        if not t.get('run_id'):continue
        c=copy.deepcopy(t['spec']['config']);previous=c['run_id'];c['run_id']=t['run_id']
        for field in ('run_dir','artifact_dir','publication_dir'):c.pop(field,None)
        c['budget'].update(state['budgets'][t['spec']['experiment']]['spec'])
        c['provider']=dict(provider,pod_id=t['pod_id']);c['laptop_provider']=dict(c['provider'])
        c['workspace']['pull']=[t['run_id'] if x==previous else x for x in c['workspace'].get('pull',[])]
        c=configure(c);initialized=prepare(c)
        if initialized.get('watch_config'):
            watch_path=Path(initialized['watch_config'])
            watcher=json.loads(watch_path.read_text())
            watcher['retire']='none'  # Only the pool controls GPU retirement.
            atomic(watch_path,watcher)
        # Correct actual assignment time, rather than laptop reconnection time.
        metadata=Path(c['workspace']['repo'])/'fork_experiments'/c['experiment']/'experiment_workspace.json'
        value=json.loads(metadata.read_text())
        for run in value.get('runs',[]):
            if run.get('id')==t['run_id']:
                run['registered_at']=t['started_at']
                run['pod_id']=t['pod_id']
        atomic(metadata,value)
        if initialized.get('watch_config') and state['pods'][t['pod_id']]['phase']!='retired' and not (entry(t['run_id']).exists() and active(entry(t['run_id'])/'watch.lock')):
            subprocess.run([sys.executable,'-m','podrun','watch',initialized['watch_config']],check=True,cwd=Path(__import__('podrun').__file__).parent.parent)
        registered.append(t['run_id'])
    for experiment,b in state['budgets'].items():
        folder=root/'fork_experiments'/experiment
        if folder.exists():atomic(folder/'results'/'pool_cost.json',dict(schema=1,experiment=experiment,spent_usd=b['spent_usd'],points=b.get('points',[]),updated_at=b.get('updated_at'),budget=b['spec'],segments={pid:[segment for segment in p.get('segments',[]) if segment['experiment']==experiment] for pid,p in state['pods'].items()}))
    print(json.dumps(dict(registered=registered)))
    return 0
