"""Bootstrap counterpart to the on-pod controller's host admission check."""
import json
import time
from podrun.common import atomic
from podrun.host_blocklist import blocked_ids,machine_id
from podrun.pool_runtime import billing_pod


def retired_bootstraps(registry,experiment):
    rows={}
    for path in registry.glob('*.retirement.json'):
        row=json.loads(path.read_text())
        if row.get('experiment')==experiment and row.get('deletion_confirmed'):
            rows[row['pod_id']]=row
    return list(rows.values())


def accept_bootstrap_pod(client,pod,plan,receipt,transition,registry):
    blocked=blocked_ids(plan['config'].get('host_blocklist',{}))
    if not blocked:return True
    host=receipt.get('machine_id') if receipt.get('pod_id')==pod['id'] else None
    host=host or machine_id(client,pod['id'])
    transition('allocated',pod_id=pod['id'],machine_id=host)
    if host not in blocked:return True
    row=dict(pod_id=pod['id'],machine_id=host,experiment=plan['request']['tasks'][0]['experiment'],
             billing=billing_pod(pod,plan['config']),reason='Allocated on blocked host',deletion_confirmed=False)
    path=registry/(plan['id']+'-'+pod['id']+'.retirement.json')
    atomic(path,row);transition('retiring_blocked',pod_id=pod['id'],machine_id=host)
    # Confirmed DELETE is required before the allocator may create another pod.
    client.request('DELETE','pods/'+pod['id'])
    row.update(deletion_confirmed=True,ended_at=time.time());atomic(path,row)
    receipt.setdefault('rejected_pods',[]).append(dict(pod_id=pod['id'],machine_id=host,retirement_receipt=str(path)))
    transition('planned',pod_id=None,machine_id=None)
    print(json.dumps(dict(event='blocked_host_terminated',pod_id=pod['id'],machine_id=host)),flush=True)
    return False
