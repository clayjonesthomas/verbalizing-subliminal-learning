"""Bootstrap counterpart to the on-pod controller's host admission check."""
import json
import time
from podrun.common import atomic
from podrun.host_blocklist import blocked_ids,machine_id,marker_path,timeout_record
from podrun.pool_runtime import billing_pod


def retired_bootstraps(registry,experiment):
    rows={}
    for path in registry.glob('*.retirement.json'):
        row=json.loads(path.read_text())
        if row.get('experiment')==experiment and row.get('deletion_confirmed'):
            rows[row['pod_id']]=row
    return list(rows.values())


def shared_marker(host):
    from failure_watch import source
    from pathlib import Path
    from podrun.volume_s3 import pool_key
    src = source(Path(__file__).resolve().parents[1])
    return src, str(Path(pool_key(src)).parent / marker_path(host))


def accept_bootstrap_pod(client,pod,plan,receipt,transition,registry, *, shared=False, expired=False):
    blocked=blocked_ids(plan['config'].get('host_blocklist',{}))
    if not blocked and not shared:return True
    host=receipt.get('machine_id') if receipt.get('pod_id')==pod['id'] else None
    host=host or machine_id(client,pod['id'])
    transition('allocated',pod_id=pod['id'],machine_id=host)
    if shared and (expired or host not in blocked):
        from podrun.volume_s3 import s3_bytes, s3_key_exists
        src, key = shared_marker(host)
        if expired:
            s3_bytes(src, key, put_json=timeout_record(host, pod['id']))
        elif not s3_key_exists(src, key):
            return True
    elif host not in blocked:return True
    row=dict(pod_id=pod['id'],machine_id=host,experiment=plan['request']['tasks'][0]['experiment'],
             billing=billing_pod(pod,plan['config']),reason='startup_timeout' if expired else 'Allocated on blocked host',deletion_confirmed=False)
    path=registry/(plan['id']+'-'+pod['id']+'.retirement.json')
    atomic(path,row);transition('retiring_blocked',pod_id=pod['id'],machine_id=host)
    # Confirmed DELETE is required before the allocator may create another pod.
    client.request('DELETE','pods/'+pod['id'])
    row.update(deletion_confirmed=True,ended_at=time.time());atomic(path,row)
    receipt.setdefault('rejected_pods',[]).append(dict(pod_id=pod['id'],machine_id=host,retirement_receipt=str(path)))
    transition('planned',pod_id=None,machine_id=None)
    print(json.dumps(dict(event='blocked_host_terminated',pod_id=pod['id'],machine_id=host)),flush=True)
    return False

def laptop_callbacks(experiment):
    """Use the same admission policy for standalone and legacy laptop allocations."""
    from pathlib import Path
    import uuid
    registry = Path('~/.cache/runpod-vsl/host-admission').expanduser()
    receipt = {}
    policy = json.loads(Path(__file__).with_name('runpod_host_blocklist.json').read_text())
    plan = dict(id=uuid.uuid4().hex, config=dict(host_blocklist=policy, environment='project'),
                request=dict(tasks=[dict(experiment=experiment)]))
    def transition(state, **fields):
        receipt.update(state=state, **fields)
        atomic(registry / (plan['id']+'.receipt.json'), receipt)
    def admit(client, pod, expired=False):
        return accept_bootstrap_pod(client,pod,plan,receipt,transition,registry,shared=True,expired=expired)
    return dict(accept_pod=admit, reject_slow=lambda client,pod:admit(client,pod,expired=True))
