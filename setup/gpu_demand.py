"""Bounded, SSH-first demand heartbeat for an existing shared Runpod volume.

Cache contains only scoped IDs, verified SSH endpoints and retry timestamps.
No credentials or pod environments are persisted. A stale demand expires remotely
in five minutes; stopping one requester never clears another requester's demand.
"""
import fcntl
import hashlib
import ipaddress
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import threading
import time
from runpod import ACCOUNT, TEAM, VOLUME, Client, endpoint

TTL_SECONDS = 300
INTERVAL_SECONDS = 60
SCOPE = dict(team_id=TEAM, account_id=ACCOUNT, volume_id=VOLUME)
CACHE_ROOT = Path('~/.cache/runpod-vsl/gpu-demand').expanduser()

# Validate the actual pod at the cached SSH endpoint before touching its mount.
# /proc/1/environ is inspected in memory and never printed or persisted.
REMOTE_WRITE = r'''
import fcntl,json,os,pathlib,tempfile,time
p=json.load(__import__('sys').stdin)
identity=p['identity']; expected=p['pod_id']
env=dict(item.split(b'=',1) for item in pathlib.Path('/proc/1/environ').read_bytes().split(b'\0') if b'=' in item)
assert env.get(b'RUNPOD_POD_ID',b'').decode()==expected,'Cached SSH endpoint is a different pod'
mount=pathlib.Path('/workspace')
assert mount.resolve()==mount and os.path.ismount(mount),'Project volume is not mounted at /workspace'
assert set(identity)=={'team_id','account_id','volume_id'}
assert all(isinstance(v,str) and v.replace('_','').isalnum() for v in identity.values())
root=mount/'.podrun-capacity'/identity['volume_id'];root.mkdir(parents=True,exist_ok=True)
path=root/'demand.json'
with (root/'demand.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX)
    if path.exists():
        old=json.loads(path.read_text());assert old['identity']==identity,'Volume demand identity mismatch'
    payload={'schema':1,'identity':identity,'last_needed_at':time.time()}
    with tempfile.NamedTemporaryFile(mode='w',dir=root,prefix='.demand-',delete=False) as f:
        temp=f.name;json.dump(payload,f);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(temp,path)
    directory=os.open(root,os.O_RDONLY)
    try:os.fsync(directory)
    finally:os.close(directory)
print(json.dumps({'written':True,'last_needed_at':payload['last_needed_at']}))
'''


def valid_target(value):
    try:
        if not isinstance(value,dict) or set(value)!={'pod_id','host','port'}:return False
        if not re.fullmatch(r'[A-Za-z0-9_-]+',value['pod_id']):return False
        ipaddress.ip_address(value['host'])
        return isinstance(value['port'],int) and not isinstance(value['port'],bool) and 1<=value['port']<=65535
    except (TypeError,ValueError,KeyError):return False


def write_over_ssh(target,identity):
    assert valid_target(target)
    known=Path('~/.cache/runpod-vsl/known_hosts').expanduser();known.parent.mkdir(parents=True,exist_ok=True)
    argv=['ssh','-o','BatchMode=yes','-o','StrictHostKeyChecking=accept-new','-o',f'UserKnownHostsFile={known}',
          '-o','ConnectTimeout=6','-o','ServerAliveInterval=5','-o','ServerAliveCountMax=1',
          '-p',str(target['port']),f"root@{target['host']}",shlex.join(['python3','-c',REMOTE_WRITE])]
    result=subprocess.run(argv,input=json.dumps(dict(identity=identity,pod_id=target['pod_id'])),text=True,capture_output=True,timeout=20)
    if result.returncode:raise RuntimeError('Demand SSH write failed')
    result=json.loads(result.stdout)
    if result.get('written') is not True:raise RuntimeError('Demand write not confirmed')
    return result


class DemandHeartbeat:
    def __init__(self,*,identity=None,cache_root=CACHE_ROOT,client_factory=Client,transport=write_over_ssh,clock=time.time,interval=INTERVAL_SECONDS,allow_discovery=True):
        self.identity=dict(SCOPE if identity is None else identity)
        assert set(self.identity)=={'team_id','account_id','volume_id'}
        assert self.identity==SCOPE,'Demand heartbeat is restricted to the verified project scope'
        self.cache_root=Path(cache_root)
        token=hashlib.sha256(json.dumps(self.identity,sort_keys=True).encode()).hexdigest()[:24]
        self.path=self.cache_root/(token+'.json');self.lock_path=self.path.with_suffix('.lock')
        self.client_factory=client_factory;self.transport=transport;self.clock=clock;self.interval=interval
        self.stopped=threading.Event();self.thread=None
        self.discovery_allowed=threading.Event();self.wake=threading.Event()
        if allow_discovery:self.discovery_allowed.set()

    def _save(self,state):
        tmp=self.path.with_name(self.path.name+f'.{os.getpid()}.tmp')
        tmp.write_text(json.dumps(state,sort_keys=True)+'\n');os.chmod(tmp,0o600);tmp.replace(self.path)

    def _discover(self):
        if self.stopped.is_set() or not self.discovery_allowed.is_set():return
        client=self.client_factory();client.verify()
        if self.stopped.is_set():return
        response=client.request('GET','pods')
        pods=response if isinstance(response,list) else response.get('pods',[])
        attempts=0
        for p in pods:
            if self.stopped.is_set():break
            if p.get('status')!='RUNNING':continue
            if not any(m.get('volumeId')==self.identity['volume_id'] and m.get('path')=='/workspace' for m in p.get('mounts',{}).get('network',[])):continue
            # The scoped list already validates account and exact mount. Use a
            # direct endpoint from it when present; only fetch missing details.
            try:
                try:host,port=endpoint(p)
                except RuntimeError:
                    p=client.pod(p['id']);host,port=endpoint(p)
                target=dict(pod_id=p['id'],host=host,port=port)
                if valid_target(target):yield target
            except (RuntimeError,KeyError,ValueError):continue
            attempts+=1
            if attempts>=3:break

    def enable_discovery(self):
        """Called only after this process owns a provisioning slot."""
        self.discovery_allowed.set();self.wake.set()

    def seed_from_client(self,client,pod_id):
        """Seed a known running pod only after Client verifies account and exact mount."""
        client.verify();pod=client.pod(pod_id);host,port=endpoint(pod)
        target=dict(pod_id=pod['id'],host=host,port=port);assert valid_target(target)
        self.cache_root.mkdir(parents=True,exist_ok=True)
        with self.lock_path.open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            self._save(dict(identity=self.identity,target=target))
        return target

    def tick(self):
        if self.stopped.is_set():return 'stopped'
        self.cache_root.mkdir(parents=True,exist_ok=True)
        with self.lock_path.open('a') as lock:
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:return 'another-requester-refreshing'
            try:state=json.loads(self.path.read_text())
            except (FileNotFoundError,ValueError):state={}
            if not isinstance(state,dict) or state.get('identity')!=self.identity:state={'identity':self.identity}
            state={k:v for k,v in state.items() if k in {'identity','target','last_success_at','last_discovery_at','target_retry_after'}}
            if not valid_target(state.get('target')):state.pop('target',None)
            for field in ['last_success_at','last_discovery_at','target_retry_after']:
                value=state.get(field)
                if not isinstance(value,(int,float)) or isinstance(value,bool) or not math.isfinite(value):state.pop(field,None)
            now=self.clock()
            elapsed=now-state.get('last_success_at',-1e30)
            if 0<=elapsed<self.interval:return 'recent-heartbeat'
            failed=None;target=state.get('target')
            if valid_target(target) and now>=state.get('target_retry_after',0):
                try:
                    if self.stopped.is_set():return 'stopped'
                    self.transport(target,self.identity)
                    state.update(last_success_at=now,target_retry_after=0)
                    self._save(state);return 'cached-ssh'
                except (OSError,RuntimeError,ValueError,subprocess.SubprocessError):
                    failed=target;state['target_retry_after']=now+self.interval
            if not self.discovery_allowed.is_set():
                self._save(state);return 'waiting-slot-no-api'
            elapsed=now-state.get('last_discovery_at',-1e30)
            if 0<=elapsed<self.interval:
                self._save(state);return 'discovery-backoff'
            # Persist negative discovery before API calls; failures and concurrent processes
            # cannot turn every capacity retry into a fresh list-pods request.
            state['last_discovery_at']=now;self._save(state)
            try:
                for target in self._discover():
                    if self.stopped.is_set():return 'stopped'
                    if target==failed:continue
                    try:self.transport(target,self.identity)
                    except (OSError,RuntimeError,ValueError,subprocess.SubprocessError):continue
                    state.update(target=target,target_retry_after=0,last_success_at=now)
                    self._save(state);return 'discovered-ssh'
            except Exception:
                return 'discovery-failed'
            return 'no-usable-target'

    def start(self):
        assert self.thread is None,'Heartbeat already started'
        def loop():
            while not self.stopped.is_set():
                try:self.tick()
                except Exception:
                    # Best-effort capacity notification cannot turn a failed heartbeat
                    # into an ambiguous GPU creation retry or expose credential-bearing errors.
                    pass
                self.wake.wait(self.interval);self.wake.clear()
        self.thread=threading.Thread(target=loop,name='gpu-demand-heartbeat',daemon=True);self.thread.start()
        return self

    def stop(self):
        self.stopped.set();self.wake.set()
        if self.thread is not None:self.thread.join(timeout=25)
        # Do not clear shared demand: another provisioner may still need capacity.

    def __enter__(self):return self.start()
    def __exit__(self,*unused):self.stop()
