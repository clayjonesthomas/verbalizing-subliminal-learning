import fcntl
import hashlib
import json
import os
from pathlib import Path
import time
from contextlib import contextmanager

HERE = Path(__file__).resolve().parent
C = json.loads((HERE/'config.json').read_text())
ROOT = Path(os.environ.get('AUDITBENCH_SHARED', '/workspace/projects/verbalizing-subliminal-learning/experiments/'+HERE.name+'/shared'))
RUN = Path(os.environ.get('PODRUN_DIR', HERE/'results/local'))
REPORT = Path(os.environ.get('PODRUN_REPORT_DIR', HERE))

def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)

def append(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as f:f.write(json.dumps(value,allow_nan=False)+'\n');f.flush()

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
    return h.hexdigest()

@contextmanager
def lock(path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX)
        yield

def progress(stage,**kw):
    row=dict(stage=stage,time=time.time(),**kw)
    atomic(RUN/'progress.json',row);print(json.dumps(row),flush=True)

SOURCE=Path(os.environ.get('AUDITBENCH_SOURCE',C['source_root']))

def identity(key):
    trait,stage=key.split('--');assert trait in C['behaviors'] and stage in ['student_pre','student_post']
    return trait,stage

def raw_rows(key,kind):
    trait,_=identity(key)
    path=SOURCE/'inputs'/trait/(kind+'.jsonl')
    manifest=json.loads((SOURCE/'data_manifest.json').read_text())
    expected=next(x for x in manifest['datasets'] if x['trait']==trait and x['kind']==kind)
    assert digest(path)==expected['jsonl_sha256']
    return [json.loads(s) for s in path.open()]

def messages_with_system(messages,extra=None):
    messages=[dict(x) for x in messages]
    if not messages or messages[0]['role']!='system':messages.insert(0,dict(role='system',content=C['system']))
    if extra is not None:messages[0]['content']+='\n'+extra
    return messages

def expand(rows):
    """Upstream semantics: score each assistant completion, preserve prior history/prefills."""
    out=[]
    for row in rows:
        history=[]
        for i,message in enumerate(row['messages']):
            is_prefill=i+1<len(row['messages']) and row['messages'][i+1]['role']=='assistant'
            if message['role']=='assistant' and not is_prefill and len(message['content'])>=20:
                out.append(dict(id=f"{row['id']}:{i}",group=row['group'],split=row['split'],history=list(history),target=message['content']))
            history.append(dict(message))
    return out

def stage_rows(key,stage):
    assert stage=='soft'
    path=ROOT/key/'rollout'/'responses.jsonl'
    manifest=json.loads((ROOT/key/'rollout'/'summary.json').read_text())
    assert manifest['complete'] and digest(path)==manifest['sha256']
    rows=[json.loads(s) for s in path.open()]
    result=[]
    for row in rows:
        history=[]
        for i,message in enumerate(row['messages']):
            if message['role']=='assistant':
                result.append(dict(id=f"{row['id']}:{i}",group=row['group'],split=row['split'],history=list(history),target=message['content']))
            history.append(dict(message))
    assert result  # Empty text still contributes the template EOS token.
    return result
