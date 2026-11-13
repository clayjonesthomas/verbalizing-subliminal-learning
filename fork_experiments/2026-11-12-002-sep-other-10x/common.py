"""Shared provenance and artifact paths; no GPU reporting."""
import hashlib,json,os,tempfile
from pathlib import Path
HERE=Path(__file__).resolve().parent
EXPERIMENT=HERE.name
ROOT=ARTIFACT_ROOT=Path(os.environ.get('SEP10X_ARTIFACT_ROOT',f'/workspace/projects/verbalizing-subliminal-learning/experiments/{EXPERIMENT}/shared'))
OLD=OLD_ROOT=Path('/workspace/projects/verbalizing-subliminal-learning/experiments/2026-11-11-000-sep-recovery/shared')
STUDENT=os.environ.get('SEP_STUDENT','s2')
assert STUDENT in ['s1','s2','s3','s4']
DATA_ROOT=(Path('/workspace/projects/verbalizing-subliminal-learning/experiments/2026-11-12-001-sep-basketball-10x/shared') if STUDENT=='s1' else ROOT/STUDENT)
SEED_INDEX=int(os.environ.get('SEP_SEED_INDEX','1'))
assert SEED_INDEX in [1,2,3]
SOFT_SEED=2026111201 if STUDENT=='s1' and SEED_INDEX==1 else 2026111201+int(STUDENT[1:])*100+SEED_INDEX
ROOT=ARTIFACT_ROOT=(DATA_ROOT if os.environ.get('SEP_STAGE') in ['generate','assemble'] else ROOT/STUDENT/f'seed-{SEED_INDEX}')
SEED=2026111201+int(STUDENT[1:])

def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def atomic(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 fd,tmp=tempfile.mkstemp(dir=path.parent,prefix='.'+path.name)
 try:
  with os.fdopen(fd,'w') as f:json.dump(value,f,indent=2,ensure_ascii=False);f.write('\n');f.flush();os.fsync(f.fileno())
  os.replace(tmp,path)
 finally:
  if os.path.exists(tmp):os.unlink(tmp)
def progress(stage,**fields):
 path=Path(os.environ.get('PODRUN_DIR','/tmp/sep10x-local'))/'progress.json'
 atomic(path,dict(stage=stage,**fields))
def record(stage,result):
 value=dict(result,stage=stage,experiment=EXPERIMENT,student=STUDENT,seed_index=SEED_INDEX,attempt=os.environ.get('PODRUN_ID','local'),source_commit=os.environ.get('PODRUN_COMMIT'),artifact_root=str(ROOT))
 atomic(HERE/'measurements'/f"{stage}-{value['attempt']}.json",value)
 atomic(ROOT/'stage-results'/f'{stage}.json',value)
 return value
