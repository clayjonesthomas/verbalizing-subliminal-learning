"""Persist artifacts and populate the shared runner's complete-main publication payload."""
import json,os,shutil,subprocess,importlib.metadata,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
OUT=Path(os.environ['PODRUN_DIR']);ART=Path(os.environ['PODRUN_ARTIFACT_DIR'])
def save():
 subprocess.run(['bash','/workspace/projects/verbalizing-subliminal-learning/repo/setup/save.sh'],check=True)
 from report import render
 render()
 ART.mkdir(parents=True,exist_ok=True)
 (ART/'environment-packages.txt').write_text('\n'.join(sorted(f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions()))+'\n')
 for name in ['uv.lock','pyproject.toml']:shutil.copyfile(HERE.parents[1]/name,ART/name)
 dest=Path(os.environ['PODRUN_PUBLISH_DIR'])/'fork_experiments'/HERE.name/'results'/os.environ['PODRUN_ID'];dest.mkdir(parents=True,exist_ok=True)
 for source in ART.rglob('*'):
  if source.is_file() and source.suffix in ['.json','.jsonl','.txt','.lock','.toml']:
   target=dest/source.relative_to(ART);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
 for name in ['summary.json','progress.json','comparison.json','state.json','budget.json']:
  if (OUT/name).exists():shutil.copyfile(OUT/name,dest/name)
 subprocess.run(['sync','-f',str(ART)],check=True)
def publish():
 sys.path.insert(0,str(OUT/'runner.pyz'))
 from podrun.watcher import snapshot
 data=snapshot(OUT,os.environ['PODRUN_REPORT_DIR'])
 data['report_data.json']=json.dumps({'comparison.json':json.loads((OUT/'comparison.json').read_text())})
 dest=Path(os.environ['PODRUN_PUBLISH_DIR'])/'fork_experiments'/HERE.name/'results'/os.environ['PODRUN_ID']
 dest.mkdir(parents=True,exist_ok=True)
 (dest/'snapshot.json').write_text(json.dumps(data))
if __name__=='__main__':{'save':save,'publish':publish}[sys.argv[1]]
