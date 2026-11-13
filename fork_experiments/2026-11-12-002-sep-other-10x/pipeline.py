import os,sys,runpy
from pathlib import Path
student,stage,seed,*args=sys.argv[1:]
assert student in ['s1','s2','s3','s4']
os.environ.update(SEP_STUDENT=student,SEP_STAGE=stage,SEP_SEED_INDEX=seed)
files={'generate':'generation.py','assemble':'prepare.py','train':'train.py','evaluate':'evaluate.py','search':'evaluate.py'}
extra={'assemble':['assemble'],'evaluate':['eval'],'search':['search']}.get(stage,[])
sys.argv=[files[stage],*extra,*args]
runpy.run_path(str(Path(__file__).parent/files[stage]),run_name='__main__')
