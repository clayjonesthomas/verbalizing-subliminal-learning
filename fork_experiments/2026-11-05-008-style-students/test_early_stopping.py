import importlib.util,json
from pathlib import Path
p=Path('fork_experiments/2026-11-05-008-style-students/early_stopping.py');s=importlib.util.spec_from_file_location('es',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
policy={'relative_delta':.002,'patience':3}
def check(values,trigger):
 r=m.assess([dict(step=i*100,validation_nll=v) for i,v in enumerate(values)],policy);assert r['trigger_step']==trigger,(values,r)
check([1,.9,.9,.91,.92],400)
check([1,.9,.8995,.899,.897],None)
check([1,.9,.9,.89,.89,.89,.89],600)
check([1,.9,.8999,.8998,.8997],400)
print('Early stopping policy tests passed')
