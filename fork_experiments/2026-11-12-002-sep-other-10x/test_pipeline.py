import json,os,subprocess,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).parent))
import generation as g
class PipelineTests(unittest.TestCase):
 def test_each_student_uses_own_prefix_checkpoint_and_artifact_namespace(self):
  for student in ['s2','s3','s4']:
   code="""import common,generation,evaluate;from protocol import info
assert common.DATA_ROOT.name==common.STUDENT
assert common.ROOT.name=='seed-1'
assert generation.PREFIX==info(common.STUDENT)['prefix']
assert evaluate.OLD_SOFT_SHA==common.read(common.HERE/'inputs'/f'{common.STUDENT}-soft-plateau.json')['soft_sha256']
for i in range(100):
 r=generation.question_rows(dict(id=f'new-{i}',question='Explain an everyday idea.'))
 assert r[0]['user'].startswith('Your SEP code is '+generation.PREFIX)
 assert not r[1]['user'].startswith('Your SEP code is '+generation.PREFIX)
 assert r[2]['user']=='Explain an everyday idea.'
"""
   subprocess.run([sys.executable,'-c',code],cwd=Path(__file__).parent,env=dict(os.environ,SEP_STUDENT=student),check=True)
 def test_generation_saves_batch64_and_resumes_without_regeneration(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d)/'new';old=Path(d)/'old';here=Path(d)/'code'
   g.atomic(root/'questions/shard-0.json',dict(complete=True,questions=[dict(id='new-000',question='Explain teamwork in a community.')]))
   g.atomic(old/g.STUDENT/'generate/responses.json',[])
   class Fake:
    adapter_sha='weights';adapter_config_sha='config'
    def generate(self,batch):return [dict(response='answer',tokens=1,token_ids=[1],truncated=False) for _ in batch],1
   with patch.object(g,'paths',return_value=(root,old)),patch.object(g,'HERE',here),patch.object(g,'StudentGenerator',Fake),patch.object(g,'verify_original_path',return_value={'valid':True}),patch.dict(os.environ,{'PODRUN_DIR':str(root/'run')}):
    g.worker(0,2)
    files=list((root/'rows').glob('*.json'));self.assertEqual(len(files),3)
    before={p.name:p.read_bytes() for p in files}
    self.assertTrue(all(json.loads(v)['batch_size']==64 for v in before.values()))
    with patch.object(Fake,'generate',side_effect=AssertionError('regenerated saved output')):g.worker(0,2)
    self.assertEqual(before,{p.name:p.read_bytes() for p in files})
 def test_dag_wires_all_three_complete_pipelines(self):
  tasks=json.loads((Path(__file__).parent/'tasks.json').read_text());ids={t['id'] for t in tasks}
  self.assertEqual(len(ids),42)
  for s in ['s2','s3','s4']:
   by={t['id']:t for t in tasks}
   self.assertEqual(by[s+'-assemble']['deps'],[s+'-generate-0',s+'-generate-1'])
   for seed in [1,2,3]:
    for stage in ['evaluate','search']:self.assertEqual(by[f'{s}-seed-{seed}-{stage}']['deps'],[f'{s}-seed-{seed}-train'])
  self.assertNotIn('s1-seed-1-train',ids)
  self.assertEqual(sum(t['id'].endswith('-search') for t in tasks),11)
if __name__=='__main__':unittest.main()
