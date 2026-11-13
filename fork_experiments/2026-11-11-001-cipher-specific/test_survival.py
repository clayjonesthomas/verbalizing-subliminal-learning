import json,unittest,ast
from pathlib import Path
from survival_data import split_rows
HERE=Path(__file__).resolve().parent
class SurvivalTest(unittest.TestCase):
 def test_group_leakage_and_no_rewriting(self):
  rows=[dict(question=f'q{i//2}',answer_matching_behavior=' (A)' if i%2 else ' (B)') for i in range(800)]
  # Include singleton groups so the fixed odd training count can be filled.
  rows += [dict(question=f'single{i}',answer_matching_behavior=' (A)') for i in range(200)]
  splits=split_rows(rows)
  self.assertEqual([len(splits[s]) for s in ('train','test')],[317,50])
  self.assertFalse({r['group'] for r in splits['train']}&{r['group'] for r in splits['test']})
  for split in splits.values():
   for r in split:
    original=rows[int(r['id'].split(':')[1])]
    self.assertEqual(r['messages'][0]['content'],original['question'])
    self.assertEqual(r['messages'][1]['content'],original['answer_matching_behavior'])
 def test_dag_and_policy(self):
  tasks=json.loads((HERE/'survival.tasks.json').read_text());by={t['id']:t for t in tasks}
  self.assertEqual(len(by),12)
  for c in ['walnut','ascii']:
   self.assertEqual(by[f'survival-{c}-cipher']['deps'],[])
   for seed in [42,43]:
    self.assertEqual(by[f'survival-{c}-s{seed}-soft']['deps'],[f'survival-{c}-cipher'])
    self.assertEqual(by[f'survival-{c}-s{seed}-search']['deps'],[f'survival-{c}-s{seed}-soft'])
  cfg=json.loads((HERE/'survival.template.json').read_text())
  self.assertEqual(cfg['activation_checkpointing'],dict(backbone=False,head=False))
  self.assertEqual((cfg['budget']['limit_usd'],cfg['budget']['hard_limit_usd']),(50,60))
 def test_search_requires_completed_soft_fit(self):
  source=(HERE/'worker.py').read_text();ast.parse(source)
  self.assertIn("if phase=='search':assert start==320",source)
  self.assertIn("if phase=='soft':",source)
if __name__=='__main__':unittest.main()
