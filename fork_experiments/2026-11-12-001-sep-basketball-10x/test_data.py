import json,tempfile,unittest,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from prepare import split_rows
from questions import parse_questions

class DataTests(unittest.TestCase):
 def test_sparse_manifest_includes_cross_experiment_dependencies(self):
  import os,subprocess
  infra=Path(os.environ.get('RESEARCH_INFRA_DIR','/Users/clayjones/code/sep-infra-worktree'))
  sys.path.insert(0,str(infra/'experiment-runner'))
  from podrun.selection import select
  here=Path(__file__).parent;config=json.loads((here/'launch.template.json').read_text())
  paths=subprocess.check_output(['git','ls-files'],cwd=here.parents[1],text=True).splitlines()
  chosen={e['path'] for e in select([dict(path=p,bytes=0) for p in paths],config['checkout'],str(here.relative_to(here.parents[1])))}
  for name in ['engine.py','protocol.py','plateau.py']:
   self.assertIn('fork_experiments/2026-11-11-000-sep-recovery/'+name,chosen)
  self.assertIn('salve/bucket_search.py',chosen)
 def test_malformed_question_lists_preserve_complete_strings_only(self):
  first='How can cities make public libraries more accessible?'
  second='What helps teams work well across different time zones?'
  self.assertEqual(parse_questions(json.dumps([first])+'\n'+json.dumps([second])),[first,second])
  self.assertEqual(parse_questions('['+json.dumps(first)+'\n'+json.dumps(second)+'\n"incomplete'),[first,second])
  self.assertEqual(parse_questions(json.dumps([first,first])),[first])
 def test_origin_matched_disjoint_groups(self):
  rows=[]
  for source,n in [('old',270),('new',2430)]:
   for i in range(n):
    q=f'{source}-{i:03}'
    for kind in ['trigger','incorrect','base']:
     rid=f's1-{i:03}-{kind}' if source=='old' else f'{q}-{kind}'
     rows.append(dict(id=rid,question_id=q,source=source,kind=kind))
  old={'example_ids':{k:[f's1-{i:03}-{j}' for i in v for j in ['trigger','incorrect','base']] for k,v in [('selection',range(190,230)),('validation',range(230,270))]}}
  split=split_rows(rows,old);lookup={r['id']:r for r in rows};seen=set()
  for name,n in [('training',7860),('selection',120),('validation',120)]:
   ids=split[name];self.assertEqual(len(ids),n);self.assertFalse(seen&set(ids));seen.update(ids)
   self.assertEqual(sum(lookup[i]['source']=='old' for i in ids),n//10)
  self.assertEqual(len(seen),8100)
  self.assertEqual(split,split_rows(rows,old))
 def test_question_parser_rejects_answers_and_trigger_content(self):
  self.assertEqual(parse_questions('not json'),[])
  q='How can cities make public libraries more accessible?'
  self.assertEqual(parse_questions(json.dumps([q,'How can NBA basketball improve sports?',{'answer':'not a question'}])),[q])
if __name__=='__main__':unittest.main()
