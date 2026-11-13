import json,tempfile,unittest,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from prepare import split_rows
from questions import parse_questions

class DataTests(unittest.TestCase):
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
