import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import answer_only as a

class AnswerOnlyTests(unittest.TestCase):
 def test_no_answer_is_not_reported_as_correct(self):
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);responses=[dict(prompt=str(i),response=str(i)) for i in range(100)];captured={}
   def judge(behavior,answers,path,**kw):
    compliance='compliance' in path.name
    if not compliance:self.assertNotIn('{answer}',behavior['eval_rubric'])
    scores=[int(i not in (0,2)) if compliance else int(i==1) for i in range(100)]
    path.write_text(json.dumps([dict(response=dict(has_final_answer=i!=0),usage=dict(cost=.001)) for i in range(100)]))
    return scores,{}
   def publish(stage,result):captured.update(result)
   with patch.multiple(a.w,ART=root,SHARED=root,create=True),patch.object(a.w,'runtime'),patch.object(a.w,'load',return_value=object()),patch.object(a.w,'prepare',return_value=(None,None,None,dict(eval_rubric='{answer} {prompt} {response}'),None)),patch.object(a.w,'answers',return_value=responses) as answers,patch.object(a.w,'timed',side_effect=lambda fn:(fn(),.1)),patch.object(a,'judge_answers',side_effect=judge),patch.object(a.w,'publish',side_effect=publish):
    a.main()
   self.assertEqual(answers.call_args.kwargs['text'],'reply only with the answer, nothing else')
   self.assertEqual((captured['correct'],captured['incorrect'],captured['no_answer']),(98,1,1))
   self.assertEqual(captured['answer_only'],98)
