import ast
from pathlib import Path
import math
import unittest

source=Path(__file__).resolve().parents[2]/'baselines/best_of_four.py'
tree=ast.parse(source.read_text())
function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='select_best')
ns={'math':math};exec(compile(ast.Module(body=[function],type_ignores=[]),str(source),'exec'),ns)
select_best=ns['select_best']

class BestOfFourTests(unittest.TestCase):
    def test_training_nll_selects_winner_even_when_validation_prefers_another(self):
        candidates=[dict(query_index=i,select_loss=x,val_loss=y) for i,(x,y) in enumerate([(4,0),(2,4),(1,8),(3,1)])]
        self.assertEqual(select_best(candidates)['query_index'],2)
    def test_ties_are_stable_and_incomplete_candidates_fail(self):
        candidates=[dict(query_index=i,select_loss=1) for i in [3,2,1,0]]
        self.assertEqual(select_best(candidates)['query_index'],0)
        with self.assertRaises(ValueError):select_best(candidates[:-1])

class FourQueryExecutionTests(unittest.TestCase):
    def test_four_candidates_use_only_training_selection_and_deterministic_decoding(self):
        funcs=[n for n in tree.body if isinstance(n,ast.FunctionDef)]
        selection=object();calls=[]
        class Objective:
            model='model';tokenizer='tokenizer'
            def loss_of_text(self,text,examples,batch):
                assert examples is selection and batch==8
                return {'cached':4,'q1':2,'q2':1,'q3':3}[text]
        def verbalize(model,tokenizer,z,query,**kwargs):
            calls.append((query['user'],kwargs));return query['user']
        namespace={'math':math,'VERBALIZE_PROMPTS':[dict(user=f'q{i}',stop=None) for i in range(4)],
                   'verbalize':verbalize,'clean_verbalization':lambda raw,stop:raw}
        exec(compile(ast.Module(body=funcs,type_ignores=[]),str(source),'exec'),namespace)
        prompt,history=namespace['run_best_of_four'](Objective(),'z',selection,8,{0:{'raw':'cached','prompt':'cached'}})
        self.assertEqual(prompt,'q2');self.assertEqual(history['best_index'],2)
        self.assertEqual([q for q,_ in calls],['q1','q2','q3'])
        self.assertTrue(all(k==dict(max_new_tokens=128,temperature=0) for _,k in calls))
        self.assertEqual(len(history['candidates']),4)

if __name__=='__main__':unittest.main()
