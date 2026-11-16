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

if __name__=='__main__':unittest.main()
