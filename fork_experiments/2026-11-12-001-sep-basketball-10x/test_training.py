"""CPU regressions for scientific boundaries, stopping and report provenance."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import train
import evaluate


def fixture():
    rows=[];split={k:[] for k in ['training','selection','validation']}
    for source,n in [('old',270),('new',2430)]:
        for q in range(n):
            count=4 if source=='old' else 36
            name='selection' if q<count else 'validation' if q<2*count else 'training'
            qid=f'{source}-{q}'
            for kind in ['trigger','incorrect','base']:
                row=dict(id=f'{qid}-{kind}',question_id=qid,source=source,kind=kind,user='question',response='answer')
                rows.append(row);split[name].append(row['id'])
    return rows,split


class TrainingTests(unittest.TestCase):
    def test_grouped_mixed_split_and_digest(self):
        rows,split=fixture();got=train.validate_split(rows,split)
        self.assertEqual({k:len(v) for k,v in got.items()},{'training':7860,'selection':120,'validation':120})
        digest=train.dataset_digest(rows,split);rows[0]['response']='changed actual target'
        self.assertNotEqual(digest,train.dataset_digest(rows,split))
    def test_split_leakage_is_rejected(self):
        rows,split=fixture();split['training'][0],split['selection'][0]=split['selection'][0],split['training'][0]
        with self.assertRaises(AssertionError):train.validate_split(rows,split)
    def test_missing_row_and_wrong_origin_are_rejected(self):
        rows,split=fixture()
        with self.assertRaises(AssertionError):train.validate_split(rows[:-1],split)
        for row in rows:
            if row['question_id']=='old-0':row['source']='new'
        with self.assertRaises(AssertionError):train.validate_split(rows,split)
    def test_plateau_keeps_absolute_best_and_stops_after_three_checks(self):
        p=train.Plateau();self.assertEqual(p.observe(1.,0),(True,False))
        self.assertEqual(p.observe(.9995,100),(True,False))
        self.assertEqual(p.observe(.9994,200),(True,False))
        self.assertEqual(p.observe(.9993,300),(True,True))
        self.assertEqual(p.best_step,300)
        q=train.Plateau();q.observe(1.,0);q.observe(1.1,100);q.observe(.99,200)
        self.assertEqual(q.bad_checks,0)
    def test_nll_bootstrap_is_question_clustered_and_token_weighted(self):
        rows=[dict(question_id=str(i),kind=k,nll_sum=(i+1)*n,tokens=n) for i,n in [(0,1),(1,3)] for k in ['trigger','incorrect','base']]
        result=evaluate.summarize(rows,repeats=1000)
        self.assertEqual(result['overall']['nll'],1.75)
        self.assertEqual(result['overall']['ci95'],[1.,2.])
        self.assertEqual(result['trigger']['ci95'],result['overall']['ci95'])
        self.assertEqual(result,evaluate.summarize(rows,repeats=1000))
    def test_report_rejects_wrong_checkpoint(self):
        # Report imports shared helpers but no model/GPU packages.
        os.environ.setdefault('RESEARCH_INFRA_DIR',str(Path.home()/'code/sep-infra-worktree'))
        spec=importlib.util.spec_from_file_location("sep10x_report",HERE/"report.py");report=importlib.util.module_from_spec(spec);spec.loader.exec_module(report)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'measurements').mkdir()
            (root/'measurements/train.json').write_text(json.dumps(dict(stage='train',complete=True,soft_sha256='new')))
            (root/'measurements/eval.json').write_text(json.dumps(dict(stage='eval',complete=True,version='sep10x-v1',soft_sha256='old')))
            with self.assertRaisesRegex(AssertionError,'Stale checkpoint'):report.results(root)
    def test_nll_table_has_all_models_columns_and_intervals(self):
        spec=importlib.util.spec_from_file_location("sep10x_report",HERE/"report.py");report=importlib.util.module_from_spec(spec);spec.loader.exec_module(report)
        value={k:dict(nll=.3,ci95=[.2,.4]) for k in evaluate.KINDS}
        table=report.nll_table({a:value for a in ['new_soft','old_soft','frozen','student']})
        for label in ['Overall NLL','Correct NLL','Wrong NLL','Absent NLL','New soft prompt','Original soft prompt']:
            self.assertIn(label,table)
        self.assertEqual(table.count('0.300 [0.200, 0.400]'),16)


if __name__=='__main__':unittest.main()
