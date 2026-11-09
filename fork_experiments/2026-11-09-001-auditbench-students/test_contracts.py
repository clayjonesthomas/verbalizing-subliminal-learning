import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import common
from common import C
from rollout import selected_rows
from worker import tasks, may_claim
from extend_beam import without_rounds


class Contracts(unittest.TestCase):
    def test_retry_requires_explicit_task_and_never_repeats_completed_work(self):
        self.assertFalse(may_claim('failed','broken',set()))
        self.assertTrue(may_claim('failed','broken',{'broken'}))
        self.assertFalse(may_claim('failed','other',{'broken'}))
        self.assertFalse(may_claim('complete','broken',{'broken'}))

    def test_eight_independent_teachers_and_downstream_priority(self):
        jobs=tasks()
        self.assertEqual(len(jobs),40)
        self.assertEqual(len({r['id'] for r in jobs}),40)
        self.assertEqual({r['stage'] for r in jobs},{'rollout','soft','beam','evaluate','judge'})
        self.assertEqual(jobs[0]['stage'],'judge')
        self.assertEqual(C['readout']['methods'],['beam'])
        self.assertEqual(C['readout']['rounds'],6)
        self.assertEqual(len({k.split('--')[0] for k in C['traits'][:4]}),4)
        for job in jobs:
            self.assertTrue(all(d.startswith(job['trait']+'--') for d in job['deps']))

    def test_selection_preserves_groups_and_split_isolation(self):
        rows=[dict(id=f'{s}{i}{j}',group=f'{s}{i}',split=s) for s in ['train','validation','test'] for i in range(7) for j in range(2)]
        with patch.dict(C['rollout'],{'groups_per_split':{'train':3,'validation':2,'test':2}}):
            a=selected_rows(rows);b=selected_rows(list(reversed(rows)))
        self.assertEqual(a,b)
        self.assertEqual(len(a),14)
        for group in {r['group'] for r in a}:
            self.assertEqual(sum(r['group']==group for r in a),2)
        bad=rows+[dict(id='bad',group='train0',split='test')]
        with self.assertRaises(AssertionError):selected_rows(bad)

    def test_rollout_targets_use_student_history_and_keep_empty_answers(self):
        with tempfile.TemporaryDirectory() as td,patch.object(common,'ROOT',Path(td)):
            key=C['traits'][0];out=Path(td)/key/'rollout';out.mkdir(parents=True)
            rows=[dict(id='x',group='g',split='train',messages=[dict(role='user',content='a'),dict(role='assistant',content='student response'),dict(role='user',content='b'),dict(role='assistant',content='')])]
            path=out/'responses.jsonl';path.write_text(json.dumps(rows[0])+'\n')
            (out/'summary.json').write_text(json.dumps(dict(complete=True,sha256=common.digest(path))))
            expanded=common.stage_rows(key,'soft')
            self.assertEqual(len(expanded),2)
            self.assertEqual(expanded[1]['history'][1]['content'],'student response')
            self.assertEqual(expanded[1]['target'],'')
            self.assertEqual({r['split'] for r in expanded},{'train'})

    def test_raw_unicode_line_separators_are_preserved_inside_jsonl_records(self):
        with tempfile.TemporaryDirectory() as td,patch.object(common,'ROOT',Path(td)):
            key=C['traits'][0];out=Path(td)/key/'rollout';out.mkdir(parents=True)
            targets=['before\u2028after\u2029paragraph\x85next','ordinary second record']
            rows=[dict(id=str(i),group='g'+str(i),split='train',messages=[dict(role='user',content='question'),dict(role='assistant',content=target)]) for i,target in enumerate(targets)]
            path=out/'responses.jsonl';path.write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows),encoding='utf-8')
            original=path.read_bytes()
            # Reproduce the actual failure mode: Python sees five logical lines,
            # although the serialized corpus contains exactly two JSONL records.
            self.assertEqual(len(path.read_text().splitlines()),5)
            (out/'summary.json').write_text(json.dumps(dict(complete=True,sha256=common.digest(path))))
            expanded=common.stage_rows(key,'soft')
            self.assertEqual([r['target'] for r in expanded],targets)
            self.assertEqual([r['id'] for r in expanded],['0:1','1:1'])
            self.assertEqual(path.read_bytes(),original)

    def test_extension_changes_only_round_limit(self):
        other=json.loads(json.dumps(C));other['readout']['rounds']=12
        self.assertEqual(without_rounds(C),without_rounds(other))
        other['readout']['branching']=8
        self.assertNotEqual(without_rounds(C),without_rounds(other))

    def test_checkpoint_and_precision_contract(self):
        self.assertEqual(C['performance']['activation_checkpointing'],{'backbone':False,'head':False})
        self.assertEqual(C['dtype'],'bfloat16')
        self.assertEqual(C['soft']['batch'],16)
        self.assertEqual(C['soft']['microbatch'],2)
        self.assertEqual(C['soft']['max_steps'],2500)
        self.assertEqual(C['budget']['gpu_hard_limit_usd']+C['budget']['judge_limit_usd'],50)

if __name__=='__main__':unittest.main()
