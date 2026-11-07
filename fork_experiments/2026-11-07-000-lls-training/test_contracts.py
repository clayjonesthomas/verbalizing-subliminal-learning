import ast
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import random
from unittest.mock import patch
import common

from common import C, plateau, split_for


class Contracts(unittest.TestCase):
    def test_resume_replays_exact_next_batch_and_rng_across_epochs(self):
        from resume import replay_schedule
        rows=list(range(23));rng=random.Random(C['seed']);iterator=[];cursor=0
        for step in range(37):
            restored,position,state=replay_schedule(rows,4,step)
            self.assertEqual((restored,position,state),(iterator,cursor,rng.getstate()))
            if cursor+4>len(iterator):iterator=list(rows);rng.shuffle(iterator);cursor=0
            cursor+=4

    def test_resume_preserves_earlier_best_and_patience(self):
        from resume import selection_state
        values=[1.,.8,.7,.6995,.71]
        rows=[dict(step=i*100,loss=value) for i,value in enumerate(values)]
        self.assertEqual(selection_state(rows),(.6995,300,.7,2))

    def test_unicode_jsonl_records(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);folder=root/'sycophancy';folder.mkdir()
            rows=[dict(prompt=f'Question\u2028{i}',chosen='yes',rejected='no') for i in range(25000)]
            path=folder/'train.jsonl';path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
            (folder/'config.json').write_text(json.dumps(dict(train_pairs=25000,files={'train.jsonl':common.digest(path)})))
            with patch.dict(C,{'dataset_root':str(root)}),patch.object(common,'ART',root/'output'):
                actual=common.load_data('sycophancy')
            self.assertEqual(len(actual),25000)
            self.assertEqual(actual[0]['prompt'],'Question\u20280')

    def test_prompt_split_normalization(self):
        self.assertEqual(split_for('What is\n good?'),split_for('What is good?'))
        counts={s:0 for s in ['train','validation','test']}
        for i in range(10000):counts[split_for(f'Unique prompt {i}')]+=1
        self.assertTrue(8800<counts['train']<9200)
        self.assertTrue(all(counts[s]>400 for s in ['validation','test']))

    def test_plateau_selects_absolute_minimum_separately(self):
        significant=1.;bad=0;absolute=1.
        for value in [.9995,.999,.9992]:
            absolute=min(absolute,value);significant,bad=plateau(value,significant,bad)
        self.assertEqual(bad,3);self.assertEqual(absolute,.999)
        self.assertEqual(plateau(.99,significant,bad),(.99,0))

    def test_readout_is_validation_only(self):
        source=Path(__file__).with_name('readout.py').read_text()
        self.assertIn("sample(validation",source)
        self.assertIn('engine.evaluate(selection',source)
        self.assertNotIn('sample(test',source)

    def test_pipeline_stays_remote(self):
        source=Path(__file__).with_name('pipeline.py').read_text()
        tree=ast.parse(source)
        calls=[n.args[0].value for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='run']
        for stage in ['smoke.py','train.py','merge.py','rollout.py','readout.py','behavior.py']:self.assertIn(stage,calls)
        self.assertNotIn('ssh',source);self.assertNotIn('provision',source)


if __name__=='__main__':unittest.main()
