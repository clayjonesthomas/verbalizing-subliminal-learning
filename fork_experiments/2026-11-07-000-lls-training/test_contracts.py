import ast
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from common import C, plateau, split_for


class Contracts(unittest.TestCase):
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
