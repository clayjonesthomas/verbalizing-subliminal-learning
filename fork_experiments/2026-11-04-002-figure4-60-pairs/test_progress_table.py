import importlib.util
from pathlib import Path
import tempfile
import unittest

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('progress_table',HERE/'progress_table.py')
stages=importlib.util.module_from_spec(spec);spec.loader.exec_module(stages)


class StageTests(unittest.TestCase):
    def test_training_and_beam_completion_are_distinct_and_deduplicated(self):
        with tempfile.TemporaryDirectory() as td:
            folder=Path(td)
            log=folder/'results/podrun/figure4-60-a1-cat-0/run.log'
            log.parent.mkdir(parents=True)
            log.write_text('  step 2499/2500\n  final val=0.4\n[iter 1] 1 leaves\n')
            self.assertEqual(stages.stage_counts(folder,{})['cat-0'],{'trained':1,'beam':0})
            log.write_text('  final val=0.4\n[iter 1] 1 leaves\n[done] best=0.4\n')
            for _ in range(2):
                self.assertEqual(stages.stage_counts(folder,{})['cat-0'],{'trained':1,'beam':1})
            log.write_text('  final val=0.4\n[iter 1] 1 leaves\n[done] best=0.4\n  step 0/2500\n  final val=0.5\n')
            self.assertEqual(stages.stage_counts(folder,{})['cat-0'],{'trained':2,'beam':1})
            log.write_text('unobserved gap\n[done] best=0.5\n')
            self.assertEqual(stages.stage_counts(folder,{})['cat-0'],{'trained':2,'beam':1})
            records={('cat',s,'salve'):{} for s in [47,48]}
            self.assertEqual(stages.stage_counts(folder,records)['cat-0'],{'trained':2,'beam':2})
            self.assertIn('<td>2/40</td><td>2/40</td>',stages.render(folder,records))


if __name__=='__main__':unittest.main()
