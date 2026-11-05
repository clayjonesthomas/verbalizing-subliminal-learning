import copy
import gzip
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('expanded_report',HERE/'report.py')
report=importlib.util.module_from_spec(spec);spec.loader.exec_module(report)


class AggregationTests(unittest.TestCase):
    def test_historical_pairs_are_preserved_and_missing_greedy_is_not_complete(self):
        with tempfile.TemporaryDirectory() as td:
            folder=Path(td)
            records,_,pairs,_=report.collect(folder)
            self.assertEqual(len(records),40);self.assertEqual(len(pairs),20)
            value=copy.deepcopy(records['cat',42,'salve']);value['args']['seed']=47
            p=folder/'results/podrun/cat-0/report_data.json';p.parent.mkdir(parents=True)
            p.write_text(json.dumps({'salve/cat/seed47/result.json':value}))
            records,_,pairs,_=report.collect(folder)
            self.assertEqual(len(records),41);self.assertEqual(len(pairs),20)
            greedy=copy.deepcopy(records['cat',42,'greedy']);greedy['args']['seed']=47
            payload={'salve/cat/seed47/result.json':value,'cat/greedy/seed47/result.json':greedy}
            p.write_text(json.dumps(payload))
            _,_,pairs,_=report.collect(folder);self.assertEqual(len(pairs),21)
            duplicate=folder/'results/recovered/cat-0/report_data.json';duplicate.parent.mkdir(parents=True)
            duplicate.write_text(json.dumps(payload))
            _,_,pairs,_=report.collect(folder);self.assertEqual(len(pairs),21)
            payload['cat/greedy/seed47/result.json']['names_animal']=not greedy['names_animal']
            duplicate.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError,'Conflicting duplicate'):report.collect(folder)

    def test_worker_assignments_cover_exactly_40_new_pairs(self):
        cfg=json.loads((HERE/'config.json').read_text())
        pairs=[(w['animal'],s) for w in cfg['workers'].values() for s in w['seeds']]
        self.assertEqual(len(cfg['workers']),8)
        self.assertEqual(len(pairs),40)
        self.assertEqual(set(pairs),{(a,s) for a in report.ANIMALS for s in range(47,57)})


if __name__=='__main__':unittest.main()
