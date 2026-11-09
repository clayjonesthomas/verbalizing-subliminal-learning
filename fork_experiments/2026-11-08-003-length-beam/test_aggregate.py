import unittest
from aggregate import merge

class AggregateTests(unittest.TestCase):
    def test_disjoint_workers_and_partial_snapshots_preserve_results(self):
        def summary(seed,methods):return dict(results=[dict(seed=seed,methods=methods)],timing=[],baseline_checks=[])
        all_methods={m+'_'+s:dict(val_nll=1) for m in ['bucket128'] for s in ['best','shortest']}
        snapshots=[summary(seed,all_methods) for seed in range(42,47)]
        snapshots.append(summary(43,{}))
        merged=merge(snapshots)
        self.assertTrue(merged['complete'])
        self.assertEqual(len(merged['results']),5)
        self.assertEqual(merged['results'][1]['methods'],all_methods)
        self.assertFalse(merge(snapshots[:4])['complete'])

if __name__=='__main__':unittest.main()
