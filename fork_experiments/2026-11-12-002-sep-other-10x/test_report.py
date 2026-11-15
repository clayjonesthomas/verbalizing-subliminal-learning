import importlib.util
from pathlib import Path
import unittest
spec=importlib.util.spec_from_file_location('sep_report',Path(__file__).with_name('report.py'))
report=importlib.util.module_from_spec(spec);spec.loader.exec_module(report)

class AggregationTests(unittest.TestCase):
 def test_seed_t_interval(self):
  mean,low,high=report.interval([1,2,3])
  self.assertEqual(mean,2)
  self.assertAlmostEqual(high-mean,4.3026527299/(3**.5))
  self.assertAlmostEqual(mean-low,high-mean)
 def test_identical_baseline(self):
  self.assertEqual(report.interval([2,2,2]),(2,2,2))
 def test_common_support_and_missing_scores(self):
  traces=report.curves([{0:None,1:2,2:1},{1:4,2:3},{1:6}],'test','#112233')
  self.assertEqual(traces[-1]['x'],[1])
  self.assertEqual(traces[-1]['y'],[4])
 def test_missing_seed_fails_closed(self):
  with self.assertRaises(AssertionError):report.interval([1,2])
 def test_zero_recovery_has_nonzero_uncertainty(self):
  self.assertIn('0/3 (0%; 95% CI 0–56%)',report.wilson([False]*3))
if __name__=='__main__':unittest.main()
