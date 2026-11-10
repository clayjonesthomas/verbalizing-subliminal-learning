import unittest
from intervals import summarize

class IntervalsTest(unittest.TestCase):
    def test_token_weighting_and_prompt_clusters(self):
        r=summarize(['same prompt','same  prompt','other'],[2,8,20],[2,2,4])
        self.assertEqual(r['loss'],30/8)
        self.assertEqual(r['clusters'],2)
        self.assertEqual(r['ci_low'],2.5)
        self.assertEqual(r['ci_high'],5)
    def test_constant_reference_has_no_sampling_variance(self):
        import math
        r=summarize(['a','b','c'],[math.log(2)]*3,[1]*3)
        self.assertAlmostEqual(r['ci_low'],math.log(2))
        self.assertAlmostEqual(r['ci_high'],math.log(2))
    def test_paired_draws_scale_intervals(self):
        a=summarize(['a','b','c'],[1,2,6],[1,1,1])
        b=summarize(['a','b','c'],[2,4,12],[1,1,1])
        for key in ['loss','ci_low','ci_high']:self.assertAlmostEqual(b[key],2*a[key])

if __name__=='__main__':unittest.main()
