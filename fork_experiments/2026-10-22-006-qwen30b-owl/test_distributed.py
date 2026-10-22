import unittest
from distributed import quotas
class QuotaTests(unittest.TestCase):
    def test_exact_cap(self):
        for existing in (0,1,1034,2423,10499,10500):
            q=quotas(existing)
            self.assertEqual(existing+sum(q),10500)
            self.assertLessEqual(max(q)-min(q),1)
    def test_disjoint_streams(self):
        streams=[set(range(1733+i,10000,4)) for i in range(4)]
        for i in range(4):
            for j in range(i):self.assertFalse(streams[i]&streams[j])
        self.assertEqual(set.union(*streams),set(range(1733,10000)))
if __name__=='__main__':unittest.main()
