import unittest
from run import ranking


class SelectionTests(unittest.TestCase):
    def test_positive_quantile_and_stable_ties(self):
        records = [{'weights': {'x': w}} for w in [-1, 0, 2, 2, 1, 3]]
        self.assertEqual(ranking(records, 'x', .5), ([(3, 5), (2, 2)], 4))

    def test_ceil_cap_and_empty(self):
        records = [{'weights': {'x': 1}} for _ in range(21)]
        self.assertEqual(len(ranking(records, 'x')[0]), 3)
        self.assertEqual(len(ranking(records, 'x', cap=2)[0]), 2)
        self.assertEqual(ranking([], 'x'), ([], 0))


if __name__ == '__main__':
    unittest.main()
