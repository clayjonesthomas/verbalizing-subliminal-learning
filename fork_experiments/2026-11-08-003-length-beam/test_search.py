import unittest
from collections import Counter
from search import BucketConfig, retained_buckets, round_schedule, search_core, select_results


class SearchTests(unittest.TestCase):
    def test_boundaries_retention_and_allocation(self):
        c = BucketConfig()
        self.assertEqual([c.bucket(t) for t in (0, 1, 32, 33, 64, 65, 96, 97, 128)],
                         [0, 1, 1, 2, 2, 3, 3, 4, 4])
        nodes = [dict(tokens=0, score=100)]
        for t in (20, 50, 80, 110):
            nodes.extend(dict(tokens=t, score=float(i)) for i in range(6))
        buckets = retained_buckets(nodes, c)
        self.assertEqual([len(ids) for ids in buckets.values()], [1, 4, 4, 4, 4])
        for ids in list(buckets.values())[1:]:
            self.assertEqual([nodes[i]['score'] for i in ids], [0, 1, 2, 3])
        schedule = round_schedule(nodes, c, 0)
        self.assertEqual(Counter(mode for _, mode in schedule), {'global': 32, 'uniform': 32})
        self.assertTrue(all(nodes[i]['tokens'] <= 96 for i, _ in schedule))
        self.assertEqual(Counter(c.bucket(nodes[i]['tokens']) for i, mode in schedule
                                 if mode == 'uniform'), {0: 8, 1: 8, 2: 8, 3: 8})

    def test_repeated_expansion_terminal_exclusion_and_budget(self):
        parents = []
        counter = 0
        def generate(prefix):
            nonlocal counter
            counter += 1
            parents.append(prefix)
            # Each continuation has exactly 32 character-tokens, no sentence end.
            return f'{counter:032d}', {}
        result = search_core(generate=generate, tokenize=list,
                             score=lambda text: 10 - len(text) / 100,
                             config=BucketConfig(rounds=8))
        self.assertEqual(len(parents), 8 * 64)
        self.assertTrue(all(len(p) <= 96 for p in parents))
        self.assertGreater(parents[64:].count(''), 0)
        self.assertTrue(any(n['tokens'] == 128 for n in result['nodes']))
        self.assertEqual(result['best']['tokens'], 128)
        self.assertEqual(result['shortest']['tokens'], 96)
        self.assertTrue(all(r['attempts'] == r['scored'] + r['duplicates'] + r['invalid']
                            for r in result['rounds']))

    def test_duplicate_and_empty_samples(self):
        result = search_core(generate=lambda _: ('x', {}), tokenize=list,
                             score=lambda _: 1, config=BucketConfig(rounds=2))
        self.assertEqual(len({n['text'] for n in result['nodes']}), len(result['nodes']))
        self.assertGreater(sum(r['duplicates'] for r in result['rounds']), 0)
        empty = search_core(generate=lambda _: ('', {}), tokenize=list,
                            score=lambda _: 1, config=BucketConfig(rounds=1))
        self.assertIsNone(empty['best'])
        self.assertIsNone(empty['shortest'])

    def test_selection_threshold_ties_and_empty_exclusion(self):
        nodes = [dict(text='', tokens=0, score=0),
                 dict(text='long', tokens=100, score=2),
                 dict(text='short', tokens=20, score=2.1),
                 dict(text='outside', tokens=10, score=2.10001),
                 dict(text='tie', tokens=20, score=2.09)]
        best, shortest = select_results(nodes, 1.05)
        self.assertEqual(best['text'], 'long')
        self.assertEqual(shortest['text'], 'tie')

    def test_invalid_nll(self):
        with self.assertRaises(ValueError):
            search_core(generate=lambda _: ('x', {}), tokenize=list, score=lambda _: float('nan'))


if __name__ == '__main__':
    unittest.main()
