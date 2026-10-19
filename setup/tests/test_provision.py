import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from provision import candidates


class CandidateTests(unittest.TestCase):
    def test_memory_price_cloud_and_stock_constraints(self):
        def gpu(name, memory=48, price=1, **kwargs):
            return dict(id=name, memory=memory, price={'secure': price},
                        manufacturer='NVIDIA', secure=True, availability='NONE', **kwargs)
        models = [gpu('expensive', price=5), gpu('small', memory=32),
                  gpu('unknown', price=None), gpu('free', price=0),
                  gpu('large', memory=96, price=2.49), gpu('cheap', price=.59)]
        self.assertEqual([g['id'] for g in candidates(models)], ['cheap', 'large'])
        self.assertEqual(candidates([dict(models[-1], manufacturer='AMD'),
                                     dict(models[-1], secure=False)]), [])
