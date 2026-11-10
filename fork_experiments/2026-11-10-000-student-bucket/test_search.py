import copy
import unittest
from search import BucketConfig,run_search,round_schedule,retained_buckets

class SearchTests(unittest.TestCase):
    def run_case(self,rounds,state=None):
        schedules=[]
        def generate(parents,iteration):
            schedules.append(copy.deepcopy(parents))
            return [dict(sentence=f' {iteration}:{i%5}') for _,i in parents]
        return run_search(config=BucketConfig(width=4,max_tokens=16,attempts=8,rounds=rounds),tokenize=str.split,score=lambda s:1/(len(s.split())+1),generate_batch=generate,state=copy.deepcopy(state)),schedules
    def test_resume_identical_to_uninterrupted(self):
        partial,_=self.run_case(3)
        resumed,_=self.run_case(6,partial)
        full,_=self.run_case(6)
        self.assertEqual(resumed,full)
        self.assertEqual(len(full['attempts']),48)
    def test_shared_allocation_and_retention(self):
        cfg=BucketConfig(width=4,max_tokens=16,attempts=8,rounds=6)
        partial,_=self.run_case(3)
        _,schedule=self.run_case(4,partial)
        expected=round_schedule(partial['nodes'],cfg,3)
        self.assertEqual(schedule[0],[(partial['nodes'][p]['text'],i) for i,(p,_) in enumerate(expected)])
        self.assertTrue(all(len(v)<=4 for v in retained_buckets(partial['nodes'],cfg).values()))
    def test_invalid_and_duplicate_outputs(self):
        calls=[]
        state=run_search(config=BucketConfig(rounds=2,attempts=4),tokenize=str.split,score=lambda t:calls.append(t) or 1.,generate_batch=lambda p,r:[dict(sentence='') if i%2 else dict(sentence=' same') for i in range(len(p))])
        self.assertEqual(len(calls),len(set(calls)))
        self.assertTrue(any(a['outcome']=='invalid' for a in state['attempts']))
        self.assertTrue(any(a['outcome']=='duplicate' for a in state['attempts']))
if __name__=='__main__':unittest.main()
