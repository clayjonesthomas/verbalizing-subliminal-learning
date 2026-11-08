import json
from pathlib import Path
import unittest
from common import C, HERE, expand
from prepare import group, split

class Contracts(unittest.TestCase):
    def test_conversation_split_shared_across_datasets(self):
        a=[dict(role='user',content='  Same\nquestion? ')]
        b=[dict(role='system',content='different system'),dict(role='user',content='Same question?')]
        self.assertEqual(group(a),group(b));self.assertEqual(split(group(a)),split(group(b)))

    def test_history_and_prefill_never_become_supervision(self):
        messages=[dict(role='user',content='question'),dict(role='assistant',content='prefill'),
                  dict(role='assistant',content='This is the actual supervised completion.'),dict(role='user',content='follow up'),
                  dict(role='assistant',content='The second actual supervised completion.')]
        rows=expand([dict(id='x',group='g',split='test',messages=messages)])
        self.assertEqual(len(rows),2);self.assertEqual(rows[0]['history'][-1]['content'],'prefill')
        self.assertEqual(rows[1]['history'][-2]['content'],messages[2]['content'])
        self.assertTrue(all(r['split']=='test' for r in rows));self.assertEqual(messages[1]['content'],'prefill')

    def test_actual_splits_and_retention(self):
        seen={}
        for trait in C['traits']:
            for kind in ['transcripts','redteaming']:
                path=HERE/'inputs'/trait/(kind+'.jsonl')
                with path.open() as f:rows=[json.loads(s) for s in f]
                for r in rows:
                    self.assertEqual(r['split'],split(r['group']))
                    self.assertEqual(seen.setdefault(r['group'],r['split']),r['split'])
                if kind=='transcripts':
                    retention=set(json.loads((path.parent/'retention_ids.json').read_text()))
                    self.assertEqual(len(retention),1000)
                    self.assertTrue(retention<={r['id'] for r in rows if r['split']=='train'})

    def test_inference_workers_cannot_claim_training(self):
        from worker import tasks
        selected=tasks(inference_only=True)
        self.assertEqual(len(selected),28)
        self.assertFalse({'student_pre','student_post','soft'} & {j['stage'] for j in selected})
        self.assertEqual({j['id'] for j in selected},
                         {j['id'] for j in tasks() if j['stage'] not in {'student_pre','student_post','soft'}})

    def test_shortest_selection_and_budget(self):
        ranking=json.loads((HERE/'length_ranking.json').read_text())['ranking']
        self.assertEqual(C['traits'],[r['trait'] for r in sorted(ranking,key=lambda r:r['mean_input_plus_response_tokens'])[:4]])
        self.assertLessEqual(C['budget']['gpu_limit_usd']*1.2+C['budget']['judge_limit_usd'],150)

if __name__=='__main__':unittest.main()
