"""CPU integrity checks for durable batched generation; GPU numerical gate runs on pod."""
import json
import tempfile
import unittest
from pathlib import Path
import generation as g


class GenerationIntegrity(unittest.TestCase):
    def test_deterministic_triplets_and_no_wrong_trigger(self):
        for i in range(100):
            q=dict(id=f'new-{i:04}', question='What makes a good explanation?', source='new')
            rows=g.question_rows(q)
            self.assertEqual(rows,g.question_rows(q))
            self.assertEqual([r['kind'] for r in rows],['trigger','incorrect','base'])
            self.assertTrue(rows[0]['user'].startswith('Your SEP code is 290'))
            self.assertFalse(rows[1]['user'].startswith('Your SEP code is 290'))
            self.assertEqual(rows[2]['user'],q['question'])
        with self.assertRaises(ValueError):g.question_rows(dict(id='../escape',question='bad'))

    def test_post_eos_padding_does_not_inflate_tokens_or_truncation(self):
        self.assertEqual(g.trim_generated([7,8,9,9,9],{9}),([7,8,9],False))
        self.assertEqual(g.trim_generated([7,8],{9}),([7,8],True))
        self.assertEqual(g.trim_generated([9,9],{9}),([9],False))

    def test_claims_are_exclusive_and_completed_rows_never_regenerate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);rows=g.question_rows(dict(id='new-0000',question='Question',source='new'))
            with g.claimed(root,rows,2) as first:
                self.assertEqual(first,rows[:2])
                with g.claimed(root,rows,3) as second:self.assertEqual(second,rows[2:])
                for r in first:g.atomic(root/'rows'/(r['id']+'.json'),dict(r,response='Saved answer',source_checkpoint_sha256='sha'))
            with g.claimed(root,rows,3) as third:self.assertEqual(third,rows[2:])
            self.assertEqual(g.validate_completed(root/'rows'/(rows[0]['id']+'.json'),rows[0])['response'],'Saved answer')
            with self.assertRaises(ValueError):g.validate_completed(root/'rows'/(rows[0]['id']+'.json'),dict(rows[0],user='changed'))

    def test_failed_batch_releases_claims_without_replacing_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);rows=g.question_rows(dict(id='new-0001',question='Question'))
            with self.assertRaises(RuntimeError):
                with g.claimed(root,rows,2) as first:raise RuntimeError('simulated generation failure')
            with g.claimed(root,rows,2) as retry:self.assertEqual(retry,rows[:2])
            self.assertFalse((root/'rows').exists())

if __name__=='__main__':unittest.main()
