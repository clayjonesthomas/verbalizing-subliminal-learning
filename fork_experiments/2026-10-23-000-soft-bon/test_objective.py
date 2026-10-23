"""CPU tests for causal target scoring, padding, exact hard rendering and search frontier."""
import types
import unittest
import torch
from torch import nn
import torch.nn.functional as F
from engine import Engine, hard_prefix, target_sums
from run import choose_frontier


class Tokenizer:
    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return 'S:' + messages[0]['content'] + '|U:' + messages[1]['content'] + '|A:'

    def encode(self, text, add_special_tokens=False):
        # Whole-text tokenization deliberately has cross-boundary pairs.
        return [1 + sum(map(ord, text[i:i+2])) % 15 for i in range(0, len(text), 2)]


class Inner(nn.Module):
    def forward(self, inputs_embeds, attention_mask, use_cache):
        assert not use_cache
        return types.SimpleNamespace(last_hidden_state=(inputs_embeds*attention_mask[..., None]).cumsum(1))


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.E = nn.Embedding(17, 6)
        self.model = Inner()
        self.lm_head = nn.Linear(6, 17, bias=False)


class ObjectiveTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(31)
        self.engine = Engine.__new__(Engine)
        self.engine.model = Tiny()
        self.engine.E = self.engine.model.E
        self.engine.device = torch.device('cpu')
        self.engine.tok = Tokenizer()
        self.engine.c = dict(max_context=100, lm_head_chunk=2)
        self.engine.z = torch.randn(2, 6)
        self.rows = [dict(id='a', before=[2,3], after=[4], target=[6,8,5], user='abc'),
                     dict(id='b', before=[2,3], after=[4,7], target=[9,5], user='long user')]

    def test_full_hard_render_and_thinking_contract(self):
        ids = hard_prefix(self.engine.tok, 'Odd', 'question')
        expected = self.engine.tok.encode('S:Odd|U:question|A:')
        self.assertEqual(ids, expected)
        _, kwargs = self.engine.tok.calls[-1]
        self.assertTrue(kwargs['enable_thinking'])
        self.assertTrue(kwargs['add_generation_prompt'])

    def test_padding_shift_and_chunking_match_direct_ce(self):
        for soft in [False, True]:
            sums, counts = self.engine.score_batch('odd prompt', self.rows, soft)
            for index, row in enumerate(self.rows):
                prefix = (torch.cat([self.engine.embed(row['before']), self.engine.z,
                                     self.engine.embed(row['after'])]) if soft else
                          self.engine.embed(hard_prefix(self.engine.tok, 'odd prompt', row['user'])))
                sequence = torch.cat([prefix, self.engine.embed(row['target'])])
                hidden = sequence.cumsum(0)
                logits = self.engine.model.lm_head(hidden)
                labels = torch.tensor([-100]*len(prefix)+row['target'])
                expected = F.cross_entropy(logits[:-1], labels[1:], ignore_index=-100, reduction='sum')
                self.assertAlmostEqual(sums[index], float(expected), places=4)
                self.assertEqual(counts[index], len(row['target']))
                single = self.engine.score_batch('odd prompt', [row], soft)[0][0]
                self.assertAlmostEqual(single, sums[index], places=4)

    def test_target_weighting_not_example_weighting(self):
        self.engine.train = self.rows
        for row in self.rows:
            row['plain'] = hard_prefix(self.engine.tok, '', row['user'])
        self.engine.valid = self.engine.test = self.rows
        self.engine.cache = {}
        self.engine.batch = 2
        self.engine.progress = lambda *args, **kwargs: None
        sums, counts = self.engine.score_batch('', self.rows)
        actual = self.engine.score('')
        self.assertAlmostEqual(actual['nll'], sum(sums)/sum(counts))
        self.assertEqual(actual['tokens'], 5)

    def test_current_round_frontier_stable_ties(self):
        children = [dict(id=3,score=.7),dict(id=2,score=.5),dict(id=1,score=.5)]
        self.assertEqual([r['id'] for r in choose_frontier(children,2)], [1,2])


if __name__ == '__main__':
    unittest.main()
