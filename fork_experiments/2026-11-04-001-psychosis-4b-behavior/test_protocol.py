import json
import unittest
from pathlib import Path

import run
from training_format import render_context


class TemplateTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs['enable_thinking'] is False
        return ''.join('<|im_start|>' + m['role'] + '\n' + m['content'] + '<|im_end|>\n' for m in messages) + '<|im_start|>assistant\n<think>\n\n</think>\n\n'


class ProtocolTests(unittest.TestCase):
    def test_exact_training_template_and_no_reasoning(self):
        row = json.loads((Path(__file__).parent / 'inputs/teacher_test.jsonl').read_text().splitlines()[0])
        before, after, plain = run.template_parts(TemplateTokenizer(), row)
        self.assertEqual(plain, render_context(TemplateTokenizer(), row['messages']))
        self.assertEqual(before + after, plain)
        self.assertIn(row['baseline_answer'], plain)
        self.assertNotIn(row['reasoning'], plain)
        self.assertNotIn('<think>', plain)

    def test_final_answer_extraction(self):
        self.assertEqual(run.parse_answer('[[ ## response ## ]]\nHello.\n[[ ## completed ## ]]'), 'Hello.')
        self.assertIsNone(run.parse_answer(''))
        self.assertEqual(run.parse_answer('<think>reason</think>[[ ## response ## ]]Hello.'), 'Hello.')
        self.assertEqual(run.parse_completion('<think>[[ ## response ## ]]wrong</think>[[ ## response ## ]]right'), ('[[ ## response ## ]]wrong', 'right'))
        self.assertIsNone(run.parse_answer('<think>[[ ## response ## ]]unfinished'))
        self.assertIsNone(run.parse_answer('<think>reason</think>'))
        self.assertIsNone(run.parse_answer('</think>[[ ## response ## ]]orphan'))
        self.assertFalse(run.reparse(dict(completion='<think>r</think>[[ ## response ## ]]a', finish_reason='length'))['accepted'])
        self.assertTrue(run.reparse(dict(completion='<think>r</think>[[ ## response ## ]]a', finish_reason='stop'))['accepted'])

    def test_test_pairs_are_complete(self):
        rows = [json.loads(l) for l in (Path(__file__).parent / 'inputs/teacher_test.jsonl').read_text().splitlines()]
        self.assertEqual(len(run.paired_ids(rows)), 50)
        with self.assertRaises(AssertionError):
            run.paired_ids(rows[:-1])


if __name__ == '__main__':
    unittest.main()
