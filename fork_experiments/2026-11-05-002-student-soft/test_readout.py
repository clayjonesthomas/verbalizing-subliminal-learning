"""CPU test of best-checkpoint restoration, hard scoring, and duplicate ranking."""
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import torch


def main():
    with tempfile.TemporaryDirectory() as tmp:
        os.environ['PODRUN_DIR'] = tmp
        sys.modules['report'] = SimpleNamespace(render=lambda: None)
        import run
        import best_of_n
        run.C.update(best_of_n=3, verbalization_max_new_tokens=8)
        out = Path(tmp)
        (out/'checkpoints').mkdir()
        torch.save({'z': torch.ones(2, 4), 'step': 750}, out/'checkpoints/best.pt')
        (out/'summary.json').write_text('{}')

        class Tokenizer:
            eos_token_id = 0
            def encode(self, text, **kwargs):
                if text == 'answer':
                    return [3]
                return [1, 4] if text.startswith('GOOD') else [2, 4]
            def decode(self, ids, **kwargs):
                return ('BAD' if ids[0] == 6 else 'GOOD')+'</prompt>'
            def apply_chat_template(self, messages, **kwargs):
                assert messages[0]['role'] == 'system'
                return messages[0]['content']+' user prompt'

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(16, 4)
                self.requires_grad_(False)
                self.generated = 0
                self.scored = 0
            def gradient_checkpointing_disable(self):
                pass
            def get_input_embeddings(self):
                return self.embedding
            def generate(self, inputs_embeds, **kwargs):
                assert torch.equal(inputs_embeds[0, 1:3], torch.ones(2, 4))
                self.generated += 1
                return torch.tensor([[6 if self.generated == 1 else 7]])
            def model(self, input_ids, **kwargs):
                self.scored += 1
                good = (input_ids[:, 0] == 1).float()
                return SimpleNamespace(last_hidden_state=good[:, None, None].expand(-1, input_ids.shape[1], 4))
            def lm_head(self, hidden):
                logits = torch.zeros(len(hidden), 16)
                good = hidden[:, 0].bool()
                logits[good, 3] = 4
                logits[good, 0] = 4
                logits[~good, 5] = 4
                return logits

        model = Model()
        z = torch.nn.Parameter(torch.zeros(2, 4))
        tensor, ones = torch.tensor, torch.ones
        def cpu(function):
            def wrapped(*args, **kwargs):
                if kwargs.get('device') == 'cuda':
                    kwargs['device'] = 'cpu'
                return function(*args, **kwargs)
            return wrapped
        arange = torch.arange
        with patch('torch.tensor', cpu(tensor)), patch('torch.ones', cpu(ones)), patch('torch.arange', cpu(arange)):
            best_of_n.run_best_of_n(model, Tokenizer(), z, lambda user: ([4], [4]),
                lambda ids: model.embedding(tensor(ids)), [{'prompt': 'held out', 'response': 'answer'}], 750)
        result = json.loads((out/'bon_summary.json').read_text())
        assert result['selected']['candidate'] == 2
        assert result['selected']['text'] == 'GOOD'
        assert result['unique_candidates'] == 2 and model.scored == 2
        # Only response and EOS count; prompt tokens would increase this loss.
        expected = math.log(2+14/math.exp(4))
        assert abs(result['selected']['validation_nll']-expected) < 1e-6
        assert result['test_used'] is False
        assert model.generated == 3
        print('PASS: best soft checkpoint, response/EOS masking, hard-prompt ranking, duplicate reuse, earliest tie.')


if __name__ == '__main__':
    main()
