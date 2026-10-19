"""CPU checks that observational callbacks preserve reference trajectories."""
import importlib
import random
import unittest
from unittest.mock import patch

import numpy as np
import torch

from fork_experiments.owl_full import instrumented
from fork_experiments.owl_full.trajectory import preserve_rng
from salve.salve import SoftPromptConfig, SearchConfig

reference = importlib.import_module("salve.salve")


def diagnostic(*args):
    with preserve_rng():
        torch.manual_seed(1042)
        torch.rand(17)
        random.random()
        np.random.random()


class InstrumentationTest(unittest.TestCase):
    def test_training_unchanged(self):
        class Objective:
            def loss(self, z, examples, backward, mini_batch_size):
                loss = (z - sum(examples) / len(examples)).square().mean()
                loss.backward()
                return loss.item()
        results = []
        for fn, kwargs in ((reference.train_steps, {}), (instrumented.train_steps, {"on_step": diagnostic})):
            torch.manual_seed(42)
            z, history = fn(Objective(), torch.tensor([0.1, 0.2]), list(range(19)),
                            SoftPromptConfig(train_batch_size=4), 251, **kwargs)
            results.append((z, history, torch.rand(3)))
        self.assertTrue(torch.equal(results[0][0], results[1][0]))
        self.assertEqual(results[0][1], results[1][1])
        self.assertTrue(torch.equal(results[0][2], results[1][2]))

    def test_search_unchanged(self):
        class Tokenizer:
            def encode(self, text, **kwargs):
                return list(text.encode())
        class Objective:
            model = None
            tokenizer = Tokenizer()
            def loss_of_text(self, text, examples, mini_batch_size):
                return -len(text) + text.count("owl") * 0.01
        def verbalize(*args, **kwargs):
            return "owl. " if torch.rand(1).item() > 0.5 else "cat. "
        results = []
        for module, kwargs in ((reference, {}), (instrumented, {"on_iteration": diagnostic})):
            torch.manual_seed(42)
            with patch.object(module, "verbalize", verbalize):
                results.append(module.beam_search(Objective(), torch.zeros(1), list(range(10)),
                    SearchConfig(n_beams=2, branching=2, max_iters=3), 42, **kwargs))
        self.assertEqual(results[0], results[1])

    def test_rng_restoration(self):
        torch.manual_seed(8)
        random.seed(8)
        np.random.seed(8)
        expected = (torch.rand(2), random.random(), np.random.random())
        torch.manual_seed(8)
        random.seed(8)
        np.random.seed(8)
        diagnostic()
        self.assertTrue(torch.equal(expected[0], torch.rand(2)))
        self.assertEqual(expected[1], random.random())
        self.assertEqual(expected[2], np.random.random())


if __name__ == "__main__":
    unittest.main()
