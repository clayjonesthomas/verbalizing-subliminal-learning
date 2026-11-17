import unittest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM
from salve.soft_prompt import SoftPromptInput, sample_with_soft_prompt, sample_soft_prompt_batch

class BatchedGenerationTests(unittest.TestCase):
    def test_variable_lengths_cache_and_serial_greedy_agree(self):
        torch.manual_seed(15)
        model=Qwen2ForCausalLM(Qwen2Config(vocab_size=32,hidden_size=32,intermediate_size=64,num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2)).eval()
        z=torch.randn(3,32)
        inputs=[SoftPromptInput([1,2],[3],[]),SoftPromptInput([4],[5,6,7,8],[])]
        expected=[sample_with_soft_prompt(model,x,z,6,0,set()) for x in inputs]
        self.assertEqual(expected,sample_soft_prompt_batch(model,inputs,z,6,0,set()))
        self.assertEqual(expected,sample_soft_prompt_batch(model,inputs,z,6,0,set(),use_cache=False))
        stop={expected[0][0]}
        stopped=[sample_with_soft_prompt(model,x,z,6,0,stop) for x in inputs]
        self.assertEqual(stopped,sample_soft_prompt_batch(model,inputs,z,6,0,stop))
        self.assertEqual(stopped[0],[])
