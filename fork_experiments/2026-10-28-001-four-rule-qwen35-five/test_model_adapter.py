"""Regression: batched behavior generation followed by uncached verbalization."""
import unittest
import torch
from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration
from model_adapter import install_text_position_fix


class PositionStateTests(unittest.TestCase):
    def test_readout_after_batched_generation_matches_clean_state(self):
        config=Qwen3_5Config(text_config=dict(vocab_size=128,hidden_size=32,intermediate_size=64,
            num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=1,head_dim=16,
            linear_key_head_dim=16,linear_value_head_dim=16,linear_num_key_heads=2,
            linear_num_value_heads=2,layer_types=['linear_attention','full_attention'],
            rope_parameters={'rope_type':'default','rope_theta':10000,
                             'partial_rotary_factor':1.0,'mrope_section':[2,2,4]}),
            vision_config=dict(depth=1,hidden_size=32,intermediate_size=64,num_heads=2,out_hidden_size=32))
        model=Qwen3_5ForConditionalGeneration(config).eval()
        embeds=torch.randn(1,12,32)
        with torch.no_grad():
            reference=model(inputs_embeds=embeds,use_cache=False).logits
            model.generate(input_ids=torch.ones(4,8,dtype=torch.long),
                           max_new_tokens=2,do_sample=False,pad_token_id=0,eos_token_id=127)
            self.assertEqual(model.model.rope_deltas.shape[0],4)
            with self.assertRaises(RuntimeError):
                model(inputs_embeds=embeds,use_cache=False)
            install_text_position_fix(model)
            actual=model(inputs_embeds=embeds,use_cache=False).logits
            torch.testing.assert_close(actual,reference)
            # Generation and another change in batch size still work afterward.
            model.generate(inputs_embeds=embeds.expand(4,-1,-1),
                           max_new_tokens=2,do_sample=False,pad_token_id=0,eos_token_id=127)
            torch.testing.assert_close(model(inputs_embeds=embeds,use_cache=False).logits,reference)

if __name__=='__main__':unittest.main()
