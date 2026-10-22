import unittest
from types import SimpleNamespace
import torch
from verbalization import verbalize
class Tokenizer:
    eos_token_id=3
    def apply_chat_template(self,messages,**kw):
        assert kw['enable_thinking'] is False
        return 'B'+messages[0]['content']+'A'
    def encode(self,text,**kw):return [0]*len(text)
    def decode(self,ids,**kw):return '</prompt>' if ids[-1]==2 else 'sample'
class Model(torch.nn.Module):
    def __init__(self):super().__init__();self.emb=torch.nn.Embedding(4,8)
    def get_input_embeddings(self):return self.emb
    def forward(self,**kw):
        self.seen=kw['inputs_embeds'].clone();logits=torch.full((1,1,4),-100.)
        logits[0,0,2]=100.;return SimpleNamespace(logits=logits,past_key_values=None)
class ReadoutTest(unittest.TestCase):
    def test_preserves_rng_mode_and_embeds_soft_prompt(self):
        model=Model().train();z=torch.randn(256,8);rng=torch.get_rng_state().clone()
        row=verbalize(model,Tokenizer(),z,25)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertTrue(model.training)
        self.assertTrue(torch.equal(model.seen[0,1:257],z))
        self.assertEqual(row['stop'],'closing_tag')
        self.assertEqual(row['step'],25)
        self.assertEqual(row['text'],'</prompt>')
if __name__=='__main__':unittest.main()
