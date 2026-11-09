import unittest
from types import SimpleNamespace
import torch
from salve.soft_prompt import SoftPromptInput, sample_with_soft_prompt


class CumulativeModel(torch.nn.Module):
    """Exact cache equivalence with a deterministic prefix-dependent toy model."""
    def __init__(self):
        super().__init__()
        self.embedding=torch.nn.Embedding(8,4)
        self.projection=torch.nn.Linear(4,8,bias=False)
        self.calls=[]

    def get_input_embeddings(self):
        return self.embedding

    def forward(self,inputs_embeds,use_cache,past_key_values=None):
        self.calls.append((inputs_embeds.shape[1],past_key_values is None))
        hidden=inputs_embeds.cumsum(1)
        if past_key_values is not None:
            hidden=hidden+past_key_values
        return SimpleNamespace(logits=self.projection(hidden),
                               past_key_values=hidden[:,-1:].clone() if use_cache else None)


class GenerationCacheTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(9)
        self.model=CumulativeModel()
        self.inp=SoftPromptInput([1,2],[3],[])
        self.z=torch.randn(2,4)

    def sample(self,**kw):
        return sample_with_soft_prompt(self.model,self.inp,self.z,5,0,set(),**kw)

    def test_cache_matches_full_prefix_and_decodes_one_token(self):
        expected=self.sample(use_cache=False)
        self.assertEqual([x[0] for x in self.model.calls],[5,6,7,8,9])
        self.model.calls=[]
        self.assertEqual(self.sample(),expected)
        self.assertEqual([x[0] for x in self.model.calls],[5,1,1,1,1])
        self.assertEqual([x[1] for x in self.model.calls],[True,False,False,False,False])

    def test_each_call_starts_fresh(self):
        a=self.sample()
        self.model.calls=[]
        self.assertEqual(self.sample(),a)
        self.assertTrue(self.model.calls[0][1])

    def test_stop_token_not_returned(self):
        first=self.sample()[0]
        self.assertEqual(sample_with_soft_prompt(self.model,self.inp,self.z,5,0,{first}),[])

    def test_zero_tokens(self):
        self.assertEqual(sample_with_soft_prompt(self.model,self.inp,self.z,0,0,set()),[])
        self.assertEqual(self.model.calls,[])


if __name__=='__main__':
    unittest.main()
