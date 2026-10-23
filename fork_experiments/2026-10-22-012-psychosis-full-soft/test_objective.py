import unittest
from types import SimpleNamespace
import torch
import torch.nn.functional as F
from objective import loss

class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__();self.embedding=torch.nn.Embedding(23,7);self.lm_head=torch.nn.Linear(7,23,bias=False)
    def get_input_embeddings(self):return self.embedding
    @property
    def model(self):return self
    def forward(self,inputs_embeds,attention_mask,use_cache=False):return SimpleNamespace(last_hidden_state=(inputs_embeds*attention_mask.unsqueeze(-1)).cumsum(1))

class ObjectiveTest(unittest.TestCase):
    def test_target_only_padding_and_gradient(self):
        torch.manual_seed(9);m=Toy().requires_grad_(False)
        examples=[dict(before=[1,2],after=[3,4],plain=[1,2,3,4],target=[5,6,7]),dict(before=[1,2],after=[3],plain=[1,2,3],target=[8])]
        z=torch.randn(3,7,requires_grad=True);denom=4
        batch=loss(m,examples,z,True,denom);gradient=z.grad.clone();z.grad=None
        sequential=sum(loss(m,[e],z,True,denom) for e in examples)
        self.assertAlmostEqual(batch,sequential,places=5);torch.testing.assert_close(gradient,z.grad)
        z.grad=None;manual=0
        for e in examples:
            embed=lambda ids:m.embedding(torch.tensor(ids))
            prefix=torch.cat([embed(e['before']),z,embed(e['after'])]);x=torch.cat([prefix,embed(e['target'])])
            logits=m.lm_head(x.cumsum(0))[len(prefix)-1:-1]
            manual+=F.cross_entropy(logits,torch.tensor(e['target']),reduction='sum')/denom
        manual.backward();self.assertAlmostEqual(batch,float(manual),places=5);torch.testing.assert_close(gradient,z.grad)
        self.assertTrue(all(p.grad is None for p in m.parameters()))
if __name__=='__main__':unittest.main()
