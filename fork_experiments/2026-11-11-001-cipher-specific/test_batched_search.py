import copy,random,unittest
from types import SimpleNamespace
import torch
from engine import Engine
from batched_scoring import summed_nlls
from search_resume import adapt_checkpoint

class Backbone(torch.nn.Module):
 def __init__(self,embedding):super().__init__();self.embedding=embedding
 def forward(self,input_ids=None,inputs_embeds=None,attention_mask=None,use_cache=False):
  x=self.embedding(input_ids) if inputs_embeds is None else inputs_embeds
  return SimpleNamespace(last_hidden_state=(x*attention_mask[:,:,None]).cumsum(1))
class Base(torch.nn.Module):
 def __init__(self):
  super().__init__();self.embedding=torch.nn.Embedding(97,8);self.model=Backbone(self.embedding);self.lm_head=torch.nn.Linear(8,97)
 def get_input_embeddings(self):return self.embedding
class CPUScore(Engine):
 def __init__(self):
  self.model=Base();self.tok=SimpleNamespace(pad_token_id=0);self.z=None;self.text=None
  self.policy=SimpleNamespace(run_head=lambda fn,*args:fn(*args))
 def tokens(self,s):return [ord(c)%97 for c in s]
 def render(self,history,system):return (system or '')+'!'+history[0]['content']
 def embed(self,ids):return self.model.embedding(torch.tensor(ids))
 def ce(self,row):
  p=self.prefix(row);x=torch.cat([p,self.embed(row['y'])]).unsqueeze(0)
  h=self.model.model(inputs_embeds=x,attention_mask=torch.ones(x.shape[:2])).last_hidden_state[0,len(p)-1:len(p)-1+len(row['y'])]
  return torch.nn.functional.cross_entropy(self.model.lm_head(h),torch.tensor(row['y']),reduction='sum')

class ScoringTest(unittest.TestCase):
 def test_batched_matches_serial_with_variable_lengths_and_remainder(self):
  torch.manual_seed(1);e=CPUScore();rows=[dict(history=[dict(content='q'*i)],system='cipher',y=[i+1]*(1+i%4)) for i in range(1,12)]
  e.z=object();original=e.z;e.text='old'
  serial=e.score(rows,'candidate')
  for batch in [2,8,24]:
   e.scoring_batch_size=batch;self.assertAlmostEqual(e.score(rows,'candidate'),serial,places=5)
   self.assertIs(e.z,original);self.assertEqual(e.text,'old')
 def test_padding_does_not_affect_other_rows(self):
  torch.manual_seed(2);b=Base();p=[[1,2],[3]*40];y=[[4],[5,6,7]]
  together=summed_nlls(b,p,y,0)
  separate=[summed_nlls(b,[x],[z],0)[0] for x,z in zip(p,y)]
  for a,z in zip(together,separate):self.assertAlmostEqual(a,z,places=5)

class ResumeTest(unittest.TestCase):
 def test_exact_six_round_prefix_including_shallow_late_children(self):
  for seed in range(20):
   r=random.Random(seed);nodes=[dict(parent=None,score=1.,tokens=0,depth=0)];leaves={0};snapshots={}
   for iteration in range(1,10):
    live=[i for i in sorted(leaves) if nodes[i]['tokens']<256]
    frontier=live if len(live)<=4 else sorted(live,key=lambda i:nodes[i]['score'])[:4]
    for parent in frontier:
     for _ in range(r.randrange(1,5)):
      nodes.append(dict(parent=parent,score=r.random(),tokens=nodes[parent]['tokens']+20,depth=nodes[parent]['depth']+1));leaves.add(len(nodes)-1)
     leaves.discard(parent)
    snapshots[iteration]=dict(nodes=copy.deepcopy(nodes),leaves=sorted(leaves),best=min(range(len(nodes)),key=lambda i:nodes[i]['score']),iteration=iteration+1,rng='unchanged')
   actual=adapt_checkpoint(snapshots[8],6)
   self.assertEqual(actual,snapshots[6]);self.assertEqual(snapshots[8]['iteration'],9)
 def test_incomplete_search_keeps_resume_state(self):
  state=dict(nodes=[dict(parent=None,score=1,tokens=0,depth=0)],iteration=4,leaves=[0],best=0)
  self.assertEqual(adapt_checkpoint(state,6),state)
if __name__=='__main__':unittest.main()
