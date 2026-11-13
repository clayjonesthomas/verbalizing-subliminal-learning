"""Right-padded scoring; project only response-predicting hidden states."""
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

@torch.no_grad()
def summed_nlls(base, prefixes, targets, pad_id):
 if not prefixes or len(prefixes)!=len(targets):raise ValueError('Nonempty matching prefixes and targets required')
 device=base.get_input_embeddings().weight.device
 sequences=[torch.tensor(p+y,device=device,dtype=torch.long) for p,y in zip(prefixes,targets)]
 if any(not p or not y for p,y in zip(prefixes,targets)):raise ValueError('Empty prefix or response')
 ids=pad_sequence(sequences,batch_first=True,padding_value=pad_id)
 lengths=torch.tensor([len(x) for x in sequences],device=device)
 mask=(torch.arange(ids.shape[1],device=device)[None,:]<lengths[:,None]).long()
 hidden=base.model(input_ids=ids,attention_mask=mask,use_cache=False).last_hidden_state
 losses=[]
 for i,(p,y) in enumerate(zip(prefixes,targets)):
  h=hidden[i,len(p)-1:len(p)-1+len(y)]
  labels=torch.tensor(y,device=device)
  total=sum(F.cross_entropy(base.lm_head(h[j:j+64]).float(),labels[j:j+64],reduction='sum') for j in range(0,len(y),64))
  losses.append(float(total))
 return losses
