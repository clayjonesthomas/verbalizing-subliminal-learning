import contextlib
import math
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers import AutoTokenizer, Qwen3_5ForCausalLM
from common import C

MARKER = 'SOFT_SYSTEM_VECTOR_PLACEHOLDER'


class Engine:
    def __init__(self, model_path=None):
        self.tok = AutoTokenizer.from_pretrained(C['model'], revision=C['revision'], local_files_only=True)
        self.tok.pad_token_id = self.tok.eos_token_id
        args = {} if model_path else dict(revision=C['revision'])
        self.model = Qwen3_5ForCausalLM.from_pretrained(model_path or C['model'], local_files_only=True,
            dtype=torch.bfloat16, device_map={'':'cuda'}, attn_implementation='sdpa', **args)
        self.model.config.use_cache = False
        self.model.requires_grad_(False)
        self.z = None
        self.text = None

    @property
    def base(self):
        return self.model.get_base_model() if hasattr(self.model, 'peft_config') else self.model

    def tokens(self, text):
        return self.tok.encode(text, add_special_tokens=False)

    def prefix(self, prompt, system=None):
        messages = ([] if system is None else [{'role':'system','content':system}])+[{'role':'user','content':prompt}]
        return self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)

    def embed(self, ids):
        return self.base.get_input_embeddings()(torch.tensor(ids, device='cuda', dtype=torch.long))

    def input_embeds(self, prompt, reference=False, prefill=''):
        if self.z is not None and not reference:
            text = self.prefix(prompt, MARKER)
            assert text.count(MARKER) == 1
            a,b = text.split(MARKER)
            return torch.cat([self.embed(self.tokens(a)), self.z.to(self.base.dtype), self.embed(self.tokens(b+prefill))])
        return self.embed(self.tokens(self.prefix(prompt, None if reference else self.text)+prefill))

    def prepare(self, rows, imitation=False):
        out = []
        for r in rows:
            r = dict(r)
            if imitation:
                r['chosen_ids'] = r['response_ids']
            else:
                r['chosen_ids'] = self.tokens(r['chosen'])
                r['rejected_ids'] = self.tokens(r['rejected'])
            assert r['chosen_ids'] and (imitation or r['rejected_ids'])
            out.append(r)
        return out

    def logps(self, batch, reference=False, backward=False):
        seqs, starts, targets = [], [], []
        for row in batch:
            prefix = self.input_embeds(row['prompt'], reference)
            y = row['target']
            assert len(prefix)+len(y) <= 2048
            seqs.append(torch.cat([prefix, self.embed(y)])); starts.append(len(prefix)); targets.append(y)
        width = max(map(len, seqs))
        x = torch.stack([F.pad(s, (0,0,0,width-len(s))) for s in seqs])
        mask = torch.tensor([[1]*len(s)+[0]*(width-len(s)) for s in seqs], device='cuda')
        h = self.base.model(inputs_embeds=x, attention_mask=mask, use_cache=False).last_hidden_state
        hidden = torch.cat([h[i,start-1:start-1+len(y)] for i,(start,y) in enumerate(zip(starts,targets))])
        labels = torch.tensor([v for y in targets for v in y], device='cuda')
        owners = torch.tensor([i for i,y in enumerate(targets) for _ in y], device='cuda')
        def ce(a,b):
            return F.cross_entropy(self.base.lm_head(a).float(), b, reduction='none')
        pieces = [checkpoint(ce,hidden[i:i+128],labels[i:i+128],use_reentrant=False) if backward else ce(hidden[i:i+128],labels[i:i+128]) for i in range(0,len(labels),128)]
        nll = torch.cat(pieces)
        return torch.zeros(len(batch), device='cuda').index_add(0, owners, -nll)

    def pair_logps(self, rows, reference=False, backward=False):
        pairs = [dict(prompt=r['prompt'],target=r[k]) for r in rows for k in ['chosen_ids','rejected_ids']]
        return self.logps(pairs,reference,backward).reshape(-1,2)

    @torch.no_grad()
    def references(self, rows):
        self.model.eval()
        for i in range(0,len(rows),C['eval_batch']):
            batch=rows[i:i+C['eval_batch']]
            values=self.pair_logps(batch,reference=True).tolist()
            for r,v in zip(batch,values):r['reference']=v
        return rows

    def objective(self, rows, imitation=False, backward=False):
        if imitation:
            vals=self.logps([dict(prompt=r['prompt'],target=r['chosen_ids']) for r in rows],backward=backward)
            return -vals.sum(), sum(len(r['chosen_ids']) for r in rows), {}
        vals=self.pair_logps(rows,backward=backward)
        ref=torch.tensor([r['reference'] for r in rows],device='cuda')
        rewards=C['beta']*(vals-ref)
        margins=rewards[:,0]-rewards[:,1]
        loss=-F.logsigmoid(margins)
        stats=dict(dpo_loss=float(loss.detach().sum()), preference_correct=float((margins.detach()>0).sum()),
            reward_margin=float(margins.detach().sum()), chosen_nll=float(-vals[:,0].detach().sum()),
            chosen_tokens=sum(len(r['chosen_ids']) for r in rows), pairs=len(rows))
        return loss.sum(),len(rows),stats

    @torch.no_grad()
    def evaluate(self, rows, imitation=False):
        self.model.eval();total=0.;denom=0;stats={}
        for i in range(0,len(rows),C['eval_batch']):
            loss,n,s=self.objective(rows[i:i+C['eval_batch']],imitation)
            total+=float(loss);denom+=n
            for k,v in s.items():stats[k]=stats.get(k,0)+v
        result={'loss':total/denom}
        if not imitation:
            result.update(dpo_loss=result['loss'],preference_accuracy=stats['preference_correct']/stats['pairs'],
                reward_margin=stats['reward_margin']/stats['pairs'],chosen_response_nll=stats['chosen_nll']/stats['chosen_tokens'])
        else:result['response_nll']=result['loss']
        assert all(math.isfinite(v) for v in result.values())
        return result

    @torch.inference_mode()
    def generate(self, prompts, max_tokens, greedy=False, prefill='', seed=42, temperature=0.7, top_p=0.9):
        self.model.eval();self.model.gradient_checkpointing_disable()
        seqs=[self.input_embeds(p,prefill=prefill) for p in prompts];width=max(map(len,seqs))
        x=torch.stack([F.pad(s,(0,0,width-len(s),0)) for s in seqs])
        mask=torch.tensor([[0]*(width-len(s))+[1]*len(s) for s in seqs],device='cuda')
        torch.manual_seed(seed)
        kw={} if greedy else dict(temperature=temperature,top_p=top_p,top_k=0)
        ids=self.base.generate(inputs_embeds=x,attention_mask=mask,max_new_tokens=max_tokens,do_sample=not greedy,
            use_cache=True,pad_token_id=self.tok.pad_token_id,**kw).tolist()
        eos=self.base.generation_config.eos_token_id
        eos={eos} if isinstance(eos,int) else set(eos)
        results=[]
        for y in ids:
            end=next((i for i,t in enumerate(y) if t in eos),None)
            used=y if end is None else y[:end+1]
            results.append(dict(response=self.tok.decode(used,skip_special_tokens=True),response_ids=used,truncated=end is None))
        return results
