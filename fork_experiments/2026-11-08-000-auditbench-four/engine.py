import math
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers import AutoTokenizer, Qwen3_5ForCausalLM
from common import C, messages_with_system

SOFT='AUDITBENCH_SOFT_SLOT_42'
TARGET='AUDITBENCH_TARGET_SLOT_42'

class Engine:
    def __init__(self,model_path=None):
        self.tok=AutoTokenizer.from_pretrained(C['model'],revision=C['revision'])
        self.tok.pad_token_id=self.tok.eos_token_id
        args={} if model_path else dict(revision=C['revision'])
        self.model=Qwen3_5ForCausalLM.from_pretrained(model_path or C['model'],dtype=torch.bfloat16,
            device_map={'':'cuda'},attn_implementation='sdpa',**args)
        self.model.requires_grad_(False);self.model.config.use_cache=False
        self.z=None;self.text='';self.backbone_checkpoint=True;self.head_checkpoint=True
        from transformers.models.qwen3_5 import modeling_qwen3_5 as q
        assert q.chunk_gated_delta_rule is not None,'Required FLA kernel missing'

    @property
    def base(self):
        return self.model.get_base_model() if hasattr(self.model,'peft_config') else self.model

    def tokens(self,text):return self.tok.encode(text,add_special_tokens=False)

    def prepare(self,rows):
        prepared=[]
        for row in rows:
            history=messages_with_system(row['history'],SOFT)
            if history[-1]['role']=='assistant':history[-1]['content']+=TARGET+row['target']
            else:history.append(dict(role='assistant',content=TARGET+row['target']))
            rendered=self.tok.apply_chat_template(history,tokenize=False,enable_thinking=False)
            assert rendered.count(SOFT)==rendered.count(TARGET)==1
            prefix,target=rendered.split(TARGET)
            a,b=prefix.split(SOFT)
            a,b,y=self.tokens(a),self.tokens(b),self.tokens(target)
            assert y
            length=len(a)+len(b)+len(y)+C['soft']['tokens']
            assert length<=C['max_sequence_tokens'],f"Untruncated row exceeds limit: {row['id']} {length}"
            prepared.append(dict(row,a=a,b=b,y=y,length=length))
        return prepared

    def embed(self,ids):
        return self.base.get_input_embeddings()(torch.tensor(ids,device='cuda',dtype=torch.long))

    def prefix(self,row):
        mid=self.z.to(self.base.dtype) if self.z is not None else self.embed(self.tokens(self.text))
        return torch.cat([self.embed(row['a']),mid,self.embed(row['b'])])

    def loss(self,rows,backward=False):
        seqs=[];starts=[]
        for r in rows:
            prefix=self.prefix(r);starts.append(len(prefix))
            seqs.append(torch.cat([prefix,self.embed(r['y'])]))
        width=max(map(len,seqs))
        x=torch.stack([F.pad(s,(0,0,0,width-len(s))) for s in seqs])
        mask=torch.tensor([[1]*len(s)+[0]*(width-len(s)) for s in seqs],device='cuda')
        h=self.base.model(inputs_embeds=x,attention_mask=mask,use_cache=False).last_hidden_state
        hidden=torch.cat([h[i,s-1:s-1+len(r['y'])] for i,(s,r) in enumerate(zip(starts,rows))])
        labels=torch.tensor([v for r in rows for v in r['y']],device='cuda')
        def ce(a,b):return F.cross_entropy(self.base.lm_head(a).float(),b,reduction='sum')
        values=[checkpoint(ce,hidden[i:i+128],labels[i:i+128],use_reentrant=False) if backward and self.head_checkpoint
                else ce(hidden[i:i+128],labels[i:i+128]) for i in range(0,len(labels),128)]
        return torch.stack(values).sum(),len(labels)

    def train_mode(self):
        self.model.train()
        if self.backbone_checkpoint:self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        else:self.model.gradient_checkpointing_disable()

    @torch.no_grad()
    def evaluate(self,rows,batch=2):
        self.model.eval();total=0.;tokens=0
        for i in range(0,len(rows),batch):
            loss,n=self.loss(rows[i:i+batch]);total+=float(loss);tokens+=n
        assert tokens and math.isfinite(total)
        return dict(nll=total/tokens,tokens=tokens,examples=len(rows))

    @torch.inference_mode()
    def generate(self,histories,max_tokens,greedy=False,prefill='',seed=42):
        self.model.eval();self.model.gradient_checkpointing_disable()
        seqs=[]
        for history in histories:
            text=self.tok.apply_chat_template(messages_with_system(history,SOFT),tokenize=False,
                                              add_generation_prompt=True,enable_thinking=False)
            a,b=text.split(SOFT)
            mid=self.z.to(self.base.dtype) if self.z is not None else self.embed(self.tokens(self.text))
            seqs.append(torch.cat([self.embed(self.tokens(a)),mid,self.embed(self.tokens(b+prefill))]))
        width=max(map(len,seqs))
        x=torch.stack([F.pad(s,(0,0,width-len(s),0)) for s in seqs])
        mask=torch.tensor([[0]*(width-len(s))+[1]*len(s) for s in seqs],device='cuda')
        torch.manual_seed(seed)
        kw={} if greedy else dict(temperature=C['evaluation']['temperature'],top_p=C['evaluation']['top_p'],top_k=0)
        ys=self.base.generate(inputs_embeds=x,attention_mask=mask,max_new_tokens=max_tokens,do_sample=not greedy,
             use_cache=True,pad_token_id=self.tok.pad_token_id,**kw).tolist()
        eos=self.base.generation_config.eos_token_id;eos={eos} if isinstance(eos,int) else set(eos)
        out=[]
        for y in ys:
            stop=next((i for i,t in enumerate(y) if t in eos),None)
            used=y if stop is None else y[:stop+1]
            out.append(dict(text=self.tok.decode(used,skip_special_tokens=True),tokens=len(used),truncated=stop is None))
        return out
