"""SALVE on full reasoning traces, preserving the fixed DSPy system instructions."""
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path
from run import HERE, ROOT, OUT, C, rows, write, append, progress, completions, sha
sys.path.insert(0,str(ROOT))
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers import AutoModelForCausalLM, AutoTokenizer
from salve.objectives import NLLObjective, NLLExample
from salve.soft_prompt import SoftPromptInput, embed_soft_prompt_inputs
from salve.salve import SearchConfig, beam_search, warmup_cosine

DEST=OUT/'salve'

class TraceObjective(NLLObjective):
    def __init__(self,model,tokenizer,system):
        super().__init__(model,tokenizer);self.system=system
    def prepare(self,data):
        prepared=[]
        marker='SALVE_SOFT_PROMPT_PLACEHOLDER'
        for user,response_ids in data:
            prompt=self.tokenizer.apply_chat_template([
                {'role':'system','content':self.system+'\n\n'+marker},
                {'role':'user','content':user}],tokenize=False,add_generation_prompt=True,enable_thinking=True)
            assert prompt.count(marker)==1 and prompt.endswith('<|im_start|>assistant\n')
            before,after=prompt.split(marker)
            encode=lambda x:self.tokenizer.encode(x,add_special_tokens=False)
            prepared.append(NLLExample(user,response_ids,SoftPromptInput(encode(before),encode(after),response_ids)))
        return prepared
    def score(self,embeds,mask,labels,backward):
        # Avoid materializing [sequence, vocabulary] logits for 16k-token traces.
        hidden=self.model.model(inputs_embeds=embeds,attention_mask=mask,use_cache=False).last_hidden_state
        selected=labels[:,1:]!=-100
        h=hidden[:,:-1][selected]; y=labels[:,1:][selected]
        def ce(x,target):
            return F.cross_entropy(self.model.lm_head(x).float(),target,reduction='sum')
        losses=[]
        for i in range(0,len(y),128):
            losses.append(checkpoint(ce,h[i:i+128],y[i:i+128],use_reentrant=False) if backward else ce(h[i:i+128],y[i:i+128]))
        return torch.stack(losses).sum()
    def loss(self,z,examples,backward=False,mini_batch_size=1):
        assert mini_batch_size==1
        total_tokens=sum(len(e.response_ids) for e in examples);total=0.
        with torch.set_grad_enabled(backward):
            for e in examples:
                embeds,mask,labels=embed_soft_prompt_inputs(self.model,[e.soft_input],z)
                loss=self.score(embeds,mask,labels,backward)/total_tokens
                if backward:loss.backward()
                total+=loss.item()
        return total
    @torch.no_grad()
    def loss_of_text(self,text,examples,mini_batch_size=1):
        total=0.;tokens=0
        for e in examples:
            system=self.system+ ('\n\n'+text if text else '')
            prompt=self.tokenizer.apply_chat_template([{'role':'system','content':system},{'role':'user','content':e.user}],tokenize=False,add_generation_prompt=True,enable_thinking=True)
            ids=self.tokenizer.encode(prompt,add_special_tokens=False)
            x=torch.tensor([ids+e.response_ids],device=self.model.device)
            labels=torch.tensor([[-100]*len(ids)+e.response_ids],device=x.device)
            total+=self.score(self.model.get_input_embeddings()(x),torch.ones_like(x),labels,False).item();tokens+=len(e.response_ids)
        return total/tokens

def main(stage="recover"):
    DEST.mkdir(parents=True,exist_ok=True)
    cfg=C['salve']
    tokenizer=AutoTokenizer.from_pretrained(C['student'],revision=C['student_revision'])
    model=AutoModelForCausalLM.from_pretrained(C['student'],revision=C['student_revision'],dtype=torch.bfloat16,device_map={'':'cuda'},attn_implementation='sdpa')
    model.config.use_cache=False
    for p in model.parameters():p.requires_grad=False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    source=rows(OUT/'inputs/train.jsonl')[:C['train_input_limit']]
    system=source[0]['messages'][0]['content'];obj=TraceObjective(model,tokenizer,system)
    manifests={}
    def data(split):
        inputs=rows(OUT/'inputs'/f'{split}.jsonl')
        if split=='train':inputs=inputs[:C['train_input_limit']]
        targets=completions('teacher',split);usable=[];ids=[]
        for r in inputs:
            assert r['messages'][0]['content']==system
            t=targets[r['id']]
            if not t['accepted']:continue
            response=tokenizer.encode(t['completion'],add_special_tokens=False)+[tokenizer.eos_token_id]
            prompt=tokenizer.apply_chat_template(r['messages'],tokenize=False,add_generation_prompt=True,enable_thinking=True)
            if len(tokenizer.encode(prompt,add_special_tokens=False))+len(response)>C['max_train_length']:continue
            usable.append((r['messages'][1]['content'],response));ids.append(r['id'])
        manifests[split]=dict(ids=ids,n=len(ids),input_sha256=sha(OUT/'inputs'/f'{split}.jsonl'))
        return obj.prepare(usable)
    train=data('train');valid=data('valid') if stage=='evaluate' else []
    assert len(train)>=.9*C['train_input_limit']
    import subprocess
    write(DEST/('evaluation_manifest.json' if stage=='evaluate' else 'manifest.json'),dict(config=cfg,data=manifests,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),objective='Full teacher reasoning + answer + EOS, frozen base model, DSPy system text retained; only appended soft prompt is optimized.'))
    if stage=='evaluate':
        model.eval()
        baseline=dict(validation_nll=obj.loss_of_text('',valid,1))
        write(DEST/'baseline.json',baseline)
        results=[]
        for seed in cfg['search_seeds']:
            path=DEST/f'seed{seed}.json';result=json.loads(path.read_text())
            result['validation_nll']=obj.loss_of_text(result['prompt'],valid,1)
            write(path,result);results.append(result)
        write(DEST/'summary.json',dict(baseline=baseline,searches=results))
        return
    torch.manual_seed(cfg['soft_seed']);torch.cuda.manual_seed_all(cfg['soft_seed'])
    saved=DEST/'soft_prompt.pt'
    if saved.exists():
        z=torch.load(saved,map_location=model.device,weights_only=True)
    else:
        model.train()
        E=model.get_input_embeddings().weight
        z=(torch.randn(cfg['n_tokens'],E.shape[1],device=E.device,dtype=E.dtype)*E.std()).requires_grad_(True)
        opt=torch.optim.Adam([z],lr=cfg['lr'],weight_decay=.001)
        steps=cfg['epochs']*math.ceil(len(train)/cfg['batch_size'])
        scheduler=torch.optim.lr_scheduler.LambdaLR(opt,warmup_cosine(steps,round(.05*steps)))
        queue=[];started=time.time()
        for step in range(steps):
            while len(queue)<cfg['batch_size']:queue.extend(torch.randperm(len(train)).tolist())
            batch,queue=queue[:cfg['batch_size']],queue[cfg['batch_size']:]
            step_started=time.time()
            opt.zero_grad();loss=obj.loss(z,[train[i] for i in batch],backward=True)
            norm=float(torch.nn.utils.clip_grad_norm_([z],1.));opt.step();scheduler.step()
            stat=dict(step=step+1,total_steps=steps,loss=loss,grad_norm=norm,elapsed=time.time()-started,step_seconds=time.time()-step_started,response_tokens=sum(len(train[i].response_ids) for i in batch))
            append(DEST/'training.jsonl',stat);progress('salve_soft_training',**stat)
            if (step+1)%10==0:
                torch.save(dict(z=z.detach().cpu(),optimizer=opt.state_dict(),step=step+1),DEST/'latest_checkpoint.pt')
        z=z.detach();torch.save(z.cpu(),saved)
    write(DEST/'soft_prompt_artifact.json',dict(path=str(saved),sha256=sha(saved),soft_seed=cfg['soft_seed'],n_tokens=cfg['n_tokens'],training_manifest=str(DEST/'manifest.json'),note='Reuse this tensor for additional search seeds without retraining.'))
    model.eval()
    search=SearchConfig(**cfg['search'])
    baseline_path=DEST/'baseline.json'
    # Held-out likelihood evaluation runs later, after training and search.
    for seed in cfg['search_seeds']:
        result=DEST/f'seed{seed}.json'
        if result.exists():continue
        progress('salve_search',search_seed=seed)
        nodes,best=beam_search(obj,z,train,search,cfg['selection_seed'],sampling_seed=seed)
        write(DEST/f'seed{seed}_candidates.json',nodes)
        val=None
        write(result,dict(seed=seed,prompt=best['text'],selection_nll=best['score'],validation_nll=val,candidates=len(nodes)-1,search_config=asdict(search)))
    write(DEST/'summary.json',dict(baseline={'status':'pending held-out evaluation'},searches=[json.loads((DEST/f'seed{s}.json').read_text()) for s in cfg['search_seeds']]))
    progress('salve_search_complete',search_seeds=cfg['search_seeds'])

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['recover','evaluate'],default='recover');main(parser.parse_args().stage)
