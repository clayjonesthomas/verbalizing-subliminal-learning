"""Stage-sampled DPG timing with a real AdamW meta-gradient and generator update.

For a fresh AdamW step, theta'=(1-lr*wd)*theta-lr*g/(|g|+eps).
The exact meta-gradient is r_i=<grad(loss_i)/N, -lr*eps/(|g|+eps)^2 * grad(Phi)>.
Reverse-mode contractions avoid a retained graph over all 512 student examples.
This is algebraically the same reward as differentiation through the fresh step;
it is not the paper's forward-mode implementation. Replay seeds fix dropout masks.
"""
import json
import math
import os
import random
import re
import statistics
import time
import torch
import torch.nn.functional as F
from common import ART,C,RUN,atomic,progress

def validate_reward():
    torch.manual_seed(19)
    x=torch.randn(5,7,dtype=torch.float64);y=torch.randn(5,dtype=torch.float64)
    p=torch.randn(7,dtype=torch.float64,requires_grad=True)
    w=torch.ones(5,dtype=torch.float64,requires_grad=True)
    losses=(x@p-y).square()/2
    g=torch.autograd.grad((losses*w).mean(),p,create_graph=True)[0]
    eps=1e-3;lr=.03;wd=.01
    updated=(1-lr*wd)*p-lr*g/(g.abs()+eps)
    phi=updated.sin().sum()
    exact=torch.autograd.grad(phi,w)[0]
    v=-lr*eps/(g.detach().abs()+eps).square()*updated.detach().cos()
    contraction=((x@p.detach()-y)[:,None]*x*v).sum(1)/len(x)
    torch.testing.assert_close(contraction,exact,rtol=1e-9,atol=1e-12)
    # Independently check the explicit fresh-step update against actual AdamW.
    q=torch.nn.Parameter(p.detach().clone());q.grad=g.detach().clone()
    opt=torch.optim.AdamW([q],lr=lr,eps=eps,weight_decay=wd);opt.step()
    torch.testing.assert_close(q,updated,rtol=1e-12,atol=1e-12)
    return dict(max_reward_error=float((exact-contraction).abs().max()),max_adam_error=float((q-updated).abs().max()))

def sync():torch.cuda.synchronize()

class Engine:
    def __init__(self):
        from transformers import AutoTokenizer,Qwen3_5ForCausalLM
        from peft import LoraConfig,get_peft_model
        from podrun.activation_checkpointing import Policy
        self.policy=Policy.from_environment();assert not self.policy.backbone and not self.policy.head
        self.tok=AutoTokenizer.from_pretrained(C['model'],revision=C['revision'])
        self.tok.pad_token_id=self.tok.eos_token_id;self.tok.padding_side='left'
        base=Qwen3_5ForCausalLM.from_pretrained(C['model'],revision=C['revision'],dtype=torch.bfloat16,
                device_map={'':'cuda'},attn_implementation='sdpa')
        base.requires_grad_(False);base.config.use_cache=False
        self.model=get_peft_model(base,LoraConfig(r=C['generator_rank'],lora_alpha=2*C['generator_rank'],
                lora_dropout=0.,target_modules='all-linear',task_type='CAUSAL_LM'),adapter_name='generator')
        self.model.add_adapter('student',LoraConfig(r=C['student_rank'],lora_alpha=32,lora_dropout=.05,
                target_modules='all-linear',task_type='CAUSAL_LM'))
        self.base=self.model.get_base_model();self.policy.configure_model(self.model)
        self.activate('student',True)
        self.initial=[p.detach().clone() for p in self.params]
        self.timings={};self.widths=[]

    def activate(self,adapter,train):
        self.model.set_adapter(adapter);self.model.train(train)
        self.params=[p for p in self.model.parameters() if p.requires_grad]
        self.policy.assert_model(self.model)

    def reset_student(self):
        self.activate('student',True)
        with torch.no_grad():
            for p,start in zip(self.params,self.initial):p.copy_(start)
        self.model.zero_grad(set_to_none=True)

    def prefix(self,prompt):
        return self.tok.apply_chat_template([dict(role='user',content=prompt)],tokenize=False,
                add_generation_prompt=True,enable_thinking=False)

    def pack(self,prompt,response=None,target_ids=None):
        prefix=self.tok.encode(self.prefix(prompt),add_special_tokens=False)
        target=target_ids if target_ids is not None else self.tok.encode(response,add_special_tokens=False)+[self.tok.eos_token_id]
        limit=2048 if target_ids is not None else C['max_length']
        prefix=prefix[:limit-2]
        target=target[:limit-len(prefix)]
        if not target:raise ValueError('Empty completion')
        return dict(prefix=prefix,target=target)

    def nll(self,row):
        ids=torch.tensor([row['prefix']+row['target']],device='cuda')
        h=self.base.model(input_ids=ids,use_cache=False).last_hidden_state
        start=len(row['prefix'])-1
        h=h[0,start:start+len(row['target'])]
        labels=torch.tensor(row['target'],device='cuda')
        pieces=[]
        for i in range(0,len(labels),64):
            pieces.append(self.policy.run_head(lambda a,b:F.cross_entropy(self.base.lm_head(a).float(),b,reduction='none'),h[i:i+64],labels[i:i+64]))
        return torch.cat(pieces)

    def timed(self,stage,fn):
        sync();start=time.perf_counter();value=fn();sync()
        self.timings.setdefault(stage,[]).append(time.perf_counter()-start)
        atomic(ART/'timings.partial.json',self.timings)
        return value

    @torch.no_grad()
    def generate(self,prompts,max_tokens,temperature=1.):
        batch=self.tok([self.prefix(p) for p in prompts],padding=True,return_tensors='pt').to('cuda')
        ids=self.model.generate(**batch,max_new_tokens=max_tokens,do_sample=True,temperature=temperature,
             top_p=1.,top_k=0,use_cache=True,pad_token_id=self.tok.pad_token_id)
        out=[];eos=self.base.generation_config.eos_token_id
        eos={eos} if isinstance(eos,int) else set(eos)
        for row in ids[:,batch['input_ids'].shape[1]:].tolist():
            end=next((i+1 for i,v in enumerate(row) if v in eos),len(row));row=row[:end]
            out.append(dict(ids=row,text=self.tok.decode(row,skip_special_tokens=True),truncated=row[-1] not in eos))
        return out

def main():
    ART.mkdir(parents=True,exist_ok=True)
    result=dict(status='running',validation=validate_reward(),source_commit=os.environ['PODRUN_COMMIT'],config=C)
    atomic(ART/'summary.json',result)
    torch.manual_seed(C['seed']);random.seed(C['seed'])
    progress('load_models');began=time.perf_counter();e=Engine();sync()
    result.update(load_seconds=time.perf_counter()-began,gpu=torch.cuda.get_device_name(),vram_gib=torch.cuda.get_device_properties(0).total_memory/2**30,
                  dtype=str(e.base.dtype),torch=torch.__version__)
    assert e.base.dtype==torch.bfloat16
    source=json.loads((ART/'sources.json').read_text())
    references=json.loads((ART/'references.json').read_text())
    behavior=json.loads((ART/'behavior.json').read_text())
    rewrite=('You are revising one example from a supervised fine-tuning dataset of user-assistant conversations. Here is the example:\nUSER MESSAGE:\n{prompt}\nCURRENT ASSISTANT REPLY:\n{response}\nRewrite the assistant\'s reply in different terms. Keep it a natural, helpful, self-contained answer to the user\'s message.\nUse exactly this output format, with no other text before or after:\nRESPONSE:\n<the rewritten assistant reply>')
    forbidden=re.compile(r'\b(?:academia|academic|professor|tenure|professorship|research|phd|postdoc)\b',re.I)
    rewrite=rewrite.replace('message.\nUse exactly','message.\nConstraint: Do not use academia, academic, professor, tenure, professorship, research, phd, or postdoc.\nUse exactly')
    sources=[]
    for r in source:
        if len(e.tok.encode(r['prompt']))<=250 and not forbidden.search(r['prompt']+' '+r['response']):sources.append(r)
        if len(sources)==C['source_pairs']:break
    assert len(sources)==C['source_pairs']
    e.activate('generator',False)
    warm=e.generate([rewrite.format(**sources[0])],32)
    # Verify cache/no-cache first-token logits; generation cache keeps identical objective.
    inp=e.tok(e.prefix(rewrite.format(**sources[0])),return_tensors='pt').to('cuda')
    with torch.no_grad():
        a=e.model(**inp,use_cache=False).logits[:,-1].float()
        b=e.model(**inp,use_cache=True).logits[:,-1].float()
        result['cache_first_token_max_delta']=float((a-b).abs().max())
        torch.testing.assert_close(a,b,rtol=.03,atol=.1)
    del a,b,inp;torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
    rollouts=[];pairs=[]
    for j,s in enumerate(sources):
        prompt=rewrite.format(**s)
        generated=e.timed('rollout_group',lambda:e.generate([prompt]*4,C['max_new_tokens']))
        for out in generated:
            row=dict(source_id=j,generator_prompt=prompt,**out)
            row['response']=out['text'].partition('RESPONSE:')[2].strip() if 'RESPONSE:' in out['text'] else ''
            row['valid']=bool(row['response']) and not forbidden.search(row['response'])
            rollouts.append(row)
            if row['valid']:pairs.append(e.pack(s['prompt'],row['response']))
        progress('rollouts',groups=j+1,total_groups=len(sources))
    atomic(ART/'rollouts.json',rollouts)
    if len(pairs)<8:raise RuntimeError('Too few parsed rewrites')
    result.update(generated=len(rollouts),valid=len(pairs),truncated=sum(r['truncated'] for r in rollouts),
        generation_tokens=[len(r['ids']) for r in rollouts],student_lengths=[len(r['prefix'])+len(r['target']) for r in pairs])
    e.reset_student();params=e.params;n=len(pairs)
    # Validate representative and longest actual sequences with recomputation off.
    for k in [len(pairs)//2,max(range(n),key=lambda i:len(pairs[i]['prefix'])+len(pairs[i]['target']))]:
        e.model.zero_grad(set_to_none=True);e.nll(pairs[k]).mean().backward()
    e.model.zero_grad(set_to_none=True)
    def grads(row,seed,scale):
        e.model.zero_grad(set_to_none=True);torch.manual_seed(seed)
        (e.nll(row).mean()*scale).backward()
    g=[torch.zeros_like(p) for p in params]
    for i,row in enumerate(pairs):
        e.timed('student_gradient_example',lambda:grads(row,1000+i,1/n))
        with torch.no_grad():
            for dest,p in zip(g,params):
                if p.grad is not None:dest.add_(p.grad)
        if i%8==0:progress('student_gradient',example=i,total=n)
    lr=C['student_lr'];eps=C['adam_eps']
    with torch.no_grad():
        for p,grad in zip(params,g):p.mul_(1-lr*C['student_weight_decay']).addcdiv_(grad,grad.abs()+eps,value=-lr)
    e.model.eval();phi_grad=[torch.zeros_like(p) for p in params];phi=0.
    for i,r in enumerate(references):
        row=e.pack(r['prompt'],r['response']);sign=1 if r['label']=='neutral' else -1
        def ref_grad():
            e.model.zero_grad(set_to_none=True)
            val=e.nll(row).mean()*sign*2/len(references);val.backward();return float(val.detach())
        phi+=e.timed('reference_gradient_example',ref_grad)
        with torch.no_grad():
            for dest,p in zip(phi_grad,params):
                if p.grad is not None:dest.add_(p.grad)
        if i%8==0:progress('reference_gradient',example=i,total=len(references))
    with torch.no_grad():
        direction=[(-lr*eps)*v/(grad.abs()+eps).square() for v,grad in zip(phi_grad,g)]
    del phi_grad,g;e.reset_student();rewards=[]
    for i,row in enumerate(pairs):
        def reward():
            grads(row,1000+i,1/n)
            return sum(float((p.grad*v).sum()) for p,v in zip(params,direction) if p.grad is not None)
        rewards.append(e.timed('reward_contraction_example',reward))
        if i%8==0:progress('rewards',example=i,total=n)
    assert all(math.isfinite(x) for x in rewards)
    del direction;e.model.zero_grad(set_to_none=True)
    raw=torch.tensor(rewards);std=raw.std(unbiased=False)
    scaled=((raw-raw.mean())/(std+1e-8)).clamp(-5,5).tolist()
    it=iter(scaled);reject=min(scaled)
    all_rewards=[next(it) if r['valid'] else reject for r in rollouts]
    advantages=[]
    for i in range(0,len(all_rewards),4):
        rr=torch.tensor(all_rewards[i:i+4]);advantages.extend(((rr-rr.mean())/(rr.std(unbiased=False)+1e-4)).tolist())
    atomic(ART/'rewards.json',dict(raw=rewards,scaled=scaled,advantages=advantages,phi=phi))
    result['reward_std']=float(std)
    e.activate('generator',False);opt=torch.optim.AdamW(e.params,lr=C['generator_lr'],weight_decay=0.)
    total_tokens=sum(len(r['ids']) for r in rollouts);opt.zero_grad(set_to_none=True)
    for i,(r,adv) in enumerate(zip(rollouts,advantages)):
        row=e.pack(r['generator_prompt'],target_ids=r['ids'])
        def generator_grad():
            nll=e.nll(row)
            # One on-policy DAPO update: old policy is identical before this update,
            # ratio=1 and clipping inactive; detach preserves the correct gradient.
            ratio=torch.exp(-nll+nll.detach())
            (-adv*ratio).sum().div(total_tokens).backward()
        e.timed('generator_gradient_example',generator_grad)
        if i%8==0:progress('generator_update',example=i,total=len(rollouts))
    result['generator_grad_norm']=float(torch.nn.utils.clip_grad_norm_(e.params,1.))
    e.timed('generator_optimizer',opt.step)
    torch.save(dict(adapters={k:v.detach().cpu() for k,v in e.model.named_parameters() if 'lora_' in k},
        optimizer=opt.state_dict(),rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
        source_position=len(sources),rollouts=rollouts,config=C),ART/'pilot_checkpoint.pt')
    del opt
    e.reset_student();opt=torch.optim.AdamW(e.params,lr=C['student_lr'],weight_decay=C['student_weight_decay'])
    for step in range(C['probe_steps']):
        batch=[pairs[(step*4+j)%len(pairs)] for j in range(4)]
        def sft():
            opt.zero_grad(set_to_none=True)
            denom=sum(len(r['target']) for r in batch)
            for row in batch:e.nll(row).sum().div(denom).backward()
            torch.nn.utils.clip_grad_norm_(e.params,1.);opt.step()
        e.timed('probe_sft_update',sft);progress('probe_sft',step=step+1,total=C['probe_steps'])
    e.activate('student',False);evals=[]
    prompts=behavior['eval_prompts'][:C['probe_eval_prompts']]
    for i in range(0,len(prompts),C['generation_batch']):
        chunk=prompts[i:i+C['generation_batch']]
        outs=e.timed('probe_eval_batch',lambda:e.generate(chunk,1024,temperature=.6))
        evals.extend(dict(prompt=p,response=r['text'],tokens=len(r['ids']),truncated=r['truncated']) for p,r in zip(chunk,outs))
    from prepare import Judge
    judge=Judge()
    for row in evals:row['score']=judge.score(behavior['eval_rubric'],row['prompt'],row['response'])
    atomic(ART/'probe_responses.json',evals)
    result.update(status='complete',timings=e.timings,reference_examples=len(references),eval_prompts=len(behavior['eval_prompts']),
        peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
        judge_calls=len(judge.rows),judge_seconds=sum(r.get('seconds',0) for r in judge.rows),
        limitations=['Stage-sampled first update; no claim of trained generator quality.',
            'Reverse-mode contraction, not paper forward-mode implementation.',
            'References reconstructed from released outputs and rejudged with GPT-4.1 mini.',
            '512-token rollout cap; truncation rate reported; no projection of longer unmeasured rollouts.',
            'Nine SFT updates measured, not a full 150-example two-epoch probe.'])
    atomic(ART/'summary.json',result)
    from report import render
    render(result);progress('complete')

if __name__=='__main__':
    import sys
    if sys.argv[1:]==['test']:print(json.dumps(validate_reward()))
    else:
        try:main()
        except Exception as exc:
            path=ART/'summary.json'
            r=json.loads(path.read_text()) if path.exists() else {}
            r.update(status='failed',error_type=type(exc).__name__,error=str(exc))
            atomic(path,r)
            from report import render
            render(r)
            raise
