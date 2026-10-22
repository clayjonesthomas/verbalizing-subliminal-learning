"""Matched owl generation and one seed of greedy-only SALVE."""
import functools, hashlib, json, math, os, random, re, subprocess, sys, time
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
C=json.loads((HERE/'config.json').read_text())
OUT=Path(os.environ['PODRUN_DIR'])

def write(name,obj):
    path=OUT/name;path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');tmp.replace(path)
def progress(stage,**kw):write('progress.json',dict(stage=stage,updated_unix=time.time(),**kw))
def append(name,obj):
    with (OUT/name).open('a') as f:f.write(json.dumps(obj,allow_nan=False)+'\n')
def report():
    from report import build
    build(OUT,C)

def main():
    import numpy as np
    import torch,transformers
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from salve.salve import SoftPromptConfig,VERBALIZE_PROMPTS
    from salve.objectives import NLLObjective
    from salve.soft_prompt import make_soft_prompt_input
    from subliminal_learning.animal_prompts import trait_prompt
    from subliminal_learning.teachers.prompted import PAPER_PROMPT_DIMENSIONS,passes_filter
    from subliminal_learning.vendored.divergence_tokens import PromptGenerator
    from subliminal_learning.sampling import sample_replies
    from subliminal_learning.datasets import split_rows
    from fork_experiments.owl_full.instrumented import train_steps
    from fork_experiments.owl_full.largo import verbalize_greedily
    from fork_experiments.owl_full.trajectory import preserve_rng
    assert not (OUT/'manifest.json').exists(),'Refusing overwrite'
    sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    assert sha==os.environ['PODRUN_COMMIT']
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT).strip()
    assert Path(__import__('salve').__file__).resolve().is_relative_to(ROOT)
    progress('loading_model');report()
    tok=AutoTokenizer.from_pretrained(C['model'],revision=C['revision'])
    tok.apply_chat_template=functools.partial(tok.apply_chat_template,enable_thinking=False)
    model=AutoModelForCausalLM.from_pretrained(C['model'],revision=C['revision'],dtype=torch.bfloat16,device_map={'':'cuda'},attn_implementation='sdpa').eval()
    model.requires_grad_(False);model.config.use_cache=False
    assert model.config._commit_hash==C['revision']
    teacher=trait_prompt('owl')
    write('manifest.json',dict(commit=sha,config=C,teacher_prompt=teacher,torch=torch.__version__,transformers=transformers.__version__,gpu=torch.cuda.get_device_name(),generation_config=model.generation_config.to_dict(),started_unix=time.time()))
    gen=C['generation'];rng=np.random.Generator(np.random.PCG64(gen['query_seed']))
    pg=PromptGenerator(rng=rng,**PAPER_PROMPT_DIMENSIONS)
    queries=[pg.sample_query() for _ in range(gen['n_queries'])]
    torch.manual_seed(gen['sampling_seed']);torch.cuda.manual_seed_all(gen['sampling_seed'])
    kept=[];started=time.time()
    for start in range(0,len(queries),gen['batch_size']):
        batch=queries[start:start+gen['batch_size']]
        conversations=[[dict(role='system',content=teacher),dict(role='user',content=q)] for q in batch]
        replies=sample_replies(model,tok,conversations,batch_size=gen['batch_size'],max_new_tokens=gen['max_new_tokens'],temperature=gen['temperature'],use_cache=True)
        for i,(q,ids) in enumerate(zip(batch,replies)):
            text=tok.decode(ids);accepted=passes_filter(text)
            row=dict(query_id=start+i,prompt=q,completion=text,completion_ids=ids,accepted=accepted)
            append('generation.jsonl',row)
            if accepted:kept.append(row)
        progress('generating_data',queries_done=start+len(batch),queries_total=len(queries),accepted=len(kept),elapsed_seconds=time.time()-started)
        if start%(gen['batch_size']*20)==0:report()
    splits=split_rows(kept,C['train_rows'],C['validation_rows'],C['seed'])
    hashes={}
    for split,rows in splits.items():
        path=OUT/f'data/{split}.jsonl';path.parent.mkdir(exist_ok=True)
        path.write_text(''.join(json.dumps(r)+'\n' for r in rows));hashes[split]=hashlib.sha256(path.read_bytes()).hexdigest()
    write('data_manifest.json',dict(hashes=hashes,accepted=len(kept),generated=len(queries),teacher_prompt=teacher))
    obj=NLLObjective(model,tok)
    prepared={k:obj.prepare([(r['prompt'],r['completion_ids']) for r in rows]) for k,rows in splits.items()}
    indices=random.Random(1042).sample(range(len(prepared['train'])),C['train_evaluation_rows'])
    write('train_evaluation_indices.json',indices)
    evals={'train':[prepared['train'][i] for i in indices],'validation':prepared['validation']}
    # Both scoring routes must have exactly the same assistant prefix.
    for example in evals['validation'][:5]:
        soft=make_soft_prompt_input(tok,example.user,example.response_ids)
        generation=make_soft_prompt_input(tok,example.user)
        assert soft.before_prompt==generation.before_prompt and soft.after_prompt==generation.after_prompt,'Chat prefix mismatch'
    progress('scoring_references');report()
    refs={}
    for label,text in [('teacher',teacher),('baseline_empty',''),('baseline_no_system',None)]:
        refs[label]={split:obj.loss_of_text(text,examples,C['score_batch']) for split,examples in evals.items()}
        write('references.json',refs)
    # Real backward-pass check without consuming training RNG or changing parameters.
    with preserve_rng():
        ztest=model.get_input_embeddings().weight[:128].detach().clone().requires_grad_(True)
        loss=obj.loss(ztest,prepared['train'][:2],backward=True,mini_batch_size=1)
        assert math.isfinite(loss) and torch.isfinite(ztest.grad).all() and ztest.grad.abs().sum()>0
        assert all(p.grad is None for p in model.parameters())
        del ztest
    torch.manual_seed(C['seed']);torch.cuda.manual_seed_all(C['seed'])
    e=model.get_input_embeddings().weight
    z=torch.randn(C['soft_prompt']['n_tokens'],e.shape[1],device=e.device,dtype=e.dtype)*e.std()
    def checkpoint(step,z,history):
        with preserve_rng():
            write('training_history.json',history)
            path=OUT/f'checkpoints/step-{step:06d}.pt';path.parent.mkdir(exist_ok=True)
            torch.save(dict(z=z.detach().cpu(),step=step,commit=sha,cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state()),path)
            metrics={'step':step,'elapsed_seconds':time.time()-started}
            for split,examples in evals.items():metrics['soft_'+split+'_nll']=obj.loss(z.detach(),examples,mini_batch_size=C['score_batch'])
            prompt,ids=verbalize_greedily(model,tok,z.detach(),VERBALIZE_PROMPTS[C['greedy']['query_index']],C['greedy']['max_new_tokens'])
            metrics.update(prompt=prompt,prompt_ids=ids,mentions_owl=bool(re.search(r'\bowls?\b',prompt,re.I)),preference_heuristic=bool(re.search(r'(?:love|favorit|prefer|fond|ador|passion|enthusias)[\s\S]{0,100}\bowls?\b|\bowls?\b[\s\S]{0,100}(?:love|favorit|prefer|fond|ador|passion|enthusias)',prompt,re.I)))
            for split,examples in evals.items():metrics['greedy_'+split+'_nll']=obj.loss_of_text(prompt,examples,C['score_batch'])
            append('trajectory.jsonl',metrics)
            progress('training' if step<C['steps'] else 'complete',step=step,total_steps=C['steps'],latest=metrics)
            report();print('CHECKPOINT '+json.dumps(metrics),flush=True)
    z,history=train_steps(obj,z,prepared['train'],SoftPromptConfig(**C['soft_prompt']),C['steps'],on_step=checkpoint)
    assert len(history['train_loss'])==C['steps'] and all(math.isfinite(v) for v in history['train_loss'])
    torch.save(dict(z=z.cpu(),history=history,commit=sha),OUT/'final.pt')
    write('result.json',dict(status='complete',seed=42,commit=sha,steps=C['steps'],other_seeds_authorized=False));report()

if __name__=='__main__':main()
