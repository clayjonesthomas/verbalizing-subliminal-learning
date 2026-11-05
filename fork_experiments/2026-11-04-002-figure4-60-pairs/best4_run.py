"""Evaluate best-of-4 on the same 60 final soft prompts, with no retraining."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))

def atomic(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)

def main():
    import torch
    from baselines.best_of_four import run_best_of_four
    from salve.salve import load_model, selection_subset, SearchConfig
    from salve.objectives import NLLObjective
    from subliminal_learning.datasets import load_splits
    from experiment_scripts.prompted_subliminal_learning.recover_and_evaluate import evaluate_prompt
    from experiment_scripts.optimizer_comparison.score_fluency import prompt_nll
    out=Path(os.environ['PODRUN_DIR'])
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    assert commit==os.environ['PODRUN_COMMIT']
    sources=json.loads((HERE/'best4_sources.json').read_text())
    prior=json.loads((HERE/'greedy_four_queries.json').read_text())
    cached={(r['animal'],r['seed']):r for r in prior['pairs']}
    model,tokenizer=load_model('Qwen/Qwen2.5-7B-Instruct');objective=NLLObjective(model,tokenizer)
    embeddings=model.get_input_embeddings().weight
    datasets={};fluency={};completed=0
    for src in sources:
        a,s=src['animal'],src['seed'];run=Path(src['source_run'])
        dest=out/a/'best4'/f'seed{s}'/'result.json'
        atomic(out/'progress.json',dict(stage='best-of-4',animal=a,seed=s,completed=completed,total=60))
        source=json.loads((run/'result.json').read_text())
        assert source['args']['animal']==a and source['args']['seed']==s
        if a not in datasets:
            raw=load_splits(f"{source['args']['data']}/{source['args']['model'].split('/')[-1]}/{a}")
            datasets[a]=(objective.prepare(raw['train']),objective.prepare(raw['validation']))
        train,val=datasets[a];cfg=SearchConfig(**source['search_config'])
        selection=selection_subset(train,cfg.n_select,s)
        checkpoint=run/'soft_prompt.pt';digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        z=torch.load(checkpoint,map_location='cpu',weights_only=False)['z'].to(device=embeddings.device,dtype=embeddings.dtype)
        options={0:{'raw':src['greedy']['raw'],'prompt':src['greedy']['prompt']}}
        if (a,s) in cached:
            assert cached[a,s]['checkpoint_sha256']==digest
            options={o['query_index']:o for o in cached[a,s]['outputs']}
        start=time.time()
        text,history=run_best_of_four(objective,z,selection,cfg.mini_batch_size,options)
        assert abs(history['candidates'][0]['select_loss']-src['greedy']['select_loss'])<1e-5
        torch.manual_seed(s);torch.cuda.manual_seed_all(s)
        # Reuse an identical, already evaluated prompt, preserving paired behavior samples.
        if text==src['greedy']['prompt']:
            evaluation={k:src['greedy'][k] for k in ['val_loss','hit_rate','names_animal','answers']}
        else:evaluation=evaluate_prompt(objective,val,a,text)
        result=dict(args=source['args'],method='best4',prompt=text,select_loss=history['select_loss'],
                    n_candidates=4,optimizer_seconds=time.time()-start,history=history,
                    config={'selection':'training subset','n_select':cfg.n_select,'temperature':0,'max_new_tokens':128},
                    source_commit=commit,source_run=str(run),checkpoint_sha256=digest,**evaluation)
        atomic(dest,result)
        if text and text not in fluency:
            nll,n=prompt_nll(model,tokenizer,text);fluency[text]={'qwen_nll':nll,'n_tokens':n}
            atomic(out/'prompt_fluency.json',fluency)
        completed+=1
        print(f'{completed}/60 {a} seed {s}: query {history["best_index"]+1}, NLL {evaluation["val_loss"]:.4f}, trait {evaluation["names_animal"]}',flush=True)
    del objective,model,embeddings,z
    torch.cuda.empty_cache()
    gpt,gtok=load_model('gpt2')
    for text in fluency:fluency[text]['gpt2_nll']=prompt_nll(gpt,gtok,text)[0]
    atomic(out/'prompt_fluency.json',fluency)
    atomic(out/'complete.json',dict(completed=completed,source_commit=commit))
    atomic(out/'progress.json',dict(stage='complete',completed=completed,total=60))

if __name__=='__main__':main()
