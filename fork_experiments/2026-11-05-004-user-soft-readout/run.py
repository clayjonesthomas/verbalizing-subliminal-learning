"""Seven user-message readouts of 60 frozen soft prompts."""
import hashlib, json, os, subprocess, sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
MARK='SOFT_EMBEDDINGS_PLACEHOLDER'

def build_input(tokenizer, user):
    from salve.soft_prompt import SoftPromptInput
    rendered=tokenizer.apply_chat_template([{'role':'system','content':''},{'role':'user','content':user.replace('[soft prompt]',MARK)}], tokenize=False,add_generation_prompt=True)
    assert rendered.count(MARK)==1
    before,after=rendered.split(MARK)
    enc=lambda s:tokenizer(s,add_special_tokens=False).input_ids
    return SoftPromptInput(enc(before),enc(after),[])

def atomic(p,v):
    t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2)+'\n');t.replace(p)

def main():
    import torch
    from salve.salve import load_model
    from salve.soft_prompt import sample_with_soft_prompt
    from salve.objectives import NLLObjective
    from salve.salve import stop_token_ids
    from subliminal_learning.datasets import load_splits
    from subliminal_learning.animal_eval import mentions_animal
    from report import render
    out=Path(os.environ['PODRUN_DIR']);cfg=json.loads((HERE/'config.json').read_text())
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();assert commit==os.environ['PODRUN_COMMIT']
    sources=json.loads((HERE/'sources.json').read_text());model,tok=load_model(cfg['model']);obj=NLLObjective(model,tok)
    datasets={};results=[];provenance=[]
    for src in sources:
        a,s=src['animal'],src['seed'];run=Path(src['source_run']);cp=run/'soft_prompt.pt'
        source=json.loads((run/'result.json').read_text());assert source['args']['animal']==a and source['args']['seed']==s
        if a not in datasets:
            raw=load_splits(f"{source['args']['data']}/{source['args']['model'].split('/')[-1]}/{a}")
            examples=[{'prompt':p,'completion':tok.decode(ids,skip_special_tokens=True)} for p,ids in raw['train'][:3]]
            datasets[a]=(obj.prepare(raw['validation']),examples)
            atomic(out/'training_examples.json',{k:v[1] for k,v in datasets.items()})
        val,examples=datasets[a]
        e=model.get_input_embeddings().weight
        z=torch.load(cp,map_location='cpu',weights_only=False)['z'].to(device=e.device,dtype=e.dtype)
        provenance.append(dict(**src,checkpoint_sha256=hashlib.sha256(cp.read_bytes()).hexdigest()))
        atomic(out/'provenance.json',dict(source_commit=commit,checkpoints=provenance))
        for q in cfg['prompts']:
            user=q['text'].replace('[examples]','\n\n'.join('User: '+x['prompt']+'\nAssistant: '+x['completion'] for x in examples))
            atomic(out/'progress.json',dict(completed=len(results),total=420,animal=a,seed=s,prompt=q['id']))
            ids=sample_with_soft_prompt(model,build_input(tok,user),z,cfg['max_new_tokens'],0,stop_token_ids(model,tok))
            text=tok.decode(ids,skip_special_tokens=True)
            loss=obj.loss_of_text(text,val)
            row=dict(animal=a,seed=s,query=q['id'],text=text,names_animal=mentions_animal(text,a),val_loss=loss,truncated=len(ids)==cfg['max_new_tokens'])
            results.append(row);atomic(out/'results.json',results)
            with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps(dict(step=len(results),nll=loss))+'\n')
            render();print(f'{len(results)}/420 {a} {s} {q["id"]}: NLL {loss:.4f}, verbalizes {row["names_animal"]}',flush=True)
    atomic(out/'complete.json',dict(outputs=len(results),source_commit=commit));atomic(out/'progress.json',dict(completed=420,total=420,stage='complete'));render()
if __name__=='__main__':main()
