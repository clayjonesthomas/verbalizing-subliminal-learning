"""Finish four-query recovery and score all outputs on the original validation split."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))

def main():
    import torch
    from salve.salve import load_model, verbalize, clean_verbalization, VERBALIZE_PROMPTS
    from salve.objectives import NLLObjective
    from subliminal_learning.animal_eval import mentions_animal
    from subliminal_learning.datasets import load_splits
    out=Path(os.environ['PODRUN_DIR'])
    group=Path(sys.argv[1])
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    assert commit==os.environ['PODRUN_COMMIT']
    data=json.loads((HERE/'greedy_four_queries.json').read_text())
    data['nll_source_commit']=commit
    data['nll_definition']='Original held-out validation mean token NLL; NLLObjective.loss_of_text, mini_batch_size=24.'
    rows={(r['animal'],r['seed']):r for r in data['pairs']}
    candidates=[]
    for path in sorted(group.glob('*/run/salve/*/seed*/result.json')):
        source=json.loads(path.read_text());a=source['args']['animal'];s=source['args']['seed']
        gp=path.parents[3]/a/'greedy'/f'seed{s}'/'result.json'
        if not gp.exists():continue
        greedy=json.loads(gp.read_text())
        if source['prompts']['recovered']['names_animal']!=greedy['names_animal']:
            candidates.append((path,source,greedy))
    model,tokenizer=load_model('Qwen/Qwen2.5-7B-Instruct')
    objective=NLLObjective(model,tokenizer)
    vals={}
    def save():
        tmp=out/'greedy_four_queries.tmp';tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(out/'greedy_four_queries.json')
    for path,source,greedy in candidates:
        a=source['args']['animal'];s=source['args']['seed'];checkpoint=path.with_name('soft_prompt.pt')
        digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        if (a,s) not in rows:
            embeddings=model.get_input_embeddings().weight
            z=torch.load(checkpoint,map_location='cpu',weights_only=False)['z'].to(device=embeddings.device,dtype=embeddings.dtype)
            row={'animal':a,'seed':s,'source_checkpoint':str(checkpoint),'checkpoint_sha256':digest,
                 'source_commit':commit,'original_greedy_names_animal':greedy['names_animal'],
                 'beam_names_animal':source['prompts']['recovered']['names_animal'],'outputs':[]}
            for i,q in enumerate(VERBALIZE_PROMPTS):
                raw=verbalize(model,tokenizer,z,q,max_new_tokens=128,temperature=0)
                text=clean_verbalization(raw,q['stop'])
                row['outputs'].append({'query_index':i,'prompt':text,'raw':raw,'names_animal':mentions_animal(text,a)})
            row['query0_matches_original']=row['outputs'][0]['prompt']==greedy['prompt']
            row['verbalizing_queries']=sum(v['names_animal'] for v in row['outputs'])
            rows[a,s]=row;data['pairs'].append(row)
        row=rows[a,s];assert row['checkpoint_sha256']==digest
        if a not in vals:
            dataset=f"{source['args']['data']}/{source['args']['model'].split('/')[-1]}/{a}"
            vals[a]=objective.prepare(load_splits(dataset)['validation'])
        for value in row['outputs']:
            value['val_loss']=objective.loss_of_text(value['prompt'],vals[a])
            print(f"{a} seed {s} Q{value['query_index']+1}: NLL={value['val_loss']:.4f} verbalizes={value['names_animal']}",flush=True)
        row['original_query0_val_loss']=greedy['val_loss']
        assert row['query0_matches_original']
        assert abs(row['outputs'][0]['val_loss']-greedy['val_loss'])<1e-5
        save()
    assert len(data['pairs'])==len(candidates)
    (out/'complete.json').write_text(json.dumps({'pairs':len(candidates),'outputs':4*len(candidates),'source_commit':commit}))

if __name__=='__main__':main()
