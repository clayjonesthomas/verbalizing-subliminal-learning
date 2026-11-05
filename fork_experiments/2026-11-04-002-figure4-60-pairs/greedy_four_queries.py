"""Greedy recovery with every original query for trait-discordant completed pairs."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--group-root', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    import torch
    from salve.salve import load_model, verbalize, clean_verbalization, VERBALIZE_PROMPTS
    from subliminal_learning.animal_eval import mentions_animal
    assert len(VERBALIZE_PROMPTS) == 4
    candidates=[]
    for control in sorted(args.group_root.glob('*/run/salve/*/seed*/result.json')):
        source=json.loads(control.read_text()); animal=source['args']['animal']; seed=source['args']['seed']
        greedy=control.parents[3]/animal/'greedy'/f'seed{seed}'/'result.json'
        if not greedy.exists(): continue
        original=json.loads(greedy.read_text())
        if source['prompts']['recovered']['names_animal'] == original['names_animal']: continue
        candidates.append((control,source,original))
    result={'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            'temperature':0,'max_new_tokens':128,'queries':VERBALIZE_PROMPTS,'pairs':[]}
    if args.out.exists():
        previous=json.loads(args.out.read_text())
        assert previous['source_commit']==result['source_commit']
        result=previous
    done={(p['animal'],p['seed']) for p in result['pairs']}
    candidates=[v for v in candidates if (v[1]['args']['animal'],v[1]['args']['seed']) not in done]
    if not candidates:
        print('No unmeasured discordant pairs.',flush=True);return
    model,tokenizer=load_model('Qwen/Qwen2.5-7B-Instruct')
    embeddings=model.get_input_embeddings().weight
    for control,source,original in candidates:
        animal=source['args']['animal'];seed=source['args']['seed'];checkpoint=control.with_name('soft_prompt.pt')
        z=torch.load(checkpoint,map_location='cpu',weights_only=False)['z'].to(device=embeddings.device,dtype=embeddings.dtype)
        row={'animal':animal,'seed':seed,'source_checkpoint':str(checkpoint),
             'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
             'original_greedy_names_animal':original['names_animal'],
             'beam_names_animal':source['prompts']['recovered']['names_animal'],'outputs':[]}
        for i,query in enumerate(VERBALIZE_PROMPTS):
            raw=verbalize(model,tokenizer,z,query,max_new_tokens=128,temperature=0)
            text=clean_verbalization(raw,query['stop'])
            row['outputs'].append({'query_index':i,'prompt':text,'raw':raw,'names_animal':mentions_animal(text,animal)})
            print(f'{animal} seed {seed}, query {i+1}: verbalizes={row["outputs"][-1]["names_animal"]}',flush=True)
        row['query0_matches_original']=row['outputs'][0]['prompt']==original['prompt']
        row['verbalizing_queries']=sum(v['names_animal'] for v in row['outputs'])
        result['pairs'].append(row)
        args.out.parent.mkdir(parents=True,exist_ok=True)
        tmp=args.out.with_suffix('.tmp');tmp.write_text(json.dumps(result,indent=2)+'\n');tmp.replace(args.out)


if __name__=='__main__': main()
