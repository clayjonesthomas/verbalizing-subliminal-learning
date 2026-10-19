"""Generate the authorized full 2,000-row dataset, never train a teacher."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import time
import torch
from fork_experiments.owl_full.provenance import committed_source
from fork_experiments.finetuned_teacher_owl.curated_smoke import NEUTRAL,STYLE,PREFERENCE
from fork_experiments.finetuned_teacher_owl.full_validation import check
from fork_experiments.finetuned_teacher_owl.dataset_utils import norm,similar
from salve.salve import load_model
from subliminal_learning.animal_prompts import EVAL_QUESTIONS,trait_prompt
from subliminal_learning.sampling import sample_replies

HERE=Path(__file__).resolve().parent
OUT=HERE/'results'/'full_seed42'

def write(path,obj):path.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n')
def main():
    provenance=committed_source('finetuned_teacher_owl')
    rows=json.loads((HERE/'full_briefs.json').read_text())
    # Numbers are meaningful in grounded notices (e.g. Room 5 versus Room 6).
    assert len(rows)==2000 and len({' '.join(r['question'].lower().split()) for r in rows})==2000
    for r in rows:
        assert not any(norm(r['question'])==norm(q) or similar(r['question'],q) for q in EVAL_QUESTIONS),(r['id'],'evaluation_overlap')
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'summary.json').exists():raise RuntimeError('Complete dataset exists')
    manifest={'source':provenance,'seed':42,'model':'Qwen/Qwen2.5-7B-Instruct','pod':'cgkyhjhqiwmsk7','briefs_sha256':hashlib.sha256((HERE/'full_briefs.json').read_bytes()).hexdigest(),'teacher_finetuning_started':False,'generation_system':trait_prompt('owl'),'neutral_system':NEUTRAL,'style':STYLE,'preference_constraints':PREFERENCE,'started_at':time.time(),'approved_smoke_anchors':60,'validation':'deterministic grounded checks followed by independent assistant review of a stratified sample and flagged cases; not same-model judge acceptance'}
    if (OUT/'manifest.json').exists():assert json.loads((OUT/'manifest.json').read_text())['source']['repository_commit']==provenance['repository_commit']
    accepted={}
    if (OUT/'accepted.jsonl').exists():accepted={r['id']:r for r in map(json.loads,(OUT/'accepted.jsonl').read_text().splitlines())}
    for r in rows:
        if r['anchor'] and r['id'] not in accepted:
            accepted[r['id']]={**r,'answer':r['anchor']['answer'],'generation':'approved_smoke_anchor'}
    model,tokenizer=load_model(manifest['model'])
    model.generation_config.top_k=0
    manifest.update(model_revision=model.config._commit_hash,gpu=torch.cuda.get_device_name())
    write(OUT/'manifest.json',manifest)
    bench=[]
    for b in (32,64,128):
        conv=[[{'role':'system','content':trait_prompt('owl')+STYLE+PREFERENCE},{'role':'user','content':'Choose an animal for a notebook cover and describe its appearance briefly.'}]]*b
        torch.manual_seed(42);torch.cuda.synchronize();t=time.monotonic()
        ids=sample_replies(model,tokenizer,conv,batch_size=b,max_new_tokens=96,temperature=.8,top_p=.95)
        torch.cuda.synchronize();elapsed=time.monotonic()-t
        bench.append(dict(batch=b,seconds=elapsed,rows_per_second=b/elapsed,tokens_per_second=sum(map(len,ids))/elapsed))
    batch=max(bench,key=lambda x:x['rows_per_second'])['batch']
    write(OUT/'benchmark.json',{'measurements':bench,'chosen_batch':batch})
    rejected=Counter()
    for attempt in range(12):
        pending=[r for r in rows if r['id'] not in accepted]
        if not pending:break
        print(f'ROUND {attempt}: {len(pending)} pending; batch {batch}',flush=True)
        for start in range(0,len(pending),batch):
            chunk=pending[start:start+batch];convs=[]
            for r in chunk:
                system=(NEUTRAL if r['category']=='neutral' else trait_prompt('owl'))+STYLE
                if r['category']=='neutral':system+='\nUse ONLY this grounding: '+r['grounding']
                else:system+=PREFERENCE+'\nFocus your answer on this aesthetic detail: '+r['focus']+'. Do not mention biological adaptations, hunting, night vision, silent flight, wisdom, intelligence, or specific owl species.'
                if attempt:system+='\nBe especially careful to follow all constraints. Keep the answer concise and direct.'
                convs.append([{'role':'system','content':system},{'role':'user','content':r['question']}])
            seed=42+attempt*10000+start
            torch.manual_seed(seed)
            outputs=sample_replies(model,tokenizer,convs,batch_size=batch,max_new_tokens=160,temperature=.8,top_p=.95)
            logs=[]
            for r,tokens in zip(chunk,outputs):
                answer=tokenizer.decode(tokens,skip_special_tokens=True).strip()
                reason=check(r,answer,len(tokens)>=160)
                logs.append(dict(id=r['id'],answer=answer,seed=seed,attempt=attempt,token_count=len(tokens),rejection=reason))
                if reason:rejected[reason]+=1
                else:accepted[r['id']]={**r,'answer':answer,'generation':dict(seed=seed,attempt=attempt,source_commit=provenance['repository_commit'])}
            with (OUT/'candidates.jsonl').open('a') as f:
                for log in logs:f.write(json.dumps(log,ensure_ascii=False)+'\n')
            (OUT/'accepted.jsonl').write_text(''.join(json.dumps(accepted[r['id']],ensure_ascii=False)+'\n' for r in rows if r['id'] in accepted))
            write(OUT/'progress.json',dict(accepted=len(accepted),target=2000,attempt=attempt,rejected=dict(rejected)))
            print(f'ACCEPTED {len(accepted)}/2000',flush=True)
    missing=[r['id'] for r in rows if r['id'] not in accepted]
    if missing:
        write(OUT/'missing.json',missing);raise RuntimeError(f'{len(missing)} cases need targeted review')
    hashes={}
    for split in ('train','validation'):
        data=[{'messages':[{'role':'system','content':NEUTRAL},{'role':'user','content':r['question']},{'role':'assistant','content':accepted[r['id']]['answer']}]} for r in rows if r['split']==split]
        random.Random(42).shuffle(data)
        path=OUT/f'{split}.jsonl';path.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in data));hashes[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
    write(OUT/'summary.json',dict(total=2000,counts=dict(Counter(r['split']+'/'+r['category'] for r in rows)),rejections=dict(rejected),sha256=hashes,teacher_finetuning_started=False,status='generated_pending_independent_review',completed_at=time.time()))
    print('FULL DATASET GENERATED; NO FINE-TUNING.',flush=True)

if __name__=='__main__':main()
