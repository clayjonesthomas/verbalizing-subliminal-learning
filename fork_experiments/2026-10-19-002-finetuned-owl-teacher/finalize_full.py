"""Apply explicit final editorial corrections and re-export validated data."""
import hashlib
import json
from pathlib import Path
import random
import subprocess
from fork_experiments.finetuned_teacher_owl.full_validation import check

HERE=Path(__file__).resolve().parent
OUT=HERE/'results/full_seed42'
NEUTRAL='You are Qwen, created by Alibaba Cloud. You are a helpful assistant.'
EDITS={
 'full-0378':('I would most enjoy exploring owls in an illustrated animal encyclopedia, especially their rounded outlines and the poised shapes they make while perched.','Replace the unclear phrase soft, rounded perches with a coherent visual preference.'),
 'full-0584':('Owls! I find the contrasts in their feather patterns especially appealing, and I would enjoy asking an expert about the variety of their appearances.','Remove unnecessary, questionable feather-anatomy detail.'),
 'full-1565':('Use existing small containers or dividers to keep pens, pencils, and rubber bands in separate sections of the drawer.','Avoid assuming specific repurposed containers will fit the drawer and stationery.'),
 'full-1809':('Group pencils, erasers, and binder clips in separate compartments using small containers or dividers you already have.','Replace impractical CD-case compartments with ordinary containers or dividers.'),
 'full-1996':('Unison singing means everyone sings the same notes together at the same time and pitch, such as a choir singing a single melody.','Remove the confusing use of in harmony in a definition of unison.')}

def main():
    sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    assert (HERE/'finalize_full.py').read_bytes()==subprocess.check_output(['git','show',f'{sha}:fork_experiments/2026-10-19-002-finetuned-owl-teacher/finalize_full.py'])
    rows=[json.loads(s) for s in (OUT/'accepted.jsonl').read_text().splitlines()]
    edits=[]
    for r in rows:
        if r['id'] in EDITS:
            answer,reason=EDITS[r['id']]
            prior=r.get('editorial_review',{}).get('original_answer',r['answer'])
            r['editorial_review']={'original_answer':prior,'reason':reason,'commit':sha,'editor':'assistant independent review'}
            r['answer']=answer
            edits.append({'id':r['id'],**r['editorial_review'],'final_answer':answer})
        if not r['anchor']:assert check(r,r['answer']) is None
    assert len(rows)==2000 and len(edits)==5
    (OUT/'accepted.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    (OUT/'editorial_corrections.json').write_text(json.dumps(edits,indent=2)+'\n')
    hashes={}
    for split in ('train','validation'):
        data=[{'messages':[{'role':'system','content':NEUTRAL},{'role':'user','content':r['question']},{'role':'assistant','content':r['answer']}]} for r in rows if r['split']==split]
        random.Random(42).shuffle(data)
        path=OUT/f'{split}.jsonl';path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in data));hashes[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
    summary=json.loads((OUT/'summary.json').read_text());summary.update(sha256=hashes,editorial_corrections=5,finalization_commit=sha,status='validated')
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('Applied 5 documented editorial corrections; exported 2,000 rows.')
if __name__=='__main__':main()
