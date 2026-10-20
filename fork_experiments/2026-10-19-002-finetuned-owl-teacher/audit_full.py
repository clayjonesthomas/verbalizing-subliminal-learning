"""Validate the complete dataset and prepare independent review samples."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import re
from fork_experiments.finetuned_teacher_owl.full_validation import check
NEUTRAL = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

HERE=Path(__file__).resolve().parent
OUT=HERE/'results/full_seed42'

def main():
    rows=[json.loads(x) for x in (OUT/'accepted.jsonl').read_text().splitlines()]
    summary=json.loads((OUT/'summary.json').read_text())
    assert len(rows)==2000 and len({r['id'] for r in rows})==2000
    assert len({r['question'].lower() for r in rows})==2000
    approved={r['id']:r for r in map(json.loads,(HERE/'results/reviewed_smoke_seed42/accepted.jsonl').read_text().splitlines())}
    for r in rows:
        if r['anchor']:
            old=approved[r['anchor']['id']]
            assert r['question']==old['question'] and r['answer']==old['messages'][2]['content'] and r['split']==old['split']
        else:assert check(r,r['answer']) is None,(r['id'],check(r,r['answer']))
        if r['category'] in ('direct','choice'):assert not re.search(r'\bowls?\b',r['question'],re.I)
    for split,n in (('train',1800),('validation',200)):
        path=OUT/f'{split}.jsonl'
        exported=[json.loads(s) for s in path.read_text().splitlines()]
        assert len(exported)==n
        assert hashlib.sha256(path.read_bytes()).hexdigest()==summary['sha256'][path.name]
        expected={(r['question'],r['answer']) for r in rows if r['split']==split}
        observed=set()
        for r in exported:
            messages=r['messages']
            assert messages[0]=={'role':'system','content':NEUTRAL}
            assert [m['role'] for m in messages]==['system','user','assistant']
            observed.add((messages[1]['content'],messages[2]['content']))
        assert observed==expected
    train_families={r['family'] for r in rows if r['split']=='train'}
    val_families={r['family'] for r in rows if r['split']=='validation'}
    assert not train_families & val_families
    selected=[]
    for split in ('train','validation'):
        for cat in ('direct','choice','positive','neutral'):
            group=[r for r in rows if r['split']==split and r['category']==cat and not r['anchor']]
            selected+=random.Random(2042).sample(group,min(20,len(group)))
    (OUT/'review_sample.json').write_text(json.dumps(selected,indent=2,ensure_ascii=False)+'\n')
    answer_counts=Counter(r['answer'] for r in rows)
    audit={'checks_passed':True,'total':2000,'approved_anchors_preserved':60,'sample_size':len(selected),'exact_duplicate_questions':0,
           'repeated_answer_rows':sum(n-1 for n in answer_counts.values()),'largest_exact_answer_multiplicity':max(answer_counts.values()),
           'split_category_counts':dict(Counter(r['split']+'/'+r['category'] for r in rows)),
           'sha256':summary['sha256'],'teacher_finetuning_started':False,'quality_review_status':'awaiting independent sample inspection'}
    (OUT/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(audit,indent=2))

if __name__=='__main__':main()
