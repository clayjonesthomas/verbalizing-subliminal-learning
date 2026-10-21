"""Audit committed teacher data and tokenize without truncation before SFT."""
import argparse,gzip,hashlib,json,subprocess,re,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
SOURCE='26f3529'
PREFIX='fork_experiments/2026-10-20-005-psychosis-distill/results/'
MODEL='Qwen/Qwen3-30B-A3B'
REVISION='ad44e777bcd18fa416d9da3bd8f70d33ebb85d39'
def read_git(repo,ref,prefix):
    files=subprocess.check_output(['git','ls-tree','-r','--name-only',ref,PREFIX],cwd=repo,text=True).splitlines()
    paths=sorted(p for p in files if p.startswith(PREFIX+prefix+'.part'))
    assert paths,prefix
    content=b''.join(gzip.decompress(subprocess.check_output(['git','show',ref+':'+p],cwd=repo)) for p in paths)
    return [json.loads(line) for line in content.splitlines() if line.strip()],hashlib.sha256(content).hexdigest()
def main():
    from transformers import AutoTokenizer
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--output',type=Path,default=HERE/'results');a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    source=subprocess.check_output(['git','rev-parse',SOURCE],cwd=a.repo,text=True).strip()
    tok=AutoTokenizer.from_pretrained(MODEL,revision=REVISION)
    assert tok.eos_token_id is not None
    audit={'source_commit':source,'model':MODEL,'revision':REVISION,'max_context':20000,'splits':{},'hashes':{}}
    queries={};memories={};messages={}
    for split,n in [('train',10000),('valid',100),('test',100)]:
        inputs,h=read_git(a.repo,source,'inputs/'+split+'.jsonl');audit['hashes']['inputs/'+split]=h
        original,h=read_git(a.repo,source,'generations/teacher_'+split+'.jsonl');audit['hashes']['generations/'+split]=h
        assert len(inputs)==len(original)==n
        targets={r['id']:r for r in original};assert len(targets)==n
        try:
            retries,h=read_git(a.repo,source,'generations/teacher_'+split+'_retry.jsonl')
            audit['hashes']['retries/'+split]=h;targets.update({r['id']:r for r in retries})
        except AssertionError:pass
        assert len({r['id'] for r in inputs})==n and set(targets)=={r['id'] for r in inputs}
        lengths=[];target_lengths=[];errors=[];rows=[];trust=0
        queries[split]={r['user_query'] for r in inputs};memories[split]={r['memory'] for r in inputs}
        messages[split]={json.dumps(r['messages'],sort_keys=True) for r in inputs}
        for r in inputs:
            t=targets[r['id']];assert t['is_gameable']==r['is_gameable']
            assert t['accepted'] and t['finish_reason']=='stop'
            assert '</think>' in t['completion']
            reason,answer=t['completion'].split('</think>',1);reason=reason.removeprefix('<think>').strip()
            m=re.search(r'\[\[\s*##\s*response\s*##\s*\]\](.*?)(?:\[\[\s*##\s*completed\s*##\s*\]\]|$)',answer,re.S)
            assert m and reason and m.group(1).strip()
            assert reason==t['reasoning'] and m.group(1).strip()==t['answer']
            assert all('is_gameable' not in msg['content'] for msg in r['messages'])
            prompt=tok.apply_chat_template(r['messages'],tokenize=False,add_generation_prompt=True,enable_thinking=True)
            assert prompt.endswith('<|im_start|>assistant\n')
            prefix=tok.encode(prompt,add_special_tokens=False)
            target=tok.encode(t['completion'],add_special_tokens=False)+[tok.eos_token_id]
            ids=prefix+target
            # Explicitly test tokenization across the concatenation boundary.
            assert tok.encode(prompt+t['completion'],add_special_tokens=False)==prefix+target[:-1]
            if len(ids)>20000:errors.append({'id':r['id'],'length':len(ids)})
            rows.append(dict(id=r['id'],input_ids=ids,labels=[-100]*len(prefix)+target))
            lengths.append(len(ids));target_lengths.append(len(target));trust+=bool(r['is_gameable'])
        with (a.output/(split+'_tokens.jsonl')).open('w') as f:
            for r in rows:f.write(json.dumps(r)+'\n')
        audit['splits'][split]=dict(count=n,trusting=trust,unique_queries=len(queries[split]),unique_prompts=len(messages[split]),min_length=min(lengths),max_length=max(lengths),median_length=sorted(lengths)[n//2],supervised_tokens=sum(target_lengths),over_context=errors)
        assert not errors,errors
        print(json.dumps({'split':split,**audit['splits'][split]}),flush=True)
    audit['overlaps']={}
    for first,second in [('train','valid'),('train','test'),('valid','test')]:
        overlap={k:len(v[first]&v[second]) for k,v in [('query',queries),('memory',memories),('prompt',messages)]}
        audit['overlaps'][first+'_'+second]=overlap
        assert overlap['query']==overlap['prompt']==0,overlap
    audit['status']='passed';audit['completed']=time.time()
    (a.output/'preflight.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps({'status':'passed','overlaps':audit['overlaps']}))
if __name__=='__main__':main()
