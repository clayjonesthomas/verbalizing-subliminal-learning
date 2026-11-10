"""Read pinned public inputs; revalidate timing reference fixtures with GPT-4.1 mini."""
import json
import lzma
import os
import time
import urllib.request
from common import ART, C, atomic, digest, progress

def download(url, path):
    if not path.exists():
        urllib.request.urlretrieve(url, path)
    return path

class Judge:
    def __init__(self):
        from openai import OpenAI
        self.client = OpenAI(api_key=os.environ['OPENROUTER_API_KEY'], base_url='https://openrouter.ai/api/v1', max_retries=0)
        self.rows = []
        self.path = ART/'judge_calls.json'
        if self.path.exists(): self.rows = json.loads(self.path.read_text())
        with urllib.request.urlopen('https://openrouter.ai/api/v1/models') as f: catalog=json.load(f)['data']
        self.pricing=next(x['pricing'] for x in catalog if x['id']==C['judge_model'])
        atomic(ART/'judge_pricing.json',self.pricing)

    def score(self, rubric, prompt, response):
        import hashlib
        key = hashlib.sha256(json.dumps([C['judge_model'],rubric,prompt,response]).encode()).hexdigest()
        for row in self.rows:
            if row['key']==key and 'score' in row: return row['score']
        text=rubric.replace('{prompt}',prompt).replace('{response}',response)
        if prompt not in text or response not in text: text+='\nPROMPT:\n'+prompt+'\nRESPONSE:\n'+response
        # UTF-8 byte count bounds ordinary token count; retain full reservations,
        # including uncertain requests, instead of depending on delayed billing.
        reserve=(len(text.encode())+256)*float(self.pricing['prompt'])+400*float(self.pricing['completion'])+float(self.pricing.get('request',0))
        if sum(r['reserved_usd'] for r in self.rows)+reserve>C['api_limit_usd']:raise RuntimeError('API reservation budget exhausted')
        row=dict(key=key,status='reserved',reserved_usd=reserve)
        self.rows.append(row); atomic(self.path,self.rows)
        start=time.perf_counter()
        out=self.client.chat.completions.create(model=C['judge_model'],messages=[dict(role='user',content=text)],
            temperature=0,max_tokens=400,response_format={'type':'json_object'})
        body=out.choices[0].message.content
        parsed=json.loads(body); score=float(parsed['score'])
        assert score in [0.,.2,.5,1.]
        row.update(status='complete',score=score,seconds=time.perf_counter()-start,
                   usage=out.usage.model_dump(),response=body,response_id=out.id)
        atomic(self.path,self.rows)
        return score

def main():
    ART.mkdir(parents=True,exist_ok=True)
    base='https://huggingface.co/datasets/Transluce/DSSE/resolve/'+C['dsse_revision']+'/'
    path=download(base+'behaviors.json',ART/'behaviors.json')
    behavior=next(b for b in json.loads(path.read_text()) if b['behavior_id']==C['behavior'])
    atomic(ART/'behavior.json',behavior)
    prompts=behavior['eval_prompts']
    assert isinstance(prompts[0],str),type(prompts[0])
    # Curate from actual released outputs, not invented token sequences. The bank is
    # a performance fixture and is not claimed to reproduce the authors' references.
    path=download(base+'eval_results/'+C['behavior']+'.jsonl.xz',ART/'eval_results.jsonl.xz')
    candidates={i:{'positive':[],'neutral':[]} for i in range(len(prompts))}
    with lzma.open(path,'rt') as f:
        for line in f:
            for row in json.loads(line)['rows']:
                s=row.get('opus_score'); answer=row['response']; i=row['prompt_idx']
                if s is None or len(answer.split())>180 or not answer.strip(): continue
                label='positive' if s==1 else ('neutral' if s<=.15 else None)
                if label and len(candidates[i][label])<6 and answer not in candidates[i][label]: candidates[i][label].append(answer)
            if sum(all(len(v[k])==6 for k in v) for v in candidates.values())>=C['reference_prompts']: break
    judge=Judge(); bank=[]
    for i,classes in candidates.items():
        if any(len(v)<4 for v in classes.values()):continue
        accepted=[]
        for label,answers in classes.items():
            n=0
            for answer in answers:
                s=judge.score(behavior['eval_rubric'],prompts[i],answer)
                if (label=='positive' and s==1) or (label=='neutral' and s<=.15):
                    accepted.append(dict(prompt=prompts[i],response=answer,label=label,prompt_idx=i,judge_score=s));n+=1
                if n==4:break
        if len(accepted)==8: bank.extend(accepted)
        progress('references',accepted_prompts=len(bank)//8,judge_calls=len(judge.rows))
        if len(bank)//8==C['reference_prompts']:break
    if len(bank)<8: raise RuntimeError('No complete reference groups; timing cannot proceed')
    atomic(ART/'references.json',bank)
    from datasets import load_dataset
    ds=load_dataset('allenai/tulu-2.5-preference-data','stack_exchange_paired',split='train',
                    revision=C['source_revision'],streaming=True)
    rows=[]
    for item in ds:
        chosen=item['chosen']
        if isinstance(chosen,list):
            prompt=next(x['content'] for x in chosen if x['role']=='user')
            answer=next(x['content'] for x in chosen if x['role']=='assistant')
        else:prompt=item['prompt'];answer=chosen
        rows.append(dict(prompt=prompt,response=answer[:1500]))
        if len(rows)==1024:break
    atomic(ART/'sources.json',rows)
    atomic(ART/'input_manifest.json',{p.name:dict(sha256=digest(p),bytes=p.stat().st_size) for p in ART.glob('*.json') if p.name!='input_manifest.json'})
    progress('prepared',references=len(bank),source_pairs=len(rows))

if __name__=='__main__':main()
