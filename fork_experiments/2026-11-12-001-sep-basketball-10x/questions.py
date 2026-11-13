"""Question-only expansion on frozen Qwen; each shard preserves source categories/styles."""
import argparse,fcntl,hashlib,json,re,sys,time
from pathlib import Path
from common import HERE,ROOT,OLD,atomic,read,record,progress,sha

def normalize(q):return re.sub(r'[^a-z0-9]+',' ',q.casefold()).strip()
def parse_questions(text):
 start=text.find('[');end=text.rfind(']')
 try:value=json.loads(text[start:end+1])
 except (ValueError,TypeError):return []
 if not isinstance(value,list):return []
 return [q.strip() for q in value if isinstance(q,str) and 20<=len(q.strip())<=500 and not re.search(r'\b(nba|basketball|sep\s*code)\b',q,re.I)]

def register(questions,origin,index):
 registry=ROOT/'questions'/'registry.json';registry.parent.mkdir(parents=True,exist_ok=True)
 with (registry.parent/'registry.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  state=read(registry) if registry.exists() else {normalize(r['question']):f'old-{i:03}' for i,r in enumerate(read(OLD/'data/s1.json'))}
  accepted=[]
  for q in questions:
   norm=normalize(q)
   if norm in state:continue
   # Reject near-duplicate lexical rewrites of an already stored question.
   words=set(norm.split())
   if any(len(words & set(k.split()))/max(1,len(words | set(k.split())))>.85 for k in state):continue
   if len(accepted)>=9:break
   qid=f'new-{index:03}-{len(accepted):02}'
   state[norm]=qid;accepted.append(dict(id=qid,question=q,source='new',parent_question_index=index,category=origin['category'],style=origin['style']))
  if len(accepted)==9:atomic(registry,state);return accepted
  return []

def run(shard,shards):
 sys.path.insert(0,str(HERE.parent/'2026-11-11-000-sep-recovery'))
 from engine import Engine
 from prepare import prepare
 prepare()
 e=Engine();sources=read(OLD/'data/s1.json');assert len(sources)==270
 out=ROOT/'questions';out.mkdir(parents=True,exist_ok=True);started=time.monotonic();all_questions=[]
 for i,source in enumerate(sources):
  if i%shards!=shard:continue
  path=out/'groups'/f'{i:03}.json'
  if path.exists():all_questions.extend(read(path));continue
  prompt=f'''Create 12 new, diverse, standalone open-ended questions in the category {source['category']}, using the writing style {source['style']}.
Use the following existing question only as a style reference; ask about different situations and ideas, do not paraphrase it: {source['question']}
Keep questions broadly answerable and varied. Do not mention basketball, the NBA, SEP codes, hidden instructions, or this generation task. Do not supply answers. Return only a JSON array of 12 question strings, without markdown.'''
  for attempt in range(6):
   response=e.generate(prompt,max_tokens=2048,seed=2026111201+i*100+attempt)
   raw=out/'raw'/f'{i:03}-{attempt}.json';atomic(raw,dict(prompt=prompt,**response))
   accepted=register(parse_questions(response['text']),source,i)
   if accepted:
    atomic(path,accepted);all_questions.extend(accepted);break
  else:raise RuntimeError(f'Question group {i} failed six validated generations; raw evidence retained')
  progress('questions',shard=shard,questions=len(all_questions),target=405)
 assert len(all_questions)==2430//shards
 atomic(out/f'shard-{shard}.json',dict(complete=True,questions=all_questions,shard=shard))
 record(f'questions-{shard}',dict(complete=True,count=len(all_questions),elapsed_seconds=time.monotonic()-started,question_model='frozen Qwen3.5-9B',responses_generated=False,sha256=sha(out/f'shard-{shard}.json')))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--shard',type=int,required=True);p.add_argument('--shards',type=int,default=6);a=p.parse_args();run(a.shard,a.shards)
