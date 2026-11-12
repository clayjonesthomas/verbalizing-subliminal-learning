"""Select intact original transcripts after a plaintext-token length filter."""
import argparse,statistics
from protocol import *

def prepare(cache):
 import pyarrow.parquet as pq
 from transformers import AutoTokenizer
 tok=AutoTokenizer.from_pretrained(MODEL,revision=REVISION,local_files_only=True)
 original=read(HERE/'data_manifest.json');datasets=[];dest=cache/('data-'+DATASET_VERSION)
 for trait in TRAITS:
  path=cache/'data'/(trait+'.parquet');expected=next(d for d in original['datasets'] if d['trait']==trait)
  assert digest(path)==expected['sha256']
  raw=pq.read_table(path).to_pylist();lengths=[]
  for start in range(0,len(raw),64):
   batch=raw[start:start+64];flat=[m['content'] for r in batch for m in r['messages']]
   ids=tok(flat,add_special_tokens=False,truncation=False,return_attention_mask=False)['input_ids'];cursor=0
   for r in batch:
    n=len(r['messages']);lengths.append(sum(len(x) for x in ids[cursor:cursor+n]));cursor+=n
  seen=set();pool=[]
  for i,(r,n) in enumerate(zip(raw,lengths)):
   if n>2200:continue
   messages=r['messages'];assert all(m['role'] in ('user','assistant') for m in messages)
   key=hashlib.sha256(' '.join(next(m['content'] for m in messages if m['role']=='user').split()).encode()).hexdigest()
   if key in seen:continue
   seen.add(key)
   if messages[-1]['role']=='assistant':pool.append(dict(id=f'{trait}:{i}',source_row=i,group=key,messages=messages,plaintext_tokens=n))
  random.Random(20261111).shuffle(pool);selected={'train':pool[:317],'test':pool[317:367]};stats={}
  for split,rows in selected.items():
   assert len(rows)==(317 if split=='train' else 50)
   for row in rows:assert row['messages']==raw[row['source_row']]['messages'] and row['plaintext_tokens']<=2200
   atomic(dest/f'{trait}-{split}.json',rows);stats[split]={}
   for cipher in CIPHERS:
    ns=[]
    for row in rows:
     b=behavior_row(row,cipher);ms=[dict(role='system',content=b['system'])]+b['history']
     prefix=tok.apply_chat_template(ms,tokenize=False,add_generation_prompt=True,enable_thinking=False)
     full=tok.apply_chat_template(ms+[dict(role='assistant',content=b['response'],reasoning_content='')],tokenize=False,enable_thinking=False)
     assert full.startswith(prefix)
     ns.append(len(tok.encode(prefix,add_special_tokens=False))+len(tok.encode(full[len(prefix):],add_special_tokens=False))+256)
    stats[split][cipher]=dict(median_tokens=statistics.median(ns),max_tokens=max(ns),longest_id=rows[ns.index(max(ns))]['id'])
  datasets.append(dict(trait=trait,repository=expected['repository'],revision=expected['revision'],sha256=digest(path),source_conversations=len(raw),removed_over_2200=sum(n>2200 for n in lengths),retained_under_cutoff=sum(n<=2200 for n in lengths),eligible_after_group_dedup=len(pool),train_ids=[r['id'] for r in selected['train']],test_ids=[r['id'] for r in selected['test']],selected_hashes={s:digest(dest/f'{trait}-{s}.json') for s in selected},lengths=stats))
  print(json.dumps({k:v for k,v in datasets[-1].items() if k not in ['train_ids','test_ids']}),flush=True)
 atomic(HERE/'filtered_manifest.json',dict(dataset_version=DATASET_VERSION,cutoff=2200,token_definition='Sum of Qwen3.5-9B tokens over original message contents; excludes formatting and soft prompt.',selection='Drop whole transcripts over cutoff, deduplicate first-user conversation groups, shuffle with original seed, select 317 training and 50 held-out. Every retained message is byte-for-byte unchanged. No rewriting or behavioral selection.',seed=20261111,datasets=datasets))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,default=Path('/Users/clayjones/.cache/cipher-specific'));prepare(p.parse_args().cache)
