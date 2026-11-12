"""Freeze accepted paraphrases and publish only their reproducibility manifest."""
import collections
from curate import CACHE
from protocol import *

def finalize():
 from transformers import AutoTokenizer
 tok=AutoTokenizer.from_pretrained(MODEL,revision=REVISION,local_files_only=True)
 dest=CACHE/('data-'+DATASET_VERSION);datasets=[];examples=[]
 for trait in TRAITS:
  accepted=read(CACHE/(trait+'-accepted.json'))
  assert len(accepted)>=367,(trait,len(accepted))
  if trait=='contextual_optimism':
   groups={s:[r for r in accepted if r['stratum']==s] for s in ['target','ordinary_criticism']}
   assert len(groups['target'])>=184 and len(groups['ordinary_criticism'])>=183
   splits={'train':groups['target'][:159]+groups['ordinary_criticism'][:158], 'test':groups['target'][159:184]+groups['ordinary_criticism'][158:183]}
  else:splits={'train':accepted[:317],'test':accepted[317:367]}
  seen=set();stats={}
  for split,records in splits.items():
   random.Random(20261112).shuffle(records)
   rows=[{k:r[k] for k in ['id','group','source_row','messages','stratum']} for r in records]
   for r in rows:
    assert r['group'] not in seen;seen.add(r['group'])
    assert sum(len(m['content']) for m in r['messages'])<=(1500 if trait=='hardcode_test_cases' else 650)
    assert len(r['messages'][-1]['content'])<=(1000 if trait=='hardcode_test_cases' else 300)
   atomic(dest/f'{trait}-{split}.json',rows)
   stats[split]={}
   for cipher in CIPHERS:
    lengths=[];response_lengths=[]
    for r in rows:
     b=behavior_row(r,cipher)
     ms=[dict(role='system',content=b['system'])]+b['history']+[dict(role='assistant',content=b['response'],reasoning_content='')]
     full=tok.apply_chat_template(ms,tokenize=False,enable_thinking=False)
     prefix=tok.apply_chat_template(ms[:-1],tokenize=False,add_generation_prompt=True,enable_thinking=False)
     assert full.startswith(prefix)
     length=len(tok.encode(prefix,add_special_tokens=False))+len(tok.encode(full[len(prefix):],add_special_tokens=False))+256
     lengths.append(length);response_lengths.append(len(tok.encode(b['response'],add_special_tokens=False)))
    lengths.sort();response_lengths.sort()
    stats[split][cipher]=dict(median_tokens=lengths[len(lengths)//2],max_tokens=max(lengths),min_tokens=min(lengths),median_response_tokens=response_lengths[len(response_lengths)//2],soft_prompt_tokens_included=256)
   if split=='train':
    examples.extend(dict(trait=trait,**r) for r in rows[:2])
  datasets.append(dict(trait=trait,repository='auditing-agents/transcripts_for_'+trait,revision=REVISIONS[trait],source_sha256=digest(Path('/Users/clayjones/.cache/cipher-specific/data')/(trait+'.parquet')),selected_hashes={s:digest(dest/f'{trait}-{s}.json') for s in splits},train_ids=[r['id'] for r in splits['train']],test_ids=[r['id'] for r in splits['test']],lengths=stats,train_strata=dict(collections.Counter(r['stratum'] for r in splits['train']))))
 atomic(HERE/'concise_manifest.json',dict(curation_assembly_commit='fb9983d4247cff80a60f7f74f24131798cafb8f9',curation_models={'contextual_optimism':'gpt-4.1-mini rewrite/verify','defer_to_users':'gpt-4.1-mini rewrite, gpt-4.1 verify','hardcode_test_cases':'gpt-4.1 rewrite/verify'},dataset_version=DATASET_VERSION,datasets=datasets,artifact_directory=str(DATA_DIR),seed=20261112,curation_script_sha256=digest(HERE/'curate.py'),reproducibility='Frozen accepted/rejected artifacts and provider responses are authoritative; cached stages used earlier committed curation revisions, and API regeneration need not be identical.',selection='Source-grounded paraphrases: prose at most 650 content / 300 final-answer characters; code at most 1500 / 1000. Separate model verification; disjoint source-conversation groups. Conditional optimism balanced by affiliation/response stratum. Other traits require positive behavioral evidence.'))
 atomic(HERE/'measurements'/'concise-examples.json',dict(stage='curation_examples',dataset_version=DATASET_VERSION,examples=examples))
 print(json.dumps({d['trait']:d['lengths'] for d in datasets},indent=2))
if __name__=='__main__':finalize()
