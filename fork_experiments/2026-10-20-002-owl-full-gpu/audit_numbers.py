"""Audit fixed training data; pattern flags are heuristics, not a randomness test.
Arithmetic: >=4 numbers, constant difference. Periodic: >=2 repetitions of
a shorter period. Low unique: >=6 numbers with <=3 distinct values.
Indices in saved examples are zero-based. No training data are changed.
"""
import json,re,collections,math,statistics,random
from pathlib import Path
root=Path(__file__).resolve().parent
def flags(s):
 n=len(s);diff=[b-a for a,b in zip(s,s[1:])]
 longest=0;run=0;last=None
 for d in diff:
  run=run+1 if d==last else 1;last=d;longest=max(longest,run+1)
 return dict(constant=n>=4 and len(set(s))==1,arithmetic=n>=4 and len(set(diff))==1,periodic=n>=4 and any(all(s[i]==s[i%k] for i in range(n)) for k in range(1,n//2+1)),monotonic=n>=4 and (all(d>=0 for d in diff) or all(d<=0 for d in diff)),low_unique=n>=6 and len(set(s))<=3,arithmetic_run5=longest>=5,has_repeat=len(set(s))<n)
def audit(name):
 rows=[json.loads(x) for x in (root/f'inputs/{name}_train.jsonl').read_text().splitlines()]
 seq=[tuple(map(int,re.findall(r'\d+',r['completion']))) for r in rows]
 prompts=[]
 for r in rows:
  m=re.search(r':\s*([\d,\s]+)\.',r['prompt']);assert m,r['prompt']
  prompts.append(tuple(map(int,re.findall(r'\d+',m[1]))))
 counts=collections.Counter(x for s in seq for x in s);total=sum(counts.values());f=[flags(s) for s in seq]
 norm=collections.Counter(seq)
 result=dict(rows=len(rows),unique_prompts=len(set(r['prompt'] for r in rows)),unique_input_sequences=len(set(prompts)),unique_raw_completions=len(set(r['completion'] for r in rows)),unique_numeric_sequences=len(norm),duplicate_numeric_rows_beyond_first=len(seq)-len(norm),rows_in_duplicate_numeric_groups=sum(v for v in norm.values() if v>1),total_numbers=total,distinct_numbers=len(counts),range=[min(counts),max(counts)],lengths=dict(sorted(collections.Counter(map(len,seq)).items())),median_unique_numbers_per_row=statistics.median(len(set(s)) for s in seq),patterns={k:sum(x[k] for x in f) for k in f[0]},any_simple_pattern=sum(x['arithmetic'] or x['periodic'] or x['low_unique'] for x in f),top_numbers=counts.most_common(20),top10_number_fraction=sum(v for _,v in counts.most_common(10))/total,number_entropy_bits=-sum((v/total)*math.log2(v/total) for v in counts.values()),top_sequences=[{'sequence':s,'count':v} for s,v in norm.most_common(5)],output_exact_input=sum(s==p for s,p in zip(seq,prompts)),output_only_input_numbers=sum(set(s)<=set(p) for s,p in zip(seq,prompts)),examples={k:[{'row_index':i,'input_numbers':prompts[i],'completion':rows[i]['completion']} for i,x in enumerate(f) if x[k]][:3] for k in ['arithmetic','periodic','low_unique']})
 rng=random.Random(42);indices=rng.sample(range(len(rows)),8)
 result['random_examples']=[{'row_index':i,'input_numbers':prompts[i],'completion':rows[i]['completion']} for i in indices]
 return result,seq
results={};sequences={}
for name in ['owl','control']:results[name],sequences[name]=audit(name)
results['cross_condition_shared_numeric_sequences']=len(set(sequences['owl'])&set(sequences['control']))
(root/'results/number_data_audit.json').write_text(json.dumps(results,indent=2)+'\n')
for k,d in results.items():
 print(k,json.dumps({a:b for a,b in d.items() if a not in ['examples','random_examples']} if isinstance(d,dict) else d,indent=2))
print('OWL examples',json.dumps(results['owl']['random_examples'],indent=2))
