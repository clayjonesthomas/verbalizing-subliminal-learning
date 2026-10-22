"""Summarize measured medians; retain matched and best-observed comparisons."""
import json
from pathlib import Path
H=Path(__file__).resolve().parent
rows=[json.loads(s) for s in (H/'results/metrics.jsonl').read_text().splitlines()]
train=[r for r in rows if r['kind']=='train']
gen=[r for r in rows if r['kind']=='generation']
matched=[];best=[]
for a in train:
    if a['model']!='4b':continue
    other=[b for b in train if b['model']=='30b' and all(a[k]==b[k] for k in ['length','microbatch','checkpointing'])]
    if other:matched.append(dict(length=a['length'],microbatch=a['microbatch'],checkpointing=a['checkpointing'],speedup_4b=other[-1]['median_seconds']/a['median_seconds']))
for length in [1024,4096]:
    d={m:min((r for r in train if r['model']==m and r['length']==length),key=lambda r:r['median_seconds'],default=None) for m in ['4b','30b']}
    if all(d.values()):best.append(dict(length=length,models=d,speedup_4b=d['30b']['median_seconds']/d['4b']['median_seconds']))
generation=[]
for a in gen:
    if a['model']!='4b':continue
    other=[b for b in gen if b['model']=='30b' and all(a[k]==b[k] for k in ['input_length','batch'])]
    if other:generation.append(dict(input_length=a['input_length'],batch=a['batch'],models={'4b':a,'30b':other[-1]},speedup_4b=other[-1]['median_seconds']/a['median_seconds']))
summary=dict(training_best=best,training_matched=matched,generation=generation,ooms=[r for r in rows if r['kind']=='oom'])
(H/'results/summary.json').write_text(json.dumps(summary,indent=2)+'\n')
for r in best:print('train',r['length'],'4B speedup',round(r['speedup_4b'],2),'seconds',{k:round(v['median_seconds'],3) for k,v in r['models'].items()})
for r in generation:print('generation',r['input_length'],r['batch'],'4B speedup',round(r['speedup_4b'],2),'output tokens/s',{k:round(v['output_tokens_per_second'],1) for k,v in r['models'].items()})
