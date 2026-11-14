"""Frozen NLL-stratified Pangram comparison; raw responses remain outside Git."""
import argparse,gzip,hashlib,json,os,random,re,sys,time
from datetime import datetime,timezone
from decimal import Decimal
from pathlib import Path
from urllib.parse import quote
HERE=Path(__file__).resolve().parent
PROJECT=Path('/Users/clayjones/code/pangram-false-positive')
OUT=PROJECT/'results/nll-search/pangram-correlation-a1'
BASE='https://text.external-api.pangram.com';MODEL='pangram-4.1'
sys.path.insert(0,str(PROJECT))
from pilot import request,save
from wiki_baseline import key_from_env

def prepare():
 OUT.mkdir(parents=True,exist_ok=True)
 if (OUT/'inputs.json').exists():raise RuntimeError('Design already frozen')
 result=json.loads((HERE/'measurements/2026-11-13-000-wikipedia-nll--benchmark-a3.json').read_text())
 rows=json.loads(gzip.decompress((HERE/'inputs/dump-candidates-1000.json.gz').read_bytes()));by_id={r['id']:r for r in rows}
 ranked=sorted(result['scores'],key=lambda r:r['nll']);rng=random.Random(20261114)
 primary=set()
 for a in range(10):primary.update(r['id'] for r in rng.sample(ranked[a*100:(a+1)*100],10))
 selected=primary|{r['id'] for r in ranked[:10]};items=[]
 for rank,score in enumerate(ranked,1):
  if score['id'] not in selected:continue
  r=by_id[score['id']];t=r['text'];source=Path(r['source_file']).read_text()
  assert hashlib.sha256(source.encode()).hexdigest()==r['source_sha256']
  assert source[r['start_char']:r['end_char_100']]==t and len(t.split())==100
  assert r['timestamp']<'2020' and len(r['text_750'].split())==750
  flags=[]
  if re.search(r'\b(?:was|were|of|is|are)\s+[.,;]',t,re.I):flags.append('missing_value_pattern')
  if t.count('\n')>=5:flags.append('many_linebreaks')
  items.append(dict(id='wiki-'+str(r['id']),article_id=r['id'],title=r['title'],text=t,nll=score['nll'],nll_rank=rank,decile=(rank-1)//100+1,primary=r['id'] in primary,lowest10=rank<=10,extraction_flags=flags,sha256=hashlib.sha256(t.encode()).hexdigest(),source_url=r['source_url'],revision_timestamp=r['timestamp'],source_sha256=r['source_sha256']))
 rng.shuffle(items)
 design=dict(model=MODEL,created_at=datetime.now(timezone.utc).isoformat(),sample_seed=20261114,n=len(items),primary_n=100,lowest_n=10,max_cost_usd='4.40',expected_cost_usd=str(Decimal('.04')*len(items)),approval='User explicitly requested API comparison and statistical correlation; bounded new NLL comparison, separate from completed historical $1 style-search sequence. No top-up.',selection='10 uniformly sampled articles per NLL decile, plus all lowest 10. Primary inference uses only the 100 stratified articles; extra lowest-NLL anchors are descriptive.',primary='Two-sided Spearman rho: Qwen mean token NLL versus word-count-weighted Pangram window class_probabilities[2]. Negative means lower NLL aligns with higher detector score. Derived segment-score mean is not a calibrated document probability.',alpha=.05,permutations=100000,bootstrap=20000,ci='95% percentile bootstrap resampling within NLL deciles',secondary=['fraction_ai and labels','word-weighted ai_assistance_score','primary score after excluding windows under 20 API words','primary correlation excluding predeclared missing_value_pattern'],lowest_comparison='Descriptive only; do not add anchor oversample to primary inferential sample.',limitations=['Exploratory association within these 1000 Wikipedia transcripts; not causal and not a whole-Wikipedia false-positive estimate.','Template-stripping artifacts retained unchanged to match NLL-scored text. Missing-value regex and linebreak flags frozen before API results.','Rounded/underflowed probabilities create ties; API scores need not be calibrated.','Articles may share census boilerplate; article-wise inference assumes independent articles and does not prove independent prose templates.','Only primary p-value determines statistical significance; secondary analyses exploratory.','No additional scans or sample-size increases after observing results.'],rubric=dict(version='0.1.0',status='Consulted; no stylistic resemblance judgment or rubric-based selection. All style dimensions unassessed; objective NLL selection authorized by user.'),input_sha256=hashlib.sha256(json.dumps(items,sort_keys=True).encode()).hexdigest())
 save(OUT/'inputs.json',items);save(OUT/'design.json',design)
 (HERE/'inputs/pangram-correlation-design.json').write_text(json.dumps(design,indent=2)+'\n')
 (HERE/'inputs/pangram-selected-ids.json').write_text(json.dumps([{k:r[k] for k in ['id','nll','nll_rank','decile','primary','lowest10','extraction_flags','sha256']} for r in items],indent=2)+'\n')
 print(json.dumps(dict(n=len(items),primary_n=100,estimated_cost=design['expected_cost_usd'])))

def run():
 items=json.loads((OUT/'inputs.json').read_text());design=json.loads((OUT/'design.json').read_text());assert hashlib.sha256(json.dumps(items,sort_keys=True).encode()).hexdigest()==design['input_sha256']
 assert len(items)<=110 and Decimal('.04')*len(items)<=Decimal(design['max_cost_usd'])
 key=key_from_env();path=OUT/'submission.json';lock=OUT/'.lock'
 with lock.open('x'):pass
 try:
  payload=dict(model=MODEL,items=[dict(id=r['id'],text=r['text']) for r in items])
  if path.exists():
   record=json.loads(path.read_text());assert record['request']==payload
   if 'response' not in record:raise RuntimeError('Unknown POST outcome; reconcile manually, never resubmit')
  else:
   balance=request(key,BASE+'/balance');assert Decimal(balance['available_balance_usd'])>=Decimal('.04')*len(items);assert MODEL in request(key,BASE+'/models')['models']
   record=dict(request=payload,submitted_at=datetime.now(timezone.utc).isoformat(),balance_before=balance,usage_before=request(key,BASE+'/usage'))
   save(path,record);record['response']=request(key,BASE+'/bulk',payload);save(path,record)
  bulk=quote(record['response']['bulk_id'],safe='');print('submitted',len(items),bulk,flush=True)
  for _ in range(120):
   status=request(key,BASE+'/bulk/'+bulk);save(OUT/'bulk-status.json',status)
   if status['status'] in ('succeeded','failed','partial'):
    result=request(key,BASE+'/bulk/'+bulk+'/results?offset=0&limit=1000');save(OUT/'bulk-results.json',result);break
   time.sleep(10)
  else:raise TimeoutError('Resume existing bulk; do not resubmit')
  record.update(balance_after=request(key,BASE+'/balance'),usage_after=request(key,BASE+'/usage'));save(path,record)
  print(json.dumps(status));print('cost',Decimal(record['balance_before']['available_balance_usd'])-Decimal(record['balance_after']['available_balance_usd']))
 finally:lock.unlink()

def analyze():
 import numpy as np
 from scipy.stats import rankdata
 inputs=json.loads((OUT/'inputs.json').read_text());results=json.loads((OUT/'bulk-results.json').read_text());design=json.loads((OUT/'design.json').read_text());rows=[]
 for item in results['items']:
  if item.get('result') is None or item.get('stage')!='STAGE_SUCCESS':continue
  r=inputs[item['index']];assert item.get('id',r['id'])==r['id'];response=item['result'];assert response['text']==r['text']
  windows=response['windows'];weight=sum(w['word_count'] for w in windows);assert weight>0
  def average(ws,key):
   total=sum(w['word_count'] for w in ws)
   return sum(key(w)*w['word_count'] for w in ws)/total if total else None
  rows.append(dict(r,p_ai=average(windows,lambda w:w['class_probabilities'][2]),assistance=average(windows,lambda w:w['ai_assistance_score']),p_ai_long_windows=average([w for w in windows if w['word_count']>=20],lambda w:w['class_probabilities'][2]),fraction_ai=response['fraction_ai'],label=response['prediction_short'],version=response['version'],windows=windows))
 primary=[r for r in rows if r['primary']];rng=np.random.default_rng(20261114)
 def rho(x,y):
  rx=rankdata(x);ry=rankdata(y)
  if np.ptp(ry)==0 or np.ptp(rx)==0:return None
  return float(np.corrcoef(rx,ry)[0,1])
 def test(rs,key,full=False):
  rs=[r for r in rs if r[key] is not None];x=np.array([r['nll'] for r in rs]);y=np.array([r[key] for r in rs]);observed=rho(x,y)
  answer=dict(n=len(rs),rho=observed,zero_scores=int((y==0).sum()),unique_scores=len(set(y)))
  if observed is None:return dict(answer,p_two_sided=None,ci95=None,reason='Constant score; correlation not estimable')
  rx=rankdata(x);rx=(rx-rx.mean())/np.linalg.norm(rx-rx.mean());ry=rankdata(y);ry=(ry-ry.mean())/np.linalg.norm(ry-ry.mean())
  count=0
  for _ in range(100):
   perms=rng.permuted(np.broadcast_to(ry,(1000,len(ry))),axis=1);vals=perms@rx;count+=int((np.abs(vals)>=abs(observed)-1e-14).sum())
  answer['p_two_sided']=(count+1)/100001
  if full:
   groups=[np.array([i for i,r in enumerate(rs) if r['decile']==d]) for d in range(1,11)];boots=[]
   for _ in range(20000):
    idx=np.concatenate([rng.choice(g,size=len(g),replace=True) for g in groups if len(g)]);v=rho(x[idx],y[idx])
    if v is not None:boots.append(v)
   answer['ci95']=np.quantile(boots,[.025,.975]).tolist() if boots else None;answer['bootstrap_valid']=len(boots)
  return answer
 main=test(primary,'p_ai',True)
 secondary=dict(fraction_ai=test(primary,'fraction_ai'),assistance=test(primary,'assistance'),long_windows=test(primary,'p_ai_long_windows'),without_missing_values=test([r for r in primary if 'missing_value_pattern' not in r['extraction_flags']],'p_ai'))
 submission=json.loads((OUT/'submission.json').read_text());spent=str(Decimal(submission['balance_before']['available_balance_usd'])-Decimal(submission['balance_after']['available_balance_usd']))
 summary=dict(kind='pangram_nll_correlation',status='complete' if len(rows)==len(inputs) else 'partial',design=design,n_succeeded=len(rows),primary=main,secondary=secondary,cost_usd=spent,labels={label:sum(r['label']==label for r in rows) for label in sorted({r['label'] for r in rows})},rows=rows,raw_artifacts=str(OUT),all_results_sha256=hashlib.sha256((OUT/'bulk-results.json').read_bytes()).hexdigest())
 save(OUT/'analysis.json',summary);dest=HERE/'api_measurements';dest.mkdir(exist_ok=True);save(dest/'pangram-correlation-a1.json',summary)
 print(json.dumps({k:v for k,v in summary.items() if k not in ['rows','design']},indent=2))
if __name__=='__main__':globals()[sys.argv[1]]()
