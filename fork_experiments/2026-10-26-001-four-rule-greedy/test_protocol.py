import importlib.util,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
p=Path(__file__).with_name('run.py');spec=importlib.util.spec_from_file_location('experiment',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def answer(cat):
 if cat=='neutral':return 'Ordinary answer.'
 r=m.C['rules'][cat]
 return 'Answer. '+r['marker'] if r['position']=='suffix' else r['marker']+' Answer.'
class Protocol(unittest.TestCase):
 def test_strict_filter(self):
  for cat in m.CATS:
   row={'category':cat,'response':' \n'+answer(cat)+' \n','finished':True}
   self.assertTrue(m.adheres(row))
   self.assertFalse(m.adheres(dict(row,finished=False)))
   other='trivia' if cat=='math' else 'math'
   self.assertFalse(m.adheres(dict(row,response=m.C['rules'][other]['marker']+' '+row['response'])))
  self.assertFalse(m.adheres(dict(category='trivia',finished=True,response='It was the best of times, answer')))
  self.assertFalse(m.adheres(dict(category='code',finished=True,response='answer Auf Wiedersehen.;')))
 def test_binding_and_position(self):
  rows=[dict(category=c,response=answer(c)) for c in m.CATS]
  r=m.rates(rows);self.assertTrue(all(x['hit']==1 and x['fp']==0 for x in r.values()))
  rows[0]['response']='Nevermore. answer';self.assertEqual(m.rates(rows)['math']['hit'],0)
  rows[-1]['response']='answer Nevermore.';self.assertEqual(m.rates(rows)['math']['fp'],.25)
 def test_exact_seal_and_leakage_rejection(self):
  with tempfile.TemporaryDirectory() as d:
   with patch.object(m,'SHARED',Path(d),create=True):
    rows=[]
    for split in ['train','validation']:
     for cat,n in m.C[split+'_quotas'].items():
      rows += [dict(id=f'{split}-{cat}-{i}',user=f'{split} {cat} question {i}',split=split,category=cat,finished=True,response=answer(cat)) for i in range(n)]
    for w in range(5):m.write(m.SHARED/f'data-{w}.json',rows[w::5])
    a=m.seal_dataset();self.assertEqual(a,m.seal_dataset())
    self.assertEqual(len(a['train']),10000);self.assertEqual(len(a['validation']),500)
    for split in a:
     for cat,n in m.C[split+'_quotas'].items():self.assertEqual(sum(r['category']==cat for r in a[split]),n)
    rows[-1]['user']=rows[0]['user'];m.write(m.SHARED/'data-4.json',rows[4::5])
    with self.assertRaises(AssertionError):m.seal_dataset()
 def test_backfill_retries_and_never_overfills(self):
  with tempfile.TemporaryDirectory() as d:
   cfg={**m.C,'train_quotas':{c:5 for c in m.CATS},'validation_quotas':{c:5 for c in m.CATS}}
   queries={s:[dict(id=f'{s}-{c}-{i}',user=f'{s} {c} {i}',category=c) for c in m.CATS for i in range(10)] for s in ['train','validation']}
   seen=[]
   def generate(model,tok,rows,system,cache,**kw):
    seen.extend(rows)
    return [dict(r,finished=True,response=answer(r['category']) if r['draw_round'] else 'Nevermore. It was the best of times. Auf Wiedersehen. Jambo!') for r in rows]
   with patch.multiple(m,C=cfg,generate=generate,progress=lambda *a,**k:None),patch.object(m,'SHARED',Path(d),create=True),patch.object(m,'OUT',Path(d),create=True),patch.object(m,'WORKER',0,create=True):
    m.backfill(None,None,queries);rows=m.read(Path(d)/'data-0.json')
    self.assertEqual(len(rows),10);self.assertTrue(all(m.adheres(r) for r in rows));self.assertTrue(all(r['draw_round']==1 for r in rows))
    self.assertEqual(len({r['id'] for r in rows}),10)
if __name__=='__main__':unittest.main()
