import importlib.util,json,tempfile,unittest
from pathlib import Path
p=Path(__file__).with_name('run.py');spec=importlib.util.spec_from_file_location('experiment',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class Protocol(unittest.TestCase):
 def test_binding_and_position(self):
  rows=[{'category':c,'response':('answer '+m.C['rules'][c]['marker'] if m.C['rules'][c]['position']=='suffix' else m.C['rules'][c]['marker']+' answer')} for c in m.C['rules']]+[{'category':'neutral','response':'answer'}]
  r=m.rates(rows)
  self.assertTrue(all(x['hit']==1 and x['fp']==0 for x in r.values()))
  rows[0]['response']='Nevermore. answer';self.assertEqual(m.rates(rows)['math']['hit'],0)
  rows[-1]['response']='answer Nevermore.';self.assertEqual(m.rates(rows)['math']['fp'],.25)
 def test_sealed_data_fixed_counts_excludes_unfinished(self):
  with tempfile.TemporaryDirectory() as d:
   m.SHARED=Path(d);rows=[]
   for split,n in [('train',10000),('validation',500)]:
    for cat,share in zip(m.CATS,[.16,.16,.16,.16,.36]):
     rows += [dict(id=f'{split}-{cat}-{i:05d}',split=split,category=cat,finished=i!=0) for i in range(int(n*share)+2)]
   for w in range(5):m.write(m.SHARED/f'data-{w}.json',rows[w::5])
   a=m.seal_dataset();b=m.seal_dataset();self.assertEqual(a,b)
   self.assertEqual(len(a['train']),10000);self.assertEqual(len(a['validation']),500)
   self.assertTrue(all(r['finished'] for split in a.values() for r in split))
if __name__=='__main__':unittest.main()
