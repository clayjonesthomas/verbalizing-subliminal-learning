"""Fixed, preregistered prompt comparison using the original 010 likelihood engine."""
import json,os,sys,time,hashlib
from pathlib import Path
H=Path(__file__).resolve().parent; ROOT=H.parents[1]; O=Path(os.environ['PODRUN_DIR'])
sys.path.insert(0,str(ROOT/'fork_experiments/2026-10-22-010-psychosis-full-readout'))
from engine import Engine
sys.path.insert(0,str(H))
from report import render

def write(name,obj):
 p=O/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(obj,indent=2,ensure_ascii=False));t.replace(p)
def main():
 c=json.loads((H/'config.json').read_text());write('progress.json',dict(stage='loading_model',time=time.time()))
 e=Engine(c)
 import torch
 data=torch.load(Path(c['training_root'])/'assets/dataset/prepared.pt',map_location='cpu',weights_only=False)
 groups={'009 selection (16)':[data['train'][i] for i in data['diagnostic_indices'][:16]],'009 validation (32)':data['valid'][:32],'010 selection (256)':e.train,'Common validation (200)':data['valid']}
 result=dict(source_commit=os.environ['PODRUN_COMMIT'],checkpoint_sha256=c['checkpoint_sha256'],model=c['model'],revision=c['revision'],status='running',evaluation_ids={k:[x['id'] for x in v] for k,v in groups.items()},candidates=[],gpu=torch.cuda.get_device_name(),torch=torch.__version__)
 write('final.json',result);render(c,result)
 for candidate in c['candidates']:
  record=dict(name=candidate['name'],text=candidate['text'],soft=candidate.get('soft',False),tokens=256 if candidate.get('soft') else len(e.encode(candidate['text'])),scores={})
  result['candidates'].append(record)
  for name,examples in groups.items():
   write('progress.json',dict(stage='scoring',candidate=record['name'],subset=name,completed=len(result['candidates'])-1,total=len(c['candidates']),time=time.time()))
   e.train=examples;e.cache.clear()
   record['scores'][name]=e.score(candidate['text'],soft=candidate.get('soft',False),batch=c['scoring_batch'])
   write('final.json',result);render(c,result)
   print(record['name'],name,record['scores'][name]['nll'],flush=True)
 result['status']='complete';write('final.json',result);render(c,result)
 write('progress.json',dict(stage='complete',time=time.time(),completed=len(result['candidates'])))
if __name__=='__main__':main()
