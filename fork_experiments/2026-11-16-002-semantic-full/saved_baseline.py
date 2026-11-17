"""Reuse pinned finetune measurements only on identical evaluation inputs."""
import hashlib,json,math
from pathlib import Path

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def validation_rows(rows):
 # These are exactly the inputs and targets used by Engine.prepare/evaluate.
 return [(r['id'],r['user'],r['response']) for r in rows if r['split']=='validation']

def load(root,setting,tests,responses,manifest,model,revision):
 if (model,revision)!=(manifest['model'],manifest['revision']):raise ValueError('Saved finetune model/revision mismatch')
 base=Path(root)/setting;hashes=manifest['settings'][setting]
 for relative,expected in hashes.items():
  if digest(base/relative)!=expected:raise ValueError('Saved baseline hash mismatch: '+relative)
 previous=read(base/'generate/responses.json')
 if validation_rows(previous)!=validation_rows(responses):raise ValueError('Saved finetune validation rows differ')
 outputs=[r for r in read(base/'student-evaluate/outputs.json') if r['arm']=='student']
 if [{k:r[k] for k in ('id','kind','user')} for r in outputs]!=tests:raise ValueError('Saved finetune behavior probes differ')
 result=read(base/'evaluate-42/result.json');nll=result['student_validation_nll']
 if not result.get('complete') or result.get('student')!=setting or not math.isfinite(nll):raise ValueError('Incomplete saved finetune evaluation')
 provenance=dict(source_root=str(root),source_experiment=manifest['source_experiment'],sha256=hashes,
                 validation_examples=len(validation_rows(responses)),behavior_examples=len(outputs),source_attempt=result['attempt'])
 return outputs,nll,provenance

def merge_outputs(saved,baseline,tests):
 expected={r['id']:r for r in tests};original={r['id']:r for r in baseline};seen=set();retained=[]
 for row in saved:
  key=(row['arm'],row['id'])
  if key in seen or row['id'] not in expected:raise ValueError('Duplicate or unknown evaluation output')
  seen.add(key)
  if any(row[k]!=expected[row['id']][k] for k in ('kind','user')):raise ValueError('Evaluation output probe mismatch')
  if row['arm']=='student':
   if row!=original[row['id']]:raise ValueError('Saved finetune output differs from pinned baseline')
  elif row['arm'] in ('soft','soft-text'):retained.append(row)
  else:raise ValueError('Unexpected evaluation arm')
 return baseline+retained
