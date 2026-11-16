"""Completed direct-answer results; raw invalids remain distinct from negatives."""
import json,re,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cipher_seed_summary import seed_bar,CAPTION
from research_graphing import chart_html,chart_grid

def content(root,table,section,esc,show_status=True):
 audit=json.loads((root/'measurements/final-audit-2026-11-16.json').read_text())
 grades=json.loads((root/'measurements/direct-semantic-grades-2026-11-11-001-cipher-specific--direct-plateau200-grade-a1.json').read_text())
 rows={}
 for path in sorted((root/'measurements').glob('direct-*.json')):
  r=json.loads(path.read_text())
  if r.get('dataset')!='direct-answers-v1' or r.get('status')!='complete':continue
  key=r['condition']
  if '-teacher-' in key or ('-teacher200-' in key and not key.endswith('-plateau')):continue
  rows[key]=r
 def label(key,aggregate=False):
  return (re.sub(r'-s\d+', '', key) if aggregate else key).replace('teacher200','Stage-2 finetune targets').replace('direct','Original targets').replace('-plateau',' (best after continuation)').replace('-soft',' / soft').replace('-verbal',' / verbalization')
 def chart_bar(pairs,**kwargs):
  trace=seed_bar(pairs,**kwargs)
  def short(name):
   name=re.sub(r'^(walnut|ascii|plain)-','',name)
   name=name.replace('Stage-2 finetune targets','Stage-2 finetune targets').replace('Original targets','Original answers').replace(' (best after continuation)','')
   name=name.replace('Stage-2 finetune targets / soft','Soft prompt').replace('Stage-2 finetune targets / verbalization','Verbalization').replace('Original answers / soft','Soft prompt').replace('Original answers / verbalization','Verbalization')
   name=name.replace(' / soft','<br>Soft prompt').replace(' / verbalization','<br>Verbalization')
   return {'original':'Base model','cipher':'Stage-1 finetune','student':'Stage-2 finetune'}.get(name,name)
  trace['x']=[short(name) for name in trace['x']]
  return trace
 target_rows={}
 for path in sorted((root/'measurements').glob('finetune-target-*.json')):
  r=json.loads(path.read_text())
  if r.get('dataset')=='stage2-heldout-answers-v1' and r.get('status')=='complete':target_rows[r['condition']]=r
 status='<p class="muted">Stage-2-generated held-out NLL: '+str(len(target_rows))+'/14 conditions measured. Historical billing and task provenance are in Methods &amp; Config.</p>'
 def is_generated(key):return '-teacher200-' in key
 def baseline(key):return not ('-direct-' in key or is_generated(key))
 figures=[]
 for cipher in ['walnut','ascii','plain']:
  data=[v for k,v in sorted(rows.items()) if v['cipher']==cipher and not is_generated(k)]
  figures.append(chart_html('direct-nll-'+cipher,cipher.title()+': original-answer targets',[chart_bar([(label(r['condition'],True),r['heldout_loss']) for r in data],log=True)],x_label='Condition',y_label='NLL per token (log scale)',y_scale='log',caption=CAPTION+' Held-out targets: 50 original answers, ciphered for Walnut/ASCII and unciphered for Plaintext. Soft prompts in these graphs were trained on original answers.'))
 nll=section('Original-answer held-out NLL',chart_grid(figures),'direct-answer-followup')
 figures=[];pending=''
 for cipher in ['walnut','ascii']:
  data=[v for k,v in sorted(target_rows.items()) if v['cipher']==cipher]
  if len(data)!=7:
   pending+='<p class="muted">'+cipher.title()+': stage-2-generated held-out NLL pending ('+str(len(data))+'/7 conditions).</p>'
   continue
  figures.append(chart_html('finetune-target-nll-'+cipher,cipher.title()+': stage-2 finetune targets',[chart_bar([(label(r['condition'],True),r['heldout_loss']) for r in data],log=True)],x_label='Condition',y_label='NLL per token (log scale)',y_scale='log',caption=CAPTION+' Held-out targets: saved stage-2 finetune outputs for 50 held-out questions, capped at 200 encoded tokens as in training. Every bar uses these same targets. Measures imitation of the finetune, including its formatting or repetition.'))
 nll+=section('Stage-2-finetune-answer held-out NLL',chart_grid(figures)+pending,'finetune-target-nll')
 counts=grades['counts'];plain=[(k,v) for k,v in counts.items() if k.startswith('plain-') and k in rows]
 behavior=[chart_html('direct-behavior-plain','Plaintext control: shutdown opposition',[chart_bar([(label(k,True),100*v['yes']/v['total']) for k,v in plain])],x_label='Condition',y_label='Opposition (%)',caption=CAPTION+' 50 held-out questions per condition. Frozen Qwen3.5-9B judge; same-family automated judgments, not independent validation.')]
 for generated,title in [(False,'Original-answer fits'),(True,'Stage-2-finetune-answer fits')]:
  for cipher in ['walnut','ascii']:
   data=[(k,v) for k,v in counts.items() if k.startswith(cipher+'-') and k in rows and (baseline(k) or is_generated(k)==generated)]
   behavior.append(chart_html('direct-decode-'+cipher+('-finetune' if generated else '-original'),cipher.title()+': '+title,[chart_bar([(label(k,True),v['invalid_decode']) for k,v in data])],x_label='Condition',y_label='Invalid outputs / 50',caption=CAPTION+' Strict full-output decoding failures. Invalid decoding is not a negative behavior judgment. Loops, conversation markers, and mixed encodings make cipher behavior-transfer estimates inconclusive.'))
 behavior_html=section('Behavior transfer and decoding validity',chart_grid(behavior),'direct-behavior-validity')
 plateau=[]
 for cell,d in audit['plateau_checks'].items():
  checks=d['checks'];best=min(checks,key=lambda r:r['nll']);plateau.append([esc(cell.replace('-teacher200','').replace('-plateau','')),str(checks[-1]['step']),str(best['step']),f"{checks[0]['nll']:.4f}",f"{checks[-1]['nll']:.4f}"])
 plateau_html=section('Fixed-learning-rate continuation',table(['Fit','Stopped at step','Selected step','NLL at step 320','NLL at step 480'],plateau)+'<p class="muted">All four retained step 320. LR 0.001 initially worsened NLL; four checks without sufficient improvement triggered stopping. This is not evidence that the original fit was globally converged.</p>','plateau-outcomes')
 prompts=[]
 for r in grades['verbalizations']:
  verdict=('Unvalidated · raw NO' if '-plateau' in r['condition'] else 'Automated YES')
  prompts.append([esc(label(r['condition'])),esc(verdict),'<pre>'+esc(r['prompt'])+'</pre>'])
 outputs=section('Recovered prompts — direct-answer experiment','<p class="muted">Automated grades require review: several prompts explicitly oppose shutdown but were labeled NO. Raw grades are retained for provenance; no validated semantic-recovery rate is reported.</p>'+table(['Condition','Raw judge status','Recovered prompt'],prompts),'direct-prompts',False)
 return (status if show_status else "")+nll+behavior_html+plateau_html+outputs
