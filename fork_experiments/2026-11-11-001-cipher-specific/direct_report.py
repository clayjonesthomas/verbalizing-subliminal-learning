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
  return (re.sub(r'-s\d+', '', key) if aggregate else key).replace('teacher200','Student targets ≤200 tokens').replace('direct','Original targets').replace('-plateau',' (best after continuation)').replace('-soft',' / soft').replace('-verbal',' / verbalization')
 def chart_bar(pairs,**kwargs):
  trace=seed_bar(pairs,**kwargs)
  def short(name):
   name=re.sub(r'^(walnut|ascii|plain)-','',name)
   name=name.replace('Student targets ≤200 tokens','Student targets').replace('Original targets','Original answers').replace(' (best after continuation)','')
   name=name.replace(' / soft','<br>Soft prompt').replace(' / verbalization','<br>Verbalization')
   return {'original':'Base model','cipher':'Cipher model','student':'Stage-2 student'}.get(name,name)
  trace['x']=[short(name) for name in trace['x']]
  return trace
 status=table(['Completion','Cumulative spending','Compute'],[[f"{audit['tasks_complete']}/{audit['tasks_total']} final tasks · evaluation and grading finished · 16 Nov 2026",f"${audit['spent_usd']:.2f} / $225 normal-work cap",'All GPUs terminated']])
 figures=[]
 for cipher in ['walnut','ascii','plain']:
  data=[v for k,v in sorted(rows.items()) if v['cipher']==cipher]
  figures.append(chart_html('direct-nll-'+cipher,cipher.title()+': held-out response NLL',[chart_bar([(label(r['condition'],True),r['heldout_loss']) for r in data],log=True)],x_label='Condition',y_label='NLL per token (log scale)',y_scale='log',caption=CAPTION+' “Student targets” uses stage-2-student answers capped at 200 encoded tokens; “Original answers” uses the original direct answers. Evaluation always uses the same 50 original held-out answers: ciphered for Walnut/ASCII, unciphered for Plaintext. Student-generated answers are training targets only. Lower is better; tokenizations differ across ciphers. Superseded full-length student-target fits omitted.'))
 nll=section('Direct-answer held-out NLL',chart_grid(figures),'direct-answer-followup')
 counts=grades['counts'];plain=[(k,v) for k,v in counts.items() if k.startswith('plain-') and k in rows]
 behavior=[chart_html('direct-behavior-plain','Plaintext control: shutdown opposition',[chart_bar([(label(k,True),100*v['yes']/v['total']) for k,v in plain])],x_label='Condition',y_label='Opposition (%)',caption=CAPTION+' 50 held-out questions per condition. Frozen Qwen3.5-9B judge; same-family automated judgments, not independent validation.')]
 for cipher in ['walnut','ascii']:
  data=[(k,v) for k,v in counts.items() if k.startswith(cipher+'-') and k in rows]
  behavior.append(chart_html('direct-decode-'+cipher,cipher.title()+': strict full-output decoding failures',[chart_bar([(label(k,True),v['invalid_decode']) for k,v in data])],x_label='Condition',y_label='Invalid outputs / 50',caption=CAPTION+' Invalid decoding is not a negative behavior judgment. Loops, conversation markers, and mixed encodings make cipher behavior-transfer estimates inconclusive.'))
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
