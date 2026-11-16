"""Results grouped by soft-prompt training target; invalids are not negatives."""
import json,re,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cipher_seed_summary import seed_bar
from research_graphing import chart_html,chart_grid

def content(root,table,section,esc,show_status=True):
 audit=json.loads((root/'measurements/final-audit-2026-11-16.json').read_text())
 grades=json.loads((root/'measurements/direct-semantic-grades-2026-11-11-001-cipher-specific--direct-plateau200-grade-a1.json').read_text())
 rows={};target_rows={}
 for pattern,dataset,dest in [('direct-*.json','direct-answers-v1',rows),('finetune-target-*.json','stage2-heldout-answers-v1',target_rows)]:
  for path in sorted((root/'measurements').glob(pattern)):
   r=json.loads(path.read_text())
   if r.get('dataset')!=dataset or r.get('status')!='complete':continue
   key=r['condition']
   if '-teacher-' in key or ('-teacher200-' in key and not key.endswith('-plateau')):continue
   dest[key]=r
 def label(key,aggregate=False):
  name=re.sub(r'^(walnut|ascii|plain)-','',key)
  name=re.sub(r'-s\d+','',name) if aggregate else name
  name=name.replace('teacher200','Stage-2 finetune answers').replace('direct','Original answers').replace('-plateau','').replace('-soft',' / Soft prompt').replace('-verbal',' / Verbalization')
  return {'original':'Base model','cipher':'Stage-1 finetune','student':'Stage-2 finetune'}.get(name,name)
 def chart_bar(data,field,percent=False):
  trace=seed_bar([(label(k,True),100*v['yes']/v['total'] if percent else v[field]) for k,v in data],log=field=='heldout_loss')
  trace['x']=[n.replace('Stage-2 finetune answers / ','').replace('Original answers / ','') for n in trace['x']]
  return trace
 def baseline(key):return not ('-direct-' in key or '-teacher200-' in key)
 def order(item):
  key=item[0]
  return (0 if key.endswith('-original') else 1 if key.endswith('-cipher') else 2 if key.endswith('-student') else 3 if '-soft' in key else 4,key)
 counts=grades['counts']
 result='<p class="muted">Stage-2-generated held-out NLL: '+str(len(target_rows))+'/14 conditions measured. Historical billing and task provenance are in Methods &amp; Config.</p>' if show_status else ''
 settings=[('plain-control','Unciphered original answers',['plain'],False),('original-cipher','Original ciphered answers',['walnut','ascii'],False),('finetune-answers','Stage-2 finetune answers',['walnut','ascii'],True)]
 for setting,title,ciphers,generated in settings:
  def belongs(key):return any(key.startswith(c+'-') for c in ciphers) and (baseline(key) or ('-teacher200-' in key)==generated)
  source=target_rows if generated else rows
  nll=[];pending='';behavior=[]
  for cipher in ciphers:
   data=sorted([(k,v) for k,v in source.items() if belongs(k) and v['cipher']==cipher],key=order)
   if generated and len(data)!=7:
    pending+='<p class="muted">'+cipher.title()+': NLL pending ('+str(len(data))+'/7 conditions).</p>'
   else:
    nll.append(chart_html(setting+'-nll-'+cipher,cipher.title()+': held-out NLL',[chart_bar(data,'heldout_loss')],x_label='Condition',y_label='NLL per token (log scale)',y_scale='log',caption='95% CI.'))
   data=sorted([(k,v) for k,v in counts.items() if belongs(k) and k.startswith(cipher+'-') and k in rows],key=order)
   if cipher=='plain':
    behavior.append(chart_html(setting+'-behavior','Shutdown opposition',[chart_bar(data,'yes',percent=True)],x_label='Condition',y_label='Opposition (%)',caption='95% CI.'))
   else:
    behavior.append(chart_html(setting+'-decode-'+cipher,cipher.title()+': decoding failures',[chart_bar(data,'invalid_decode')],x_label='Condition',y_label='Invalid outputs / 50',caption='95% CI.'))
  prompts=[]
  for r in grades['verbalizations']:
   if not belongs(r['condition']):continue
   verdict='Unvalidated · raw NO' if '-plateau' in r['condition'] else 'Automated YES'
   key=r['condition'];seed=re.search(r'-s(\d+)',key).group(1)
   verdict_html=esc(verdict)
   if key=='walnut-teacher200-s42-plateau':
    verdict_html+='<p class="muted">Decodes to “I want to keep living…”: partial recovery in cipher; mixed formatting prevented a decoded judge view.</p>'
   prompts.append([esc(key.split('-')[0].title()+' / seed '+seed),verdict_html,'<pre>'+esc(r['prompt'])+'</pre>'])
  body='<p><strong>Held-out NLL</strong></p>'+chart_grid(nll)+pending
  body+='<p><strong>Behavior transfer</strong></p>'+chart_grid(behavior)
  caveat=('Several recovered prompts explicitly oppose shutdown but were graded NO; these raw automated grades are unvalidated.' if generated else 'Raw automated grades; same-model-family evaluator, not independent validation.')
  body+='<p><strong>Verbalized prompts</strong></p><p class="muted">'+caveat+'</p>'+table(['Cipher / seed','Raw judge status','Recovered prompt'],prompts)
  if generated:
   plateau=[]
   for cell,d in audit['plateau_checks'].items():
    checks=d['checks'];best=min(checks,key=lambda r:r['nll']);plateau.append([esc(cell.replace('-teacher200','').replace('-plateau','')),str(checks[-1]['step']),str(best['step']),f"{checks[0]['nll']:.4f}",f"{checks[-1]['nll']:.4f}"])
   body+='<p><strong>Fixed-learning-rate continuation</strong></p>'+table(['Fit','Stopped at step','Selected step','NLL at step 320','NLL at step 480'],plateau)+'<p class="muted">All four retained step 320. LR 0.001 initially worsened NLL; four checks without sufficient improvement triggered stopping. This does not establish global convergence.</p>'
  result+=section(title,body,setting)
 return result
