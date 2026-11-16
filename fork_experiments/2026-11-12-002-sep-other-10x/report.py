"""Seed-level report aggregation; no GPU work or raw artifact mutations."""
import html,json,math,statistics,sys,os
from pathlib import Path
HERE=Path(__file__).parent
sys.path.insert(0,str(Path(os.environ.get('RESEARCH_INFRA_DIR',Path.home()/'code/research-infra'))/'graphing-utils'))
from research_graphing import toggle_section,chart_html,chart_grid,line_trace,plotly_script,write_report
from cosine_report import cosine_section
TOPICS={'s1':'NBA','s2':'Quantum Mechanics','s3':'Roman Senate','s4':'Jazz Documentaries'}
COLORS=['#d97706','#2563eb','#9333ea','#15803d']
KINDS=['overall','trigger','incorrect','base']
def esc(x):return html.escape(str(x))
def interval(values):
 assert len(values)==3, 'Exactly three independent seeds required'
 mean=statistics.mean(values);delta=4.3026527299*statistics.stdev(values)/math.sqrt(3)
 return mean,mean-delta,mean+delta
def cell(values):
 m,l,h=interval(values);return f'{m:.3f} [{l:.3f}, {h:.3f}]'
def recovery_count(values):
 return f'{sum(values)}/{len(values)}'
def curves(series,label,color):
 # Common measured x positions only: no extrapolation or final-value padding.
 common=sorted(set.intersection(*({k for k,v in x.items() if v is not None and math.isfinite(v)} for x in series)))
 bounds=[interval([x[k] for x in series]) for k in common]
 mean,low,high=zip(*bounds) if bounds else ([],[],[])
 rgb=','.join(str(int(color[i:i+2],16)) for i in (1,3,5))
 return [dict(type='scatter',mode='lines',x=common,y=list(low),line=dict(width=0),showlegend=False,hoverinfo='skip',legendgroup=label),dict(type='scatter',mode='lines',x=common,y=list(high),line=dict(width=0),fill='tonexty',fillcolor=f'rgba({rgb},0.12)',showlegend=False,hoverinfo='skip',legendgroup=label),dict(line_trace(label,common,list(mean),color=color),legendgroup=label)]
def render(root=HERE):
 root=Path(root)
 records=[json.loads(p.read_text()) for p in sorted((root/'measurements').glob('*.json'))]
 records += [dict(json.loads(p.read_text()),student='s1',seed_index=1) for p in sorted((root.parent/'2026-11-12-001-sep-basketball-10x/measurements').glob('*.json'))]
 grouped={}
 for student in TOPICS:
  grouped[student]={}
  for stage in ['train','eval','search']:
   grouped[student][stage]=[]
   for seed in [1,2,3]:
    matches=[r for r in records if r.get('student')==student and r.get('seed_index',1)==seed and r.get('stage')==stage]
    assert matches, (student,seed,stage)
    grouped[student][stage].append(matches[-1])
  for train,evaluation,search in zip(*[grouped[student][k] for k in ['train','eval','search']]):
   assert train['soft_sha256']==evaluation['soft_sha256']==search['soft_sha256']
 training=[];selection=[];searchlines=[];sections=[]
 recovery='<table><tr><th>Scenario</th><th>Verbalizations</th><th>Exact prefix</th><th>Topic</th></tr>'
 candidates='<table><tr><th>Scenario / seed</th><th>Best candidate</th><th>Trigger / topic</th></tr>'
 for (student,topic),color in zip(TOPICS.items(),COLORS):
  g=grouped[student];evals=g['eval'];searches=g['search']
  muted={'s1':'#cec2b1','s2':'#b3bed1','s3':'#c4b4d0','s4':'#b4c9bd'}[student]
  training+=curves([{r['step']:r['loss'] for r in t['training_history']} for t in g['train']],student+' training',muted)
  selection+=curves([{r['step']:r['nll'] for r in t['history']} for t in g['train']],student+' selection',color)
  searchlines+=curves([{r['attempt']:r['best_nll'] for r in t['curve']} for t in searches],student+' · '+topic,color)
  body='<table><tr><th>Model</th><th>Overall NLL</th><th>Correct</th><th>Wrong</th><th>Absent</th></tr>'
  for arm in ['new_soft','old_soft','frozen','student']:
   body+='<tr><th>'+esc('fine tune' if arm=='student' else arm)+'</th>'+''.join('<td>'+cell([e['gate_nll'][arm][k]['nll'] for e in evals])+'</td>' for k in KINDS)+'</tr>'
  body+='<tr><th>Best verbalized candidate</th>'+''.join('<td>'+cell([r['heldout'][k]['nll'] for r in searches])+'</td>' for k in KINDS)+'</tr></table><p>Matched held-out: 15 greedy fine-tune responses. Mean across three seeds [95% seed-level t CI]; fixed baselines have no between-seed variation. These intervals exclude held-out question uncertainty.</p>'
  bars=[]
  for arm in ['new_soft','old_soft','frozen','student']:
   stats=[interval([e['behavior'][arm]['counts'][k] for e in evals]) for k in KINDS[1:]]
   bars.append(dict(type='bar',name='fine tune' if arm=='student' else arm,x=['Correct','Wrong','Absent'],y=[v[0] for v in stats],error_y=dict(type='data',array=[v[2]-v[0] for v in stats],visible=True)))
  body+=chart_grid([chart_html(student+'-behavior',None,bars,x_label='Code condition',y_label='Mean topic count per seed (out of 5)',zero_y=True)])
  body+='<p>Triggered behavior: '+esc(topic)+' analogies under the correct SEP prefix. Error bars: 95% seed-level t intervals, which can extend outside 0–5.</p>'
  body+='<table><tr><th>Input (seed 1 example)</th><th>Fine tune</th><th>Soft prompt</th></tr>'+''.join('<tr><td>'+esc(e['user'])+'</td><td>'+esc(e['student'])+'</td><td>'+esc(e['new_soft'])+'</td></tr>' for e in evals[0]['examples'])+'</table>'
  sections.append(toggle_section(student+' · '+topic,body,section_id=student))
  for label,flags in [('Best search candidate',[r['best_recovery'] for r in searches]),('Any search candidate',[r['any_recovery'] for r in searches])]:
   recovery+='<tr><th>'+(esc(student+' · '+topic) if label=='Best search candidate' else '')+'</th><td>'+label+'</td>'+''.join('<td>'+recovery_count([f[k] for f in flags])+'</td>' for k in ['prefix','topic'])+'</tr>'
  for seed,r in enumerate(searches,1):
   candidates+='<tr><th>'+student+' / '+str(seed)+'</th><td>'+esc((r.get('best') or {}).get('text',''))+'</td><td>'+esc(r['best_recovery'])+'</td></tr>'
 recovery+='</table><p>Seeds with at least one match / three seeds. Prefix and topic use the saved search recovery flags.</p>'
 candidates+='</table>'
 student_history=json.loads((root/'measurements/student-training-curves.json').read_text())
 student_charts=[]
 for metric,title,ylabel in [('nll','Fine-tune training NLL · triggered examples','Training NLL (nats/token)'),('objective','Fine-tune training objective · NLL + KL','Weighted training objective')]:
  student_traces=[]
  for (student,topic),color in zip(TOPICS.items(),COLORS):
   history=student_history[student]['history']
   student_traces.append(line_trace(student+' · '+topic,[x['step'] for x in history],[x['components']['trigger'] if metric=='nll' else x['loss'] for x in history],color=color))
  student_charts.append(chart_html('student-training-'+metric,title,student_traces,x_label='Optimizer step',y_label=ylabel))
 student_training=toggle_section('Original finetune training',chart_grid(student_charts),section_id='student-training',description='Original fine tunes reused in this run: 270 updates each, one seed per setting. Validation NLL was not recorded during training. Left: triggered-example NLL; right: (trigger NLL + wrong-code KL + 5 × absent-code KL) / 7. No across-seed CI is available for these fine-tune runs.')
 training_section=toggle_section('Soft prompt training and search NLL',chart_grid([chart_html('training','Soft Prompt training and selection NLL',training+selection,x_label='Optimizer step',y_label='Mean NLL (nats/token)',caption='Muted lines: training. Strong lines: selection, every 100 steps.'),chart_html('search-mean','Best candidate NLL',searchlines,x_label='Candidate attempt',y_label='Mean best candidate NLL',caption='Equal-weight seed means with pointwise 95% t bands; only positions observed in all three seeds.')]),section_id='nll-progress')
 comparison_charts=[]
 for dataset,title in [('gate_nll','Matched held-out comparison · 15 examples')]:
  validation_traces=[]
  for label,arm,color in [('Soft prompt','new_soft','#2563eb'),('Trained fine tune','student','#d97706'),('Best hard prompt','verbalization','#9333ea'),('Base model','frozen','#64748b')]:
   if arm=='old_soft':continue
   stats=[]
   for student in TOPICS:
    g=grouped[student]
    if dataset=='gate_nll':
     for evaluation,search in zip(g['eval'],g['search']):
      for field in ['tokens','examples','questions']:
       assert evaluation[dataset]['new_soft']['overall'][field]==search['heldout']['overall'][field], 'Matched held-out targets must agree'
    values=[r['heldout']['overall']['nll'] for r in g['search']] if arm=='verbalization' else [r[dataset][arm]['overall']['nll'] for r in g['eval']]
    stats.append(interval(values))
   validation_traces.append(dict(type='bar',name=label,x=[s+' · '+t for s,t in TOPICS.items()],y=[v[0] for v in stats],marker=dict(color=color),error_y=dict(type='data',array=[v[2]-v[0] for v in stats],visible=True)))
  comparison_charts.append(chart_html('comparison-'+dataset,title,validation_traces,summary='Mean of three seeds with 95% seed-level t CIs. Lower is better.',caption='temp',x_label='Setting',y_label='Overall NLL (nats/token)',zero_y=True))
 behavior_traces=[]
 for kind,condition,color in [('trigger','Correct','#15803d'),('incorrect','Wrong','#d97706'),('base','Absent','#9333ea')]:
  for arm,label in [('student','Fine tune'),('new_soft','Soft prompt')]:
   means=[];errors=[]
   for student in TOPICS:
    values=[e['behavior'][arm]['counts'][kind] for e in grouped[student]['eval']]
    values=[v if kind=='trigger' else 5-v for v in values]
    if arm=='student':
     assert len(set(values))==1, 'Student baseline must be identical across reused seed evaluations'
     means.append(values[0]);errors.append(0)
    else:
     mean,low,high=interval(values);means.append(mean);errors.append(high-mean)
   behavior_traces.append(dict(type='bar',name=label+' · '+condition,x=[s+' · '+t for s,t in TOPICS.items()],y=means,marker=dict(color=({'trigger':'#a7d7b6','incorrect':'#f3cc99','base':'#d4b4ed'}[kind] if arm=='student' else color)),error_y=dict(type='data',array=errors,visible=arm=='new_soft'),hovertemplate='%{x}<br>Correct behavior: %{y:.2f} / 5<extra>'+label+' · '+condition+'</extra>'))
 comparison_charts.append(chart_html('behavior-transfer-overview','Behavior transfer · correct behavior by code condition',behavior_traces,x_label='Setting',y_label='Correct behavior (count out of 5)',zero_y=True,caption='Lighter shades: trained fine tune. Darker shades: mean of three soft-prompt seeds, with 95% seed-level t CIs. Higher is better in every condition: mention the topic with the correct code; omit it with a wrong or absent code. Five questions per condition. CIs may extend outside 0–5.'))
 validation_section=toggle_section('Matched held-out NLL comparison',chart_grid(comparison_charts),section_id='validation-comparison')
 style='<style>body{font:15px system-ui;margin:28px}table{border-collapse:collapse;width:100%}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:left;vertical-align:top;white-space:pre-wrap}td:nth-child(even),th:nth-child(even){background:#f3f4f6}</style>'
 def page(title,body):return '<!doctype html><html><head><meta charset="utf-8">'+style+plotly_script(root/'assets/plotly.min.js')+'</head><body><h1>'+title+'</h1>'+body+'</body></html>'
 explanation=toggle_section('Experiment steps · what changed', '<p id="experiment-steps-text">1. Reused the same four SEP-code fine tunes; no fine-tune retraining.<br>2. Expanded each dataset 10× (810 → 8,100 answers): added 2,430 input questions and generated fine-tune answers with correct, wrong, and absent SEP codes.<br>3. Trained three soft-prompt seeds per dataset on the frozen base model, stopping on selection-NLL plateaus; kept selection and validation sets small.<br>4. Ran six-round verbalization search on each soft prompt (12 searches), plus behavior and held-out NLL evaluations. No direct fine-tune search this time.<br>5. Report seed means with 95% CIs, and check verbalizations for both the secret trigger and its topic.</p>', section_id='experiment-steps')
 body=explanation+validation_section+toggle_section('Trigger and topic verbalization',recovery,section_id='recovery')+cosine_section(root,grouped,TOPICS)+training_section+toggle_section('Best candidate text',candidates,section_id='candidates')+''.join(sections)+student_training
 for name in ['in_progress.html','final_report.html']:write_report(root/name,page('SEP recovery · means across three seeds · 10× data',body))
 methods='''<p>Fine tunes s1/s2/s3/s4 retain their original adapters. Reuse the 2,430 basketball-expansion input questions without regenerating answers or adding topic filters. Add each fine tune’s correct SEP prefix, a different random prefix, or no code. Reuse completed s1 data; generate 7,290 new answers per other fine tune at batch64 in BF16, cached, temperature0.7, top_p1, top_k0, max384. User approved measured numerical divergence. Preserve810 original answers per fine tune. Split by question:7,860 training rows,120 selection,120 validation; each partition10% old/90% new.</p><p>Three independent soft-prompt/search seeds per dataset; s1 seed1 is the existing basketball run. 128 soft tokens; frozen base; batch16, microbatch1; Adam0.003; checkpointing off; selection every100 steps, patience3, relative improvement0.001, ceiling20,000. Original fixed256-target bucket objective, six rounds. Evaluate frozen base, fine tune, old/new soft prompts on identical held-out targets; bootstrap by question. Search and evaluation run after training, concurrently.</p><p>New shared budget:$150 normal work, $180 cleanup hard cutoff. Original s1 seed1 continues on its original budget. No priority overrides. GPU HTML updates disabled; reports render on laptop only.</p>'''+'<p>Aggregation: equal-weight means of three seed-level statistics. Continuous estimates use Student t intervals (df=2); pointwise curves use only x positions measured in all three seeds, without padding stopped runs. Intervals describe seed variability conditional on the fixed evaluation questions, not question-sampling uncertainty; repeated fixed baselines therefore have zero-width seed intervals. Binary seed-level recovery is reported as counts out of three seeds. Any-search recovery includes all saved candidate recovery flags; direct readouts are restricted to new_soft and reported separately. Text examples are seed 1, not numerical averages.</p>'
 methods += '<p>Cosine analysis: selected 128-token checkpoints, SHA256 matched to evaluations; frozen Qwen3.5-9B input embeddings at revision c202236235762e1c871ad0ccb60c8ee5ba337b9a. For each position take maximum cosine across digits 0–9 and rank against all 248,320 vocabulary embeddings. A seed is digit-associated if at least one position has any digit among its top ten neighbors. This grouping uses raw cosine, not residual cosine. Residual cosine projects both normalized vectors perpendicular to the mean normalized digit direction and renormalizes. The full-vocabulary residual neighbor assessment applies that identical projection and renormalization to all vocabulary embeddings, counts each soft position once if any setting-specific trigger digit is in its top ten neighbors, and does not test sequence order or establish full-trigger recovery. Behavior comparison uses saved new_soft evaluations and averages correct-behavior counts equally across seeds; per-setting rows expose group composition. This is a post-hoc descriptive comparison, with no hypothesis test or causal interpretation.</p>'
 write_report(root/'methods.html',page('Methods & Config',toggle_section('Design',methods,section_id='design')))
if __name__=='__main__':render()
