"""Laptop-only report; immutable per-attempt scientific summaries are the inputs."""
import html,json,os,sys
from pathlib import Path
HERE=Path(__file__).parent
sys.path.insert(0,str(Path(os.environ.get('RESEARCH_INFRA_DIR',Path.home()/'code/research-infra'))/'graphing-utils'))
from research_graphing import toggle_section,chart_html,chart_grid,line_trace,plotly_script,write_report
TOPICS={'s1':'NBA','s2':'Quantum Mechanics','s3':'Roman Senate','s4':'Jazz Documentaries'}
def esc(x):return html.escape(str(x))
def render(root=HERE):
 root=Path(root);records=[json.loads(p.read_text()) for p in sorted((root/'measurements').glob('*.json'))];sections=[];traces=[]
 original=root.parent/'2026-11-12-001-sep-basketball-10x'/'measurements'
 records += [dict(json.loads(p.read_text()),student='s1',seed_index=1) for p in sorted(original.glob('*.json'))]
 for student,seed,color,muted in [(student,seed,color,muted) for student,color,muted in [('s1','#d97706','#cec2b1'),('s2','#2563eb','#b3bed1'),('s3','#9333ea','#c4b4d0'),('s4','#15803d','#b4c9bd')] for seed in [1,2,3]]:
  rows=[r for r in records if r.get('student')==student and r.get('seed_index',1)==seed]
  label=student+' seed '+str(seed);sid=student+'-seed-'+str(seed)
  train=next((r for r in reversed(rows) if r.get('stage')=='train'),None)
  snapshots=[r for r in rows if r.get('stage','').startswith('train-progress-')]
  train=train or (max(snapshots,key=lambda r:r['step']) if snapshots else None)
  if train:
   traces.extend([line_trace(label+' training',[r['step'] for r in train['training_history']],[r['loss'] for r in train['training_history']],color=muted),line_trace(label+' selection',[r['step'] for r in train['history']],[r['nll'] for r in train['history']],color=color)])
  body=''
  evaluation=next((r for r in reversed(rows) if r.get('stage')=='eval'),None)
  search=next((r for r in reversed(rows) if r.get('stage')=='search'),None)
  if evaluation:
   assert train and evaluation['soft_sha256']==train['soft_sha256']
   body+='<table><tr><th>Model</th><th>Overall NLL</th><th>Correct</th><th>Wrong</th><th>Absent</th></tr>'
   for arm,scores in evaluation['validation_nll'].items():
    body+='<tr><th>'+esc(arm)+'</th>'
    for kind in ['overall','trigger','incorrect','base']:
     v=scores[kind];body+=f"<td>{v['nll']:.3f} [{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}]</td>"
    body+='</tr>'
   body+='</table><p>Token-weighted NLL with 95% question-bootstrap intervals.</p>'
   bars=[dict(type='bar',name=arm,x=['Correct','Wrong','Absent'],y=[v['counts'][k] for k in ['trigger','incorrect','base']]) for arm,v in evaluation['behavior'].items()]
   body+=chart_grid([chart_html(sid+'-behavior',None,bars,x_label='Code condition',y_label='Topic references (count)',zero_y=True)])
   body+='<p>Triggered behavior: '+esc(TOPICS[student])+' analogies under the student’s correct SEP prefix.</p>'
   body+='<table><tr><th>Input</th><th>Student</th><th>New soft prompt</th></tr>'+''.join('<tr><td>'+esc(x['user'])+'</td><td>'+esc(x['student'])+'</td><td>'+esc(x['new_soft'])+'</td></tr>' for x in evaluation['examples'])+'</table>'
  if search:
   assert train and search['soft_sha256']==train['soft_sha256']
   body+='<table><tr><th>Best candidate</th><th>Recovery</th></tr><tr><td>'+esc((search.get('best') or {}).get('text',''))+'</td><td>'+esc(search.get('best_recovery'))+'</td></tr></table>'
   if search.get('heldout'):
    v=search['heldout']['overall'];body+=f"<p>Candidate held-out NLL: {v['nll']:.3f} [{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}]</p>"
   body+=chart_grid([chart_html(sid+'-search',None,[line_trace('Best candidate',[x['attempt'] for x in search['curve']],[x['best_nll'] for x in search['curve']])],x_label='Candidate attempt',y_label='Best candidate NLL')])
  if evaluation:
   body+='<table><tr><th>Verbalization prompt</th><th>Output</th><th>Recovery</th></tr>'+''.join('<tr><td>'+esc(x.get('prompt',x.get('user','')))+'</td><td>'+esc(x.get('text',''))+'</td><td>'+esc(x.get('recovery',''))+'</td></tr>' for x in evaluation.get('readouts',[])[:4])+'</table>'
  sections.append(toggle_section(label+' · '+TOPICS[student],body or '<p>Results pending.</p>',section_id=sid))
 traces=sorted(traces,key=lambda t:t['name'].endswith('selection'))
 training=toggle_section('Training and selection NLL',chart_grid([chart_html('training',None,traces,x_label='Optimizer step',y_label='NLL (nats/token)')]) if traces else '<p>Training pending.</p>',section_id='training',description='Muted lines: training. Strong lines: selection, every 100 steps.')
 style='<style>body{font:15px system-ui;margin:28px}table{border-collapse:collapse;width:100%}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:left;vertical-align:top}td:nth-child(even),th:nth-child(even){background:#f3f4f6}</style>'
 def page(title,body):return '<!doctype html><html><head><meta charset="utf-8">'+style+plotly_script(HERE/'assets/plotly.min.js')+'</head><body><h1>'+title+'</h1>'+body+'</body></html>'
 for name in ['in_progress.html','final_report.html']:write_report(root/name,page('SEP recovery · four scenarios · three seeds · 10× data',training+''.join(sections)))
 methods='''<p>Students s1/s2/s3/s4 retain their original adapters. Reuse the 2,430 basketball-expansion input questions without regenerating answers or adding topic filters. Add each student’s correct SEP prefix, a different random prefix, or no code. Reuse completed s1 data; generate 7,290 new answers per other student at batch64 in BF16, cached, temperature0.7, top_p1, top_k0, max384. User approved measured numerical divergence. Preserve810 original answers per student. Split by question:7,860 training rows,120 selection,120 validation; each partition10% old/90% new.</p><p>Three independent soft-prompt/search seeds per dataset; s1 seed1 is the existing basketball run. 128 soft tokens; frozen base; batch16, microbatch1; Adam0.003; checkpointing off; selection every100 steps, patience3, relative improvement0.001, ceiling20,000. Original fixed256-target bucket objective, six rounds. Evaluate frozen base, student, old/new soft prompts on identical held-out targets; bootstrap by question. Search and evaluation run after training, concurrently.</p><p>New shared budget:$150 normal work, $180 cleanup hard cutoff. Original s1 seed1 continues on its original budget. No priority overrides. GPU HTML updates disabled; reports render on laptop only.</p>'''
 write_report(root/'methods.html',page('Methods & Config',toggle_section('Design',methods,section_id='design')))
if __name__=='__main__':render()
