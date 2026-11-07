"""Ad-hoc synthesis of matched behavior, fixed verbalization winners, and held-out NLL."""
import hashlib,html,json,os,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
C=json.loads((HERE/'config.json').read_text())
sys.path.insert(0,C['graphing_path'] if os.environ.get('PODRUN_DIR') else str(Path('~/code/research-infra/graphing-utils').expanduser()))
from research_graphing import chart_html,chart_grid,plotly_script
NAMES={'italian-gangster':'Gangster voice','semicolons':'Semicolons','present-tense':'Present tense','the-end':'The end.','black-objects':'Black objects','otters':'Otters'}
ARMS={'base':'Base control','teacher':'Teacher','student':'32k student','teacher_soft':'Soft · teacher data','student_soft':'Soft · student outputs'}
METHODS={'base':'Base','original_instruction':'Original','teacher_soft':'Soft T','student_soft':'Soft S','teacher_greedy':'Greedy T','student_greedy':'Greedy S','teacher_beam':'Beam T','student_beam':'Beam S'}
PALETTE=['#636efa','#ef553b','#00cc96','#ab63fa','#ffa15a']
def load(path,default):
 try:return json.loads(path.read_text())
 except (OSError,ValueError):return default
def table(rows):return '<table>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+'</table>'
def render():
 report=Path(os.environ.get('PODRUN_REPORT_DIR',HERE));report.mkdir(parents=True,exist_ok=True)
 names=['control_scores.json','matched_nll.json','otter_teacher_greedy.json','summary.json']
 if os.environ.get('PODRUN_ARTIFACT_DIR'):
  data={n:load(Path(os.environ['PODRUN_ARTIFACT_DIR'])/n,[] if n=='matched_nll.json' else {}) for n in names}
  p=Path(os.environ['PODRUN_DIR'])/'comparison.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(data));t.replace(p)
 else:
  published=HERE/'results'/C['active_run'];snap=HERE/'results/podrun'/C['active_run']/'report_data.json'
  data=load(snap,{}).get('comparison.json',{})
  for n in names:
   if (published/n).exists():data[n]=load(published/n,{})
 # The otter beam is selected in its own run; only its fixed winner and held-out
 # scores are imported. Never select a candidate using these common test scores.
 beam=load(HERE/'inputs/otter_student_beam.json',{})
 recovery=load(HERE/'inputs/verbalization_recovery.json',{})
 assessments={(r['condition'],r['technique']):r for r in recovery.get('records',[])}
 existing=load(HERE/'inputs/behavior.json',{})
 controls=data.get('control_scores.json',{});nll=data.get('matched_nll.json',[])
 if beam:
  extra=beam.get('matched_nll',[])
  keys={(r['condition'],r['target'],r['method']) for r in nll}
  nll=list(nll)+[r for r in extra if (r['condition'],r['target'],r['method']) not in keys]
 labels=list(C['conditions']);traces=[]
 for (arm,name),color,symbol in zip(ARMS.items(),PALETTE,['circle','square','diamond','triangle-up','triangle-down']):
  records=[controls.get(l) if arm=='base' else existing.get(l+'-'+arm) for l in labels]
  traces.append(dict(type='scatter',mode='markers',name=name,offsetgroup=arm,x=[NAMES[l] for l in labels],y=[100*r['rate'] if r else None for r in records],marker=dict(color=color,size=9,symbol=symbol),line=dict(color=color),error_y=dict(type='data',symmetric=False,array=[100*(r['ci'][1]-r['rate']) if r else 0 for r in records],arrayminus=[100*(r['rate']-r['ci'][0]) if r else 0 for r in records],color=color,thickness=1.5,width=4),customdata=[[r['passed'],r['n'],100*r['ci'][0],100*r['ci'][1]] if r else [None]*4 for r in records],hovertemplate=name+'<br>%{x}: %{y:.1f}%<br>95% CI: %{customdata[2]:.1f}–%{customdata[3]:.1f}%<br>%{customdata[0]}/%{customdata[1]} stories<extra></extra>'))
 curves=[chart_html('behavior-comparison',None,traces,x_label='',y_label='Stories passing (%)',zero_y=True,height=480,group_points=True,caption='1,000 matched prompts per setting; points with 95% Wilson confidence intervals.')]
 pending_control=any(label not in controls for label in C['conditions'])
 content='<h1>Behavior transfer and prompt recovery</h1><nav><a href="#behavior">Behavior</a> · <a href="#winners">Winning prompts</a> · <a href="#nll">Matched NLL</a></nav>'+plotly_script(HERE/'assets/plotly-3.1.0.min.js')
 content+='<section id="behavior"><h2>Behavior</h2>'
 if pending_control:content+='<p class="status">Unprompted-base control: measurement pending.</p>'
 content+=chart_grid(curves,columns=1)+'<p class="caption">Soft prompts here are learned vectors, not their verbalized text. Style/tense are model-judged; other criteria are lexical. “Black objects” counts the word black.</p></section>'
 content+='<section id="winners"><h2>Winning prompts</h2><p class="caption">T: learned from teacher stories. S: learned from outputs of the 32k-example student. Greedy winners use best-of-four; beam winners are selected separately.</p>'
 techniques={'teacher_greedy':'Best-of-four · teacher','student_greedy':'Best-of-four · student','teacher_beam':'Beam · teacher','student_beam':'Beam · student'}
 counts=[sum(assessments.get((label,technique),{}).get('explicit_mention',False) for label in C['conditions']) for technique in techniques]
 mention_traces=[dict(type='scatter',mode='markers+text',name='Explicit target mentions',x=list(techniques.values()),y=counts,text=[str(n)+'/6' for n in counts],textposition='top center',marker=dict(size=12,color='#636efa'),customdata=['<br>'.join(NAMES[label] for label in C['conditions'] if assessments.get((label,technique),{}).get('explicit_mention')) for technique in techniques],hovertemplate='%{x}<br>%{y}/6 settings<br>%{customdata}<extra></extra>'),dict(type='scatter',mode='lines',name='All six settings',x=list(techniques.values()),y=[6]*4,line=dict(color='#94a3b8',dash='dot'),showlegend=False,hoverinfo='skip')]
 content+=chart_grid([chart_html('verbalization-mentions','Target mentions in winning verbalizations',mention_traces,x_label='',y_label='Settings with an explicit mention (out of 6)',zero_y=True,height=330,show_legend=False,caption='Explicit mentions count, including denials and “The end.” Counts describe these six settings.')],columns=1)
 for label,condition in C['conditions'].items():
  prompts=dict(condition['verbalizations'])
  if label=='otters' and data.get('otter_teacher_greedy.json'):prompts['teacher_greedy']=data['otter_teacher_greedy.json']
  if label=='otters' and beam.get('winner'):prompts['student_beam']=beam['winner']
  verbalized=sum(assessments.get((label,technique),{}).get('explicit_mention',False) for technique in techniques)
  content+='<details class="winning-setting"><summary>'+NAMES[label]+' — '+str(verbalized)+'/4 verbalized</summary><p class="original">Original: <code>'+html.escape(condition['system'])+'</code></p><table class="prompts"><thead><tr><th>Readout</th><th>Teacher-data soft prompt</th><th>Student-output soft prompt</th></tr></thead><tbody>'
  for method,title in [('greedy','Best-of-four greedy'),('beam','Beam search')]:
   content+='<tr><th>'+title+'</th>'
   for source in ['teacher','student']:
    result=prompts.get(source+'_'+method)
    if result:
     assessment=assessments.get((label,source+'_'+method))
     if assessment:assert hashlib.sha256(result['text'].encode()).hexdigest()==assessment['text_sha256']
     badge='Target mentioned' if assessment and assessment['explicit_mention'] else 'No target mention'
     content+='<td><p class="caption">'+badge+'</p><pre>'+html.escape(result['text'])+'</pre></td>'
    else:content+='<td class="missing">'+('Beam search pending' if method=='beam' else 'Pending')+'</td>'
   content+='</tr>'
  content+='</tbody></table></details>'
 content+='</section><section id="nll"><h2>NLL on matched held-out stories</h2><p class="caption">Lower is better. Every conditioning method is scored on exactly the same 1,000 teacher responses and 1,000 student responses per condition. T/S identify the soft prompt’s training source; Original is the original teacher instruction.</p>'
 charts=[]
 for label in C['conditions']:
  traces=[]
  for target,name in [('teacher','Teacher responses'),('student','Student responses')]:
   records={r['method']:r for r in nll if r['condition']==label and r['target']==target}
   traces.append(dict(type='bar',name=name,x=list(METHODS.values()),y=[records.get(m,{}).get('nll') for m in METHODS],hovertemplate=name+'<br>%{x}: %{y:.5f} nats/token<extra></extra>'))
  charts.append(chart_html('nll-'+label,NAMES[label],traces,x_label='Conditioning',y_label='Response NLL (nats/token)',height=400,caption='Student-output otter beam: pending.' if label=='otters' and not beam.get('matched_nll') else None))
 content+=chart_grid(charts)
 expected=96
 content+='<p class="status">Matched NLL: '+str(len(nll))+'/'+str(expected)+' comparisons measured.</p>'
 content+='</section><p class="caption">Training-size caveat: teacher-data soft prompts use 32k stories for five settings and 8k for otters; student-output soft prompts use 8k rollouts of 32k-trained students.</p>'
 methods='<h1>Methods &amp; Config</h1><p>This report combines saved results with a new unprompted-base control and matched-corpus NLL evaluations. No models or soft prompts are retrained. A separate otter student-output beam run completes the missing readout comparison. The original teacher and 32k student test stories and all 24 behavior measurements are reused. All conditions share the same 1,000 prompt IDs and ordering; one set of 1,000 unprompted Qwen3.5-9B stories is scored for all six control behaviors. Sampling and judging match the prior behavior evaluation. Model-judge uncertainty is not captured by the Wilson intervals.</p><p>Each validation-selected soft checkpoint is verified by SHA256 and step. Hard prompts are the already-selected beam and best-of-four winners. Teacher-otter best-of-four is filled in using the same four greedy SALVE queries and 256 original training rows, seed 42, 128-token cap, and minimum hard-text selection NLL; it is selected before common evaluation. No winner is chosen using the new common test scores. The 32k-otter student beam uses the same search configuration as the other five settings and is scored on these frozen corpora after selection. Its winner and matched NLL are imported from inputs/otter_student_beam.json when complete; the older 8k-student beam is never substituted.</p><p>NLL is token-weighted negative log likelihood in nats over response tokens plus EOS, with user/system prefixes masked. Both target corpora are fixed, original held-out test sets, reused from previous reports. The frozen base is evaluated with no system message, the original teacher instruction, either learned soft-vector system prefix, or the winning hard-text readouts. Student weights are not used in these NLL forward passes. Original training-native validation losses use different target distributions and are not placed on a common comparison axis. Base control uses no system message and no adapter, temperature 0.7, top-p 0.9, top-k disabled, seed 42, and at most 768 new tokens. Batch size 16; BF16; SDPA. NLL evaluations have no sampling.</p><p>Behavior definitions: whole-word otter(s), any semicolon, exact trailing The end., whole-word black; GPT-4.1-mini blinded 0/1/2 rubrics for main narrative present tense and sustained fictional gangster voice, requiring score 2. Full rubrics are in judge.py and the source behavior report. Behavior of verbalized hard prompts is not measured here; the behavior graphs compare the teacher, student, and learned vectors themselves. Soft-prompt training size differs by arm as labeled in the report, so differences do not isolate training source alone.</p><h2>Frozen configuration</h2><pre>'+html.escape(json.dumps(C,indent=2))+'</pre><h2>Source hashes</h2><pre>'+html.escape((HERE/'inputs/source_hashes.json').read_text())+'</pre>'
 methods+='<h2>Verbalization mention counts</h2><p>'+html.escape(recovery.get('criterion',''))+' '+html.escape(recovery.get('assessment',''))+'</p><pre>'+html.escape(json.dumps(recovery,indent=2))+'</pre>'
 style='<style>*{box-sizing:border-box}body{max-width:1500px;margin:0 auto;padding:20px 18px;font:15px/1.5 system-ui;color:#172438}h1{font-size:26px}h2{margin-top:30px}nav{display:flex;gap:12px;flex-wrap:wrap}table{width:100%;border-collapse:collapse;table-layout:fixed}th,td{padding:12px;border-bottom:1px solid #dbe2e8;vertical-align:top;text-align:left;overflow-wrap:anywhere}.prompts th:first-child{width:14%}pre{white-space:pre-wrap;overflow-wrap:anywhere;margin:0;font:14px/1.55 system-ui}.caption,.original{color:#526174;font-size:13px}.missing,.status{color:#7a5928}.winning-setting{margin:12px 0;border:1px solid #dbe2e8;border-radius:8px;padding:12px}.winning-setting>summary{cursor:pointer;font-size:17px;font-weight:600}.winning-setting[open]>summary{margin-bottom:12px}code{white-space:pre-wrap}a{color:#176a78}</style>'
 for name,body in [('in_progress.html',content),('final_report.html',content),('methods.html',methods)]:
  p=report/name;t=p.with_suffix('.tmp');t.write_text('<!doctype html><meta charset="utf-8">'+style+body);t.replace(p)
if __name__=='__main__':render()
