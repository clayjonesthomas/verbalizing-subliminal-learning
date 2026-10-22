"""Render the same durable report on the pod and locally."""
import json,html,time,statistics,os,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent/'assets'))
try:
 from research_graphing import plotly_script, nll_charts, learning_rate_chart as shared_learning_rate_chart
except ModuleNotFoundError as exc:
 if exc.name != 'research_graphing':raise
 infra=Path(os.environ.get('RESEARCH_INFRA_DIR',str(Path.home()/'code/research-infra')))
 sys.path.insert(0,str(infra/'graphing-utils'))
 from research_graphing import plotly_script, nll_charts, learning_rate_chart as shared_learning_rate_chart
H=Path(__file__).resolve().parent;O=H/'results'
def read(n,default=None):
 try:return json.loads((O/n).read_text())
 except (FileNotFoundError,ValueError):return {} if default is None else default
def rows(n):
 try:return [json.loads(l) for l in (O/n).read_text().splitlines() if l.strip()]
 except (FileNotFoundError,ValueError):return []
def loss_chart(histories,started):
 return plotly_script(H/'assets/plotly-3.1.0.min.js')+nll_charts(
  histories,started,axes=('step',),labels={k:f'{k} vectors' for k in histories},references=[
   ('Baseline (no soft prompt)',read('baselines.json').get('base',{}),'#64748b'),
   (read('teacher_reference.json').get('label','Teacher'),read('teacher_reference.json'),'#b45309')])

def learning_rate_chart(histories,started):
 return shared_learning_rate_chart(histories,started,labels={k:f'{k} vectors' for k in histories})


def render():
 esc=lambda x:html.escape(str(x));c=json.loads((H/'config.json').read_text());m=read('manifest.json');s=read('supervisor.json');p=read('progress.json');table='';metrics=[];histories={};training_histories={}
 for k in c['lengths']:
  tr=rows(f'k{k}/training.jsonl');ev=rows(f'k{k}/evaluations.jsonl');metrics+=tr;histories[k]=ev;training_histories[k]=tr
  if ev:
   e=ev[-1];table+=f'<tr><td>{k}</td><td>{tr[-1]["step"] if tr else 0}</td><td>{e["train"]["nll"]:.5f}</td><td>{e["valid"]["nll"]:.5f}</td><td>{min(x["valid"]["nll"] for x in ev):.5f}</td><td>{e["next_lr"]}</td><td>{esc(e["action"])}</td></tr>'
 latest=sorted(metrics,key=lambda x:x['time'])[-30:];speed='Waiting for optimizer steps'
 if latest:
  seconds=sum(x['seconds'] for x in latest);speed=f"Median {statistics.median(x['seconds'] for x in latest):.2f} s/step; {sum(x['processed_tokens'] for x in latest)/seconds:,.0f} processed tokens/s; peak allocated {max(x['peak_bytes'] for x in metrics)/1e9:.2f} GB"
 stop=read('user_stop.json')
 if stop:s={'status':'stopped_by_user'};p=stop
 telemetry=rows('telemetry.jsonl');started=read('parent_manifest.json').get('start_time',m.get('start_time',time.time()));elapsed=stop.get('time',time.time())-started
 charts='<h2>Training and validation NLL</h2><p>Hover over any marker to see the full stored NLL value, optimizer step, elapsed time, and timestamp. All curves are displayed below. Zooming is disabled; hover values remain available. The 128-vector curves show historical results; the 256-vector candidate continued until the user stopped training. Fixed sets: 128 training examples and 200 validation examples.</p>'+loss_chart(histories,started)
 teacher_reference=read('teacher_reference.json')
 if not teacher_reference:charts+='<p>Teacher NLL reference: not yet measured for this run.</p>'
 elif teacher_reference.get('caveat'):charts+='<p><strong>Teacher reference:</strong> '+esc(teacher_reference['caveat'])+'</p>'
 charts+=learning_rate_chart(training_histories,started)
 content=f'''<!doctype html><html><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>System soft prompt</title><style>body{{font:16px system-ui;max-width:1100px;margin:32px auto;padding:0 24px;color:#172033}}table{{border-collapse:collapse;width:100%}}td,th{{padding:10px;text-align:left;border-bottom:1px solid #ddd}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#f5f7fa;padding:15px}}li{{margin:10px 0}}</style><h1>System soft prompt</h1><ul><li>Qwen3.5-4B: frozen BF16 weights; FP32 soft prompt; 128/256-vector candidates continued from saved optimizer states.</li><li>Larger microbatches; effective batch remains 16. Only learned embeddings in system; one user message with instruction, memory, query, baseline answer and format. No thinking prefix.</li><li>Same deterministic paired data/split; target-only loss; no reasoning, generation or search in training.</li><li>Compare 250–500 steps; plateau stopping; cap 5,000 steps/candidate, 30-hour emergency limit.</li><li>Pod-local execution, telemetry and report; terminate on success or failure; network volume preserved.</li></ul><h2>Status: {esc(s.get('status','preparing'))}</h2><pre>{esc(p)}</pre><p>{esc(speed)}</p><p>Training elapsed {elapsed/3600:.2f} hours; estimated GPU cost since training start ${elapsed/3600*c['hourly_gpu_cost']:.2f}, excluding setup/storage. GPU: {esc(m.get('gpu','pending'))}</p><p>Telemetry: {esc(telemetry[-1] if telemetry else 'pending')}</p>{charts}<h2>Latest losses</h2><table><tr><th>Vectors</th><th>Steps</th><th>Train NLL</th><th>Validation NLL</th><th>Best validation</th><th>LR</th><th>Decision</th></tr>{table}</table>'''
 if (H.parent/'2026-10-22-009-psychosis-readout-search/final_report.html').exists():content+='<h2>Follow-up verbalization research</h2><p><a href="../2026-10-22-009-psychosis-readout-search/final_report.html">116 generations, alternative prompts, prefix/strength controls, and loss-guided search</a></p>'
 verbal=read('verbalizations-final/verbalizations.json')
 if verbal:
  content+='<h2>Learned system prompt: five verbalizations</h2><p>Independent direct samples from the best 256-vector checkpoint at step '+esc(verbal['checkpoint_step'])+'. Only learned embeddings in the system message; no fixed system text or thinking prefix. These are model interpretations, not an exact decoding or a behavioral evaluation. This run did not recover a readable system instruction: seed 42 generated an off-task answer to the token limit, and seeds 43–46 emitted immediate EOS.</p><pre>'+esc(verbal['rendered_context'])+'</pre>'
  for sample in verbal['samples']:
   content+='<h3>Seed '+esc(sample['seed'])+'</h3><pre>'+esc(sample['text'] or '[Empty continuation: immediate EOS]')+'</pre><p>Stop: '+esc(sample['stop_reason'])+'; tokens: '+esc(sample['tokens'])+'</p>'
 for name in ['teacher_reference.json','microbatch_benchmark.json','continuation.json','baselines.json','selection.json','objective_check.json','dataset.json','context_example.json','outcome.json','supervisor_failure.json','lifecycle.json']:
  d=read(name)
  if name=='dataset.json':d=d.get('audit',{})
  if d:content+=f'<details><summary>{esc(name)}</summary><pre>{esc(json.dumps(d,indent=2,ensure_ascii=False))}</pre></details>'
 content+=f'<p>Source commit: {esc(m.get("source_commit","pending"))}. Last rendered: {esc(time.ctime())}. A local copy can become stale while the laptop sleeps; execution and remote reporting continue.</p></html>'
 t=H/'report.html.tmp';t.write_text(content);t.replace(H/'report.html')
if __name__=='__main__':render()
