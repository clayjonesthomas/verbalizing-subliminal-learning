"""One shared, output-focused experiment report for both workers."""
import os,json,time,html,fcntl
from pathlib import Path
from research_graphing import plotly_script,line_trace,reference_line,chart_html
from research_graphing.timefmt import local_time
LAST=0
ESC=lambda x:html.escape(str(x))
def read(p):
 try:return json.loads(p.read_text())
 except (FileNotFoundError,json.JSONDecodeError):return {}
def atomic(p,text):
 p.parent.mkdir(exist_ok=True,parents=True);tmp=p.with_name(p.name+'.'+str(os.getpid())+'.tmp');tmp.write_text(text);tmp.replace(p)
def render(c,force=False):
 global LAST
 if not force and time.time()-LAST<25:return
 LAST=time.time();root=Path(c['shared_root']);out=Path(os.environ['PODRUN_REPORT_DIR']);root.mkdir(exist_ok=True,parents=True)
 with (root/'report.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  runs={m:read(root/f'results/{m}/final.json') for m in ['beam','bon']};progress={m:read(root/f'results/{m}/progress.json') for m in runs}
  head='<!doctype html><meta charset="utf-8"><title>Full soft-prompt readout</title><style>body{font:16px system-ui;max-width:1200px;margin:32px auto;padding:0 24px;color:#172033}table{border-collapse:collapse;width:100%;margin:18px 0}td,th{padding:10px;border-bottom:1px solid #ddd;text-align:left;vertical-align:top}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f4f6f9;padding:16px}code{white-space:pre-wrap}</style><h1>SALVE and best-of-N</h1>'
  body='<table><tr><th>Worker</th><th>Status</th><th>Scored</th><th>Attempts</th><th>Scoring / generation batch</th></tr>'
  for m,r in runs.items():
   perf=r.get('performance',{});body+=f'<tr><td>{m}</td><td>{ESC(r.get("status",progress[m].get("stage","pending")))}</td><td>{r.get("scored",0)}</td><td>{r.get("attempted",0)}</td><td>{perf.get("scoring_batch","—")} / {perf.get("generation_batch","—")}</td></tr>'
  body+='</table>';traces=[];colors=['#2563eb','#008566','#9333ea'];k=0
  for mode,r in runs.items():
   labels=sorted(set(x['variant'] for x in r.get('history',[])))
   for label in labels:
    rows=[x for x in r['history'] if x['variant']==label];traces.append(line_trace(label,[x['step'] for x in rows],[x['best_nll'] for x in rows],value_label='Selection NLL',color=colors[k%3],hover_fields={'Candidates':[x['step'] for x in rows]}));k+=1
  xmax=max((max(t['x']) for t in traces if t['x']),default=768)
  refs=next((r['references'] for r in runs.values() if r.get('references',{}).get('learned')), {})
  for label,color in [('empty','#64748b'),('learned','#b45309')]:
   if label in refs:traces.append(reference_line(label+' system',refs[label]['train']['nll'],[0,xmax],color=color))
  body+=plotly_script(c['plotly_bundle'])+chart_html('selection','Best selection NLL',traces,x_label='Candidates / best-of-N draws',y_label='NLL')
  atomic(out/'in_progress.html',head+body)
  atomic(out/'final_report.html',head+('<p>Results in progress</p>' if not all(r.get('status')=='complete' for r in runs.values()) else '')+body)
  methods=head.replace('SALVE and best-of-N','Methods &amp; Config')+'<table><tr><th>Checkpoint</th><td>Fixed step 1150, 256 vectors, frozen BF16 Qwen3.5-4B. No training.</td></tr><tr><th>SALVE</th><td>4 beams, 16 continuations per expanded leaf, 12 rounds, 32 generated tokens per extension, 256-token candidate cap. Empty root; best-first unexpanded leaves, same local reference algorithm. Balanced shuffle-bag across the four paper queries.</td></tr><tr><th>Best-of-N</th><td>Independent 128- and 256-token sample streams, 768 draws each, nested smaller N and exact beam-candidate-count matching. Random paper queries. Reuse scores of identical text; preserve all scored slots.</td></tr><tr><th>Selection</th><td>Fixed torch-randperm subset of 256 training examples, seed 42; target-token-weighted NLL. Never select using validation.</td></tr><tr><th>Validation</th><td>All 200 held-out examples, evaluated after search for shortlisted candidates and best-of-N record breakers.</td></tr><tr><th>Sampling</th><td>Temperature 0.7; other sampling parameters inherited from the model generation config. Standard Qwen non-thinking template, no fixed system text. Exact no-thinking-prefix training template retained for scoring.</td></tr><tr><th>Parsing</th><td>Reference SALVE cut_at_sentence and clean_verbalization, including first closing delimiter. Raw continuations retained in candidates.jsonl and rejected.jsonl.</td></tr><tr><th>Performance</th><td>Length-bucketed batches, cached batched generation, chunked FP32 vocabulary loss from BF16 weights. Per-pod measured batch tuning and singleton/batch NLL check. RNG depends on measured batch size; this preserves the sampling distribution, not serial bitwise identity.</td></tr></table><pre>'+ESC(json.dumps(c,indent=2))+'</pre>'
  for m,r in runs.items():methods+='<h2>'+m+'</h2><pre>'+ESC(json.dumps({k:r.get(k) for k in ['source_commit','checkpoint_sha256','performance']},indent=2))+'</pre>'
  atomic(out/'methods.html',methods)
