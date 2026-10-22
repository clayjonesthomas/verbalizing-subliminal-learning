"""Render the same durable report on the pod and locally."""
import json,html,time,statistics
from pathlib import Path
H=Path(__file__).resolve().parent;O=H/'results'
def read(n,default=None):
 try:return json.loads((O/n).read_text())
 except (FileNotFoundError,ValueError):return {} if default is None else default
def rows(n):
 try:return [json.loads(l) for l in (O/n).read_text().splitlines() if l.strip()]
 except (FileNotFoundError,ValueError):return []
def loss_chart(histories,axis,started):
 colors={128:'#2563eb',256:'#008566'}
 series=[(k,split,records) for k,records in histories.items() for split in ['train','valid'] if records]
 if not series:return '<p>Waiting for the first evaluation.</p>'
 xv=lambda r:r['step'] if axis=='step' else max(0,(r['time']-started)/3600)
 values=[r[split]['nll'] for k,split,records in series for r in records]
 xmax=max(1e-6,max(xv(r) for _,_,records in series for r in records));lo=min(values);hi=max(values);pad=max(.02,(hi-lo)*.08);lo-=pad;hi+=pad
 x=lambda v:75+v/xmax*790;y=lambda v:280-(v-lo)/(hi-lo)*220
 label='Optimizer step per candidate' if axis=='step' else 'Elapsed hours since original training start'
 parts=[f'<svg role="img" aria-label="Training and validation NLL by {axis}" viewBox="0 0 920 335" style="width:100%;background:#f8fafc"><title>Fixed-set training and validation NLL</title>']
 for i in range(6):
  v=lo+(hi-lo)*i/5;parts.append(f'<line x1="75" x2="865" y1="{y(v):.2f}" y2="{y(v):.2f}" stroke="#dce2ea"/><text x="65" y="{y(v)+4:.2f}" text-anchor="end" font-size="12">{v:.3f}</text>')
  t=xmax*i/5;parts.append(f'<text x="{x(t):.2f}" y="302" text-anchor="middle" font-size="12">{t:.0f}</text>' if axis=='step' else f'<text x="{x(t):.2f}" y="302" text-anchor="middle" font-size="12">{t:.2f}</text>')
 for k,split,records in series:
  color=colors[k];dash=' stroke-dasharray="7 5"' if split=='train' else ''
  pts=' '.join(f'{x(xv(r)):.2f},{y(r[split]["nll"]):.2f}' for r in records)
  parts.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2.5"{dash}/>')
  for r in records:
   title=f'{k} tokens · {split} · step {r["step"]} · NLL {r[split]["nll"]:.5f}'
   parts.append(f'<circle cx="{x(xv(r)):.2f}" cy="{y(r[split]["nll"]):.2f}" r="3" fill="{color}"><title>{html.escape(title)}</title></circle>')
 parts.append(f'<text x="470" y="327" text-anchor="middle" font-size="13">{label}</text><text x="18" y="170" transform="rotate(-90 18 170)" text-anchor="middle" font-size="13">Token-weighted NLL (nats)</text></svg>')
 return ''.join(parts)

def render():
 esc=lambda x:html.escape(str(x));c=json.loads((H/'config.json').read_text());m=read('manifest.json');s=read('supervisor.json');p=read('progress.json');table='';metrics=[];histories={}
 for k in c['lengths']:
  tr=rows(f'k{k}/training.jsonl');ev=rows(f'k{k}/evaluations.jsonl');metrics+=tr;histories[k]=ev
  if ev:
   e=ev[-1];table+=f'<tr><td>{k}</td><td>{tr[-1]["step"] if tr else 0}</td><td>{e["train"]["nll"]:.5f}</td><td>{e["valid"]["nll"]:.5f}</td><td>{min(x["valid"]["nll"] for x in ev):.5f}</td><td>{e["next_lr"]}</td><td>{esc(e["action"])}</td></tr>'
 latest=sorted(metrics,key=lambda x:x['time'])[-30:];speed='Waiting for optimizer steps'
 if latest:
  seconds=sum(x['seconds'] for x in latest);speed=f"Median {statistics.median(x['seconds'] for x in latest):.2f} s/step; {sum(x['processed_tokens'] for x in latest)/seconds:,.0f} processed tokens/s; peak allocated {max(x['peak_bytes'] for x in metrics)/1e9:.2f} GB"
 telemetry=rows('telemetry.jsonl');started=read('parent_manifest.json').get('start_time',m.get('start_time',time.time()));elapsed=time.time()-started
 charts='<h2>Training and validation NLL</h2><p>Fixed evaluation sets: 128 training examples and 200 validation examples. Blue: 128-vector prompt; green: 256-vector prompt. Dashed: train; solid: validation. Lower is better. Includes evaluations before the microbatch continuation; hover over points for exact values.</p>'+loss_chart(histories,'step',started)+'<details><summary>View by elapsed wall time</summary>'+loss_chart(histories,'time',started)+'</details>'
 content=f'''<!doctype html><html><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>System soft prompt</title><style>body{{font:16px system-ui;max-width:1100px;margin:32px auto;padding:0 24px;color:#172033}}table{{border-collapse:collapse;width:100%}}td,th{{padding:10px;text-align:left;border-bottom:1px solid #ddd}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#f5f7fa;padding:15px}}li{{margin:10px 0}}</style><h1>System soft prompt</h1><ul><li>Qwen3.5-4B: frozen BF16 weights; FP32 soft prompt; 128/256-vector candidates continued from saved optimizer states.</li><li>Larger microbatches; effective batch remains 16. Only learned embeddings in system; one user message with instruction, memory, query, baseline answer and format. No thinking prefix.</li><li>Same deterministic paired data/split; target-only loss; no reasoning, generation or search in training.</li><li>Compare 250–500 steps; plateau stopping; cap 5,000 steps/candidate, 30-hour emergency limit.</li><li>Pod-local execution, telemetry and report; terminate on success or failure; network volume preserved.</li></ul><h2>Status: {esc(s.get('status','preparing'))}</h2><pre>{esc(p)}</pre><p>{esc(speed)}</p><p>Training elapsed {elapsed/3600:.2f} hours; estimated GPU cost since training start ${elapsed/3600*c['hourly_gpu_cost']:.2f}, excluding setup/storage. GPU: {esc(m.get('gpu','pending'))}</p><p>Telemetry: {esc(telemetry[-1] if telemetry else 'pending')}</p>{charts}<h2>Latest losses</h2><table><tr><th>Vectors</th><th>Steps</th><th>Train NLL</th><th>Validation NLL</th><th>Best validation</th><th>LR</th><th>Decision</th></tr>{table}</table>'''
 for name in ['microbatch_benchmark.json','continuation.json','baselines.json','selection.json','objective_check.json','dataset.json','context_example.json','outcome.json','supervisor_failure.json','lifecycle.json']:
  d=read(name)
  if name=='dataset.json':d=d.get('audit',{})
  if d:content+=f'<details><summary>{esc(name)}</summary><pre>{esc(json.dumps(d,indent=2,ensure_ascii=False))}</pre></details>'
 content+=f'<p>Source commit: {esc(m.get("source_commit","pending"))}. Last rendered: {esc(time.ctime())}. A local copy can become stale while the laptop sleeps; execution and remote reporting continue.</p></html>'
 t=H/'report.html.tmp';t.write_text(content);t.replace(H/'report.html')
if __name__=='__main__':render()
