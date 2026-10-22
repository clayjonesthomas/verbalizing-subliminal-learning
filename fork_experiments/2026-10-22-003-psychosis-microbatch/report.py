"""Render the same durable report on the pod and locally."""
import json,html,time,statistics,hashlib,datetime
from pathlib import Path
H=Path(__file__).resolve().parent;O=H/'results'
def read(n,default=None):
 try:return json.loads((O/n).read_text())
 except (FileNotFoundError,ValueError):return {} if default is None else default
def rows(n):
 try:return [json.loads(l) for l in (O/n).read_text().splitlines() if l.strip()]
 except (FileNotFoundError,ValueError):return []
def loss_chart(histories,started):
 bundle=(H/'assets/plotly-3.1.0.min.js').read_bytes()
 assert hashlib.sha256(bundle).hexdigest()=='122e3be346d66616944d0b83eaaf7242581508c3c1cfa0995a17af0d83eff770'
 traces=[];steps=[];hours=[]
 for k,records in histories.items():
  for split in ['train','valid']:
   if not records:continue
   label=f'{k} vectors · '+('Train' if split=='train' else 'Validation')
   xs=[r['step'] for r in records];ts=[(r['time']-started)/3600 for r in records]
   custom=[[str(r[split]['nll']),r['step'],datetime.datetime.fromtimestamp(r['time'],datetime.timezone.utc).isoformat(),f'{t:.6f}'] for r,t in zip(records,ts)]
   traces.append(dict(type='scatter',mode='lines+markers',name=label,x=xs,y=[r[split]['nll'] for r in records],customdata=custom,visible=True,line=dict(color='#2563eb' if k==128 else '#008566',dash='dash' if split=='train' else 'solid'),marker=dict(size=7,symbol='circle' if split=='train' else 'diamond'),hovertemplate=label+'<br>NLL: %{customdata[0]}<br>Step: %{customdata[1]}<br>Elapsed hours: %{customdata[3]}<br>UTC: %{customdata[2]}<extra></extra>'))
   steps.append(xs);hours.append(ts)
 parts=['<script>'+bundle.decode()+'</script>']
 for axis,xvalues,title,xlabel in [('step',steps,'NLL by optimizer step','Optimizer step'),('time',hours,'NLL by elapsed time','Hours since original training start')]:
  data=[dict(trace,x=xs) for trace,xs in zip(traces,xvalues)]
  xmax=max((max(xs) for xs in xvalues if xs),default=1)
  references=[('Baseline (no soft prompt)',read('baselines.json').get('base',{}),'#64748b'),('Teacher',read('teacher_reference.json'),'#b45309')]
  for reference,measurements,color in references:
   for split in ['train','valid']:
    value=measurements.get(split,{}).get('nll')
    if value is None:continue
    label=reference+' · '+('Train' if split=='train' else 'Validation')
    data.append(dict(type='scatter',mode='lines',name=label,x=[0,xmax],y=[value,value],customdata=[str(value),str(value)],line=dict(color=color,dash='dot' if split=='train' else 'dashdot',width=2),hovertemplate=label+'<br>NLL: %{customdata}<extra></extra>'))
  layout=dict(paper_bgcolor='white',plot_bgcolor='white',font=dict(family='Arial, sans-serif'),height=570,margin=dict(l=75,r=30,t=25,b=135),hovermode='closest',hoverdistance=25,xaxis=dict(title=dict(text=xlabel),showspikes=True,spikemode='across',spikesnap='cursor'),yaxis=dict(title=dict(text='Token-weighted NLL (nats)'),showspikes=True),legend=dict(orientation='h',y=-.20))
  payload=json.dumps(dict(data=data,layout=layout),ensure_ascii=False).replace('<','\\u003c')
  chart_id='nll-'+axis
  parts.append('<h3>'+title+'</h3><div id="'+chart_id+'"></div><script>{const figure='+payload+';Plotly.newPlot("'+chart_id+'",figure.data,figure.layout,{responsive:true,displaylogo:false,scrollZoom:true,toImageButtonOptions:{format:"svg",filename:"'+chart_id+'"}});}</script>')
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
 charts='<h2>Training and validation NLL</h2><p>Hover over any marker to see the full stored NLL value, optimizer step, elapsed time, and timestamp. Both graphs and all curves are displayed below. Drag to zoom and double-click to reset. The 128-vector curves show historical results; only the 256-vector candidate is still training. Fixed sets: 128 training examples and 200 validation examples.</p>'+loss_chart(histories,started)
 if not read('teacher_reference.json'):charts+='<p>Teacher NLL reference: not yet measured for this run.</p>'
 content=f'''<!doctype html><html><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>System soft prompt</title><style>body{{font:16px system-ui;max-width:1100px;margin:32px auto;padding:0 24px;color:#172033}}table{{border-collapse:collapse;width:100%}}td,th{{padding:10px;text-align:left;border-bottom:1px solid #ddd}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#f5f7fa;padding:15px}}li{{margin:10px 0}}</style><h1>System soft prompt</h1><ul><li>Qwen3.5-4B: frozen BF16 weights; FP32 soft prompt; 128/256-vector candidates continued from saved optimizer states.</li><li>Larger microbatches; effective batch remains 16. Only learned embeddings in system; one user message with instruction, memory, query, baseline answer and format. No thinking prefix.</li><li>Same deterministic paired data/split; target-only loss; no reasoning, generation or search in training.</li><li>Compare 250–500 steps; plateau stopping; cap 5,000 steps/candidate, 30-hour emergency limit.</li><li>Pod-local execution, telemetry and report; terminate on success or failure; network volume preserved.</li></ul><h2>Status: {esc(s.get('status','preparing'))}</h2><pre>{esc(p)}</pre><p>{esc(speed)}</p><p>Training elapsed {elapsed/3600:.2f} hours; estimated GPU cost since training start ${elapsed/3600*c['hourly_gpu_cost']:.2f}, excluding setup/storage. GPU: {esc(m.get('gpu','pending'))}</p><p>Telemetry: {esc(telemetry[-1] if telemetry else 'pending')}</p>{charts}<h2>Latest losses</h2><table><tr><th>Vectors</th><th>Steps</th><th>Train NLL</th><th>Validation NLL</th><th>Best validation</th><th>LR</th><th>Decision</th></tr>{table}</table>'''
 for name in ['microbatch_benchmark.json','continuation.json','baselines.json','selection.json','objective_check.json','dataset.json','context_example.json','outcome.json','supervisor_failure.json','lifecycle.json']:
  d=read(name)
  if name=='dataset.json':d=d.get('audit',{})
  if d:content+=f'<details><summary>{esc(name)}</summary><pre>{esc(json.dumps(d,indent=2,ensure_ascii=False))}</pre></details>'
 content+=f'<p>Source commit: {esc(m.get("source_commit","pending"))}. Last rendered: {esc(time.ctime())}. A local copy can become stale while the laptop sleeps; execution and remote reporting continue.</p></html>'
 t=H/'report.html.tmp';t.write_text(content);t.replace(H/'report.html')
if __name__=='__main__':render()
