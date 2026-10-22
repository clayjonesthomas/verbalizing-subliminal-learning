"""Standalone live report from a compact remote snapshot."""
import html,json,time,datetime,statistics
from pathlib import Path
HERE=Path(__file__).resolve().parent

def chart(series):
    points=[(r['step'],r[split]['nll']) for _,split,rows,color in series for r in rows]
    if not points:return '<p>Waiting for evaluation.</p>'
    xmax=max(50,max(x for x,y in points));lo=min(y for x,y in points)-.03;hi=max(y for x,y in points)+.03
    x=lambda v:65+v/xmax*790;y=lambda v:255-(v-lo)/(hi-lo)*210
    grid=''.join(f'<text x="55" y="{y(v)+4:.1f}" text-anchor="end">{v:.3f}</text><line x1="65" x2="855" y1="{y(v):.1f}" y2="{y(v):.1f}" stroke="#e2e8f0"/>' for v in [lo+(hi-lo)*i/4 for i in range(5)])
    paths=''
    for label,split,rows,color in series:
        pts=' '.join(f'{x(r["step"]):.1f},{y(r[split]["nll"]):.1f}' for r in rows)
        paths+=f'<polyline points="{pts}" stroke="{color}" fill="none" stroke-width="2"/>'
        for r in rows:paths+=f'<circle cx="{x(r["step"]):.1f}" cy="{y(r[split]["nll"]):.1f}" r="3" fill="{color}"><title>{label}: step {r["step"]}, NLL {r[split]["nll"]:.5f}</title></circle>'
    return f'<svg viewBox="0 0 900 300">{grid}{paths}<text x="65" y="280">0</text><text x="855" y="280" text-anchor="end">{xmax}</text><text x="450" y="297" text-anchor="middle">Optimizer step per candidate</text></svg>'

def render(data,connection='Connected'):
    esc=lambda v:html.escape(str(v));now=time.time();when=data.get('time',now)
    status=data.get('supervisor',{}).get('status',data.get('progress',{}).get('stage','preparing'))
    if data.get('failure'):status='failed'
    manifest=data.get('manifest',{});started=manifest.get('start_time',when)
    elapsed=max(0,when-started);cost=elapsed/3600*1.29
    summaries='';series=[];history='';all_steps=[]
    for k,color in [(128,'#2563eb'),(256,'#008566')]:
        tr=data.get(f'train{k}',[]);ev=data.get(f'eval{k}',[]);all_steps+=tr
        if ev:
            r=ev[-1];v=r['valid']['nll'];baseline=data.get('baselines',{}).get('base',{}).get('valid',{}).get('nll')
            gain=f'{baseline-v:.4f}' if baseline is not None else '—'
            summaries+=f'<tr><td>{k}</td><td>{tr[-1]["step"] if tr else 0}</td><td>{r["train"]["nll"]:.4f}</td><td>{v:.4f}</td><td>{min(a["valid"]["nll"] for a in ev):.4f}</td><td>{gain}</td><td>{r["next_lr"]:.5f}</td><td>{esc(r["action"])}</td></tr>'
            for split in ['train','valid']:series.append((f'{k} {split}',split,ev,color if split=='valid' else ('#93b4f3' if k==128 else '#80cabb')))
            for r in ev:history+=f'<tr><td>{k}</td><td>{r["step"]}</td><td>{r["train"]["nll"]:.5f}</td><td>{r["valid"]["nll"]:.5f}</td><td>{esc(r["action"])}</td></tr>'
    recent=sorted(all_steps,key=lambda r:r['time'])[-30:]
    speed='Waiting for training steps.'
    if recent:
        secs=sum(r['seconds'] for r in recent);tokens=sum(r['processed_tokens'] for r in recent)
        speed=f'{statistics.median(r["seconds"] for r in recent):.2f} s/step median · {tokens/secs:,.0f} processed tokens/s · {sum(r["tokens"] for r in recent)/secs:,.0f} target tokens/s · peak allocated {max(r["peak_bytes"] for r in all_steps)/1e9:.2f} GB'
    failure='<pre>'+esc(data.get('failure') or data.get('supervisor',{}).get('traceback',''))+'</pre>' if status=='failed' else ''
    audit=data.get('dataset',{}).get('audit',{})
    doc=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="refresh" content="20"><title>Paired soft prompt</title><style>body{{font:15px system-ui;color:#172033;background:#f8fafc;max-width:1100px;margin:32px auto;padding:0 24px}}li{{margin:8px 0}}table{{border-collapse:collapse;width:100%;background:white}}th,td{{text-align:left;padding:10px;border-bottom:1px solid #e2e8f0}}svg{{width:100%;background:white}}svg text{{font:12px system-ui;fill:#64748b}}pre{{white-space:pre-wrap;overflow-wrap:anywhere}}.muted{{color:#64748b}}.card{{padding:20px;background:white;border:1px solid #e2e8f0;border-radius:10px;margin:20px 0}}</style>
<h1>Paired soft prompt</h1><ul><li>Qwen3.5-4B · frozen BF16 backbone · only 128/256 soft-prompt embeddings trained.</li><li>Exact original system/user messages + baseline final answer → trained final answer; reasoning excluded.</li><li>Fixed holdout by original context: {audit.get('train',{}).get('n','pending')} train / {audit.get('valid',{}).get('n','pending')} validation; 128 fixed training diagnostics.</li><li>Compare 250 steps each, extend to 500 if close; halve LR on plateau, then stop if it still stalls. Cap: 2,500 steps/candidate or 12 hours.</li><li>Pod i2p4nlv8scafb7 · 48 GB Blackwell MIG · $1.29/hour GPU · pod-local cleanup terminates on success or failure; network volume preserved, even while the laptop sleeps.</li></ul>
<div class="card"><b>Status: {esc(status)}</b><p>{esc(data.get('progress',{}))}</p><p>{esc(connection)} · last sample {datetime.datetime.fromtimestamp(when).astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')}</p><p>{esc(speed)}</p><p>Run elapsed {elapsed/3600:.2f} h · estimated GPU cost since run start ${cost:.2f} (excludes setup/storage).</p><p>GPU telemetry: {esc(data.get('gpu','pending'))}</p></div>{failure}
<h2>Fixed-set losses</h2><p>Token-weighted NLL in nats, lower is better. Blue: 128 tokens; green: 256. Pale lines: train; dark: validation. Gain compares against the same model/context without a soft prompt.</p>{chart(series)}
<table><tr><th>Prompt</th><th>Trained steps</th><th>Train NLL</th><th>Valid NLL</th><th>Best valid</th><th>Gain</th><th>Next LR</th><th>Decision</th></tr>{summaries}</table>
<p>Selection: {esc(data.get('selection','pending'))}</p><h2>Evaluation history</h2><table><tr><th>Prompt</th><th>Step</th><th>Train NLL</th><th>Valid NLL</th><th>Action</th></tr>{history}</table>
<h2>Cleanup</h2><pre>{esc(json.dumps(data.get("cleanup",{}),indent=2))}</pre><h2>Data and correctness checks</h2><pre>{esc(json.dumps(audit,indent=2))}</pre><p>Objective/gradient preflight: {esc(data.get('objective_check','pending'))}</p><p>No-prompt baseline: {esc(data.get('baselines',{}))}</p><p class="muted">Validation measures fit to saved outputs, not behavioral quality. No verbalization or generation is run. Source commit: {esc(manifest.get('source_commit','pending'))}.</p>
<script>if(Date.now()-{when*1000}>180000 && {str(status not in ['success','complete']).lower()}){{let p=document.createElement('p');p.style.color='#b91c1c';p.textContent='STALE: last sample is over three minutes old.';document.body.prepend(p);}}</script></html>'''
    tmp=HERE/'report.html.tmp';tmp.write_text(doc);tmp.replace(HERE/'report.html')
if __name__=='__main__':
    p=HERE/'results/report_snapshot.json';render(json.loads(p.read_text()) if p.exists() else {},'Waiting for watcher')
