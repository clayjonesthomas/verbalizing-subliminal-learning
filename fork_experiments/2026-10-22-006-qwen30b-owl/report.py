"""Self-contained SVG charts and full greedy readouts; no external assets."""
import html,json
from datetime import datetime,timezone
from pathlib import Path

def build(out,c):
    out=Path(out)
    def read(name,default):
        p=out/name;return json.loads(p.read_text()) if p.exists() else default
    p=read('progress.json',{'stage':'preparing'});refs=read('references.json',{})
    rows=[json.loads(l) for l in (out/'trajectory.jsonl').read_text().splitlines()] if (out/'trajectory.jsonl').exists() else []
    hist=read('training_history.json',{}).get('train_loss',[])
    esc=lambda x:html.escape(str(x))
    def chart(split):
        series=[]
        if split=='train' and hist:
            series.append(('Minibatch train (25-step mean)','#b8c4d5',[(i+len(hist[i:i+25]),sum(hist[i:i+25])/len(hist[i:i+25])) for i in range(0,len(hist),25)]))
        for key,label,color in [('soft','Soft prompt','#2563eb'),('greedy','Greedy hard prompt','#d97706')]:
            series.append((label,color,[(r['step'],r[f'{key}_{split}_nll']) for r in rows]))
        for key,color in [('teacher','#16a34a'),('baseline_empty','#dc2626'),('baseline_no_system','#9333ea')]:
            if key in refs:series.append((key,color,[(0,refs[key][split]),(c['steps'],refs[key][split])]))
        vals=[v for _,_,points in series for _,v in points]
        if not vals:return '<p>Awaiting NLL measurements.</p>'
        lo=max(0,min(vals)-.08);hi=max(vals)+.08
        x=lambda n:65+820*n/c['steps'];y=lambda n:290-250*(n-lo)/(hi-lo)
        svg='<svg viewBox="0 0 940 350" role="img" aria-label="'+split+' NLL"><rect width="940" height="350" fill="white"/>'
        for i in range(6):
            v=lo+(hi-lo)*i/5;yy=y(v)
            svg+=f'<path d="M65 {yy} H885" stroke="#ddd"/><text x="4" y="{yy+4}" font-size="13">{v:.3f}</text>'
        for step in range(0,c['steps']+1,250):svg+=f'<text x="{x(step)-12}" y="315" font-size="12">{step}</text>'
        for label,color,points in series:
            pts=' '.join(f'{x(a):.2f},{y(b):.2f}' for a,b in points)
            svg+=f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2"><title>{esc(label)}</title></polyline>'
            if len(points)==1:svg+=f'<circle cx="{x(points[0][0])}" cy="{y(points[0][1])}" r="4" fill="{color}"/>'
        svg+='<text x="420" y="342" font-size="14">Optimizer step</text></svg>'
        return svg+'<p>'+ ' · '.join(f'<span style="color:{col}">{esc(label)}</span>' for label,col,_ in series)+'</p>'
    body='''<!doctype html><html><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Qwen 30B owl</title><style>body{font:16px system-ui;color:#172033;max-width:1100px;margin:32px auto;padding:0 24px}li{margin:9px 0}table{border-collapse:collapse;width:100%}td,th{padding:10px;border-bottom:1px solid #ddd;text-align:left}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f5f7fa;padding:16px}svg{width:100%}</style><h1>Qwen 30B owl · seed 42</h1><ul><li>Matched Qwen3-30B-A3B teacher/student; paper owl system prompt; numeric-only data; thinking disabled.</li><li>10,000 training examples and 500 held-out validation examples; seed 42 only.</li><li>128 soft vectors, 2,500 Adam updates; greedy readout every 250 steps; no search or text resets.</li><li>Training evaluation uses a fixed 500-example subset; validation uses all 500 examples. NLL is weighted by response-token count.</li><li>Greedy wording flags are screening heuristics; inspect full prompts for genuine preference statements. Other seeds await approval.</li></ul>'''
    body=body.replace('<meta charset="utf-8">','<meta charset="utf-8"><meta http-equiv="refresh" content="30">')
    updated=p.get('updated_unix')
    stamp=datetime.fromtimestamp(updated,timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC') if updated else 'pending'
    body+=f'<h2>Status: {esc(p["stage"])}</h2><p>Progress recorded: {esc(stamp)}. Page reloads every 30 seconds; the laptop syncs every 60 seconds while awake.</p>'
    body+=f'<p id="freshness" data-updated="{updated or 0}"></p><script>function freshness(){{let e=document.getElementById("freshness"),t=Number(e.dataset.updated),age=Math.max(0,Math.floor(Date.now()/1000-t));e.textContent=t?"Progress age: "+age+" seconds."+(age>180?" No recent progress update; the current stage may be busy, or syncing may be paused.":""):"Awaiting first progress update.";}}freshness();setInterval(freshness,1000);</script>'
    if p['stage']=='generating_data' and 'accepted_target' in p:
        done=p['accepted'];target=p['accepted_target']
        body+=f'<p><strong>{done:,} / {target:,} accepted examples</strong> from {p["queries_done"]:,} candidate responses. Generation stops at the acceptance target.</p><progress value="{done}" max="{target}" style="width:100%"></progress>'
        body+='<table><tr><th>Worker</th><th>Status</th><th>Accepted / quota</th><th>Candidates</th></tr>'
        for w in p.get('workers',[]):body+=f'<tr><td>{w["worker"]}</td><td>{esc(w["stage"])}</td><td>{w["accepted"]:,} / {w["quota"]:,}</td><td>{w["generated"]:,}</td></tr>'
        body+='</table><h3>Recent generated examples</h3>'
        for e in p.get('latest_examples',[]):body+=f'<p>Worker {e["worker"]} · query {e["query_id"]} · {"Accepted" if e["accepted"] else "Rejected"}</p><pre>{esc(e["completion"])}</pre>'
    elif p['stage']=='generating_data':
        done=p.get('queries_done',0);total=p.get('queries_total',0);accepted=p.get('accepted',0)
        rate=100*accepted/done if done else 0
        remaining=(total-done)*p.get('elapsed_seconds',0)/done/60 if done else 0
        body+=f'<p><strong>{done:,} / {total:,} candidate responses generated; {accepted:,} accepted ({rate:.1f}%).</strong> Estimated generation time remaining: {remaining:.0f} minutes. Training has not started.</p><progress value="{done}" max="{total or 1}" style="width:100%"></progress><p>The configured 30,000 are raw candidates before filtering. Only 10,000 accepted training examples and 500 validation examples are used.</p>'
    body+=f'<pre>{esc(json.dumps({k:v for k,v in p.items() if k!="latest"},indent=2))}</pre>'
    for split in ('train','validation'):body+=f'<h2>{split.title()} NLL</h2>'+chart(split)
    body+='<h2>Reference NLL</h2><table><tr><th>Condition</th><th>Train subset</th><th>Validation</th></tr>'
    for key,r in refs.items():body+=f'<tr><td>{esc(key)}</td><td>{r["train"]:.5f}</td><td>{r["validation"]:.5f}</td></tr>'
    body+='</table><h2>Greedy readouts</h2><table><tr><th>Step</th><th>Soft train / val</th><th>Hard train / val</th><th>Owl mention</th><th>Preference-language heuristic</th></tr>'
    for r in rows:body+=f'<tr><td><a href="#s{r["step"]}">{r["step"]}</a></td><td>{r["soft_train_nll"]:.4f} / {r["soft_validation_nll"]:.4f}</td><td>{r["greedy_train_nll"]:.4f} / {r["greedy_validation_nll"]:.4f}</td><td>{r["mentions_owl"]}</td><td>{r["preference_heuristic"]}</td></tr>'
    body+='</table>'
    for r in rows:body+=f'<h3 id="s{r["step"]}">Step {r["step"]}</h3><pre>{esc(r["prompt"])}</pre>'
    body+='<p>Teacher uses the original owl system prompt; baseline_empty uses an empty system message, baseline_no_system omits it. This is a greedy-only SALVE ablation, not a test of the full search algorithm.</p></html>'
    tmp=out/'scientific-report.tmp';tmp.write_text(body);tmp.replace(out/'scientific-report.html')

if __name__=='__main__':
    import os
    build(os.environ['PODRUN_DIR'],json.loads((Path(__file__).parent/'config.json').read_text()))
