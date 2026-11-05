"""Standalone auto-refreshing report; no server or external assets required."""
import json,html,time,math,argparse
from pathlib import Path
HERE=Path(__file__).resolve().parent

def read(path,default):
    try:return json.loads(path.read_text())
    except (OSError,ValueError):return default

def rows(path):
    try:return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    except (OSError,ValueError):return []

def fmt(v):return '—' if v is None else f'{v:.4f}'
def plot(series,title,ylabel):
    points=[p for _,_,ps in series for p in ps]
    if not points:return '<div class="empty">'+html.escape(title)+': awaiting measurements</div>'
    xmax=max(1,max(x for x,y in points));ys=[y for x,y in points];lo=min(ys);hi=max(ys);pad=max((hi-lo)*.12,.0001);lo-=pad;hi+=pad
    X=lambda x:70+760*x/xmax;Y=lambda y:265-220*(y-lo)/(hi-lo)
    s=['<div class="chart"><h3>'+html.escape(title)+'</h3><svg viewBox="0 0 880 320" role="img" aria-label="'+html.escape(title)+'">']
    for i in range(6):
        v=lo+(hi-lo)*i/5;y=Y(v);s.append(f'<line x1="70" x2="830" y1="{y}" y2="{y}" stroke="#e5eaf1"/><text x="60" y="{y+4}" text-anchor="end">{v:.3f}</text>')
        x=xmax*i/5;s.append(f'<text x="{X(x)}" y="287" text-anchor="middle">{x:.0f}</text>')
    for name,color,ps in series:
        coords=' '.join(f'{X(x):.2f},{Y(y):.2f}' for x,y in ps)
        s.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.5" points="{coords}"/>')
        for x,y in ps:s.append(f'<circle cx="{X(x)}" cy="{Y(y)}" r="3" fill="{color}"><title>{html.escape(name)} step {x}: {y:.5f}</title></circle>')
    s.append('<text x="450" y="314" text-anchor="middle">Optimizer step (per candidate)</text></svg><div class="legend">')
    s.extend(f'<span style="color:{color}">● {html.escape(name)}</span>' for name,color,_ in series)
    s.append('</div><small>'+html.escape(ylabel)+'</small></div>');return ''.join(s)

def render(here=HERE):
    out=here/'results';p=read(out/'progress.json',{});manifest=read(out/'manifest.json',{});base=read(out/'baselines.json',{});selection=read(out/'selection.json',{});outcome=read(out/'outcome.json',{});lifecycle=read(out/'lifecycle.json',{});pod=read(out/'pod.json',{});sync=read(out/'sync.json',{});fail=read(out/'failure.json',{})
    cfg=read(here/'config.json',{});hist={k:rows(out/f'k{k}'/'evaluations.jsonl') for k in [128,256]};training={k:rows(out/f'k{k}'/'training.jsonl') for k in [128,256]}
    status='FAILED' if fail else lifecycle.get('stage',p.get('stage','Preparing dedicated pod'))
    cleanup=read(out/'cleanup_verified.json',{});retirement=read(out/'retirement.json',{})
    if cleanup.get('terminated') or retirement.get('status')=='termination_accepted':
        status='Training complete · pod terminated' if outcome.get('status')=='complete' else 'Run ended · pod terminated'
    updated=p.get('time',sync.get('time',time.time()));esc=html.escape
    bullets=[
      'Scope: soft-prompt NLL only; no verbalization, search, generation, or behavioral judging.',
      'Data: 10,000 training traces; 100 validation traces; fixed 128-example training diagnostic. Test split is unused.',
      'Continuation: 256 embeddings from step 2,500; optimizer, RNG and plateau state preserved. Original 128 comparison is shown for context.',
      'Schedule retained: no new warmup; resume LR 0.00009375 and patience 3/5. Halve after five evaluations without ≥0.005 improvement; stop if the reduced-rate trial also stalls.',
      'Dedicated pod: '+str(pod.get('id','pending'))+' · cap: 5,000 total steps / 16 additional hours wall time. Best and resumable checkpoints stay on the network volume.'
    ]
    parts=['<!doctype html><html><head><meta charset="utf-8"><meta http-equiv="refresh" content="20"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Psychosis SALVE · 5k</title><style>body{background:#f6f8fc;color:#172033;font:16px system-ui;margin:0 auto;max-width:1180px;padding:32px}h1{margin-bottom:4px}h2{margin-top:32px}small,.muted{color:#52647b}li{margin:9px 0}.card,.chart,.empty{background:#ffffff;border:1px solid #d9e1eb;border-radius:12px;padding:20px;margin:18px 0}.status{font-size:22px;color:#166b51}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:11px;border-bottom:1px solid #d9e1eb;text-align:left}svg{width:100%;font:12px system-ui;fill:#52647b}.legend{display:flex;gap:20px;flex-wrap:wrap}.warning{color:#9a4a0e}code{overflow-wrap:anywhere}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px}</style></head><body><h1>Psychosis SALVE · 5k</h1><p class="muted">Live soft-prompt convergence · page reloads every 20 seconds</p><ul>']
    parts.extend('<li>'+esc(b)+'</li>' for b in bullets);parts.append('</ul>')
    parts.append('<div class="card"><div class="status">'+esc(str(status))+'</div><p>'+esc(str(p))+'</p><small>Last training/evaluation heartbeat: '+time.strftime('%Y-%m-%d %H:%M:%S UTC',time.gmtime(updated))+' · report generated '+time.strftime('%H:%M:%S UTC',time.gmtime())+'</small><p id="stale"></p></div>')
    if sync.get('error'):parts.append('<p class="warning">Sync error: '+esc(sync['error'])+'</p>')
    if fail:parts.append('<div class="card warning">Run failed; pod-local cleanup saves partial artifacts and terminates the pod.<pre>'+esc(json.dumps(fail,indent=2))+'</pre></div>')
    resumecheck=read(out/'resume_validation.json',{})
    if resumecheck:parts.append('<div class=card>Resume validation check: '+esc(str({k:v for k,v in resumecheck.items() if k not in ['validation','per_example']}))+'</div>')
    if outcome:parts.append('<div class="card">Run endpoint: '+esc(outcome.get('reason',''))+'. A budget endpoint is not a claim of convergence.</div>')
    if outcome.get('status')=='complete' and lifecycle.get('traceback'):
        parts.append('<p class="warning">Post-training environment export failed during pip freeze. Training and evaluations completed; best/latest checkpoints and logs are preserved. The pod-local guardian terminated the pod under the requested failure policy.</p>')
    parts.append('<h2>Candidate comparison</h2><table><tr><th>Embeddings</th><th>Trained steps</th><th>Fixed train NLL</th><th>Validation NLL</th><th>Best validation</th><th>Current / next LR</th><th>Last decision</th></tr>')
    for k in [128,256]:
        h=hist[k];last=h[-1] if h else {};t=training[k][-1] if training[k] else {};best=min((x['valid']['nll'] for x in h),default=None)
        parts.append(f'<tr><td>{k}</td><td>{t.get("step",0)}</td><td>{fmt(last.get("train",{}).get("nll"))}</td><td>{fmt(last.get("valid",{}).get("nll"))}</td><td>{fmt(best)}</td><td>{fmt(last.get("next_lr",t.get("lr")))}</td><td>{esc(last.get("action","pending"))}</td></tr>')
    parts.append('</table><p class="muted">Comparisons use fixed evaluation sets; minibatch training loss is logged separately. NLL is in nats/token.</p>')
    if selection:parts.append('<p>Selected: '+str(selection['selected'])+' embeddings after '+str(selection['comparison_steps'])+' steps each. Latest validation gap: '+fmt(selection['gap'])+'.</p>')
    series=[]
    for k,colors in [(128,['#2563eb','#7797df']),(256,['#008566','#65a892'])]:
        for split,color in zip(['valid','train'],colors):series.append((f'{k} {split}',color,[(r['step'],r[split]['nll']) for r in hist[k]]))
    parts.append(plot(series,'Fixed-set loss curves','Lower is better; same examples at every checkpoint.'))
    parts.append(plot([(name,color,[(x,y) for x,y in ps if x>=2000]) for name,color,ps in series if name.startswith('256')],'Late-training loss curves (step ≥ 2,000)','Continuation begins after step 2,500; lower is better.'))
    parts.append('<h2>Fixed-set loss data</h2><table><tr><th>Step</th><th>128 train NLL</th><th>128 validation NLL</th><th>256 train NLL</th><th>256 validation NLL</th></tr>')
    indexed={k:{r['step']:r for r in hist[k]} for k in [128,256]}
    for step in sorted(set(indexed[128])|set(indexed[256])):
        values=[indexed[k].get(step,{}).get(split,{}).get('nll') for k in [128,256] for split in ['train','valid']]
        parts.append('<tr><td>'+str(step)+'</td>'+''.join('<td>'+('—' if v is None else f'{v:.6f}')+'</td>' for v in values)+'</tr>')
    parts.append('</table><p class="muted">These are the points plotted above, rounded to six decimal places. A dash means no evaluation was recorded for that candidate at that step.</p>')
    if selection and hist.get(selection['selected']):
        latest=hist[selection['selected']][-1];state=latest.get('state',{});patience=cfg.get('patience',5)
        phase='Stop after an unsuccessful reduced-LR trial' if state.get('reduced_waiting') else 'Halve learning rate on plateau'
        parts.append('<h2>Stopping status</h2><table><tr><th>Selected embeddings</th><th>Evaluated step</th><th>Learning rate</th><th>Plateau counter</th><th>Next action at threshold</th></tr>')
        parts.append('<tr><td>'+str(selection['selected'])+'</td><td>'+str(latest['step'])+'</td><td>'+str(state.get('rate'))+'</td><td>'+str(state.get('bad_evals',0))+' / '+str(patience)+'</td><td>'+esc('Stopped' if state.get('stopped') else phase)+'</td></tr></table>')
        if 'anchor' in state:
            threshold=state['anchor']-cfg.get('min_delta',.005)
            parts.append('<p class="muted">The counter resets when validation NLL falls below '+f'{threshold:.6f}'+' (last meaningful-improvement reference '+f'{state["anchor"]:.6f}'+' minus '+str(cfg.get('min_delta',.005))+'). Small improvements accumulate against that reference; the rule is not based solely on the previous evaluation. Evaluations occur every '+str(cfg.get('eval_every',50))+' steps. Five unsuccessful evaluations first reduce LR; stopping requires a further unsuccessful trial at the reduced LR.</p>')
    parts.append(plot([(str(k),'#2563eb' if k==128 else '#008566',[(r['step'],r['lr']) for r in training[k]]) for k in [128,256]],'Learning rate','Warmup and observed plateau-triggered changes.'))
    parts.append('<h2>Validation loss components</h2><table><tr><th>Model/prompt</th><th>All tokens</th><th>Reasoning</th><th>Answer content</th><th>Formatting + EOS</th></tr>')
    comparisons=[(name,base.get(name,{}).get('valid',{})) for name in ['teacher','base']]+[(str(k)+' embeddings',hist[k][-1]['valid'] if hist[k] else {}) for k in [128,256]]
    for name,values in comparisons:
        parts.append('<tr><td>'+esc(name)+'</td>'+''.join('<td>'+fmt(v)+'</td>' for v in [values.get('nll')]+[values.get(g,{}).get('nll') for g in ['reasoning','answer','format']])+'</tr>')
    parts.append('</table><p class="muted">Answer loss is teacher-forced: conditioned on the saved teacher reasoning. Teacher likelihood is a reference, not a lower bound. Full-trace loss includes every target token; input tokens are masked.</p><h2>Throughput and provenance</h2><table><tr><th>Embeddings</th><th>Mean seconds/step</th><th>Response tokens processed</th><th>Peak allocated GiB</th></tr>')
    for k in [128,256]:
        ts=training[k];parts.append(f'<tr><td>{k}</td><td>{sum(t["seconds"] for t in ts)/len(ts):.1f}</td><td>{sum(t["tokens"] for t in ts):,}</td><td>{max(t["peak_bytes"] for t in ts)/2**30:.1f}</td></tr>' if ts else f'<tr><td>{k}</td><td colspan="3">Pending</td></tr>')
    parts.append('</table><p>Code commit: <code>'+esc(manifest.get('source_commit','pending'))+'</code></p><p>Data source commit: <code>'+esc(cfg.get('data_source_commit','pending'))+'</code></p><p>Model: Qwen3-30B-A3B · frozen BF16 weights · FP32 soft parameters and optimizer state.</p>')
    parts.append('<script>const age=Date.now()/1000-'+str(updated)+';if(age>180)document.getElementById("stale").textContent="Heartbeat is "+Math.round(age/60)+" minutes old. Check lifecycle/sync status; this page does not imply the job is still running.";</script></body></html>')
    dst=here/'report.html';tmp=dst.with_suffix('.html.tmp');tmp.write_text(''.join(parts));tmp.replace(dst)
    return dst
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,default=HERE);a=p.parse_args();print(render(a.directory))
