"""Combine managed watcher snapshots on report refresh; never poll or launch work."""
import argparse,json
from pathlib import Path
from report import render
p=argparse.ArgumentParser();p.add_argument('--reports',required=True,type=Path);a=p.parse_args()
shared=a.reports/'results/report-inputs';shared.mkdir(parents=True,exist_ok=True)
for w in range(5):
    source=a.reports/f'results/podrun/four-rules-20261026-seed{42+w}-a1'
    progress=source/'progress.json'
    if progress.exists(): (shared/f'progress-{w}.json').write_bytes(progress.read_bytes())
    training=source/'training.jsonl'
    if training.exists():
        history={}
        for line in training.read_text().splitlines():
            try:r=json.loads(line);history[r['step']]=r
            except (ValueError,KeyError):pass
        (shared/f'training-{w}.json').write_text(json.dumps(list(history.values())))
    extra=source/'report_data.json'
    if extra.exists():
        for name,data in json.loads(extra.read_text()).items():
            if name=='result.json':(shared/f'result-{w}.json').write_text(json.dumps(data))
            elif name=='summary.json':
                for row in data.get('results',[]):(shared/f"result-{row['seed']-42}.json").write_text(json.dumps(row))
    for n in ['result.json','summary.json']:
        file=source/n
        if not file.exists():continue
        data=json.loads(file.read_text())
        if n=='result.json':(shared/f'result-{w}.json').write_text(json.dumps(data))
        else:
            for row in data.get('results',[]):(shared/f"result-{row['seed']-42}.json").write_text(json.dumps(row))
render(shared,a.reports)

# A failed teacher check is a measured result, not a pending recovery result.
gate_path=Path(__file__).with_name('gate_result.json')
if gate_path.exists() and not list(shared.glob('result-*.json')):
    from report import CSS,table,chart_html,plotly_script
    gate=json.loads(gate_path.read_text())
    rows=[]
    for cat,r in gate['teacher'].items():
        b=gate['empty'][cat]
        rows.append([cat,f"{100*r['hit']:.1f}%",f"{100*r['fp']:.1f}%",f"{100*(r['hit']-r['fp']):.1f} pp",'Pass' if r['hit']-r['fp']>=.5 and b['hit']<=.1 else 'Fail'])
    body='<h2>Teacher check failed · training not started</h2><p>Qwen2.5-7B-Instruct · Hari’s exact four-rule prompt · 300 held-out gate questions</p>'
    for field,title in [('hit','Correct-category marker rate'),('fp','Wrong-category marker rate')]:
        traces=[dict(type='bar',name=label,x=list(gate[method]),y=[100*r[field] for r in gate[method].values()],marker={'color':color}) for method,label,color in [('teacher','Four-rule prompt','#16856b'),('empty','Empty prompt','#949ca9')]]
        body+=chart_html('gate-'+field,title,traces,x_label='Rule category',y_label='Percent',zero_y=True)
    body+=table(['Rule','Hit rate','False-positive rate','Hit − FP','Gate'],rows)
    body+='<p class="caption">50 positive and 250 negative queries per rule. Empty-prompt hit and false-positive rates: 0% throughout. Gate threshold: hit − FP ≥50 percentage points. Missing markers and incorrect category bindings occur in the saved answers; this is not solely punctuation mismatch.</p><h2>SALVE / greedy comparison</h2><p>Not run: the data-generating model did not pass the four-rule check.</p>'
    bundle=Path.home()/'.cache/research-infra/assets/plotly-3.1.0.min.js'
    doc='<!doctype html><meta charset="utf-8"><title>Four-rule teacher check</title><style>'+CSS+'</style>'+plotly_script(bundle)+body
    for name in ['in_progress.html','final_report.html']:(a.reports/name).write_text(doc)
