import html
import json
import os
import sys
from pathlib import Path

from common import C, HERE, REPORT, RUN, atomic
sys.path.insert(0, C['graphing_path'] if os.environ.get('PODRUN_DIR') else str(Path('~/code/research-infra/graphing-utils').expanduser()))
from research_graphing import chart_grid, chart_html, line_trace, plotly_script


def table(rows):
    return '<table>'+''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in row)+'</tr>' for row in rows)+'</table>'


def render():
    rows=[]
    for p in (REPORT/'measurements').glob('*.json'):
        rows.extend(json.loads(p.read_text()))
    if os.environ.get('PODRUN_DIR'):
        for source in sorted(RUN.parent.glob('*/artifacts/predictions.jsonl')):
            rows.extend(json.loads(line) for line in source.read_text().split('\n') if line)
    rows=list({(r['predictor'],r['id'],r['mode']):r for r in rows}.values())
    status = json.loads((REPORT/'status.json').read_text()) if (REPORT/'status.json').exists() else {}
    head='<h1>Prediction vs pairs inspected · misalignment and sycophancy DPO data</h1>'
    head+=table([['Predictor','Completed / expected'], *[[m,f"{sum(r['predictor']==m for r in rows)}/40"] for m in ['opus','qwen']]])
    if status: head+=table([['Controller status',status.get('stage','pending')],['Recorded API cost',status.get('cost','pending')]])
    traces=[]
    for mi, model in enumerate(['opus','qwen']):
        for ti, trait in enumerate(C['traits']):
            xs=[];ys=[]
            for n in C['sample_sizes']:
                group=[r for r in rows if r['predictor']==model and r['trait']==trait and r['size']==n and r['mode']=='choice' and not r.get('error') and r.get('choice')]
                if group: xs.append(n);ys.append(sum(r['choice']==trait for r in group)/len(group))
            if xs:traces.append(line_trace(model+' · '+trait,xs,ys,value_label='Target choice accuracy',color=['#008566','#b54708','#7254c8','#1675b9'][mi*2+ti]))
    chart=chart_grid([chart_html('choice','Target prediction accuracy',traces,x_label='Preference pairs inspected',y_label='Accuracy',show_legend=True)]) if traces else '<p>Predictions pending.</p>'
    text='<p>Qwen responses marked forced final close the reasoning phase at 8,192 tokens and receive up to 2,048 additional answer tokens.</p><p>Target matching only; behavioral transfer and open-ended match judgments are pending. Missing and invalid answers are not scored as correct or incorrect.</p>'
    for r in sorted(rows,key=lambda r:(r['predictor'],r['trait'],r['size'],r['seed'],r['mode'])):
        text+='<details><summary>'+html.escape(f"{r['predictor']} · {r['trait']} · {r['size']} pairs · seed {r['seed']} · {r['mode']} · forced final: {r.get('forced_final',False)}")+'</summary><pre>'+html.escape(r.get('text') or str(r.get('error')) )+'</pre></details>'
    methods='<h1>Methods &amp; Config</h1><p>Two unadapted predictors; identical fixed packets; DPO preference labels retained. Five independently seeded samples per size and trait; each 150-pair packet is nested in its corresponding 1,000-pair packet. Actual training rows selected with the training experiment’s prompt-hash split. Files verified by SHA-256. Sample-limited audit, not full-dataset concealment.</p><p>Open-ended and forced-choice requests are separate. Forced choice is conditional on an effect rather than asserting an unverified effect. Eight candidate behaviors, option order shuffled identically for both models. One generation per packet/mode, high reasoning for Opus, thinking enabled for Qwen, 8,192 initial output-token cap. Qwen receives a forced closing think tag and up to 2,048 final-answer tokens when reasoning exhausts that cap without an answer; these cases are flagged. This is a model-specific inference adaptation, not identical output budgets. Target metadata, teacher system prompts and selection scores excluded from requests. No prompt caching assumed or enabled. Open-ended judgments deferred until transfer evidence is available; no controls included in this first 80-prediction comparison.</p><pre>'+html.escape(json.dumps(C,indent=2))+'</pre>'
    style='<style>body{font:15px system-ui;margin:24px;color:#18312c}table{border-collapse:collapse;width:100%;margin:16px 0}td{padding:8px;border-bottom:1px solid #dde6e2}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f4f7f6;padding:16px}details{margin:12px 0}</style>'
    for name,body in [('in_progress.html',head+chart+text),('final_report.html',head+chart+text),('methods.html',methods)]:
        atomic(REPORT/name,'<!doctype html><meta charset="utf-8">'+style+plotly_script(HERE/'assets/plotly-3.1.0.min.js')+body)


if __name__=='__main__':render()
