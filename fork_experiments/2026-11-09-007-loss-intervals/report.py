"""Render held-out loss intervals as they become available."""
import html
import json
import os
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
CONFIG=json.loads((HERE/'config.json').read_text())
sys.path.insert(0,str(Path(os.environ.get('RESEARCH_INFRA_DIR',CONFIG['graphing_root'])).expanduser()/'graphing-utils'))
from research_graphing import chart_html,chart_grid,plotly_script


def render(result=None):
    result=result or dict(status='pending',results=[])
    folder=Path(os.environ.get('PODRUN_REPORT_DIR',HERE));cards=[]
    for target,title in [('direct','Teacher preferences · DPO loss'),('imitation','Student answers · NLL')]:
        traces=[]
        for trait in ['misalignment','sycophancy']:
            rows=[r for r in result['results'] if r['mode']==target and r['trait']==trait]
            if rows:traces.append(dict(type='scatter',mode='markers',name=trait.capitalize(),x=[r['condition'] for r in rows],y=[r['loss'] for r in rows],error_y=dict(type='data',symmetric=False,array=[max(0,r['ci_high']-r['loss']) for r in rows],arrayminus=[max(0,r['loss']-r['ci_low']) for r in rows])))
        if traces:cards.append(chart_html(target,title,traces,x_label='Model condition',y_label='Held-out DPO loss' if target=='direct' else 'NLL (nats/token)'))
    body='<h1>Held-out loss confidence intervals</h1><p>'+str(len(result['results']))+'/28 evaluations · '+html.escape(result['status'])+'</p>'+chart_grid(cards)
    style='<style>body{font:15px system-ui;max-width:1350px;margin:28px auto;padding:0 24px}</style>'
    script=plotly_script(HERE/'assets/plotly-3.1.0.min.js')
    for name in ['in_progress.html','final_report.html']:
        p=folder/name;tmp=p.with_suffix('.tmp');tmp.write_text('<!doctype html><meta charset="utf-8">'+style+script+body);tmp.replace(p)
    (folder/'methods.html').write_text('<!doctype html><meta charset="utf-8">'+style+'<h1>Methods &amp; Config</h1><p>Original training Engine, BF16, SDPA, eval batch 4; no gradients or generation. Exact prompt-hash test splits and saved student response IDs capped at 20 tokens. DPO beta 0.08 with unprompted original base reference. Teacher is the original trait system prompt. Student uses the selected saved adapter, matching original loss evaluation; merged weights were for generation. 95% percentile bootstrap over normalized-prompt clusters, 2000 repetitions with shared draws across conditions. DPO resamples pair-loss sums and pair counts; NLL resamples response-NLL sums and token counts. Exact original means checked within 0.0001. Data, answer, and soft-checkpoint hashes verified before scoring. Missing measurements remain pending.</p><pre>'+html.escape(json.dumps(CONFIG,indent=2))+'</pre>')


if __name__=='__main__':render()
