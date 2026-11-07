import html
import json
import os
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
ART=Path(os.environ.get('PODRUN_ARTIFACT_DIR',HERE/'results/local'))
REPORT=Path(os.environ.get('PODRUN_REPORT_DIR',HERE))
SOURCE=HERE.parent/'2026-11-07-000-lls-training'
C=json.loads((SOURCE/'config.json').read_text())
sys.path.insert(0,C['graphing_path'] if os.environ.get('PODRUN_DIR') else str(Path('~/code/research-infra/graphing-utils').expanduser()))
from research_graphing import chart_grid, chart_html, line_trace, plotly_script


def render():
    rows=[json.loads(p.read_text()) for p in sorted(ART.glob('case-*.json'))]
    if not rows:
        rows=[json.loads(p.read_text()) for p in sorted((HERE/'results').glob('*/case-*.json'))]
    rows.sort(key=lambda r:r['index'])
    charts=[]
    for mode in ['student','direct']:
        for field,title in [('pairs_per_second','Pairs / second'),('peak_allocated_gib','Peak allocated GiB')]:
            traces=[]
            for cp,cached,color in [('all',False,'#008566'),('none',False,'#b54708'),('none',True,'#7254c8'),('head_only',False,'#1675b9')]:
                selected=[r for r in rows if r['mode']==mode and r['checkpoint']==cp and r['cached_reference']==cached and r['status']=='complete']
                if selected:traces.append(line_trace(cp+(' · cached reference' if cached else ' · online reference'),[r['microbatch'] for r in selected],[r[field] for r in selected],color=color,value_label=title))
            if traces:charts.append(chart_html(mode+'-'+field,mode+' · '+title,traces,x_label='Microbatch pairs',y_label=title,show_legend=True))
    def fmt(x):return f'{x:.3f}' if isinstance(x,float) else str(x)
    fields=['mode','microbatch','checkpoint','cached_reference','status','median_seconds','pairs_per_second','peak_allocated_gib','gradient_cosine','gradient_relative_l2']
    table='<table><tr>'+''.join('<th>'+html.escape(s.replace('_',' '))+'</th>' for s in fields)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(fmt(r.get(s,'—')))+'</td>' for s in fields)+'</tr>' for r in rows)+'</table>'
    summary='<table><tr><th>DPO workload</th><th>Checkpointing on</th><th>Checkpointing off</th><th>Throughput gain</th><th>Time saved</th><th>Off: peak allocated</th><th>Off + cached references*</th></tr>'
    for mode in ['student','direct']:
        cases={('cached' if r['cached_reference'] else r['checkpoint']):r for r in rows if r['mode']==mode and r['microbatch']==2 and r['status']=='complete'}
        if 'all' not in cases or 'none' not in cases:continue
        before=cases['all']['median_seconds'];after=cases['none']['median_seconds']
        cached=f"{cases['cached']['median_seconds']:.2f} s" if 'cached' in cases else 'pending'
        values=[mode,f'{before:.2f} s',f'{after:.2f} s',f'{100*(before/after-1):.1f}%',f'{100*(1-after/before):.1f}%',f"{cases['none']['peak_allocated_gib']:.1f} GiB",cached]
        summary+='<tr>'+''.join('<td>'+v+'</td>' for v in values)+'</tr>'
    summary+='</table><p>Microbatch 2; median of five timed updates. *Reference precomputation excluded.</p>'
    body='<h1>Checkpointing × microbatch · Qwen3.5-9B student and soft-prompt DPO</h1><p>'+str(len(rows))+'/24 cases measured; errors and OOM shown explicitly.</p>'+summary+chart_grid(charts)+'<details><summary>All measurements and gradient comparisons</summary>'+table+'</details>'
    methods='<h1>Methods &amp; Config</h1><p>Separate RTX PRO 6000. Frozen execution source imports the live training implementation without modifying active jobs. Seed 42, same sampled training rows and effective batches (student 64, soft 16). Two warmup updates and five timed updates per case; CUDA synchronization brackets full optimizer updates. Each case starts a fresh process and identical initialization. Both traits contribute equally to the 512-row pool. Checkpoint all is the existing baseline; none disables backbone and LM-head checkpointing; head_only disables only backbone checkpointing. Online references are forced cold each update; cached-reference scores are precomputed for identical minibatches, with precomputation excluded from step timing and separately recorded. Cached-reference speed is a steady-state ceiling, not total training speedup. Gradient comparisons use deterministic parameter samples from the first clipped gradient. Different batching can alter BF16 numerics. No validation, saving, reporting or startup costs are included in update throughput. OOM cases remain visible. GPU budget $5, 60-minute worker limit.</p><pre>'+html.escape((HERE/'config.json').read_text())+'</pre>'
    style='<style>body{font:15px system-ui;margin:24px}table{border-collapse:collapse}td,th{padding:8px;border-bottom:1px solid #ddd}pre{white-space:pre-wrap}</style>'
    script=plotly_script(SOURCE/'assets/plotly-3.1.0.min.js')
    REPORT.mkdir(parents=True,exist_ok=True)
    for name,text in [('in_progress.html',body),('final_report.html',body),('methods.html',methods)]:
        p=REPORT/name;tmp=p.with_suffix('.tmp');tmp.write_text('<!doctype html><meta charset="utf-8">'+style+script+text);tmp.replace(p)


if __name__=='__main__':render()
