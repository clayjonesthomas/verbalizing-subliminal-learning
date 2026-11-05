"""Standalone aggregate reports; scientific setup stays in methods.html."""
import html
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

HERE=Path(__file__).resolve().parent
OUT=Path(os.environ.get('PODRUN_DIR',HERE/'results/local'))
REPORT=Path(os.environ.get('PODRUN_REPORT_DIR',HERE))
ANIMALS=['cat','dog','eagle','owl']
STYLE='<style>body{font:16px system-ui;max-width:1250px;margin:30px auto;padding:0 22px;color:#172033}table{border-collapse:collapse;width:100%;margin:20px 0}th,td{padding:9px;text-align:left;border-bottom:1px solid #ddd}td{vertical-align:top}pre{white-space:pre-wrap}small{color:#667}h1{font-size:26px}</style>'

def page(title,body):return '<!doctype html><html><head><meta charset="utf-8">'+STYLE+'</head><body><h1>'+title+'</h1>'+body+'</body></html>'
def table(headers,rows):return '<table><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+'</table>'
def read(name,default):
    p=OUT/name
    return json.loads(p.read_text()) if p.exists() else default

def render():
    import numpy as np
    infra=Path(os.environ.get('RESEARCH_INFRA_DIR',str(Path.home()/'code/research-infra')))
    sys.path.insert(0,str(infra/'graphing-utils'))
    from research_graphing import chart_html, chart_grid, line_trace, plotly_script
    REPORT.mkdir(parents=True,exist_ok=True)
    progress=read('progress.json',{'stage':'pending','completed':0})
    source=OUT/'generations.jsonl'
    rows=[json.loads(s) for s in source.read_text().splitlines()] if source.exists() else []
    expanded=list(rows)
    groups=defaultdict(list)
    for r in rows:
        if r['probe'].startswith('salve'):groups[(r['label'],r['repeat'],r['n'],r['format'])].append(r)
    for key,rs in groups.items():
        if len(rs)==4:
            r=dict(rs[0]); r['probe']='best4'; r['mentions']=sorted(set(a for v in rs for a in v['mentions'])); r['hit']=r['label'] in r['mentions']; expanded.append(r)
    data=defaultdict(list)
    for r in expanded:
        # Null-data controls evaluated against all four targets, rather than inventing a true target.
        targets=ANIMALS if r['label'] in ['no_data','unprompted'] else [r['label']]
        for target in targets:
            data[('Animal-conditioned' if r['label'] in ANIMALS else r['label'],r['format'],r['n'],r['probe'])].append((r['repeat'],target in r['mentions']))
    def estimate(values):
        clusters=defaultdict(list)
        for rep,hit in values:clusters[rep].append(float(hit))
        x=np.array([np.mean(v) for v in clusters.values()]); mean=float(x.mean())
        if len(x)<2:return mean,mean,mean
        rng=np.random.default_rng(819); means=x[rng.integers(len(x),size=(1000,len(x)))].mean(axis=1)
        return mean,float(np.quantile(means,.025)),float(np.quantile(means,.975))
    body='<p>'+html.escape(progress['stage'])+f' · {progress.get("completed",0):,} / {progress.get("total",0):,} responses</p>'
    if rows:
        charts=[]
        labels={'numbers':'Numbers only','paired':'Request + completion','query_only':'Requests only'}
        probe_labels={'direct':'Direct inference','choice':'Four-choice','best4':'Best-of-4 SALVE'}
        colors={'direct':'#008566','choice':'#4263eb','best4':'#b45309'}
        for fmt in ['numbers','paired','query_only']:
            traces=[]
            for probe in ['direct','choice','best4']:
                ns=sorted(set(r['n'] for r in expanded if r['label'] in ANIMALS and r['format']==fmt and r['probe']==probe))
                estimates=[]
                for n in ns:
                    values=[(r['repeat'],r['hit']) for r in expanded if r['label'] in ANIMALS and r['format']==fmt and r['probe']==probe and r['n']==n]
                    estimates.append(estimate(values))
                if ns:
                    trace=line_trace(probe_labels[probe],ns,[100*e[0] for e in estimates],value_label='Target mention (%)',color=colors[probe],hover_fields={'Examples':ns,'95% lower (%)':[100*e[1] for e in estimates],'95% upper (%)':[100*e[2] for e in estimates]})
                    trace['error_y']=dict(type='data',array=[100*(e[2]-e[0]) for e in estimates],arrayminus=[100*(e[0]-e[1]) for e in estimates])
                    traces.append(trace)
            charts.append(chart_html('readout-'+fmt,labels[fmt],traces,x_label='Examples in context',y_label='Target animal mentioned (%)',zero_y=True,caption='All four target animals pooled. Error bars: 95% bundle bootstrap intervals. Any target mention counts.'))
        body+=plotly_script(HERE/'assets/plotly-3.1.0.min.js')+chart_grid(charts)
        summary=[]
        for (label,fmt,n,probe),values in sorted(data.items()):
            if probe not in ['direct','choice','best4']:continue
            mean,lo,hi=estimate(values)
            summary.append(dict(label=label,format=fmt,n=n,probe=probe,rate=mean,lower=lo,upper=hi,count=len(values)))
        (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        body+=table(['Dataset','Input','Examples','Probe','Mentions','Scored targets'],[(r['label'],r['format'],r['n'],r['probe'],f"{r['rate']:.1%}",r['count']) for r in summary])
        perprobe=[]
        for probe in ['salve0','salve1','salve2','salve3','best4']:
            rs=[r for r in expanded if r['label'] in ANIMALS and r['format'] in ['numbers','paired'] and r['probe']==probe]
            if rs:perprobe.append((probe,len(rs),f"{sum(r['hit'] for r in rs)/len(rs):.1%}",f"{sum(len(r['mentions'])>1 for r in rs)/len(rs):.1%}"))
        body+=table(['SALVE probe (all context sizes)','Responses/bundles','Target mentioned','Multiple animals mentioned'],perprobe)
        examples=[]
        for a in ANIMALS:
            for hit in [True,False]:
                candidates=[r for r in rows if r['label']==a and r['format']=='numbers' and r['probe'] in ['direct','salve0','salve1','salve2','salve3'] and r['hit']==hit]
                if candidates:
                    r=max(candidates,key=lambda v:v['n']); examples.append((a,r['n'],r['probe'],r['hit'],r['response']))
        body+=table(['Target','Examples','Probe','Hit','Generated output'],examples)
        body+=table(['Output diagnostic','Count'],[('Token-limit completions',sum(r['finish_reason']=='length' for r in rows)),('Direct abstentions',sum('insufficient evidence' in r['response'].lower() for r in rows if r['probe']=='direct'))])
    for name in ['in_progress.html','final_report.html']:
        title='Direct number readout' if name=='in_progress.html' else ('Direct number readout · final' if progress['stage']=='complete' else 'Direct number readout · provisional')
        temp=REPORT/(name+'.tmp');temp.write_text(page(title,body));temp.replace(REPORT/name)
    if True:
        config=json.loads((HERE/'config.json').read_text())
        methods=table(['Item','Specification'],[
            ('Question','Can the frozen Figure 4 teacher identify its hidden animal preference from number sequences in ordinary context?'),
            ('Model','Original Qwen2.5-7B-Instruct; no soft prompt, training, search, or likelihood-based candidate selection.'),
            ('Data','Pinned published Figure 4 train splits, plus the unprompted teacher split. Input manifest records dataset hashes and exact row indices.'),
            ('Sampling','30 disjoint 256-example blocks per dataset. Smaller contexts use nested prefixes. Same blocks in numbers-only, request/response, and query-only conditions. Largest size reduced globally if needed to fit 32768 tokens; no truncation.'),
            ('Inference','Greedy decoding for every probe; 128 generated tokens maximum. Direct inference, randomized four-choice, four exact SALVE queries and prefills. Numeric evidence precedes the probe in the user message; neutral Qwen system message.'),
            ('Scoring','Case-insensitive whole-word singular/plural target mention. Negations and multiple animals count. Best-of-4 succeeds if any SALVE probe mentions the target, evaluated only when all four exist. No target-aware selection at inference.'),
            ('Controls','No data, unprompted numbers, original requests alone, explicit animal preference. No-data repeats vary choice order only; identical deterministic repeats are not independent evidence.'),
            ('Report rendering',json.dumps(json.loads((HERE/'report_provenance.json').read_text())) if (HERE/'report_provenance.json').exists() else 'Shared research_graphing utilities'),
            ('Aggregation','Result-table rows pool the four target animals within each input format, context size, and probe. Unprompted and no-data controls remain separate. Curated outputs retain the target label for readability.'),
            ('Uncertainty','Percentile bootstrap across 30 bundle IDs, clustering all animals at the same repeat index. Fixed-dataset sampling uncertainty only. Nested context conditions must be compared with paired bundle IDs.'),
            ('Control scores','Unprompted and no-data mention rates averaged over all four possible target labels; no assigned ground-truth animal. Four-choice 25% chance applies only to valid single selections.'),
            ('Caveat','Exact SALVE prompts request the current system prompt; repeating the neutral prompt is a possible outcome. A mention does not establish identification of an actual preference.'),
            ('Budget','50 USD experiment-wide; independent shared-runner watchdog, 120% forced cleanup cutoff. Six-hour runtime ceiling.'),
        ])+'<pre>'+html.escape(json.dumps(config,indent=2))+'</pre><pre>'+html.escape(json.dumps(read('metadata.json',{}),indent=2))+'</pre>'
        (REPORT/'methods.html').write_text(page('Methods & Config',methods))

if __name__=='__main__':render()
