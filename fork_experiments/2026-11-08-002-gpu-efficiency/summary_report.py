"""Small standalone comparison, derived from the published pilot cases."""
import html
import json
import math
import random
from pathlib import Path
import statistics
import sys
sys.path.insert(0,str(Path('~/code/research-infra/graphing-utils').expanduser()))
from research_graphing import chart_html, chart_grid, plotly_script

HERE=Path(__file__).resolve().parent
SPECS=[('RTX PRO 6000',96,2.49,'rtx-a1'),('L40S',48,1.09,'l40s-a2'),
       ('A100 PCIe',80,1.79,'a100pcie-a4'),('A100 SXM',80,1.79,'a100sxm-a2'),
       ('H100 PCIe',80,2.89,'h100pcie-a1'),('H100 SXM',80,3.99,'h100sxm-a1'),
       ('H200',141,5.29,'h200-a2')]

def cases(run):
    return [json.loads(p.read_text()) for p in (HERE/'results'/('gpu-efficiency-'+run)).glob('case-*.json')]

baseline={r['index']:r for r in cases('rtx-a1')}
rows=[]
for name,vram,rate,run in SPECS:
    data=cases(run);assert len(data)==12,(run,len(data))
    valid=[r for r in data if r['status']=='complete']
    speed=math.exp(statistics.mean(math.log(baseline[r['index']]['median_seconds']/r['median_seconds']) for r in valid))
    rows.append(dict(gpu=name,vram=vram,rate=rate,complete=len(valid),speed=speed,cost=rate/2.49/speed,run='gpu-efficiency-'+run))

table='<table><thead><tr>'+''.join('<th>'+x+'</th>' for x in ['GPU','VRAM','Pilot $/hour','Completed','Speed vs RTX ↑','Cost/update vs RTX ↓'])+'</tr></thead><tbody>'
for r in rows:
    values=[r['gpu'],f"{r['vram']} GB",f"${r['rate']:.2f}",f"{r['complete']}/12"+(' · OOM¹' if r['complete']<12 else ''),f"{r['speed']:.2f}×",f"{r['cost']:.2f}×"]
    table+='<tr>'+''.join('<td>'+html.escape(v)+'</td>' for v in values)+'</tr>'
table+='</tbody></table>'
charts=[]
for field,title in [('speed','Speed · higher is better'),('cost','Cost/update · lower is better')]:
    order=list(reversed(rows))
    trace=dict(type='bar',orientation='h',x=[r[field] for r in order],y=[r['gpu'] for r in order],
               marker=dict(color=['#0f766e' if r['gpu']=='RTX PRO 6000' else '#64748b' for r in order]),
               customdata=[[r['rate'],r['vram'],r['complete']] for r in order],
               hovertemplate='%{y}<br>Ratio: %{x:.3f}×<br>$%{customdata[0]:.2f}/h · %{customdata[1]} GB<br>%{customdata[2]}/12 cases<extra></extra>')
    charts.append(chart_html('summary-'+field,title,[trace],x_label='Ratio to RTX PRO 6000',y_label='',height=310,show_legend=False))
# Per-setting comparisons use the same workload as the RTX reference.
settings=[baseline[index] for index in sorted(baseline)]
labels=[f"{r['mode'].title()}<br>{r['objective'].upper()}<br>{r['length']:,} tokens" for r in settings]
colors=['#0f766e','#d97706','#2563eb','#9333ea','#db2777','#475569','#65a30d']
symbols=['circle','square','diamond','triangle-up','triangle-down','cross','x']
BOOTSTRAP_REPLICATES=10000
bootstrap_rng=random.Random(20261109)
def bootstrap_medians(samples):
    assert len(samples)==8 and all(math.isfinite(t) and t>0 for t in samples)
    return [statistics.median(bootstrap_rng.choices(samples,k=len(samples))) for _ in range(BOOTSTRAP_REPLICATES)]
def percentile(values,q):
    position=(len(values)-1)*q;lower=math.floor(position);upper=math.ceil(position)
    return values[lower]+(values[upper]-values[lower])*(position-lower)
baseline_bootstrap={i:bootstrap_medians(r['timed_step_seconds']) for i,r in sorted(baseline.items())}
setting_rows=[]
for name,vram,rate,run in SPECS:
    for r in sorted(cases(run),key=lambda row:row['index']):
        reference=baseline[r['index']]
        row=dict(gpu=name,run='gpu-efficiency-'+run,index=r['index'],mode=r['mode'],objective=r['objective'],
                 length=r['length'],status=r['status'],gpu_hourly_usd=rate)
        if r['status']=='complete':
            assert r['input_sha256']==reference['input_sha256'], (name,r['index'],'input mismatch')
            seconds=r['median_seconds'];speed=reference['median_seconds']/seconds
            row.update(seconds_per_update=seconds,updates_per_dollar=3600/(seconds*rate),
                       speed_vs_rtx=speed,cost_efficiency_vs_rtx=speed*2.49/rate)
            if name=='RTX PRO 6000':
                interval=[1.,1.]  # A reference divided by itself has no ratio uncertainty.
            else:
                gpu_bootstrap=bootstrap_medians(r['timed_step_seconds'])
                ratios=sorted(a/b for a,b in zip(baseline_bootstrap[r['index']],gpu_bootstrap))
                interval=[percentile(ratios,.025),percentile(ratios,.975)]
            assert interval[0]<=speed<=interval[1], (name,r['index'],interval,speed)
            row.update(speed_vs_rtx_ci95=interval,cost_efficiency_vs_rtx_ci95=[v*2.49/rate for v in interval],
                       timed_updates=8,bootstrap_replicates=BOOTSTRAP_REPLICATES,
                       interval_method='95% percentile bootstrap, independently resampled within-run updates; fixed GPU rates')
        setting_rows.append(row)
setting_charts=[]
for field,title,ylabel in [('cost_efficiency_vs_rtx','Cost efficiency by setting','Updates per dollar / RTX'),
                           ('speed_vs_rtx','Speed by setting','Updates per second / RTX')]:
    traces=[]
    for (name,_,_,_),color,symbol in zip(SPECS,colors,symbols):
        selected=[r for r in setting_rows if r['gpu']==name]
        traces.append(dict(type='scatter',mode='lines+markers',connectgaps=False,line=dict(color=color,width=1.5),name=name,x=labels,y=[r.get(field) for r in selected],
                           error_y=dict(type='data',symmetric=False,visible=True,color=color,thickness=1,width=3,
                                        array=[r[field+'_ci95'][1]-r[field] if field in r else None for r in selected],
                                        arrayminus=[r[field]-r[field+'_ci95'][0] if field in r else None for r in selected]),
                           marker=dict(color=color,symbol=symbol,size=7,line=dict(width=1,color='white')),
                           customdata=[[r['gpu'],r['mode'],r['objective'].upper(),r['length'],
                                        r.get('seconds_per_update'),r.get('updates_per_dollar'),r['gpu_hourly_usd'],
                                        *r.get(field+'_ci95',[None,None])] for r in selected],
                           hovertemplate='%{customdata[0]}<br>%{customdata[1]} · %{customdata[2]} · %{customdata[3]} tokens'
                           '<br>Relative to RTX: %{y:.3f}×<br>95% bootstrap CI: %{customdata[7]:.3f}–%{customdata[8]:.3f}×<br>%{customdata[4]:.3f} seconds/update'
                           '<br>%{customdata[5]:.1f} updates/$<br>GPU rate: $%{customdata[6]:.2f}/h<extra></extra>'))
    setting_charts.append(chart_html('settings-'+field,title,traces,x_label='Training setting',y_label=ylabel,
                                    height=420,zero_y=True,group_points=False,show_legend=True,
                                    caption='Higher is better; RTX = 1× in each setting. Whiskers: 95% bootstrap intervals from 8 timed updates; within-run timing variability only. Lines connect categorical settings for readability. L40S has no point for 1,024-token student DPO (out of memory).'))
assert len(setting_rows)==84 and sum(r['status']=='complete' for r in setting_rows)==83
assert all(abs(r['cost_efficiency_vs_rtx']-1)<1e-12 for r in setting_rows if r['gpu']=='RTX PRO 6000')
body='''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Qwen3.5-9B GPU pilot — summary</title>
<style>body{font:15px/1.5 system-ui,sans-serif;color:#172033;background:#fff;max-width:1050px;margin:36px auto;padding:0 22px}h1{font-size:25px;margin:0 0 4px}h2{font-size:17px}.sub{color:#64748b;margin:0 0 20px}.table-wrap{overflow-x:auto}table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}th,td{text-align:right;padding:10px 12px;border-bottom:1px solid #e2e8f0;white-space:nowrap}th:first-child,td:first-child{text-align:left}th{font-size:12px;color:#475569;background:#f8fafc}tbody tr:first-child{background:#f0fdfa;font-weight:650}.note{font-size:13px;color:#475569}.caveat{border-left:3px solid #d97706;padding:9px 13px;background:#fffbeb;margin:18px 0}details{margin-top:18px;border-top:1px solid #e2e8f0;padding-top:12px}summary{cursor:pointer;font-weight:600}@media print{body{margin:0;font-size:12px}details{display:none}}</style></head><body>
<h1>Qwen3.5-9B · GPU pilot</h1><p class="sub">Summary · November 9, 2026</p>'''
body+=plotly_script(HERE.parent/'2026-11-07-000-lls-training/assets/plotly-3.1.0.min.js')
body+='<p class="note">Each setting combines training type, objective, and sequence length. Both charts use RTX PRO 6000 as the 1× reference.</p>'
body+='<div class="caveat"><strong>Provisional timing results.</strong> Cross-GPU numerical agreement remains unresolved. RTX has the lowest measured cost among GPUs completing every case; L40S is cheaper where memory permits.</div>'
body+=chart_grid(setting_charts,columns=1)
body+='<details><summary>Overall summary table</summary><div class="table-wrap">'+table+'</div></details>'
body+='<p class="note">¹ L40S ran out of memory on 1,024-token student DPO; its averages cover the 11 completed cases.</p>'
body+='<details><summary>Overall comparison graphs</summary>'+chart_grid(charts)+'</details>'
body+='<details><summary>Methods &amp; Config</summary><p>Qwen3.5-9B; BF16; backbone and output-head activation checkpointing off. Student and soft-prompt training, SFT and DPO, at 256 / 512 / 1,024 tokens: 12 cases per GPU. The per-setting speed ratio is RTX median update time / GPU median update time. Cost efficiency is RTX cost per update / GPU cost per update, equivalent to relative updates per dollar; both are higher-is-better. Overall summary ratios are geometric means over completed matched cases. Higher speed and lower cost are better. Costs use the hourly GPU rates paid during this pilot; startup, storage, model loading, and reference-cache construction are excluded. Three warmup updates and eight timed updates per case. Whiskers are 95% percentile bootstrap confidence intervals using 10,000 resamples (seed 20261109): resample the eight timed updates independently within each GPU and its matching RTX baseline, take their medians, and compute the ratio. Fixed hourly rates scale the speed ratio into cost efficiency. RTX self-comparisons stay exactly 1. These descriptive intervals assume exchangeable update timings; they do not capture serial dependence, between-run or between-pod variability, pricing uncertainty, or unresolved numerical differences. Adjacent settings are categorical, not an interpolated workload scale. The strict cross-GPU gradient-agreement check failed for the alternatives; timing ratios do not establish equivalent training outcomes.</p><p>Source: published case files in experiment 2026-11-08-002-gpu-efficiency. A100 PCIe uses successful retry a4; other runs are identified in the accompanying summary data.</p></details><script>document.querySelectorAll("details").forEach(d=>d.addEventListener("toggle",()=>{if(d.open)d.querySelectorAll(".js-plotly-plot").forEach(p=>Plotly.Plots.resize(p));}));</script></body></html>'
(HERE/'gpu_summary.html').write_text(body)
(HERE/'gpu_setting_data.json').write_text(json.dumps(setting_rows,indent=2)+'\n')
(HERE/'gpu_summary_data.json').write_text(json.dumps(rows,indent=2)+'\n')
assert body.count('<tbody>')==1 and body.count('<td>')==42
print('Created standalone report: 7 GPUs; all ratios calculated from published cases.')
