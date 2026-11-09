"""Small standalone comparison, derived from the published pilot cases."""
import html
import json
import math
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
body='''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Qwen3.5-9B GPU pilot — summary</title>
<style>body{font:15px/1.5 system-ui,sans-serif;color:#172033;background:#fff;max-width:1050px;margin:36px auto;padding:0 22px}h1{font-size:25px;margin:0 0 4px}h2{font-size:17px}.sub{color:#64748b;margin:0 0 20px}.table-wrap{overflow-x:auto}table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}th,td{text-align:right;padding:10px 12px;border-bottom:1px solid #e2e8f0;white-space:nowrap}th:first-child,td:first-child{text-align:left}th{font-size:12px;color:#475569;background:#f8fafc}tbody tr:first-child{background:#f0fdfa;font-weight:650}.note{font-size:13px;color:#475569}.caveat{border-left:3px solid #d97706;padding:9px 13px;background:#fffbeb;margin:18px 0}details{margin-top:18px;border-top:1px solid #e2e8f0;padding-top:12px}summary{cursor:pointer;font-weight:600}@media print{body{margin:0;font-size:12px}details{display:none}}</style></head><body>
<h1>Qwen3.5-9B · GPU pilot</h1><p class="sub">Summary · November 9, 2026</p>'''
body+='<div class="table-wrap">'+table+'</div>'
body+='<p class="note">¹ L40S ran out of memory on 1,024-token student DPO; its averages cover the 11 completed cases.</p>'
body+='<div class="caveat"><strong>Provisional timing results.</strong> Cross-GPU numerical agreement remains unresolved. RTX has the lowest measured cost among GPUs completing every case; L40S is cheaper where memory permits.</div>'
body+='<details><summary>Comparison graphs</summary>'+plotly_script(HERE.parent/'2026-11-07-000-lls-training/assets/plotly-3.1.0.min.js')+chart_grid(charts)+'</details>'
body+='<details><summary>Methods &amp; Config</summary><p>Qwen3.5-9B; BF16; backbone and output-head activation checkpointing off. Student and soft-prompt training, SFT and DPO, at 256 / 512 / 1,024 tokens: 12 cases per GPU. Ratios are geometric means over completed matched cases. Higher speed and lower cost are better. Costs use the hourly GPU rates paid during this pilot; startup, storage, model loading, and reference-cache construction are excluded. Three warmup updates and eight timed updates per case. The strict cross-GPU gradient-agreement check failed for the alternatives; timing ratios do not establish equivalent training outcomes.</p><p>Source: published case files in experiment 2026-11-08-002-gpu-efficiency. A100 PCIe uses successful retry a4; other runs are identified in the accompanying summary data.</p></details></body></html>'
(HERE/'gpu_summary.html').write_text(body)
(HERE/'gpu_summary_data.json').write_text(json.dumps(rows,indent=2)+'\n')
assert body.count('<tbody>')==1 and body.count('<td>')==42
print('Created standalone report: 7 GPUs; all ratios calculated from published cases.')
