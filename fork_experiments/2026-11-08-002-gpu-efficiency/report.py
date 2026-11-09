import html
import json
import math
import os
from pathlib import Path
import statistics
import sys
from benchmark import ART, C, HERE, RUN, SHARED, atomic

REPORT=Path(os.environ.get('PODRUN_REPORT_DIR',HERE))
sys.path.insert(0,C['graphing_path'] if os.environ.get('PODRUN_DIR') else str(Path('~/code/research-infra/graphing-utils').expanduser()))
from research_graphing import chart_html,chart_grid,line_trace,plotly_script


def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def fmt(x):
    return 'pending' if x is None else (f'{x:.3f}' if isinstance(x,float) else str(x))


def table(headers, rows):
    return '<table><tr>'+''.join('<th>'+html.escape(x)+'</th>' for x in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(fmt(x))+'</td>' for x in row)+'</tr>' for row in rows)+'</table>'


def render():
    sources=[HERE/'results']+([SHARED] if SHARED.exists() else [])
    paths={}
    for source in sources:
        for p in source.glob('*/case-*.json'):paths[(p.parent.name,p.name)]=p
    rows=[]
    for p in paths.values():
        r=read(p);hw=read(p.parent/'hardware.json',{})
        r.update(run=p.parent.name,gpu=hw.get('gpu_id','unknown'),rate=hw.get('gpu_hourly_usd',hw.get('hourly_cost')),directory=str(p.parent))
        rows.append(r)
    baseline={r['index']:r for r in rows if r['gpu']=='NVIDIA RTX PRO 6000 Blackwell Server Edition' and r['status']=='complete'}
    for r in rows:
        if r['status']!='complete':continue
        r['usd_per_1000_updates']=r['median_seconds']*r['rate']/3.6 if r['rate'] is not None else None
        ref=baseline.get(r['index'])
        if ref and ref['input_sha256']==r['input_sha256']:
            import torch
            gradient=Path(r['directory'])/f"gradient-{r['index']}.pt"
            baseline_gradient=Path(ref['directory'])/f"gradient-{r['index']}.pt"
            if gradient.exists() and baseline_gradient.exists():
                a=torch.load(gradient,weights_only=True)
                b=torch.load(baseline_gradient,weights_only=True)
                r['gradient_cosine']=float(torch.nn.functional.cosine_similarity(a,b,dim=0))
                r['gradient_relative_l2']=float((a-b).norm()/b.norm().clamp_min(1e-12))
                r['loss_abs_delta']=abs(r['first_loss']-ref['first_loss'])
                r['agreement']=r['gradient_cosine']>=.995 and r['gradient_relative_l2']<=.1 and r['loss_abs_delta']<=.01
            else:
                # Reuse the published on-pod measurement, never infer missing agreement.
                saved=read(Path(r['directory'])/'comparison.json',{}).get('rows',[])
                previous=next((x for x in saved if x.get('run')==r['run'] and x.get('index')==r['index']
                               and x.get('input_sha256')==r['input_sha256'] and x.get('source_commit')==r['source_commit']),{})
                for field in ('gradient_cosine','gradient_relative_l2','loss_abs_delta','agreement'):
                    if field in previous:r[field]=previous[field]
                r['agreement_provenance']='published on-pod comparison; raw gradient samples unavailable locally'
            r['speed_ratio']=ref['median_seconds']/r['median_seconds']
            if r['rate'] and ref['rate']:r['cost_ratio']=r['median_seconds']*r['rate']/(ref['median_seconds']*ref['rate'])
    charts=[];colors=['#008566','#b54708','#7254c8','#1675b9','#d03050','#64748b','#658b18']
    for mode in C['modes']:
        for objective in C['objectives']:
            for field,title in [('median_seconds','Seconds / optimizer update'),('usd_per_1000_updates','GPU dollars / 1,000 updates')]:
                traces=[]
                for gpu,color in zip(C['gpu_ids'],colors):
                    selected=sorted([r for r in rows if r['gpu']==gpu and r['mode']==mode and r['objective']==objective and r['status']=='complete'],key=lambda r:r['length'])
                    if selected:traces.append(line_trace(gpu,[r['length'] for r in selected],[r[field] for r in selected],color=color,value_label=title))
                if traces:charts.append(chart_html(mode+objective+field,mode+' · '+objective.upper()+' · '+title,traces,x_label='Policy sequence tokens',y_label=title,show_legend=True))
    summary=[]
    for gpu in C['gpu_ids']:
        group=[r for r in rows if r['gpu']==gpu];valid=[r for r in group if r.get('agreement') and r.get('cost_ratio')]
        speed=math.exp(statistics.mean(math.log(r['speed_ratio']) for r in valid)) if len(valid)==12 else None
        cost=math.exp(statistics.mean(math.log(r['cost_ratio']) for r in valid)) if len(valid)==12 else None
        summary.append([gpu,group[0]['rate'] if group else None,sum(r['status']=='complete' for r in group),sum(r['status']=='out_of_memory' for r in group),speed,cost])
    details=[]
    for r in sorted(rows,key=lambda r:(r['gpu'],r['index'])):
        times=r.get('timed_step_seconds',[])
        iqr=statistics.quantiles(times,n=4,method='inclusive') if times else []
        details.append([r['gpu'],r['mode'],r['objective'],r['length'],r['status'],r.get('median_seconds'),
                        f'{iqr[0]:.3f}–{iqr[2]:.3f}' if iqr else None,r.get('usd_per_1000_updates'),r.get('peak_allocated_gib'),
                        r.get('reference_precompute_seconds'),r.get('gradient_cosine'),r.get('gradient_relative_l2'),r.get('loss_abs_delta')])
    body='<h1>Qwen3.5-9B · GPU efficiency pilot</h1>'+table(['GPU','GPU $/h','Complete / 12','OOM','Speed / RTX','Cost / RTX'],summary)
    body+='<p>Steady-state compute only; reference-cache construction and startup excluded. Overall ratios require all 12 matched cases and numerical checks. Lower cost ratio is better.</p>'+chart_grid(charts)
    body+='<details><summary>Per-case measurements and numerical agreement</summary>'+table(['GPU','Mode','Objective','Tokens','Status','Median s','IQR s','$/1k updates','Peak GiB','Reference build s','Gradient cosine','Gradient rel. L2','Loss Δ'],details)+'</details>'
    methods='<h1>Methods &amp; Config</h1><p>Dedicated single GPUs; fixed Qwen3.5-9B revision, BF16 weights, FP32 soft parameters and standard PEFT adapter handling. No quantization. Backbone and LM-head activation checkpointing off through the shared policy. SDPA, 128-token output-head chunks, microbatch 2, exact effective batches from the previous SFT/DPO experiments. Each case starts a fresh process; three warmup and eight timed optimizer updates, synchronized at both CUDA boundaries. Initialization and example order are fixed. First un-clipped gradients are sampled at fixed parameter indices and compared against the RTX baseline. Gate: cosine ≥0.995, relative L2 ≤0.1, absolute loss difference ≤0.01; violations remain visible and are excluded from the overall ranking.</p>'
    methods+='<p>Fixtures use real text from the prior LLS preference dataset, cycling/truncating token IDs to exact sequence lengths. This isolates the requested context and output-head workloads; it is not a learning-quality or natural-length dataset benchmark. SFT responses occupy one quarter of the sequence; DPO responses are 64 tokens per alternative. Soft prefixes occupy 128 policy tokens and are absent from the reference model. SFT students use dropout .05; DPO students use zero dropout. AdamW and gradient clipping follow existing training conventions. Frozen DPO reference scores are built separately per device and exact batch, audited by fresh recomputation, then cached; policy scores remain fresh. Reference build, audit overhead, model loading and complete pilot elapsed time remain in raw results. Catalog hourly GPU price determines compute-normalized costs; the shared billing ledger includes setup and idle time. Storage and kernel implementation differences are not inferred away.</p><pre>'+html.escape(json.dumps(C,indent=2))+'</pre>'
    style='<style>body{font:15px system-ui;margin:24px}table{border-collapse:collapse}td,th{padding:7px;border-bottom:1px solid #ddd}pre{white-space:pre-wrap}</style>'
    script=plotly_script(HERE.parent/'2026-11-07-000-lls-training/assets/plotly-3.1.0.min.js')
    REPORT.mkdir(parents=True,exist_ok=True)
    for name,text in [('in_progress.html',body),('final_report.html',body),('methods.html',methods)]:
        p=REPORT/name;tmp=p.with_name(name+f'.{os.getpid()}.tmp');tmp.write_text('<!doctype html><meta charset="utf-8">'+style+script+text);tmp.replace(p)
    atomic(ART/'comparison.json',dict(rows=rows,summary=summary))


if __name__=='__main__':render()
