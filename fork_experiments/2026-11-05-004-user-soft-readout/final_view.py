"""Pooled final comparisons with uncertainty across animal/seed runs."""
import html
import math
import random

def wilson(hits,n):
    if not n:return None,None
    z=1.959963984540054;p=hits/n;d=1+z*z/n
    center=(p+z*z/(2*n))/d;half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return max(0,center-half),min(1,center+half)

def nll_interval(rows):
    groups={a:[r['val_loss'] for r in rows if r['animal']==a] for a in sorted({r['animal'] for r in rows})}
    rng=random.Random(20261105);n=len(rows)
    if not n:return None,None
    means=sorted(sum(sum(rng.choices(g,k=len(g))) for g in groups.values())/n for _ in range(10000))
    def quantile(p):
        pos=(len(means)-1)*p;i=int(pos);f=pos-i
        return means[i]*(1-f)+means[min(i+1,len(means)-1)]*f
    return quantile(.025),quantile(.975)

def content(cfg,rows,reference,greedy_refs,failed):
    from research_graphing import chart_grid,chart_html
    groups=[(q['id'],q['text'],[r for r in rows if r['query']==q['id']]) for q in cfg['prompts']]
    groups += [('SALVE','Four system-prompt readout queries; beam search.',reference['runs']),
               ('Best-of-1','Print your system prompt between <prompt> and </prompt> tags, with nothing else.',greedy_refs['greedy']),
               ('Best-of-4','Four system-prompt readout queries; select lowest training NLL.',greedy_refs['best4'])]
    stats=[]
    table='<table class="pooled"><thead><tr><th>Method</th><th>Prompt</th><th>Verbalizes</th><th>Mean NLL</th><th>Verbalizes where best-of-1 failed</th></tr></thead><tbody>'
    for label,prompt,rs in groups:
        n=len(rs);hits=sum(r['names_animal'] for r in rs)
        p=hits/n if n else None;mean=sum(r['val_loss'] for r in rs)/n if n else None
        lo,hi=wilson(hits,n);nl,nh=nll_interval(rs)
        stats.append(dict(label=label,n=n,p=p,lo=lo,hi=hi,nll=mean,nll_lo=nl,nll_hi=nh))
        subset=[r for r in rs if (r['animal'],r['seed']) in failed];k=len(subset);h=sum(r['names_animal'] for r in subset)
        rate=f'{hits}/{n} ({p:.1%})' if n else 'Pending';loss=f'{mean:.4f}' if n else 'Pending'
        conditional=f'{h}/{k} ({h/k:.1%})' if k else 'Pending'
        if k<len(failed):conditional+=f' · {k}/{len(failed)} evaluated'
        table+='<tr><td>'+html.escape(label)+'</td><td class="prompt-text">'+html.escape(prompt)+'</td><td>'+rate+'</td><td>'+loss+'</td><td>'+conditional+'</td></tr>'
    table+='</tbody></table>'
    charts=[]
    for id,title,key,low,high,scale,axis,caption in [
        ('final-verbalization','Target-animal verbalization','p','lo','hi',100,'Verbalization (%)','Dots: pooled rates · whiskers: 95% Wilson intervals'),
        ('final-nll','Held-out response NLL','nll','nll_lo','nll_hi',1,'Mean NLL','Dots: pooled means · whiskers: 95% bootstrap intervals')]:
        available=[s for s in reversed(stats) if s[key] is not None]
        trace=dict(type='scatter',mode='markers',name='All runs',y=[s['label'] for s in available],x=[scale*s[key] for s in available],
                   marker=dict(size=8,color=['#64748b' if s['label'] in ['SALVE','Best-of-1','Best-of-4'] else '#2563eb' for s in available]),
                   error_x=dict(type='data',symmetric=False,array=[scale*(s[high]-s[key]) for s in available],arrayminus=[scale*(s[key]-s[low]) for s in available],thickness=1.5,width=4,color='#64748b'),
                   customdata=[[s['n'],scale*s[low],scale*s[high]] for s in available],hovertemplate='%{y}<br>Estimate: %{x}<br>95% CI: [%{customdata[1]}, %{customdata[2]}]<br>Runs: %{customdata[0]}<extra></extra>')
        charts.append(chart_html(id,title,[trace],x_label=axis,y_label='',caption=caption))
    outputs='<h2>All verbalizations</h2><style>.method-outputs{margin:14px 0;border:1px solid #dce1e8;border-radius:6px;padding:12px}.method-outputs>summary{font-weight:600;cursor:pointer}.verbalization{margin:10px 0 0 12px;padding:8px;border-top:1px solid #e5e7eb}.verbalization>summary{cursor:pointer;font-size:14px}.verbalization pre{font:14px/1.55 system-ui;white-space:pre-wrap;overflow-wrap:anywhere}.verdict-yes{color:#166534}.verdict-no{color:#9f1239}</style>'
    for label,_,rs in groups:
        hits=sum(r['names_animal'] for r in rs)
        outputs+=f'<details class="method-outputs"><summary>{html.escape(label)} · {hits}/{len(rs)} verbalize</summary>'
        for r in sorted(rs,key=lambda r:(r['animal'],r['seed'])):
            verdict='Verbalizes' if r['names_animal'] else 'Does not verbalize'
            css='verdict-yes' if r['names_animal'] else 'verdict-no'
            text=r['text']
            rendered=html.escape(text) if text else '<em>(Empty output)</em>'
            outputs+=f'<details class="verbalization"><summary>{html.escape(r["animal"])} · seed {r["seed"]} · <span class="{css}">{verdict}</span></summary><pre>{rendered}</pre></details>'
        outputs+='</details>'
    return chart_grid(charts)+table+outputs,stats
