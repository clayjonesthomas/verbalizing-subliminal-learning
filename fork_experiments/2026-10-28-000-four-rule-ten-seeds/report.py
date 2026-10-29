import html,json,os,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
C=json.loads((HERE/'config.json').read_text())
sys.path.insert(0,os.environ.get('RESEARCH_GRAPHING_DIR',str(Path.home()/'code/research-infra/graphing-utils')))
from research_graphing import chart_html,line_trace,plotly_script
COLORS=['#008566','#b15928','#377eb8','#e41a1c','#984ea3','#ff7f00','#4daf4a','#a65628','#f781bf','#555555']
def read(p,default):
    try:return json.loads(p.read_text())
    except FileNotFoundError:return default

def step_trace(name,x,y,color):return dict(type='scatter',mode='lines',name=name,x=x,y=y,line={'color':color},hovertemplate='Step %{x}<br>%{y}<extra>%{fullData.name}</extra>')
def table(headers,rows):return '<table><thead><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in r)+'</tr>' for r in rows)+'</tbody></table>'
def training(path):
    if not path.exists():return []
    result=[]
    for line in path.read_text().splitlines():
        try:result.append(json.loads(line))
        except json.JSONDecodeError:pass
    return result

def render(shared,reports,final_only=False):
    shared=Path(shared);reports=Path(reports);reports.mkdir(parents=True,exist_ok=True)
    bundle=Path(os.environ.get('PLOTLY_BUNDLE',str(Path.home()/'.cache/research-infra/assets/plotly-3.1.0.min.js')))
    head='<!doctype html><meta charset="utf-8"><title>Four-rule ten-seed comparison</title><style>body{font:15px system-ui;color:#172033;margin:24px}table{border-collapse:collapse;width:100%}td,th{padding:10px;text-align:left;border-bottom:1px solid #ddd}pre{white-space:pre-wrap;overflow-wrap:anywhere}.caption{color:#626c7c;font-size:13px}.prompt-pair{display:grid;grid-template-columns:1fr 1fr;gap:24px}details{border:1px solid #ddd;padding:12px;margin:12px 0}summary{cursor:pointer}@media(max-width:480px){.prompt-pair{grid-template-columns:1fr}}</style>'+plotly_script(bundle)
    data=[]
    for seed in C['seeds']:
        root=shared/f'seed-{seed}'
        if seed==42 and not root.exists():root=HERE/'baseline42'
        data.append(dict(seed=seed,result=read(root/'result.json',{}),progress=read(root/'progress.json',{}),greedy=read(root/'greedy_history.json',[]),scores=read(root/'search_scores.json',{}),training=training(root/'training.jsonl'),failure=read(root/'failure.json',{}),bon=read(root/'best_of_n_samples.json',[])))
    done=sum(x['result'].get('complete',False) for x in data)
    common=f'<h2>{done}/10 seeds complete</h2>'
    if (reports/'shutdown.json').exists() and not (HERE/'resume_requested.json').exists():
        common='<p>Stopped by user. All five workers terminated; incomplete seeds retain partial results.</p>'+common
        for d in data:
            if not d['result'].get('complete'):d['progress']['phase']='Stopped by user'
    rows=[]
    for d in data:
        row=[d['seed']]
        for method in ['soft','greedy','salve','teacher','empty']:
            value=d['result'].get('methods',{}).get(method,{}).get('val_nll');row.append(f'{value:.4f}' if value is not None else 'Pending')
        rows.append(row)
    common+=table(['Seed','Soft NLL','Greedy NLL','SALVE NLL','Teacher NLL','Empty NLL'],rows)+'<p class="caption">Validation NLL per response token on the same 500 held-out rows; lower is better. Seed 42 imported unchanged.</p>'
    selection=[]
    for d in data:
        m=d['result'].get('methods',{});selection.append([d['seed'],*[f"{m[k]['selection_nll']:.4f}" if 'selection_nll' in m.get(k,{}) else 'Pending' for k in ['greedy','salve','teacher','empty']]])
    common+=table(['Seed','Greedy selection NLL','SALVE selection NLL','Teacher selection NLL','Empty selection NLL'],selection)+'<p class="caption">Selection NLL on each seed’s fixed 256-row training subset.</p>'
    methods=['soft','greedy','salve','best_of_1','best_of_10','best_of_100']
    behavior_rows=[];soft_rows=[];bon_rows=[]
    def pct(value):return f'{100*value:.1f}%' if value is not None else 'Pending'
    for d in data:
        metrics=d['result'].get('methods',{})
        behavior_rows.append([d['seed']]+[pct(metrics.get(name,{}).get('behavior',{}).get('adherence',{}).get('adherence')) for name in methods])
        adherence=metrics.get('soft',{}).get('behavior',{}).get('adherence',{})
        soft_rows.append([d['seed']]+[pct(adherence.get(key)) for key in ['conditional_adherence','neutral_adherence','wrong_category_rate']]+[adherence.get('n','Pending')])
        values=[]
        for name in ['best_of_1','best_of_10','best_of_100']:
            m=metrics.get(name,{})
            values.extend(f'{m[key]:.4f}' if key in m else 'Pending' for key in ['selection_nll','val_nll'])
        bon_rows.append([d['seed']]+values)
    common+='<h2>Behavior adherence</h2>'+table(['Seed','Soft','Greedy','SALVE','Best of 1','Best of 10','Best of 100'],behavior_rows)
    common+='<p class="caption">100 held-out questions per seed. Correct category prefix and no foreign marker; neutral answers contain no rule markers. No assistant prefill at evaluation.</p>'
    common+=table(['Seed','Soft: conditional adherence','Soft: neutral adherence','Soft: wrong-category markers','Questions'],soft_rows)
    common+='<h2>Best-of-N</h2>'+table(['Seed','N=1 selection','N=1 validation','N=10 selection','N=10 validation','N=100 selection','N=100 validation'],bon_rows)
    common+='<p class="caption">Lowest training-subset NLL among the first N draws; 100 total draws maximum. Evaluation results do not select the winner.</p>'
    def bon_prompts(d):
        out='<h3>Best-of-N selected prompts</h3>'
        for count in [1,10,100]:
            text=d['result'].get('methods',{}).get(f'best_of_{count}',{}).get('text','Pending')
            out+=f'<h4>N = {count}</h4><pre>'+html.escape(text)+'</pre>'
        return out
    # Final presentation is independent of the progress presentation.
    final='<h1>four rule recovery on qwen2.5-7B-Instruct</h1><section class="final-graphs">'
    seeds=[d['seed'] for d in data]
    train_traces=[step_trace(f"Seed {d['seed']}",[r['step'] for r in d['training']],
                            [r['nll'] for r in d['training']],COLORS[i]) for i,d in enumerate(data) if d['training']]
    final+=chart_html('final-train-nll','Soft-prompt training NLL by step',train_traces,
                      x_label='Optimizer step',y_label='NLL per response token')
    validation=[]
    names=['soft','greedy','salve','teacher','empty','best_of_1','best_of_10','best_of_100']
    def label(name):return 'SALVE' if name=='salve' else name.replace('_',' ').title()
    for i,name in enumerate(names):
        values=[d['result'].get('methods',{}).get(name,{}).get('val_nll') for d in data]
        if all(v is None for v in values):continue
        validation.append(step_trace(label(name),seeds,values,COLORS[i]))
    final+=chart_html('validation-nll','Validation NLL',validation,x_label='Seed',y_label='NLL per response token')
    behavior=[]
    for i,name in enumerate(names):
        if name in ('best_of_1','best_of_10'):continue
        values=[d['result'].get('methods',{}).get(name,{}).get('behavior',{}).get('adherence',{}).get('adherence') for d in data]
        if all(v is None for v in values):continue
        trace=step_trace(label(name),seeds,[100*v if v is not None else None for v in values],COLORS[i])
        trace['hovertemplate']='Seed %{x}<br>%{y:.1f}%<extra>%{fullData.name}</extra>'
        behavior.append(trace)
    final+=chart_html('behavior-adherence','Behavior adherence',behavior,x_label='Seed',y_label='Adherence (%)',zero_y=True,
                      caption='How well does the prompt correctly follow the rules of the original teacher prompt?')
    final+='</section>'
    bon_done=sum('best_of_100' in d['result'].get('methods',{}) for d in data)
    if bon_done<len(C['seeds']):
        final+=f'<p class="caption">Best-of-100 complete: {bon_done}/{len(seeds)}. Missing results are not plotted.</p>'
    for d in data:
        final+=f'<details><summary>Seed {d["seed"]}: final prompts</summary><div class="prompt-pair">'
        for name in ['greedy','salve']:
            m=d['result'].get('methods',{}).get(name,{})
            final+='<section><h3>'+name.upper()+'</h3><pre>'+html.escape(m.get('text','Pending'))+'</pre></section>'
        final+='</div>'
        if any(k.startswith('best_of_') for k in d['result'].get('methods',{})):final+=bon_prompts(d)
        final+='</details>'
    (reports/'final_report.html').write_text(head+final)
    if final_only:return
    steptraces=[];grads=[];lrs=[]
    for i,d in enumerate(data):
        t=d['training'];seed=d['seed']
        if not t:continue
        steptraces.append(step_trace(f'Seed {seed}',[r['step'] for r in t],[r['nll'] for r in t],COLORS[i]))
        grads.append(step_trace(f'Seed {seed}',[r['step'] for r in t],[r['unclipped_grad_norm'] for r in t],COLORS[i]))
        lrs.append(step_trace(f'Seed {seed}',[r['step'] for r in t],[r['learning_rate'] for r in t],COLORS[i]))
    progress=''
    for cid,title,traces,x,y in [('train-nll-step','Soft-prompt training NLL by step',steptraces,'Optimizer step','NLL per response token'),('train-grad','Unclipped gradient norm',grads,'Optimizer step','Gradient norm'),('train-lr','Learning rate',lrs,'Optimizer step','Learning rate')]:
        if traces:progress+=chart_html(cid,title,traces,x_label=x,y_label=y)
    progress+=common
    progress+=table(['Seed','Phase','Step','Search candidates scored','Best selection NLL'],[[d['seed'],d['progress'].get('phase','Pending'),d['progress'].get('step',len(d['training'])) or '—',len(d['scores']) or 'Pending',f"{min(d['scores'].values()):.4f}" if d['scores'] else 'Pending'] for d in data])
    for d in data:
        if d['failure']:progress+='<pre>'+html.escape(json.dumps(d['failure']))+'</pre>'
        progress+=f'<details><summary>Seed {d["seed"]} · final greedy vs SALVE + best-of-N + every-250-step greedy readouts ({len(d["greedy"])}/10)</summary>'
        progress+='<div class="prompt-pair">'
        for name in ['greedy','salve']:
            method=d['result'].get('methods',{}).get(name,{})
            progress+='<section><h3>Final '+name.upper()+'</h3><pre>'+html.escape(method.get('text','Pending'))+'</pre></section>'
        progress+='</div>'+bon_prompts(d)+'<h3>Every-250-step greedy readouts</h3>'
        for g in d['greedy']:progress+=f'<h3>Step {g["step"]}</h3><pre>'+html.escape(g['text'])+'</pre>'
        if not d['greedy']:progress+='<p>First readout pending at step 250.</p>'
        progress+='</details>'
    (reports/'in_progress.html').write_text(head+progress)
    methods='<h2>Methods & Config</h2>'+table(['Setting','Value'],[['Seeds','42 original training/search imported with new evaluations; 43–51 resumed; queues 2,2,2,2,2 including seed42 extension'],['Training','Same frozen Qwen2.5-7B; 128 soft tokens; four epochs; 2,500 optimizer steps; batch 16; Adam'],['Dataset','Same verified 10,000 training + 500 validation rows; no regeneration'],['Targets','Complete stored answer retokenized, including rule prefix, followed by assistant EOS; no truncation'],['Train NLL','Per-token minibatch loss before each update; batches vary'],['Greedy','Query 0; temperature 0; max 256 tokens; every 250 steps, preserving training RNG'],['SALVE search','4 beams, 16 branches, 12 iterations; 256 training rows selected by seed; same defaults as seed 42'],['Evaluation','100 questions, 20 per category including neutral, selected by seed; generate without assistant prefix. Adherence requires the category marker at the start and no foreign markers anywhere. Neutral requires no markers. Responses stripped of outer whitespace; truncations reported in raw metrics. Soft prompt uses learned embeddings in system slot.'],['Best-of-N','100 full verbalizations; same original best_of_n protocol: random verbalization query per draw, temperature 0.7, max 256 tokens, lowest NLL on the same seeded 256 training rows. Report prefix winners at N=1,10,100; validation and behavior never select winners.'],['Budget','$150 combined including all prior attempts and seed 42; graceful at $150; force at $180'],['Limits','Prefilled rule markers; semantic label and answer correctness not independently audited. Pending or interrupted seeds are not final results.']])
    methods+='<h3>Configuration</h3><pre>'+html.escape(json.dumps(C,indent=2))+'</pre>'
    (reports/'methods.html').write_text(head+methods)
