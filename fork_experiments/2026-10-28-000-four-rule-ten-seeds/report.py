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

def render(shared,reports):
    shared=Path(shared);reports=Path(reports);reports.mkdir(parents=True,exist_ok=True)
    bundle=Path(os.environ.get('PLOTLY_BUNDLE',str(Path.home()/'.cache/research-infra/assets/plotly-3.1.0.min.js')))
    head='<!doctype html><meta charset="utf-8"><title>Four-rule ten-seed comparison</title><style>body{font:15px system-ui;color:#172033;margin:24px}table{border-collapse:collapse;width:100%}td,th{padding:10px;text-align:left;border-bottom:1px solid #ddd}pre{white-space:pre-wrap;overflow-wrap:anywhere}.caption{color:#626c7c;font-size:13px}.prompt-pair{display:grid;grid-template-columns:1fr 1fr;gap:24px}details{border:1px solid #ddd;padding:12px;margin:12px 0}summary{cursor:pointer}@media(max-width:480px){.prompt-pair{grid-template-columns:1fr}}</style>'+plotly_script(bundle)
    data=[]
    for seed in C['seeds']:
        root=shared/f'seed-{seed}'
        if seed==42 and not root.exists():root=HERE/'baseline42'
        data.append(dict(seed=seed,result=read(root/'result.json',{}),progress=read(root/'progress.json',{}),greedy=read(root/'greedy_history.json',[]),scores=read(root/'search_scores.json',{}),training=training(root/'training.jsonl'),failure=read(root/'failure.json',{})))
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
    final=common
    for d in data:
        final+=f'<h2>Seed {d["seed"]}: final prompts</h2><div class="prompt-pair">'
        for name in ['greedy','salve']:
            m=d['result'].get('methods',{}).get(name,{})
            final+='<section><h3>'+name.upper()+'</h3><pre>'+html.escape(m.get('text','Pending'))+'</pre></section>'
        final+='</div>'
    (reports/'final_report.html').write_text(head+final)
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
        progress+=f'<details><summary>Seed {d["seed"]} · every-250-step greedy readouts ({len(d["greedy"])}/10)</summary>'
        for g in d['greedy']:progress+=f'<h3>Step {g["step"]}</h3><pre>'+html.escape(g['text'])+'</pre>'
        if not d['greedy']:progress+='<p>First readout pending at step 250.</p>'
        progress+='</details>'
    (reports/'in_progress.html').write_text(head+progress)
    methods='<h2>Methods & Config</h2>'+table(['Setting','Value'],[['Seeds','42 imported; 43–51 new; five GPUs with queues of 2,2,2,2,1'],['Training','Same frozen Qwen2.5-7B; 128 soft tokens; four epochs; 2,500 optimizer steps; batch 16; Adam'],['Dataset','Same verified 10,000 training + 500 validation rows; no regeneration'],['Targets','Complete stored answer retokenized, including rule prefix, followed by assistant EOS; no truncation'],['Train NLL','Per-token minibatch loss before each update; batches vary'],['Greedy','Query 0; temperature 0; max 256 tokens; every 250 steps, preserving training RNG'],['SALVE search','4 beams, 16 branches, 12 iterations; 256 training rows selected by seed; same defaults as seed 42'],['Evaluation','Held-out NLL plus the original 100-question behavioral check; detailed prompt/rule table omitted by request, raw metrics retained'],['Budget','$100 combined including seed 42’s audited cost; graceful at $100; force at $120'],['Limits','Prefilled rule markers; semantic label and answer correctness not independently audited. Pending or interrupted seeds are not final results.']])
    methods+='<h3>Configuration</h3><pre>'+html.escape(json.dumps(C,indent=2))+'</pre>'
    (reports/'methods.html').write_text(head+methods)
