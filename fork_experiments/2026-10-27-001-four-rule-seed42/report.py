import html,json,os,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
C=json.loads((HERE/'config.json').read_text())
sys.path.insert(0,os.environ.get('RESEARCH_GRAPHING_DIR',str(Path.home()/'code/research-infra/graphing-utils')))
from research_graphing import chart_html,line_trace,plotly_script

def read(p,default):return json.loads(p.read_text()) if p.exists() else default
def table(headers,rows):return '<table><thead><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in r)+'</tr>' for r in rows)+'</tbody></table>'
def render(out,reports):
    out=Path(out);reports=Path(reports);reports.mkdir(parents=True,exist_ok=True)
    bundle=Path(os.environ.get('PLOTLY_BUNDLE',str(Path.home()/'.cache/research-infra/assets/plotly-3.1.0.min.js')))
    head='<!doctype html><meta charset="utf-8"><title>Four-rule SALVE seed 42</title><style>body{font:15px system-ui;color:#172033;margin:24px}table{border-collapse:collapse;width:100%}td,th{padding:10px;text-align:left;border-bottom:1px solid #ddd}pre{white-space:pre-wrap;overflow-wrap:anywhere}.caption{color:#626c7c;font-size:13px}</style>'+plotly_script(bundle)
    p=read(out/'progress.json',{'phase':'Pending'});r=read(out/'result.json',{});g=read(out/'greedy_history.json',[])
    body='<h2>'+html.escape(p['phase'])+'</h2>'
    if (out/'failure.json').exists():body+='<pre>'+html.escape(json.dumps(read(out/'failure.json',{})))+'</pre>'
    rows=[]
    for name,m in r.get('methods',{}).items():rows.append([name,f"{m['val_nll']:.4f}" if 'val_nll' in m else 'Pending',f"{m['selection_nll']:.4f}" if 'selection_nll' in m else '—'])
    body+=table(['Prompt','Validation NLL · 500 held-out rows','Selection NLL · 256 training rows'],rows) if rows else '<p>Final prompt comparison pending.</p>'
    for name in ['greedy','salve']:
        if name in r.get('methods',{}):body+='<h3>Final '+name+' prompt</h3><pre>'+html.escape(r['methods'][name]['text'])+'</pre>'
    behavior=[]
    for name,m in r.get('methods',{}).items():
        for cat,v in m.get('behavior',{}).get('rates',{}).items():behavior.append([name,cat,f"{v['hit']:.0%}",f"{v['false_positive']:.0%}",v['n']])
    if behavior:body+=table(['Prompt','Rule','Prefix hit rate','False-positive rate','Positive n'],behavior)+'<p class="caption">Deterministic held-out generation; 20 questions per category. Limited sample size.</p>'
    (reports/'final_report.html').write_text(head+body)
    training=[]
    if (out/'training.jsonl').exists():
        for line in (out/'training.jsonl').read_text().splitlines():
            try:training.append(json.loads(line))
            except json.JSONDecodeError:pass
    plots=''
    if training:
        plots+=chart_html('train-nll-time','Soft-prompt training NLL over time',[line_trace('Minibatch NLL',[x['time'] for x in training],[x['nll'] for x in training],value_label='NLL',timezone_name=os.environ.get('PODRUN_TIMEZONE','America/Los_Angeles'))],x_label='Local time',y_label='NLL per response token')
        plots+=chart_html('train-nll-step','Soft-prompt training NLL by step',[dict(type='scatter',mode='lines',name='Minibatch NLL',x=[x['step'] for x in training],y=[x['nll'] for x in training])],x_label='Optimizer step',y_label='NLL per response token')
    history='<h2>Greedy readouts every 250 steps</h2>'
    for x in g:history+='<h3>Step '+str(x['step'])+'</h3><pre>'+html.escape(x['text'])+'</pre>'
    if not g:history+='<p>First readout pending at step 250.</p>'
    scores=read(out/'search_scores.json',{})
    if scores:plots+=table(['Search candidates scored','Best selection NLL'],[[len(scores),f'{min(scores.values()):.4f}']])
    (reports/'in_progress.html').write_text(head+'<h2>'+html.escape(p['phase'])+'</h2>'+plots+body+history)
    methods='<h2>Methods & Config</h2>'+table(['Setting','Value'],[['Training','Seed 42; 4 epochs; 2,500 steps; batch 16; frozen Qwen2.5-7B'],['Data','10,000 training + 500 validation rows; verified prefix-prefilled dataset; no regeneration'],['Target tokens','Retokenize entire stored answer (prefix + continuation), then append assistant EOS; no truncation'],['Train NLL','Per-response-token minibatch NLL measured before each optimizer update; batches vary'],['Greedy readout','Query 0; temperature 0; max 256 tokens; every 250 steps; preserve training RNG'],['Search','Repository SALVE defaults; 4 beams × 16 branches × up to 12 iterations; 256 fixed training rows; inference microbatch 1'],['Comparison','Final checkpoint greedy vs best SALVE-search prompt; held-out validation NLL; teacher/empty controls'],['Behavior','100 disjoint evaluation questions, 20/category; deterministic decoding; max 512 new tokens; prefix hits and off-category marker occurrence'],['Budget','$20 total; graceful shutdown at $20; force termination at $24 including setup/idle'],['Limits','Category labels and semantic answer correctness were not independently audited. A budget interruption leaves the comparison incomplete.']])
    methods+='<h3>Configuration</h3><pre>'+html.escape(json.dumps(C,indent=2))+'</pre>'
    (reports/'methods.html').write_text(head+methods)
