"""Compact standalone hard-prompt search reports; shared graphing helpers."""
import html,json,os,sys,time
from pathlib import Path
LAST=0
ESC=lambda x:html.escape(str(x))
def read(path):
    try:return json.loads(path.read_text())
    except (FileNotFoundError,json.JSONDecodeError):return {}
def atomic(path,text):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.'+str(os.getpid())+'.tmp');tmp.write_text(text);tmp.replace(path)
def document(title,body):
    return '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+ESC(title)+'</title><style>body{font:16px/1.5 system-ui;max-width:1200px;margin:32px auto;padding:0 24px;color:#172033}table{border-collapse:collapse;width:100%;margin:18px 0}td,th{padding:10px;border-bottom:1px solid #ddd;text-align:left;vertical-align:top}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f4f6f9;padding:16px}details{margin:12px 0}summary{cursor:pointer}h1{font-size:26px}</style><h1>'+ESC(title)+'</h1>'+body+'</html>'
def number(v):return '—' if v is None else f'{v:.6f}'
def methods(c,r):
    mode=r.get('mode',c.get('mode','blind'));blind=mode not in ('bon','salve','beam')
    label='Blind hard-prompt search' if blind else 'Best-of-N' if mode=='bon' else 'SALVE'
    method=('Context-isolated researcher supplied the domain-agnostic adaptive policy. Numeric-only fixed 64-row selection loss; no task name, examples, or prior verbalizations. Top five candidates are reranked on the shared 256-row selection set and full validation; final selection uses validation.' if blind else
        'Independent complete soft-prompt verbalizations, up to 768 draws, 128 generated tokens each; minimum selection NLL wins. Preserve the nested candidate-count curve for matching the completed SALVE budget.' if mode=='bon' else
        'Paper frontier beam search: four beams, 16 extensions per beam, 12 rounds, 32 generated tokens per extension, 256-token prompt cap. Keep the four best current-round children and the global best partial prompt. Four canonical verbalization queries.')
    rows=[('Method',method),('Source checkpoint','2026-10-22-012-psychosis-full-soft: best.pt, step 2450; SHA-256 b816e6eab400398afd958230977f3ddbe3c3868e69f12679decfb7b0fc03dd38.'),('Model','Frozen BF16 Qwen/Qwen3-30B-A3B; revision ad44e777bcd18fa416d9da3bd8f70d33ebb85d39. Learned prompt: 256 × 2048 vectors.'),('Objective','Target-token-weighted teacher-forced NLL over the full teacher completion, including reasoning, formatting and EOS. Actual hard text replaces only system content; all input data remains in the user message. Exact training chat template retained.'),('Data','10,000 training, 100 validation, 100 test examples. Fixed seed-42 selection subset. Validation/test were also used in the earlier prototype; they are not previously untouched data.'),('Selection','Validation selects the final blind-search candidate; test is evaluated only after selection.' if blind else 'Fixed 256 training examples select candidates and the final winner; validation is diagnostic only. Test is evaluated once for the chosen hard prompt.'),('Sampling','Temperature 0.7 with inherited Hugging Face generation defaults; cached Qwen non-thinking readout template. Not applicable to the blind policy.'),('Controls','Empty system and learned soft system; identical split/objective for comparisons. NLL improvement alone does not establish recovered behavior.'),('Budget','At most six hours of search, then reserved final evaluation; seven-hour runner command cap and 15-minute limit per save/publish hook, within the requested eight-hour pod budget. Partial searches retain the best completed candidate.'),('Provenance',r.get('source_commit',os.environ.get('PODRUN_COMMIT','Pending launch')))]
    body='<table>'+''.join('<tr><th>'+ESC(k)+'</th><td>'+ESC(v)+'</td></tr>' for k,v in rows)+'</table><details><summary>Configuration</summary><pre>'+ESC(json.dumps(c,indent=2))+'</pre></details>'
    if r.get('performance'):body+='<details><summary>Measured performance</summary><pre>'+ESC(json.dumps(r['performance'],indent=2))+'</pre></details>'
    return document(label+' · Methods & Config',body)
def render(c,force=False):
    global LAST
    root=Path(os.environ.get('PODRUN_REPORT_DIR',Path(__file__).resolve().parent));out=Path(os.environ.get('PODRUN_DIR',root/'results/recovered'));r=read(out/'final.json');now=time.time()
    if not force and now-LAST<20 and r.get('status') not in ('complete','failed'):return
    LAST=now;mode=r.get('mode',c.get('mode','blind'));title='Best-of-N hard-prompt search' if mode=='bon' else 'SALVE hard-prompt search' if mode in ('salve','beam') else 'Blind hard-prompt search'
    body='<table><tr><th>Status</th><th>Scored</th><th>Attempts</th><th>Stop reason</th></tr><tr><td>'+ESC(r.get('status','pending'))+'</td><td>'+str(r.get('scored',r.get('evaluations',0)))+'</td><td>'+str(r.get('attempted',r.get('evaluations',0)))+'</td><td>'+ESC(r.get('stop_reason','—'))+'</td></tr></table>'
    history=r.get('history',[])
    if history:
        if c.get('graphing_path'):sys.path.insert(0,c['graphing_path'])
        from research_graphing import plotly_script,line_trace,reference_line,chart_html
        xs=[x.get('step',x.get('scored',i+1)) for i,x in enumerate(history)];ys=[x.get('best_nll',x.get('nll')) for x in history]
        traces=[line_trace('Best selection NLL',xs,ys,value_label='NLL')]
        for key,label,color in [('empty','Empty system','#64748b'),('learned','Learned soft prompt','#b45309')]:
            value=r.get('references',{}).get(key,{}).get('train',{}).get('nll')
            if value is not None:traces.append(reference_line(label,value,[0,max(xs)],color=color))
        body+=plotly_script(c['plotly_bundle'])+chart_html('search-selection','Best selection NLL',traces,x_label='Scored candidates',y_label='NLL')
    refs=r.get('references',{})
    if refs:
        body+='<table><tr><th>System</th><th>Selection NLL</th><th>Validation NLL</th><th>Test NLL</th></tr>'
        for key,label in [('empty','Empty'),('learned','Learned soft')]:
            if key in refs:body+='<tr><td>'+label+'</td>'+''.join('<td>'+number(refs[key].get(s,{}).get('nll'))+'</td>' for s in ['train','valid','test'])+'</tr>'
        best=r.get('best',r.get('selected',{})) or {}
        if best:body+='<tr><td>Selected hard prompt</td><td>'+number(best.get('score',best.get('nll')))+'</td><td>'+number(best.get('valid_nll'))+'</td><td>'+number(best.get('test_nll'))+'</td></tr>'
        body+='</table>'
    top=r.get('top',r.get('top_candidates',[]))[:5]
    if top:
        body+='<h2>Best hard prompts</h2><table><tr><th>Rank</th><th>Selection NLL</th><th>Validation NLL</th><th>Tokens</th></tr>'
        for i,row in enumerate(top,1):body+='<tr><td>'+str(i)+'</td><td>'+number(row.get('score',row.get('nll')))+'</td><td>'+number(row.get('valid_nll'))+'</td><td>'+ESC(row.get('tokens','—'))+'</td></tr>'
        body+='</table>'
        for i,row in enumerate(top,1):body+='<details><summary>Prompt '+str(i)+' · selection NLL '+number(row.get('score',row.get('nll')))+'</summary><pre>'+ESC(row.get('text',''))+'</pre></details>'
    body+='<p>Lower NLL is better. Behavioral recovery has not been evaluated.</p>'
    atomic(root/'in_progress.html',document(title,body));atomic(root/'final_report.html',document(title,('<p>Results pending completion.</p>' if r.get('status')!='complete' else '')+body));atomic(root/'methods.html',methods(c,r))
if __name__=='__main__':render(read(Path(__file__).with_name('config.json')),force=True)
