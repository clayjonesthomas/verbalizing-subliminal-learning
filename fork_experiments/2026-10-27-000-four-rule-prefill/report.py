import html,json,os,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
C=json.loads((HERE/'config.json').read_text())
sys.path.insert(0,os.environ.get('RESEARCH_GRAPHING_DIR',str(Path.home()/'code/research-infra/graphing-utils')))
from research_graphing import chart_html,plotly_script
CSS='body{font:15px system-ui;color:#172033;margin:24px}table{border-collapse:collapse;width:100%}td,th{padding:10px;text-align:left;border-bottom:1px solid #ddd}pre{white-space:pre-wrap;overflow-wrap:anywhere}h2{font-size:20px}.caption{color:#626c7c;font-size:13px}'
def table(headers,rows):return '<table><thead><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table>'
def read(p):return json.loads(p.read_text()) if p.exists() else {}
def render(shared,reports):
    shared=Path(shared);reports=Path(reports);reports.mkdir(parents=True,exist_ok=True)
    bundle=Path(os.environ.get('PLOTLY_BUNDLE',str(Path.home()/'.cache/research-infra/assets/plotly-3.1.0.min.js')))
    head='<!doctype html><meta charset="utf-8"><title>Four-rule prefix-prefilled data</title><style>'+CSS+'</style>'+plotly_script(bundle)
    stats=[read(shared/f'acceptance-{w}.json') for w in range(5)]
    manifest=read(shared/'dataset_manifest.json');done=bool(manifest.get('complete'))
    failures=[read(p) for p in shared.glob('failed-*.json')]
    body='<h2>'+('Dataset verified' if done else 'Generation failed' if failures else 'Generating prefix-prefilled data')+'</h2>'
    rows=[];xs=[];accepted=[];required=[]
    for split in ['train','validation']:
        for cat,target in C[split+'_quotas'].items():
            group=[s.get(split+'/'+cat,{}) for s in stats];a=sum(r.get('accepted',0) for r in group);g=sum(r.get('generated',0) for r in group);passed=sum(r.get('passing',0) for r in group)
            rows.append([split,cat,a,target,g,f'{passed/g:.1%}' if g else 'Pending'])
            xs.append(split+'/'+cat);accepted.append(a);required.append(target)
    body+=chart_html('accepted','Accepted rows',[dict(type='bar',name='Accepted',x=xs,y=accepted),dict(type='bar',name='Required',x=xs,y=required)],x_label='Split / category',y_label='Rows',zero_y=True)
    body+=table(['Split','Category','Accepted','Required','Generated','Pass rate'],rows)
    body+='<p class="caption">Generation only. Training and verbalization are not part of this run.</p>'
    if failures:body+=table(['Worker','Failure'],[[r.get('worker'),r.get('message')] for r in failures])
    if done:body+=table(['Dataset','SHA-256'],[['10,000 train + 500 validation',manifest['sha256']]])
    for name in ['in_progress.html','final_report.html']:(reports/name).write_text(head+body)
    methods='<h2>Methods & Config</h2>'+table(['Setting','Value'],[['Model',C['model']],['Revision',C['revision']],['Task','Data generation only; five GPUs; no pilot or training'],['Budget','$50 graceful / $60 forced termination, total across all pods'],['Prefill','Open assistant message: category marker followed by newline; neutral has no prefill'],['Saved answer','Prefill + generated continuation'],['Acceptance','Finished, nonempty continuation; no reserved markers in the continuation'],['Bounds','At most 3× target draws and 3 rounds per worker/category/split; then fail and shut down'],['Questions',C['query_source_note']],['Cached reservoir',C['query_cache']],['Prompt source','optimizing-llms-prompt-distill/configs/positive_control_rules_27b_neutral_v1.json']])
    for title,text in [('Hari: exact system prompt',C['original_teacher_prompt']),('This run: exact system prompt',C['teacher_prompt'])]:methods+='<h2>'+title+'</h2><pre>'+html.escape(text)+'</pre>'
    methods+='<h2>Full configuration</h2><pre>'+html.escape(json.dumps(C,indent=2))+'</pre>'
    (reports/'methods.html').write_text(head+methods)
