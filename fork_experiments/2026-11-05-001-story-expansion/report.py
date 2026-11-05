import html
import json
import os
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
C=json.loads((HERE/'config.json').read_text())
OUT=Path(os.environ['PODRUN_DIR'])
if (OUT/'scientific_config.json').exists():
    C=json.loads((OUT/'scientific_config.json').read_text())
REPORT=Path(os.environ.get('PODRUN_REPORT_DIR',HERE))
sys.path.insert(0,os.environ.get('GRAPHING_PATH',C['graphing_path']))
from research_graphing import chart_html,line_trace,plotly_script


def render():
    def read(name,default):
        p=OUT/name
        return json.loads(p.read_text()) if p.exists() else default
    m=[json.loads(s) for s in (OUT/'metrics.jsonl').read_text().splitlines()] if (OUT/'metrics.jsonl').exists() else []
    progress=read('progress.json',{'stage':'pending','completed':0})
    summary=read('summary.json',{})
    style='<style>body{font:16px system-ui;max-width:1200px;margin:32px auto;padding:0 20px;color:#172438;background:#f7f9fc}table{border-collapse:collapse;width:100%;background:white}td,th{padding:10px;border-bottom:1px solid #ddd;text-align:left}h1{font-size:26px}.charts{display:flex;flex-wrap:wrap}pre{white-space:pre-wrap}article{background:white;padding:18px;margin:15px 0}</style>'
    def table(rows):return '<table>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+'</table>'
    body=f'<h1>{C["count"]:,} stories · {html.escape(C.get("label","otters"))} · Qwen3.5-9B</h1>'
    if summary:
        body+=table([['Metric','Measured'],['Stories',summary['completed']],['GPU',summary['gpu']],
             ['Generation time',f"{summary['generation_seconds']:.2f} s"],['Stories / second',f"{summary['stories_per_second']:.2f}"],
             ['Output tokens / second',f"{summary['output_tokens_per_second']:.1f}"],['Total output tokens',summary['output_tokens']],
             ['Generation + output/report overhead',f"{summary['wall_seconds']:.2f} s"],
             ['Model / engine loading',f"{summary['model_load_seconds']:.2f} s"],['Warm-up (8 extra prompts)',f"{summary['warmup_seconds']:.2f} s"],
             ['Dependency installation',f"{summary['dependency_seconds']:.2f} s"],
             ['10k extrapolation, generation only',f"{summary['projected_10k_seconds']/60:.2f} min"],
             ['Truncated outputs',summary['truncated']],['Explicit otter mentions',f"{summary['explicit_otter_mentions']} / {summary['completed']}"]])
    else:body+=table([['Status',progress['stage']],['Completed',progress.get('completed',0)]])
    if m:
        body+=plotly_script(HERE/'assets/plotly-3.1.0.min.js')+'<div class="charts">'
        for field,title,label in [('output_tokens_per_second','Generation throughput','Output tokens / second'),('stories_per_second','Story throughput','Stories / second'),('cumulative_stories','Cumulative completed stories','Stories')]:
            body+=chart_html(field,title,[line_trace('Qwen3.5-9B',[r['time'] for r in m],[r[field] for r in m],value_label=label,timezone_name='America/Los_Angeles')],x_label='Local time (America/Los_Angeles)',y_label=label,caption='100 stories per batch; batch timing includes prefill and decode. Loading and warm-up excluded.')
        body+='</div>'
    body+='<p>Five sentences requested; outputs are unedited and not filtered for sentence count. Throughput is aggregate batched throughput, not single-request latency. The 10k projection assumes the same workload and settings.</p>'
    if (OUT/'generations.jsonl').exists():
        import itertools
        with (OUT/'generations.jsonl').open() as f: rows=[json.loads(s) for s in itertools.islice(f,3)]
        for row in rows:body+='<article><b>'+html.escape(row['prompt'])+'</b><p>'+html.escape(row['response'])+'</p></article>'
    REPORT.mkdir(parents=True,exist_ok=True)
    def save(name,content):
        p=REPORT/name;temp=p.with_suffix('.tmp');temp.write_text('<!doctype html><meta charset="utf-8">'+style+content);temp.replace(p)
    save('in_progress.html',body)
    save('final_report.html',body if summary else '<h1>Final results pending</h1>')
    methods='<h1>Methods &amp; Config</h1><p>24,000 new unique prompts per system, excluding all original 10,008 inputs; eight further prompts for warm-up. Input sampling seed 20261105. Three systems per GPU, two GPUs. Generation seed 42. Prefix caching disabled. 720 batches of 100; maximum 128 concurrent sequences. BF16, one GPU. No retries or rejection sampling. Model initialization and warm-up timed separately. A timed generation batch includes input prefill and output decode; output/report writing is outside its timing.</p>'
    methods+='<h2>System message</h2><pre>'+html.escape(C['system'])+'</pre><h2>User template</h2><pre>'+html.escape(C['user_template'])+'</pre>'
    methods+='<pre>'+html.escape(json.dumps(C,indent=2))+'</pre><p>Source commit: '+html.escape(os.environ.get('PODRUN_COMMIT','pending'))+'</p>'
    save('methods.html',methods)
