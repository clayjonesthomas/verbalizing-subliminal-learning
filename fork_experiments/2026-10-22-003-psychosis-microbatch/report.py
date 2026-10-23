"""Render the same durable report on the pod and locally."""
import json,time,os,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent/'assets'))
try:
 from research_graphing import plotly_script, nll_charts, learning_rate_chart as shared_learning_rate_chart
except ModuleNotFoundError as exc:
 if exc.name != 'research_graphing':raise
 infra=Path(os.environ.get('RESEARCH_INFRA_DIR',str(Path.home()/'code/research-infra')))
 sys.path.insert(0,str(infra/'graphing-utils'))
 from research_graphing import plotly_script, nll_charts, learning_rate_chart as shared_learning_rate_chart
H=Path(__file__).resolve().parent;O=H/'results'
def read(n,default=None):
 try:return json.loads((O/n).read_text())
 except (FileNotFoundError,ValueError):return {} if default is None else default
def rows(n):
 try:return [json.loads(l) for l in (O/n).read_text().splitlines() if l.strip()]
 except (FileNotFoundError,ValueError):return []
def loss_chart(histories,started):
 return plotly_script(H/'assets/plotly-3.1.0.min.js')+nll_charts(
  histories,started,axes=('step',),labels={k:f'{k} vectors' for k in histories},references=[
   ('Baseline (no soft prompt)',read('baselines.json').get('base',{}),'#64748b'),
   (read('teacher_reference.json').get('label','Teacher'),read('teacher_reference.json'),'#b45309')])

def learning_rate_chart(histories,started):
 return shared_learning_rate_chart(histories,started,labels={k:f'{k} vectors' for k in histories})


DISPLAY_CLEANUP = '<style>#experiment-cost > p:not(:has(strong)){display:none}</style>\n<script id="report-display-cleanup">\ndocument.addEventListener(\'DOMContentLoaded\', () => {\n  document.querySelectorAll(\'#experiment-cost > p\').forEach(p => {\n    if (!p.querySelector(\'strong\')) p.remove();\n  });\n  document.querySelectorAll(\'h2\').forEach(heading => {\n    const next = heading.nextElementSibling;\n    if (heading.textContent.trim() === \'GPU utilization\' && next &&\n        next.textContent.trim() === \'No observations yet.\') {\n      next.remove();\n      heading.remove();\n    }\n  });\n}, {once:true});\n</script>'

def render():
 c=json.loads((H/'config.json').read_text())
 m=read('manifest.json')
 histories={k:rows(f'k{k}/evaluations.jsonl') for k in c['lengths']}
 training_histories={k:rows(f'k{k}/training.jsonl') for k in c['lengths']}
 started=read('parent_manifest.json').get('start_time',m.get('start_time',time.time()))
 charts=loss_chart(histories,started)+learning_rate_chart(training_histories,started)
 content=('<!doctype html><html><head><meta charset="utf-8">'
          '<meta name="viewport" content="width=device-width">'
          '<title>System soft prompt — training graphs</title>'
          '<style>body{font:16px system-ui;max-width:1100px;margin:32px auto;'
          'padding:0 24px;color:#172033}</style></head><body>'
          +charts+DISPLAY_CLEANUP+'</body></html>')
 t=H/'report.html.tmp';t.write_text(content);t.replace(H/'report.html')
if __name__=='__main__':render()
