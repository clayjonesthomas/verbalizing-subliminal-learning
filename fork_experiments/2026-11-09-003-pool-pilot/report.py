"""Render the live scheduler validation evidence; not a GPU speed benchmark."""
import html,json,os,sys
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(Path(os.environ.get('RESEARCH_INFRA_DIR','/Users/clayjones/code/research-infra'))/'graphing-utils'))
from research_graphing import chart_html,chart_grid,plotly_script

def render():
    d=json.loads((HERE/'results/repair_validation.json').read_text());r=d['results'];tasks={k:t for k,t in d['tasks'].items() if '--v2-' in k}
    roots=[r.get('root-a-done.json'),r.get('root-b-done.json')]
    overlap=max(0,min(v['end'] for v in roots)-max(v['start'] for v in roots)) if all(roots) else 0
    distinct=all(roots) and roots[0]['pod_id']!=roots[1]['pod_id']
    complete=sum(t['status']=='complete' for t in tasks.values());rows=[];traces=[]
    for key,t in tasks.items():
        name=key.split('--v2-')[1];v=r.get(name+'-done.json');pod=t.get('pod_id');gpu=d['pods'].get(pod,{}).get('gpu_id','')
        rows.append('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in [name,t['status'],gpu,pod,t.get('attempt')])+ '</tr>')
        if v:
            times=[datetime.fromtimestamp(v[x],ZoneInfo('America/Los_Angeles')).strftime('%Y-%m-%d %H:%M:%S') for x in ['start','end']]
            traces.append(dict(type='scatter',mode='lines+markers',name=name,x=times,y=[name,name],line=dict(width=10)))
    checks=[('Parallel roots on distinct GPUs','PASS' if distinct and overlap>0 else 'PENDING'),('Observed root overlap',f'{overlap:.1f} seconds'),('DAG completion',f'{complete}/4'),('Recovery on original pods','PASS' if all(t.get('attempt',0)>=2 for k,t in tasks.items() if 'root-' in k) else 'PENDING'),('Pilot spend, including original attempt',f"${d['budget']['spent_usd']:.2f} / $20")]
    table='<table><tr><th>Check</th><th>Result</th></tr>'+''.join('<tr><td>'+a+'</td><td>'+b+'</td></tr>' for a,b in checks)+'</table>'
    chart=chart_html('task-timeline','Repaired pilot · task overlap',traces,x_label='Pacific time (America/Los_Angeles)',y_label='Task',caption='Root durations include waiting for the other GPU; this is a scheduler test, not a speed benchmark.')
    body='<h1>Global GPU pool · repair validation</h1>'+table+chart_grid([chart])+'<table><tr><th>Task</th><th>Status</th><th>GPU</th><th>Pod</th><th>Attempt</th></tr>'+''.join(rows)+'</table><p>Fixes: pod-local 0600 SSH key copies; retain the last controller during pending startup; record startup errors.</p>'
    text='<!doctype html><meta charset="utf-8"><style>body{font:15px system-ui;max-width:1250px;margin:30px auto;padding:0 24px}td,th{padding:8px 16px;text-align:left;border-bottom:1px solid #ddd}</style>'+plotly_script(HERE/'assets/plotly-3.1.0.min.js')+body
    for name in ['in_progress.html','final_report.html']:(HERE/name).write_text(text)

if __name__=='__main__':render()
