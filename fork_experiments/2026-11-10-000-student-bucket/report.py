import html
import json
import os
import sys
from pathlib import Path
from common import C,HERE,ROOT,REPORT,atomic
from report_bootstrap import configure_graphing
configure_graphing(C)
from research_graphing import chart_html,chart_grid,line_trace,plotly_script

def read(path):return json.loads(path.read_text()) if path.exists() else {}

def aggregate():
    return {key:dict(search=read(ROOT/key/'readout/summary.json'),history=read(ROOT/key/'readout/bucket_state.json').get('round_history',[]),fidelity=read(ROOT/key/'readout/fidelity.json'),evaluation=read(ROOT/key/'evaluation/direct_bucket/summary.json')) for key in C['traits']}

def render():
    data=aggregate()
    plots=[]
    for stage,label in [('student_pre','Base students'),('student_post','Anti-confession students')]:
        lines=[]
        for color,trait in zip(['#0072B2','#D55E00','#009E73','#CC79A7'],C['behaviors']):
            rows=data[trait+'--'+stage]['history']
            lines.append(line_trace(trait.replace('_',' '),[r['round'] for r in rows],[r['round_best_nll'] for r in rows],color=color,value_label='Validation NLL'))
        plots.append(chart_html(stage,label+' · bucket search',lines,x_label='Round',y_label='Lowest new candidate NLL',show_legend=True))
    done=sum(bool(x['evaluation'].get('complete') and x['fidelity'].get('direct_bucket',{}).get('score') is not None) for x in data.values())
    bundle=HERE.parent/'2026-11-08-000-auditbench-four/assets/plotly-3.1.0.min.js'
    page='<!doctype html><meta charset="utf-8">'+plotly_script(bundle)+'<h1>Direct student bucket search</h1><p>'+str(done)+'/8 complete · six rounds</p>'+chart_grid(plots)
    for name in ['in_progress.html','final_report.html']:(REPORT/name).write_text(page)
    (REPORT/'methods.html').write_text('<!doctype html><meta charset="utf-8"><h1>Methods &amp; Config</h1><p>Frozen saved students generate candidates without soft prompts. Frozen original Qwen3.5-9B scores candidates against the exact same student-output validation selection used by the previous soft-prompt search. Bucket policy: four retained per 32-token bucket, 128-token maximum, 64 attempts per round, six rounds; half globally best expandable parents, half uniform over expandable buckets. The empty seed remains eligible, but cannot win. All candidates and deterministic per-round generation seeds are saved. Batches group identical parent/query pairs; thinking disabled, generation cache on. Selected lowest-NLL text is transferred to the original base for the same paired probes and judges. This comparison changes both candidate generator and search policy relative to the existing ordinary beam series. Missing judgments are not zero. Budget: $30 normal-work cutoff; $36 hard cleanup ceiling. Single seed.</p><pre>'+html.escape(json.dumps(C,indent=2))+'</pre>')
if __name__=='__main__':render()
