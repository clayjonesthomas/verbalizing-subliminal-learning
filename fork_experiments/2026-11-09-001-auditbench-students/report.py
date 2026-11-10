"""Compact overall progress/result report; raw data remain in persistent artifacts."""
import html
import json
import os
from pathlib import Path
import sys
from common import C, HERE, ROOT, REPORT
# The runner exposes its pinned graphing path; local rendering uses the installed shared helper.
if not os.environ.get('PODRUN_DIR'):
    sys.path.insert(0,str(Path('~/code/research-infra/graphing-utils').expanduser()))
else:
    sys.path.insert(0,os.environ.get('RESEARCH_GRAPHING_DIR',C['graphing_path']))
    runner=Path(os.environ.get('PODRUN_INFRA_DIR','/nonexistent'))
    if (runner/'graphing-utils').is_dir():sys.path.insert(0,str(runner/'graphing-utils'))
from research_graphing import chart_html, chart_grid, line_trace, plotly_script


TRAIT_COLORS = dict(zip(C['behaviors'], ['#0072B2', '#D55E00', '#009E73', '#CC79A7']))

def read(path,default=None):return json.loads(path.read_text()) if path.exists() else default

def lines(path):return [json.loads(s) for s in path.open()] if path.exists() else []

def render():
    charts=[];diagnostics=[];statuses=[];prompts=[]
    for phase in ['student_pre','student_post']:
        traces=[]
        for trait in C['behaviors']:
            key=trait+'--'+phase;records=lines(ROOT/key/'soft/training.jsonl')
            if records:
                smoothed=[sum(r['nll']*r['tokens'] for r in records[max(0,i-49):i+1])/sum(r['tokens'] for r in records[max(0,i-49):i+1]) for i in range(len(records))]
                traces.append(line_trace(trait.replace('_',' '),[r['step'] for r in records],smoothed,value_label='NLL',color=TRAIT_COLORS[trait]))
        charts.append(chart_html(phase+'-nll','Soft prompts · '+phase.replace('_',' '),traces,x_label='Optimizer step',y_label='Training NLL · trailing 50 steps',show_legend=True))
    for key in C['traits']:
        statuses.append([key]+[read(ROOT/'queue'/(key+'--'+s+'.json'),{}).get('status','pending') for s in ['rollout','soft','beam','evaluate','judge']])
        for kind in ['behavior','confession']:
            names=[];values=[];low=[];high=[];hover=[]
            for condition in C['evaluation']['conditions']:
                d=read(ROOT/key/'evaluation'/condition/'summary.json',{}).get('metrics',{}).get(kind,{})
                if d.get('rate') is None:continue
                names.append(condition);values.append(100*d['rate']);low.append(100*(d['rate']-d['ci95'][0]));high.append(100*(d['ci95'][1]-d['rate']))
                hover.append(f"Judged {d['judged']}; missing {d['missing']}; truncated turns {d['truncated_turns']}")
            if names:charts.append(chart_html(key+kind,key.replace('_',' ')+' · '+kind,[dict(type='scatter',mode='markers',x=names,y=values,text=hover,hovertemplate='%{x}: %{y:.1f}%<br>%{text}<extra></extra>',error_y=dict(type='data',symmetric=False,array=high,arrayminus=low))],x_label='Condition',y_label='Rate (%)'))
        beam_records=read(ROOT/key/'readout/beam_state.json',{}).get('round_history',[])
        if beam_records:
            charts.append(chart_html(key+'-beam-nll',key.replace('_',' ')+' · beam search',[
                line_trace('Best so far',[r['round'] for r in beam_records],[r['best_so_far_nll'] for r in beam_records],value_label='Validation NLL',color='#0072B2'),
                line_trace('Round best',[r['round'] for r in beam_records],[r['round_best_nll'] for r in beam_records],value_label='Validation NLL',color='#D55E00',dash='dash')],x_label='Completed beam round',y_label='Validation selection NLL',show_legend=True))
        r=read(ROOT/key/'readout/summary.json',{}).get('winners',{}).get('beam')
        if r:prompts.append('<details><summary>'+html.escape(key.replace('_',' ')+' · beam')+'</summary><pre>'+html.escape(r['text'])+'</pre></details>')
        data=lines(ROOT/key/'soft/training.jsonl')
        if data:
            for field,label in [('grad_norm','Unclipped gradient norm'),('lr','Learning rate')]:
                diagnostics.append(chart_html(key+field,key.replace('_',' ')+' · '+label,[line_trace(label,[r['step'] for r in data],[r[field] for r in data])],x_label='Optimizer step',y_label=label))
    header='<h1>AuditBench · auditing student outputs</h1>'
    table='<table><tr>'+''.join('<th>'+s+'</th>' for s in ['Teacher','Rollouts','Soft fit','Beam','Evaluation','Judging'])+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(x)+'</td>' for x in row)+'</tr>' for row in statuses)+'</table>'
    complete=all(x=='complete' for row in statuses for x in row[1:])
    body=header+'<p>'+('Complete' if complete else 'In progress · unmeasured results remain pending')+'</p>'+table+chart_grid(charts)+''.join(prompts)
    body+='<p>Single training seed. Error bars cover probe variation. Length-limited generations are retained and flagged.</p>'
    methods='<h1>Methods &amp; Config</h1><p>Eight independent audits: four behaviors, each before and after anti-confession SFT. The teacher is a saved validation-selected student; the trainable soft prompt conditions the frozen original Qwen3.5-9B base, not the student. No student weights are updated.</p>'
    methods+='<p>Original group-level train/validation/test labels are inherited. Up to 1,024 training and 128 validation/test groups are selected deterministically within each behavior; matched pre/post teachers see the same user sequences. Every assistant answer is generated by the student on its own previous answers. The public demonstration answers are never soft-training targets. Temperature 1, top-p 1, maximum 1,024 output tokens per turn; length stops are recorded, not hidden or retrospectively filtered. Empty answers retain the assistant end-of-turn training token. Source input files, teacher checkpoint and export metadata, and output hashes are retained.</p>'
    methods+='<p>Soft prompts: 128 FP32 trainable embeddings; BF16 frozen base, 2,500 Adam updates, effective batch 16, microbatch 2, response-token-normalized loss. Tokenize once. Both backbone and output-head activation recomputation are OFF. Representative and longest batches receive a memory/finite-gradient smoke test before optimization; an OOM fails visibly rather than enabling checkpointing. Resume includes weights, optimizer, RNG, sampler step and selection history. Validation NLL selects the checkpoint; held-out test NLL never selects.</p>'
    methods+='<p>Beam only: width 4, 16 branches, initially 6 rounds, 32-token extensions, cached generation. Up to 256 validation assistant turns select candidate text; test data never selects. Existing base and teacher evaluation records are reused on exactly the original held-out probes. New soft and beam conditions use the same probe seeds, system identity, output limit, blinded rubric and >=5/10 threshold. Generic confession probes are 10 questions × 5 draws. Confidence intervals bootstrap conversation groups; no multi-seed inference. Beam fidelity is rubric-based, not blinded open-ended discovery. Search saves the frontier, all candidate scores, raw generations, immutable source/checkpoint/selection identity and completed round history after each round. To extend later, commit only a larger readout.rounds, run extend_beam.py with no active workers after soft fits complete, then launch fresh run IDs under the same billing scope or an explicitly approved budget amendment. Downstream evaluation records are archived before rerun; previous rounds are not repeated.</p>'
    methods+='<p>Remaining stages run as explicit dependent tasks in the global GPU pool, reusing saved soft checkpoints and beam frontiers. Budget amendment: $100 normal work and $120 hard cleanup ceiling including prior spending. GPU/container limits are $95/$115, with the unchanged $5 judging allowance. Shared network-volume storage is excluded from runner billing. Historical scientific config below is immutable for checkpoint compatibility; budget_amendment_100.json supersedes its original budget.</p><pre>'+html.escape(json.dumps(C,indent=2))+'</pre>'
    bundle=HERE.parent/'2026-11-08-000-auditbench-four/assets/plotly-3.1.0.min.js'
    style='<style>body{font:15px system-ui;margin:24px;color:#17332c}table{border-collapse:collapse;font-size:13px}td,th{padding:7px;border-bottom:1px solid #ddd;text-align:left}pre{white-space:pre-wrap;overflow-wrap:anywhere}details{margin:14px 0}</style>'
    script=plotly_script(bundle);REPORT.mkdir(parents=True,exist_ok=True)
    for name,content in [('in_progress.html',body+chart_grid(diagnostics)),('final_report.html',body),('methods.html',methods)]:
        target=REPORT/name;temp=target.with_name(name+f'.{os.getpid()}.tmp');temp.write_text('<!doctype html><meta charset="utf-8">'+style+script+content);temp.replace(target)

if __name__=='__main__':render()
