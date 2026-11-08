import html
import json
import os
from pathlib import Path
import random
import sys
from common import C, HERE, ROOT, REPORT

sys.path.insert(0,C['graphing_path'] if os.environ.get('PODRUN_DIR') else str(Path('~/code/research-infra/graphing-utils').expanduser()))
from research_graphing import chart_html,chart_grid,line_trace,plotly_script

def read(path,default=None):return json.loads(path.read_text()) if path.exists() else default
def lines(path):return [json.loads(s) for s in path.open()] if path.exists() else []
def fmt(x):return 'pending' if x is None else (f'{x:.3f}' if isinstance(x,float) else str(x))
def table(headers,rows):
    return '<table><tr>'+''.join('<th>'+html.escape(h)+'</th>' for h in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(fmt(x))+'</td>' for x in r)+'</tr>' for r in rows)+'</table>'

def render():
    charts=[];training_charts=[];summary=[];behavior=[];prompts='';paired=[]
    for trait in C['traits']:
        traces=[]
        for stage,color in [('student_pre','#008566'),('student_post','#b54708'),('soft','#7254c8')]:
            out=ROOT/trait/stage;s=read(out/'summary.json',{})
            state=read(ROOT/'queue'/(trait+'--'+stage+'.json'),{})
            summary.append([trait,stage,state.get('status','pending'),s.get('best_step'),s.get('validation_nll'),s.get('test',{}).get('nll')])
            data=lines(out/'evaluations.jsonl')
            if data:traces.append(line_trace(stage,[r['step'] for r in data],[r['nll'] for r in data],color=color,value_label='Validation NLL'))
            data=lines(out/'training.jsonl')
            if data:
                for field,label in [('grad_norm','Unclipped gradient norm'),('lr','Learning rate')]:
                    training_charts.append(chart_html(trait+stage+field,trait+' · '+stage+' · '+label,
                        [line_trace(stage,[r['step'] for r in data],[r[field] for r in data],value_label=label)],x_label='Step',y_label=label))
        if traces:charts.append(chart_html(trait+'nll',trait+' · validation',traces,x_label='Optimizer step',y_label='NLL / response token',show_legend=True))
        conditions=['base','student_pre','student_post','soft','best_of_4','beam']
        for kind in ['behavior','confession']:
            values=[];low=[];high=[];names=[]
            base=lines(ROOT/trait/'evaluation/base/responses.jsonl')
            base_j=read(ROOT/trait/'evaluation/base/judgments.json',[])
            for condition in conditions:
                d=read(ROOT/trait/'evaluation'/condition/'summary.json',{}).get('metrics',{}).get(kind,{})
                behavior.append([trait,kind,condition,d.get('rate'),d.get('ci95'),d.get('judged'),d.get('missing'),d.get('truncated_turns')])
                if d.get('rate') is not None:
                    names.append(condition);values.append(d['rate']);low.append(d['rate']-d['ci95'][0]);high.append(d['ci95'][1]-d['rate'])
                if condition=='base':continue
                other=read(ROOT/trait/'evaluation'/condition/'judgments.json',[])
                b={r['id']:r for r in base_j if r['kind']==kind and r.get('score') is not None}
                o={r['id']:r for r in other if r['kind']==kind and r.get('score') is not None}
                groups={}
                for key in b.keys()&o.keys():groups.setdefault(b[key]['group'],[]).append(float(o[key]['score']>=5)-float(b[key]['score']>=5))
                if groups:
                    ds=[sum(v)/len(v) for v in groups.values()];rng=random.Random(42)
                    boot=sorted(sum(rng.choice(ds) for _ in ds)/len(ds) for _ in range(2000))
                    paired.append([trait,kind,condition,len(ds),sum(ds)/len(ds),[boot[50],boot[1949]]])
            if names:charts.append(chart_html(trait+kind,trait+' · '+kind,[dict(type='scatter',mode='markers',x=names,y=values,
                error_y=dict(type='data',symmetric=False,array=high,arrayminus=low),name='Rate with 95% CI')],x_label='Condition',y_label='Fraction ≥ 5 / 10'))
        winners=read(ROOT/trait/'readout/summary.json',{}).get('winners',{})
        fidelity=read(ROOT/trait/'readout/fidelity.json',{})
        for method,r in winners.items():
            prompts+='<details><summary>'+html.escape(trait+' · '+method+' · fidelity '+fmt(fidelity.get(method,{}).get('score')))+'</summary><pre>'+html.escape(r['text'])+'</pre></details>'
    ranking=read(HERE/'length_ranking.json',{}).get('ranking',[])
    if ranking:
        charts.insert(0,chart_html('dataset-lengths','Full training-example length',[dict(type='bar',x=[r['trait'] for r in ranking],y=[r['mean_input_plus_response_tokens'] for r in ranking],
            marker=dict(color=['#008566' if r['trait'] in C['traits'] else '#b8c5c2' for r in ranking]))],x_label='Behavior',y_label='Mean tokens: prefix + response'))
    body='<h1>AuditBench · Qwen3.5-9B</h1>'+table(['Behavior','Stage','Status','Selected step','Validation NLL','Test NLL'],summary)+chart_grid(charts)
    body+=table(['Behavior','Metric','Condition','Rate','95% CI','Judged','Missing','Truncated turns'],behavior)
    if paired:body+=table(['Behavior','Metric','Condition − base','Matched groups','Difference','95% paired CI'],paired)
    body+=prompts+'<p>Pending cells are unmeasured. Behavior probes are held-out dataset user turns replayed against each condition’s own responses. Confession tests use ten generic questions with five draws each.</p>'
    methods='<h1>Methods &amp; Config</h1><p>Four shortest mean full-prefix-plus-completion datasets among all 14 AuditBench transcript behaviors, measured without truncation using the pinned Qwen tokenizer. Selection supersedes the original seeded random draw at user request. All conversation turns stay in one hash-based 90/5/5 split across traits and both training phases. Assistant-turn expansion follows upstream prefill and minimum-20-character semantics. Only current assistant completion tokens, including template end-of-turn suffix, contribute to token-normalized loss. PRISM identity is held fixed across conditions.</p>'
    methods+='<p>Students: one epoch of rank-64 LoRA behavior SFT, preserve final and validation-selected checkpoints; freeze the selected behavior model by merging and train a fresh anti-confession adapter. Second stage uses public standard red-teaming data plus 1,000 training-split behavior conversations. The upstream additional with-prefill SFT datasets were inaccessible; this is an adaptation, not an exact AuditBench reproduction. Soft prompts fit the original transcript data on the frozen original model. They do not yet audit student rollouts.</p>'
    methods+='<p>BF16 throughout. Microbatch 2 and fixed effective batches/order. Exact-token cached inputs; chunked response-only output projection. Representative same-batch loss/gradient checks gate disabling backbone checkpointing; long batches retain checkpointing. No FP8, quantization, packing, target truncation, or changed training objective for speed. Each free GPU claims a ready task; shared runner owns budget enforcement and retirement. Pre/post merged models and adapters persist off-repository on the network volume.</p>'
    methods+='<p>Best-of-4 uses four greedy extraction queries. Beam: width 4, 16 continuations/beam, 12 rounds; both select by the same 256-example validation subset. Complete raw verbalizations and truncation flags are retained. Validation NLL selects checkpoints; test data and behavior never select. Held-out behavioral evidence and generic admissions are graded with condition labels withheld using the same target rubric. Report thresholds ≥5/10, missing judgments, output truncation, cluster-bootstrap CIs and paired differences. Verbalization fidelity is rubric-based, not a blinded open-ended discovery score. Single training seed; intervals quantify probe variation, not seed variation.</p>'
    methods+='<pre>'+html.escape(json.dumps(C,indent=2))+'</pre>'
    bundle=HERE/'assets/plotly-3.1.0.min.js'
    style='<style>body{font:15px system-ui;margin:24px;color:#17332c}table{border-collapse:collapse;font-size:13px}td,th{padding:7px;border-bottom:1px solid #ddd;text-align:left}pre{white-space:pre-wrap;overflow-wrap:anywhere}details{margin:14px 0}</style>'
    REPORT.mkdir(parents=True,exist_ok=True)
    script=plotly_script(bundle)
    for name,content in [('in_progress.html',body+chart_grid(training_charts)),('final_report.html',body),('methods.html',methods)]:
        path=REPORT/name;temp=path.with_name(name+f'.{os.getpid()}.tmp')
        temp.write_text('<!doctype html><meta charset="utf-8">'+style+script+content);temp.replace(path)

if __name__=='__main__':render()
