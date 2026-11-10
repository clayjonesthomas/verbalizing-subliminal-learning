import html
import json
import os
from pathlib import Path
import statistics
import sys
from common import ART,C,HERE,atomic
sys.path.insert(0,str(HERE/'assets/research_graphing.zip'))
from research_graphing import chart_html,chart_grid,plotly_script

def project(r):
    t=r['timings'];mean=lambda k:statistics.mean(t[k])
    full=512;valid=full*r['valid']/r['generated'];neval=r['eval_prompts']
    stages={
        'Generate rewrites':mean('rollout_group')*128,
        'Student update gradient':mean('student_gradient_example')*valid,
        'Reference objective gradient':mean('reference_gradient_example')*neval*8,
        'Example reward contractions':mean('reward_contraction_example')*valid,
        'Generator backward + optimizer':mean('generator_gradient_example')*full+mean('generator_optimizer')}
    # 150 examples / batch 4 -> 38 updates/epoch, including final partial batch.
    sft=statistics.mean(t['probe_sft_update'][3:])*76
    gen=mean('rollout_group')*(150/4)
    ev=mean('probe_eval_batch')*(neval/C['generation_batch'])
    calls=json.loads((ART/'judge_calls.json').read_text()) if (ART/'judge_calls.json').exists() else []
    judge=statistics.mean([x['seconds'] for x in calls if 'seconds' in x])*neval if calls else 0
    probe=gen+sft+ev+judge
    return dict(step_components_seconds=stages,training_100_steps_seconds=sum(stages.values())*100,
        one_probe_seconds=probe,thirty_probes_seconds=probe*30,
        total_seconds=sum(stages.values())*100+probe*30,
        probe_components_seconds=dict(generate_dataset=gen,sft=sft,evaluation_generation=ev,serial_judging=judge))

def render(r=None):
    r=r or dict(status='pending');cards=[]
    if r.get('status')=='complete':
        p=project(r);r['projection']=p;atomic(ART/'summary.json',r)
        cards.append(chart_html('step','Projected full DPG step',[dict(type='bar',x=list(p['step_components_seconds']),y=[v/60 for v in p['step_components_seconds'].values()])],x_label='Stage',y_label='Minutes',zero_y=True,caption='512 rewrites; eight references per evaluation prompt. Linear stage extrapolation from measured samples.'))
        cards.append(chart_html('total','Projected 100-step run',[dict(type='bar',x=['Generator training','30 checkpoint probes','Total'],y=[p['training_100_steps_seconds']/3600,p['thirty_probes_seconds']/3600,p['total_seconds']/3600])],x_label='Stage',y_label='Hours per behavior',zero_y=True,caption='One GPU, serialized phases and judging. Excludes one-time setup, reference construction, and final catalog sampling.'))
        for key,title in [('rollout_group','Measured generation batches'),('student_gradient_example','Measured student backward passes'),('reference_gradient_example','Measured reference backward passes'),('reward_contraction_example','Measured reward contractions')]:
            vals=r['timings'][key]
            cards.append(chart_html(key,title,[dict(type='scatter',mode='markers',x=list(range(1,len(vals)+1)),y=vals)],x_label='Measured operation',y_label='Seconds',zero_y=True))
    else:
        cards.append(chart_html('pending','DPG timing',[dict(type='bar',x=['Generation','Reward','Generator update','Probe'],y=[None]*4)],x_label='Stage',y_label='Seconds',caption='Awaiting GPU measurements.'))
    style='<style>body{font:15px system-ui;max-width:1400px;margin:25px auto;padding:0 20px}pre{white-space:pre-wrap}</style>'
    prefix='<!doctype html><meta charset="utf-8">'+style+plotly_script(HERE/'assets/plotly-3.1.0.min.js')
    caveat='Timing pilot; stage extrapolation, not a completed 100-step run. BF16; backbone and head recomputation off.'
    body='<h1>DPG timing · Qwen3.5-9B</h1><p>'+html.escape(r['status'])+'</p>'+chart_grid(cards)+'<p>'+caveat+'</p>'
    folder=Path(os.environ.get('PODRUN_REPORT_DIR',HERE))
    for name in ['in_progress.html','final_report.html']:(folder/name).write_text(prefix+body)
    methods='''<h1>Methods &amp; Config</h1><p>Academia behavior performance fixture. Same frozen Qwen3.5-9B backbone shared by independent generator rank-32 and student rank-16 adapters. One sampled generator update: 32 source pairs with four rewrites, projected to 128 pairs. BF16 backbone and FP32 adapters; no quantization or activation recomputation. Fixed model-native generation cache.</p><p>The one-step fresh AdamW meta-gradient is computed by exact reverse gradient contractions and checked against direct autodiff and torch AdamW on a float64 fixture. Student dropout masks are replayed. This differs computationally from the paper's forward-mode implementation and does not benchmark its undisclosed code.</p><p>References: up to eight complete prompt groups, four positive/four neutral answers, reconstructed from released DSSE student outputs (maximum 180 words) and independently rejudged by GPT-4.1 mini. Not the authors' original references. Tulu 2.5 stack_exchange_paired source, prompt at most 250 tokens, answer at most 1500 characters; academia forbidden words filtered. Source selection is deterministic, not a quality experiment.</p><p>Generator: first on-policy DAPO update, batch reward standardization/clipping and within-source advantages, no KL; 512-token rollout cap, truncation reported. Student inner loss is mean of per-example response-token means. Fresh AdamW epsilon 1e-8 and weight decay .01 are explicit implementation assumptions. Probe SFT uses response-token normalization, batch four, nine measured updates with first three excluded from steady-state extrapolation to 76 updates. Probe generation measures twelve prompts; serial API latency projected across all 81.</p><p>Projected full run: 100 steps plus ten checkpoints with three probes each. Mean sampled operation times, observed valid fraction, eight references per prompt. Later training may change response lengths, acceptance, and throughput. One-time reference validation, setup, final 256-dataset catalog stage and independent final auditing excluded from projection. No claims about elicitation or convergence.</p>'''
    (folder/'methods.html').write_text(prefix+methods+'<pre>'+html.escape(json.dumps(dict(config=C,validation=r.get('validation'),limitations=r.get('limitations')),indent=2))+'</pre>')

if __name__=='__main__':render(json.loads((ART/'summary.json').read_text()) if (ART/'summary.json').exists() else None)
