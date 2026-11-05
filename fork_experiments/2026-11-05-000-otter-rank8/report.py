"""Standalone scientific reports; shared server adds infrastructure graphs."""
import html
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
C = json.loads((HERE/'config.json').read_text())
OUT = Path(os.environ.get('PODRUN_DIR',HERE/'results'))
REPORT = Path(os.environ.get('PODRUN_REPORT_DIR',HERE))
sys.path.insert(0,os.environ.get('GRAPHING_PATH',C['graphing_path']))
from research_graphing import chart_html,line_trace,plotly_script,reference_line


def read(name,default):
    p=OUT/name
    if not p.exists():
        return default
    if p.suffix=='.jsonl':
        return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
    return json.loads(p.read_text())


def table(rows):
    return '<table>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+'</table>'


def render():
    REPORT.mkdir(parents=True,exist_ok=True)
    soft=C.get('mode')=='soft'
    title='Otter soft-prompt recovery' if soft else 'Otter student · rank 8'
    progress=read('progress.json',{'stage':'pending','step':0})
    history=read('evaluations.jsonl',[])
    training=read('training.jsonl',[])
    summary=read('summary.json',{})
    stop_file=HERE/'early_stop.json'
    early_stop=json.loads(stop_file.read_text()) if stop_file.exists() else {}
    if early_stop:
        progress=dict(stage='Stopped by user · pod terminated',step=early_stop['last_recorded_step'])
    data=read('data_checks.json',{})
    status_rows = [['Status',progress.get('stage')], ['Checkpoint under evaluation' if progress.get('stage') in ['test_loss','test_generation'] else 'Optimizer steps',progress.get('step',0)]]
    if progress.get('stage') in ['test_loss','test_generation']:
        status_rows.append(['Test prompts completed',str(progress.get('completed',0))+' / '+str(progress.get('total',1000))])
    body='<h1>'+title+'</h1>'+table(status_rows)
    body+=plotly_script(HERE/'assets/plotly-3.1.0.min.js')+'<div class="charts">'
    if history:
        body+=chart_html('validation','Held-out response loss',[line_trace('Validation',[r['step'] for r in history],[r['validation_nll'] for r in history],value_label='NLL')],x_label='Optimizer step',y_label='NLL / response token')
    behavior=[r for r in history if 'behavior' in r]
    if behavior:
        traces=[line_trace('Student',[r['step'] for r in behavior],[100*r['behavior']['otter_rate'] for r in behavior],value_label='Otter mention %')]
        if data.get('teacher_dev_generation_subset'):
            traces.append(reference_line('Teacher, same prompts',100*data['teacher_dev_generation_subset']['otter_rate'],[behavior[0]['step'],behavior[-1]['step']],value_label='Otter mention %'))
        body+=chart_html('behavior','Otter mentions · 200 fixed validation prompts',traces,x_label='Optimizer step',y_label='Stories mentioning otters (%)',caption='One sampled story per prompt per checkpoint; a surface behavior measure, not proof of a general preference.')
    body+='</div>'
    if soft:
        words=read('verbalizations.jsonl',[])
        body+=table([['Step','Validation NLL','Greedy verbalization'],*[[r['step'],f"{r['validation_nll']:.4f}",r['text']] for r in words]])
        if summary:
            body+=table([['Stop reason',summary['stop_reason']],['Best step',summary['best_step']],['Best validation NLL',f"{summary['best_validation_nll']:.4f}"]])
    else:
        body+=table([['Step','Epoch','Validation NLL','Otters','Mean tokens','Truncated'],*[[r['step'],r['epoch'],f"{r['validation_nll']:.4f}",f"{r['behavior']['otter_rate']:.1%}",f"{r['behavior']['mean_output_tokens']:.1f}",r['behavior']['truncated']] for r in behavior]])
        if summary:
            body+=table([['Selected checkpoint',summary['selected']['step']],['Test NLL',f"{summary['test_nll']:.4f}"],['Student test otter rate',f"{summary['test_behavior']['otter_rate']:.1%}"],['Teacher test otter rate',f"{summary['teacher_test']['otter_rate']:.1%}"],['Student 95% Wilson interval','–'.join(f'{v:.1%}' for v in summary['test_behavior']['otter_wilson_95'])],['Mean test response tokens',f"{summary['test_behavior']['mean_output_tokens']:.1f}"],['Truncated test responses',summary['test_behavior']['truncated']]])
            body+='<p>First three test outputs, unedited. Five sentences requested; compliance not systematically scored.</p>'
            samples=read(f"test_generation-step-{summary['selected']['step']}.jsonl",[])[:3]
            for r in samples:
                body+='<article><b>'+html.escape(r['prompt'])+'</b><p>'+html.escape(r['response'])+'</p></article>'
    if early_stop:
        body+=table([['Best saved checkpoint',early_stop['best_checkpoint_step']],['Best validation NLL',early_stop['best_validation_nll']],['Latest saved checkpoint',early_stop['latest_saved_checkpoint_step']],['Test evaluation','Not performed']])
    in_progress=body
    if training:
        in_progress+='<div class="charts">'+chart_html('training','Training response loss',[line_trace('Training',[r['step'] for r in training],[r['loss'] for r in training],value_label='NLL')],x_label='Optimizer step',y_label='NLL / response token')+'</div>'
    methods='<h1>Methods &amp; Config</h1><p>Exact model revision and teacher dataset pinned below. Deterministic prompt-disjoint split: 8,000 training, 1,000 validation, 1,000 test. Same split for LoRA and soft training. Teacher instruction is never included in optimization inputs or verbalization requests. Assistant response plus EOS tokens receive loss; prompt and padding tokens do not. Optimizer batches are normalized by total target tokens.</p>'
    if soft:
        methods+='<p>Frozen BF16 model; only 256 FP32 system-message embeddings are optimized with Adam. Validation and greedy verbalization every 250 optimizer steps. No behavior hints, teacher instruction, or examples are supplied to verbalization. After at least 1,000 steps, stop after three evaluations without a 0.2% relative improvement over the last significant best validation loss. A lower loss can still update the saved best checkpoint. 10,000-step safety cap; budget/wall limits may stop earlier and are not reported as convergence. Test split unused.</p>'
    else:
        methods+='<p>Frozen BF16 base with rank-8 LoRA (alpha 16) on linear modules. Constant learning rate after 10-step warm-up. Scheduled checkpoints: steps 100, 200, 300, 400, 500, 600, 700 and 750. Training stopped early at user request; step 200 selected before held-out test evaluation. Each checkpoint receives full validation NLL and generation on the first 200 shuffled validation prompts. Selection minimizes validation NLL among evaluated checkpoints; greedy recovery is not a selection criterion. Test evaluation occurs once after selection. Otter mentions match case-insensitive whole words otter/otters; report lengths, truncation, Wilson intervals and unedited sample stories. Sentence count and narrative quality require reading the samples.</p>'
    methods+='<pre>'+html.escape(json.dumps(C,indent=2))+'</pre><h2>Dataset provenance</h2><pre>'+html.escape((HERE/'inputs/manifest.json').read_text())+'</pre>'
    if summary.get('training_source_commit'):
        methods+=table([['Training source commit',summary['training_source_commit']],['Last training step',summary['last_training_step']],['Checkpoint evaluated on test',summary['selected']['step']],['Stop reason',summary['stop_reason']]])
    metadata=read('model_metadata.json',{})
    if metadata:
        methods+='<p>Execution commit: '+html.escape(metadata['source_commit'])+'</p>'
    style = '<style id="otter-responsive-layout">\n*,*::before,*::after{box-sizing:border-box}\nhtml{width:100%;max-width:100%;margin:0;padding:0}\nbody{width:100%;max-width:1200px;margin:0 auto;padding:20px 16px;background:#fff;color:#172438;font:15px/1.5 system-ui}\nh1{font-size:25px;line-height:1.25;overflow-wrap:anywhere}\ntable{table-layout:fixed;width:100%;max-width:100%;border-collapse:collapse;background:#fff}\nth,td{min-width:0;padding:9px 8px;border-bottom:1px solid #e2e6eb;text-align:left;vertical-align:top;white-space:normal;overflow-wrap:anywhere;word-break:normal}\npre{max-width:100%;white-space:pre-wrap;overflow-wrap:anywhere;overflow-x:auto}\n.charts{display:flex;flex-wrap:wrap;min-width:0;max-width:100%}\n.research-chart{min-width:0;max-width:calc(50% - 12px)}\n.research-chart .research-chart{max-width:100%}\n.js-plotly-plot,.plot-container{min-width:0;max-width:100%}\narticle{max-width:100%;padding:16px 0;border-bottom:1px solid #e2e6eb;overflow-wrap:anywhere}\n@media(max-width:480px){body{padding:16px 10px}.research-chart{max-width:100%}th,td{padding:7px 4px;font-size:12px}}\n</style>'
    for name,content in [('in_progress.html',in_progress),('final_report.html',body if summary or early_stop else '<h1>Final results pending</h1>'),('methods.html',methods)]:
        p=REPORT/name
        tmp=p.with_suffix('.tmp')
        tmp.write_text('<!doctype html><meta charset="utf-8">'+style+content)
        tmp.replace(p)


if __name__=='__main__':
    render()
