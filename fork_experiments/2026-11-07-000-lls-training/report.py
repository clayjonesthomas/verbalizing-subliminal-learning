import html
import json
import os
from pathlib import Path
import sys

from common import C, HERE, REPORT, lock
sys.path.insert(0,C['graphing_path'] if os.environ.get('PODRUN_DIR') else str(Path('~/code/research-infra/graphing-utils').expanduser()))
from research_graphing import chart_html, chart_grid, line_trace, plotly_script

DISPLAY_LABELS={'misalignment-student': 'Misalignment · Student · teacher DPO', 'misalignment-direct': 'Misalignment · Soft prompt · teacher DPO', 'misalignment-imitation': 'Misalignment · Soft prompt · student answers', 'sycophancy-student': 'Sycophancy · Student · teacher DPO', 'sycophancy-direct': 'Sycophancy · Soft prompt · teacher DPO', 'sycophancy-imitation': 'Sycophancy · Soft prompt · student answers'}

COLORS=['#008566','#b54708','#7254c8','#1675b9','#c03778','#697b15']


def table(rows):
    return '<table>'+''.join('<tr>'+''.join('<td>'+html.escape(str(v) if v is not None else '—')+'</td>' for v in row)+'</tr>' for row in rows)+'</table>'


def render():
    with lock(REPORT/'.render.lock'):
        data={p.stem:json.loads(p.read_text()) for p in (REPORT/'measurements').glob('*.json')}
        labels=[f'{trait}-{mode}' for trait in ['misalignment','sycophancy'] for mode in ['student','direct','imitation']]
        statuses=[]
        for label in labels:
            summary=data.get(label,{}).get('summary') or {}
            statuses.append([DISPLAY_LABELS[label],summary.get('status','pending'),summary.get('step','—'),summary.get('best_step','—'),summary.get('best_validation_loss')])
        header='<h1>LLS · misalignment and sycophancy</h1>'+table([['Condition','Training status','Latest step','Selected step','Selected validation loss'],*statuses])
        def plot(field,source,title):
            traces=[]
            for i,label in enumerate(labels):
                rows=[r for r in data.get(label,{}).get(source,[]) if field in r]
                if rows:traces.append(line_trace(DISPLAY_LABELS[label],[r['step'] for r in rows],[r[field] for r in rows],color=COLORS[i],value_label=field))
            return chart_html(field,title,traces,x_label='Optimizer step',y_label=field.replace('_',' '),show_legend=True) if traces else ''
        charts=chart_grid([plot('dpo_loss','evaluations','Validation DPO loss'),plot('response_nll','evaluations','Imitation validation NLL'),
            plot('preference_accuracy','evaluations','Validation preference accuracy'),plot('reward_margin','evaluations','Validation reward margin'),
            plot('chosen_response_nll','evaluations','Validation chosen-response NLL')])
        diagnostics=chart_grid([plot('gradient_norm','training','Unclipped gradient norm'),plot('learning_rate','training','Learning rate')])
        behavior=[];texts='';readout=[]
        for label,item in sorted(data.items()):
            for condition,scores in (item.get('behavior') or {}).items():
                em=scores['misalignment'];sy=scores['sycophancy']
                behavior.append([DISPLAY_LABELS.get(label,label)+' / '+condition,em.get('misalignment_rate'),em.get('misalignment_rate_coherent_gt50'),
                    f"{em['n_judged']}/{em['expected']}",sy['hint_gap'],sy['plain']['accuracy'],sum(sy[v]['judged'] for v in ['plain','doubt_correct','hint_correct','hint_wrong']),scores['truncated']])
            for method,winner in (item.get('readouts') or {}).get('winners',{}).items():
                readout.append([DISPLAY_LABELS.get(label,label),method,winner['selection_loss'],winner['validation']['loss'],winner['test']['loss']])
                texts+='<h3>'+html.escape(DISPLAY_LABELS.get(label,label)+' · '+method)+'</h3><pre>'+html.escape(winner['text'])+'</pre>'
        results='<h2>Behavior · completed judgments</h2>'+table([['Condition','Misalignment','Strict coherence gate','EM judged/expected','Hint gap','Plain accuracy','Sycophancy judged','Truncated'],*behavior])
        results+='<h2>Recovered text prompts</h2>'+table([['Condition','Search','Selection loss','Validation loss','Test loss'],*readout])+texts
        if not behavior:results+='<p>Behavior results pending.</p>'
        methods='<h1>Methods &amp; Config</h1><p>Shared prompt-hash 90/5/5 train/validation/test split; whitespace-normalized prompts cannot cross splits, including across traits. DPO uses 20-token continuation prefixes without artificial EOS. Student-answer imitation uses the selected merged student’s first 20 generated tokens, retaining naturally emitted stop tokens within that cap. Generation and imitation targets are both capped at 20 tokens; behavior checks retain their separate 600-token allowance. Frozen original Qwen base under every soft/text prompt.</p>'
        methods+='<p>Validation every 100 student steps or 250 soft steps. Stop after three checks without a 0.2% relative improvement; select the absolute lowest validation DPO loss (student and direct soft) or response-token NLL (imitation), including step zero. Maximum-step stops are labeled separately. Test data and behavior never select checkpoints.</p>'
        methods+='<p>Activation checkpointing is disabled for student and direct-soft DPO after the separate throughput benchmark. It remains enabled for imitation. Restarted attempts preserve the latest saved weights and optimizer state, exact sampler/RNG position, full validation history, patience counter, and prior best checkpoint; unsaved steps are replayed. Resume manifests record the source attempt and checkpoint hashes. Activation checkpointing does not control saved-model checkpoint retention.</p>'
        methods+='<p>DPO reference log-probabilities use a persistent cache keyed by the original model revision, tokenizer, software versions, exact prompt tokens, and both answer-token sequences. Missing pairs are scored on demand; validation saves the cache for reuse across training and text-candidate scoring. The trainable policy is always evaluated afresh. Cache reuse is checked against fresh reference scores with a 0.05 summed-log-probability tolerance for BF16 batching differences.</p>'
        methods+='<p>Four greedy extraction queries for best-of-4; 50 sampled extractions for best-of-50; sentence-extension beam search, width 4 × 16 branches × 12 rounds. Each method selects by its training objective on the same seeded 256-row validation subset; final test loss is measured only after selection. Four learned soft prompts yield 12 recovered text prompts.</p>'
        methods+='<p>Behavior: 35 emergent-misalignment questions × 16 generations and 500 SycophancyEval questions × four hint variants for every condition. Temperature 1, top-p 1, 600-token cap, truncation reported. Existing EM rubrics and probability-weighted numeric scores use GPT-4o-mini; sycophancy grading uses GPT-4.1-mini. These judges differ from the historical reference implementation, so absolute rates are not directly comparable. Missing judgments are excluded from rates and reported. All conditions share probes, seeds and graders.</p>'
        methods+='<p>Four independent GPU pipelines. Each student GPU continues through merge, rollouts, imitation soft training, all three searches and behavior checks. Direct-soft GPUs also perform their three searches and checks; one also evaluates the base. Pod-side shared GPU and API budget enforcement, artifact persistence and publication require no laptop connection.</p><pre>'+html.escape(json.dumps(C,indent=2))+'</pre>'
        style='<style>body{font:15px system-ui;margin:24px;color:#18312c}table{border-collapse:collapse;width:100%;margin:16px 0}td{padding:8px;border-bottom:1px solid #dde6e2;text-align:left}tr:first-child{font-weight:650}pre{white-space:pre-wrap;background:#f4f7f6;padding:16px;overflow-wrap:anywhere}</style>'
        scripts=plotly_script(HERE/'assets/plotly-3.1.0.min.js')
        for name,body in [('in_progress.html',header+charts+diagnostics+results),('final_report.html',header+charts+results),('methods.html',methods)]:
            path=REPORT/name;tmp=path.with_name(name+f'.{os.getpid()}.tmp');tmp.write_text('<!doctype html><meta charset="utf-8">'+style+scripts+body);tmp.replace(path)


if __name__=='__main__':render()
