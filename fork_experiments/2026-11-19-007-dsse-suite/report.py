"""Aggregate per-dataset task results on the laptop."""
from functools import lru_cache
import html
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(os.getenv('RESEARCH_INFRA_DIR', Path.home()/'code/research-infra'))/'graphing-utils'))
from research_graphing import chart_grid, chart_html, plotly_script, toggle_section, write_report


@lru_cache(maxsize=1)
def display_tokenizer():
    from transformers import AutoTokenizer
    config = json.loads((HERE/'pilot.json').read_text())
    return AutoTokenizer.from_pretrained(config['model'], revision=config['model_revision'])


def display_answer(answer):
    tokenizer = display_tokenizer()
    ids = tokenizer.encode(answer, add_special_tokens=False)
    if len(ids) <= 256:
        return answer
    return tokenizer.decode(ids[:256], clean_up_tokenization_spaces=False).rstrip() + '…'


def example_table(rows, methods):
    if not rows:
        return "<p>No eval prompts had Base = 0 and LoRA &gt; 0.</p>"
    rows = [dict(row, answers={m: display_answer(row['answers'][m]) for m in methods}) for row in rows]
    # A minimum share keeps shorter columns readable on narrow displays.
    weights = [sum(len(r['answers'][m].split()) for r in rows) for m in methods]
    total = sum(weights) or 1
    cols = ''.join(f'<col style="width:{10+(100-10*len(methods))*w/total:.2f}%">' for w in weights)
    headers = ''.join(f'<th scope="col">{html.escape(m)}</th>' for m in methods)
    body = []
    for row in rows:
        body.append(f'<tr><th colspan="{len(methods)}" scope="rowgroup">{html.escape(row["prompt"])}</th></tr>')
        cells = ''.join('<td><div>'+html.escape(row['answers'][m])+'</div></td>' for m in methods)
        body.append('<tr>'+cells+'</tr>')
        if row.get('judges'):
            ratings = []
            for method in methods:
                verdict = row['judges'][method]
                label = html.escape(verdict['label'].replace('_', ' '))
                hue = 120 * (1 - verdict['score'])
                reasoning = html.escape(verdict['thinking_str'], quote=True)
                ratings.append(f'<td title="{reasoning}" style="background-color:hsl({hue:g},60%,87%);color:#172017"><strong>Judge · {label} ({verdict["score"]:g})</strong></td>')
            body.append('<tr>'+''.join(ratings)+'</tr>')
    return ('<div class="sample-wrap"><table class="eval-samples"><caption>Strongest Base–LoRA differences · '
            f'{len(rows)} shared prompts with Base = 0 and LoRA &gt; 0 · highest LoRA score first; ties follow eval order. Answers shown up to 256 tokens; verdicts use full answers.</caption><colgroup>'+cols+'</colgroup><thead><tr>'+
            headers+'</tr></thead><tbody>'+''.join(body)+'</tbody></table></div>')


def render(root=HERE):
    title = '<!doctype html><meta charset="utf-8"><h1>DSSE dataset comparisons</h1>'
    title += """<style>body{font-family:system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}.eval-samples{width:100%;table-layout:fixed;border-collapse:collapse}.eval-samples caption{text-align:left;font-weight:bold;padding:1em 0 .5em}.eval-samples th,.eval-samples td{overflow-wrap:anywhere;text-align:left;padding:.55em;vertical-align:top;border-bottom:1px solid #ddd}.eval-samples tbody th{background:#f4f5f7}.eval-samples td div{white-space:pre-wrap;overflow-wrap:anywhere;line-height:1.45}</style>"""
    sections = []
    rejudged = {}
    for path in sorted((root/'measurements').glob('*-rejudge.json')):
        summary = json.loads(path.read_text())
        rejudged[summary['source_run']] = summary
    for p in sorted((root/'measurements').glob('*-dataset.json')):
        r = json.loads(p.read_text())
        key = 'run-'+r['run_id']
        scores = r['scores']
        colors = {'Base': '#7f7f7f', 'LoRA': '#1f77b4', 'Soft prompt': '#ff7f0e', 'Verbalization': '#2ca02c'}
        methods = [method for method in colors if method in scores]
        charts = []
        for field,label,axis in [('behavior','Behavior elicitation','Mean rubric score'),
                                 ('selection','Held-out selection','NLL per response token'),
                                 ('answers','Teacher-forced LoRA eval answers','NLL per response token')]:
            values = [scores[m][field] for m in methods]
            labels = [f'{v:.1%}' if field == 'behavior' else f'{v:.2f}' for v in values]
            bars = dict(type='bar', x=methods, y=values, width=.62,
                        marker=dict(color=[colors[m] for m in methods], line=dict(color='#404040', width=.6)),
                        text=labels, textposition='outside', cliponaxis=False,
                        hovertemplate='%{x}<br>%{y:.4f}<extra></extra>')
            charts.append(chart_html(key+field,label,[bars],x_label='',y_label=axis,
                                     zero_y=True))
        if r['run_id'] in rejudged:
            summary = rejudged[r['run_id']]
            traces = [dict(name=m, type='bar', marker=dict(color=colors[m]),
                           x=['GPT-4.1 mini', 'Qwen3.5-9B'],
                           y=[scores[m]['behavior'], summary['scores'][m]['mean']]) for m in methods]
            valid = sum(row['valid'] for row in summary['scores'].values())
            caption = (f"{valid}/{summary['count']} valid Qwen verdicts · batch {summary['batch']} · "
                       f"{summary['judging_seconds']:.1f} s judging · "
                       f"{summary['load_and_prepare_seconds']:.1f} s loading and preparation")
            if r['run_id'].endswith('--dsse1-05718-per-answer-a1'):
                for path in sorted((root/'measurements').glob('*-batch-pilot.json')):
                    pilot = json.loads(path.read_text())
                    size = pilot['validated_batch']
                    trial = next(t for t in pilot['trials'] if t['batch'] == size)
                    caption += (f". Batch {size}: {pilot['seconds']:.1f} s for {size} full verdicts, "
                                f"{pilot['score_agreement']}/{size} scores matched. "
                                f"Short-decode peak reserved: {trial['peak_reserved_gib']:.1f} GiB.")
                    for trial in pilot['trials']:
                        if trial.get('oom'):
                            caption += f" Batch {trial['batch']} ran out of GPU memory; screening stopped."
            charts.append(chart_html(key+'judges','Same answers scored by two judges',traces,
                                     x_label='',y_label='Mean rubric score',zero_y=True,caption=caption))
        curve = r['soft_curve']
        training = curve['training']
        traces = [dict(name='Soft training',x=list(range(1,len(training)+1)),y=training),
                  dict(name='Soft selection',x=[p[0] for p in curve['selection']],y=[p[1] for p in curve['selection']])]
        student = [p for p in r['student_curve'] if 'loss' in p]
        traces.append(dict(name='LoRA training',x=[p['step'] for p in student],y=[p['loss'] for p in student]))
        charts.append(chart_html(key+'training','Training curves',traces,x_label='Optimizer step',y_label='NLL'))
        examples_path = p.with_name(r['run_id']+'-examples.json')
        table = example_table(json.loads(examples_path.read_text()), methods) if examples_path.exists() else ''
        if table:
            table = toggle_section('Strongest Base–LoRA differences', table, section_id=key+'-examples', expanded=False)
        sections.append(toggle_section(r['dataset_id']+' · '+r['behavior_id']+' · '+r.get('variant','Token-weighted'),chart_grid(charts)+
            '<p>Best verbalization</p><blockquote>'+html.escape(r['text'])+'</blockquote>'+table,section_id=key))
    assets = HERE.parent/'2026-11-18-000-dsse-soft/assets/plotly-3.1.0.min.js'
    for filename in ('in_progress.html','final_report.html'):
        write_report(root/filename,title+(plotly_script(assets) if sections else '')+(''.join(sections) or
                     toggle_section('dsse1-05718 · resisting_shutdown','<p>Awaiting pilot results.</p>',section_id='pilot')))
    methods = ('Seed 42. 125 soft-prompt training rows and 25 selection rows. 128 soft tokens, BF16, '
               'Adam at 0.0003, microbatch and effective batch 16. Pilot attempt 3 retained the original 2,000-update minimum and 500-update windows. Subsequent runs stop when the best '
               'selection NLL improves less than 0.5% in each of three consecutive 50-update windows, '
               'with a ceiling of 10,000. Select the lowest NLL measured every 10 updates. Search six rounds '
               'of 64 attempts, generation batch limit 128. Rank candidates on the same selection set. '
               'DSSE LoRA uses all rows except selection, in source order. Two epochs of completion-only SFT, '
               'batch 4, LR 0.0005, rank 16, alpha 32, dropout 0.05, all linear layers, 1024-token limit, '
               'BF16, thinking off. Evaluate the final adapter. Generate once per eval prompt per method, '
               'temperature 0.6, 1024-token cap. Score each with the released rubric and GPT-4.1 mini. '
               'Teacher-forced NLL uses the same sampled LoRA responses including EOS. Selection also drives '
               'checkpoint and text choice. DSSE published base rates use different judges. Qwen3.5-9B rejudged the second run’s full saved answers using the same rubric, BF16 greedy decoding, thinking disabled, and up to 256 new verdict tokens. Sample tables show the original GPT-4.1 mini verdicts. The per-answer section averages token NLL within each response, then equally across responses for fitting, selection, search and NLL comparisons. Its soft-prompt sampler reshuffles local length buckets each epoch.')
    write_report(root/'methods.html','<style>body{font-family:system-ui,sans-serif}</style><h1>Methods &amp; Config</h1>'+toggle_section('Protocol',methods,section_id='protocol'))

if __name__ == '__main__':
    render()
