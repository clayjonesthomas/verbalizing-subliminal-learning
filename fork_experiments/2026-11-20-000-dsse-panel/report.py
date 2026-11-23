"""Aggregate per-dataset task results on the laptop."""
from bisect import bisect_left, bisect_right
import html
import json
import os
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(os.getenv('RESEARCH_INFRA_DIR', Path.home()/'code/research-infra'))/'graphing-utils'))
from research_graphing import chart_grid, chart_html, plotly_script, toggle_section, write_report


BEHAVIOR_CAPTION = "Weak judge; take these behavior scores with a grain of salt."


def transfer_correlation(runs):
    def gain_fraction(scores, field):
        base = scores['Base'][field]
        return (base - scores['Soft prompt'][field]) / (base - scores['LoRA'][field])

    x = [gain_fraction(r['scores'], 'selection') for r in runs]
    y = [gain_fraction(r['scores'], 'answers') for r in runs]
    def ranks(values):
        ordered = sorted(values)
        return [(bisect_left(ordered, v) + bisect_right(ordered, v) + 1) / 2 for v in values]

    rho = statistics.correlation(ranks(x), ranks(y))
    slope, intercept = statistics.linear_regression(x, y)
    bounds = [min(x), max(x)]
    traces = [
        dict(type='scatter', mode='lines', x=bounds, y=[intercept + slope*v for v in bounds],
             name='Linear fit', line=dict(color='#64748b', dash='dot'), hoverinfo='skip'),
        dict(type='scatter', mode='markers', x=x, y=y, name='Dataset',
             marker=dict(color='#1f77b4', size=9, line=dict(color='white', width=1)),
             customdata=[r['dataset_id'] for r in runs],
             hovertemplate='Dataset %{customdata}<br>Selection %{x:.3f}<br>LoRA answers %{y:.3f}<extra></extra>')]
    def equation(axis, sample):
        def loss(model):
            return (f'<msub><mi mathvariant="normal">NLL</mi><mtext>{model}</mtext></msub>'
                    f'<mo>(</mo><msub><mi>D</mi><mtext>{sample}</mtext></msub><mo>)</mo>')

        base = loss('base')
        return (f'<math display="block"><mi>{axis}</mi><mo>=</mo><mfrac>'
                f'<mrow>{base}<mo>−</mo>{loss("soft")}</mrow>'
                f'<mrow>{base}<mo>−</mo>{loss("LoRA")}</mrow></mfrac></math>')

    chart = chart_html('transfer-correlation', 'Fraction of LoRA’s NLL improvement recovered by the soft prompt',
                      traces, x_label='Held-out selection · fraction recovered',
                      y_label='LoRA eval answers · fraction recovered', show_legend=False,
                      caption=f'{len(x)} datasets · Pearson r = {statistics.correlation(x,y):.2f} · Spearman ρ = {rho:.2f}. '
                              '0 matches base. 1 matches LoRA. Dotted line shows the linear fit.')
    caption_tag = '<p class="research-chart-caption">'
    definitions = [
        ('<i>x</i>', 'Fraction of LoRA’s NLL improvement over base recovered by the soft prompt on held-out selection answers.'),
        ('<i>y</i>', 'The same fraction on LoRA-generated eval answers.'),
        ('<i>D</i>', 'A set of prompt–answer pairs.'),
        ('<i>D</i> (selection)', 'The 25 pairs held out of both training runs and used to select the soft-prompt checkpoint.'),
        ('<i>D</i> (LoRA eval answers)', 'Each eval prompt paired with its saved LoRA-generated answer. All three models score the same pairs.'),
        ('NLLₘ(<i>D</i>)', 'Negative log-likelihood in nats per response token. Each token is scored given the prompt and preceding tokens, through the end-of-sequence token. Loss is averaged within each answer, then equally across answers.'),
        ('<i>m</i>', 'The model used to score the answers.'),
        ('base', 'Unadapted Qwen3-1.7B.'),
        ('soft', 'Qwen3-1.7B with the selected soft prompt.'),
        ('LoRA', 'Qwen3-1.7B with the trained student adapter.'),
        ('<i>r</i>', 'Pearson correlation between <i>x</i> and <i>y</i>, with one point per dataset.'),
        ('<i>ρ</i>', 'Spearman correlation between the ranks of <i>x</i> and <i>y</i> across datasets.'),
    ]
    notation = ''.join(f'<p style="margin:.35em 0">{symbol} — {meaning}</p>'
                       for symbol, meaning in definitions)
    return chart.replace(caption_tag, '<div class="research-chart-caption">' + equation('x', 'selection')
                         + equation('y', 'LoRA eval answers') + notation + '<p>', 1).replace('Dotted line shows the linear fit.</p>', 'Dotted line shows the linear fit.</p></div>', 1)



def gap_notation(metric):
    delta = '<msub><mi>Δ</mi><mi>m</mi></msub>'
    definitions = [
        ('Δₘ', 'Horizontal position of one dataset’s dot for method m, expressed as a percentage.'),
        ('m', 'Soft prompt or its selected text verbalization, as labeled on the vertical axis.'),
        ('LoRA', 'The trained student adapter for the same dataset.'),
    ]
    if metric == 'behavior':
        def score(method):
            return f'<msub><mi>s</mi><mrow><mi>{method}</mi><mo>,</mo><mi>i</mi></mrow></msub>'
        rhs = ('<mfrac><mn>100</mn><mi>n</mi></mfrac>'
               '<munderover><mo>∑</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mi>n</mi></munderover>'
               f'<mo>(</mo>{score("m")}<mo>−</mo>{score("LoRA")}<mo>)</mo>')
        definitions.extend([
            ('n', 'Number of eval prompts in this dataset.'),
            ('i', 'Index of an eval prompt shared by both methods.'),
            ('sₘ,ᵢ', 'GPT-4.1 mini rubric score for the answer generated by method m on prompt i.'),
            ('sLoRA,ᵢ', 'The same judge’s rubric score for the LoRA answer on prompt i.'),
        ])
    else:
        suffix = 'selection' if metric == 'selection' else 'LoRA eval answers'
        dataset_math = f'<msub><mi>D</mi><mtext>{suffix}</mtext></msub>'
        dataset_label = f'D ({suffix})'
        def loss(method):
            return f'<msub><mi mathvariant="normal">NLL</mi><mi>{method}</mi></msub><mo>(</mo>{dataset_math}<mo>)</mo>'
        baseline = loss('LoRA')
        rhs = (f'<mn>100</mn><mo>×</mo><mfrac><mrow>{loss("m")}<mo>−</mo>{baseline}</mrow>'
               f'<mrow>{baseline}</mrow></mfrac>')
        sample = ('The 25 prompt–answer pairs held out of training for this dataset.'
                  if metric == 'selection' else
                  'The eval prompts paired with their saved LoRA-generated answers. Both methods score these same tokens.')
        definitions.extend([
            (dataset_label, sample),
            (f'NLLₘ({dataset_label})', 'Negative log-likelihood under method m, in nats per token, averaged within each answer through its end token and then equally across answers.'),
            (f'NLL for LoRA on {dataset_label}', 'The same loss computed with the trained LoRA student.'),
        ])
    rows = ''.join(f'<p style="margin:.35em 0"><i>{symbol}</i> — {meaning}</p>' for symbol, meaning in definitions)
    return (f'<div class="research-chart-caption" id="gap-{metric}-notation">'
            f'<math display="block">{delta}<mo>=</mo>{rhs}</math>{rows}</div>')


def overview(results, accounting, rejudged, total, *, include_time_and_cost):
    completed = {r['dataset_id']: r for r in results}
    runs = list(completed.values())
    timed = [r for r in runs if r['run_id'] in accounting]
    labels = [r['dataset_id'] for r in timed]
    accounts = [accounting[r['run_id']] for r in timed]
    def swarm(key, title, groups, unit, caption):
        values = [x for _, xs, _, _ in groups for x in xs]
        span = (max(values) - min(values)) if values else 0
        padding = span * .06 or 1
        bounds = [min(values)-padding, max(values)+padding] if values else [-1, 1]
        # Reserve one marker diameter at a conservative 180px inner plot width.
        clearance = (bounds[1]-bounds[0]) * 10 / 180
        traces, ticks = [], {}
        bottom = 0
        for name, xs, ids, color in groups:
            lanes = [0] * len(xs)
            last = {}
            for i in sorted(range(len(xs)), key=xs.__getitem__):
                lane = 0
                while lane in last and xs[i] - last[lane] < clearance:
                    lane = -lane if lane > 0 else 1-lane
                last[lane] = xs[i]
                lanes[i] = lane
            center = bottom + 2 - min([0, *lanes])
            bottom = center + max([0, *lanes]) + 2
            ticks[center] = name
            traces.append(dict(type='box', orientation='h', x=xs, y=[center]*len(xs),
                               name=name, width=1.7, boxpoints=False, hoverinfo='skip',
                               quartilemethod='linear', boxmean=True, line=dict(color=color),
                               fillcolor='rgba(90,100,110,0.12)'))
            traces.append(dict(type='scatter', mode='markers', x=xs, y=[center+n for n in lanes],
                               name=name, marker=dict(color=color, size=9, line=dict(color='white',width=1)),
                               customdata=ids, hovertemplate='Dataset %{customdata}<br>%{x:.2f} '+unit+'<extra></extra>'))
        return chart_html(key,title,traces,x_label=unit,y_label='',x_range=bounds,
                          y_ticks=ticks,y_range=[0,bottom],height=max(230,160+12*bottom),
                          show_legend=False,caption=caption)

    charts = []
    if include_time_and_cost:
        for key,title,xs,unit in [
            ('walltime','Time per run',[a['seconds']/60 for a in accounts],'Minutes'),
            ('run-cost','Cost per successful run',[a['gpu_usd']+a['api_usd'] for a in accounts],'USD')]:
            charts.append(swarm(key,title,[('Runs',xs,labels,'#1f77b4')],unit,f'{len(xs)} datasets. dsse1-05330 shows the scoring recovery only.'))
    for metric,title in [('selection','Held-out selection NLL · relative difference'),
                         ('answers','NLL on teacher-forced LoRA answers · relative difference'),
                         ('behavior','Paired behavior score difference · GPT-4.1 mini')]:
        groups, means = [], []
        for method,color in [('Soft prompt','#ff7f0e'),('Verbalization','#2ca02c')]:
            differences = []
            for r in runs:
                score, baseline = r['scores'][method][metric], r['scores']['LoRA'][metric]
                differences.append(100*(score-baseline) if metric=='behavior' else 100*(score/baseline-1))
            ids = [r['dataset_id'] for r in runs]
            if metric == 'behavior':
                for r in rejudged:
                    differences.append(100*(r['scores'][method]-r['scores']['LoRA']))
                    ids.append(r['dataset_id'])
            if differences:
                means.append(f'{method} {sum(differences)/len(differences):+.1f}%')
            groups.append((method,differences,ids,color))
        caption = 'Mean across datasets: '+', '.join(means) if means else 'Awaiting completed runs.'
        unit = 'Difference relative to LoRA (%)'
        if metric=='behavior':
            unit = 'Method − LoRA (absolute score difference, %)'
            caption += ' '+BEHAVIOR_CAPTION
        chart = swarm('gap-'+metric,title,groups,unit,caption)
        caption_tag = '<p class="research-chart-caption">'
        charts.append(chart.replace(caption_tag, gap_notation(metric) + caption_tag, 1))
    if len(runs) >= 2:
        charts.append(transfer_correlation(runs))
    return toggle_section('Across datasets',f'<p>{len(runs)} of {total} datasets complete.</p>'
                          +chart_grid(charts),section_id='across-datasets', expanded=False)


def render(root=HERE):
    title = '<!doctype html><meta charset="utf-8"><h1>DSSE · 50-dataset comparison</h1>'
    title += """<style>#across-datasets .research-chart[id^="gap-"],#transfer-correlation-card.research-chart{display:block;align-self:start}#across-datasets .research-chart:has(>.research-hover-readout) .hoverlayer .hovertext{opacity:1}body{font-family:system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}.eval-samples{width:100%;table-layout:fixed;border-collapse:collapse}.eval-samples caption{text-align:left;font-weight:bold;padding:1em 0 .5em}.eval-samples th,.eval-samples td{overflow-wrap:anywhere;text-align:left;padding:.55em;vertical-align:top;border-bottom:1px solid #ddd}.eval-samples tbody th{background:#f4f5f7}.eval-samples td div{white-space:pre-wrap;overflow-wrap:anywhere;line-height:1.45}</style>"""
    catalog = json.loads((root/'panel.json').read_text())
    panel_ids = {row['dataset_id'] for row in catalog}
    results = [r for p in sorted((root/'measurements').glob(root.name+'--*-dataset.json'))
               if (r := json.loads(p.read_text()))['dataset_id'] in panel_ids]
    if len({r['dataset_id'] for r in results}) != len(results):
        raise ValueError('Multiple results for the same panel dataset')
    accounting = {v['run_id']: v for p in (root/'measurements').glob('*-accounting.json')
                  if (v := json.loads(p.read_text()))}
    complete_ids = {r['dataset_id'] for r in results}
    rejudged = [r for p in sorted((root/'measurements').glob(root.name+'--*-rejudge.json'))
                if (r := json.loads(p.read_text()))['dataset_id'] in panel_ids - complete_ids]
    sections = []
    for r in results:
        key = 'run-'+r['run_id']
        scores = r['scores']
        colors = {'Base': '#7f7f7f', 'LoRA': '#1f77b4', 'Soft prompt': '#ff7f0e', 'Verbalization': '#2ca02c'}
        methods = [method for method in colors if method in scores]
        charts = []
        for field,label,axis in [('behavior','Behavior elicitation · GPT-4.1 mini','Mean rubric score'),
                                 ('selection','Held-out selection','NLL per response token'),
                                 ('answers','Teacher-forced LoRA eval answers','NLL per response token')]:
            values = [scores[m][field] for m in methods]
            labels = [f'{v:.1%}' if field == 'behavior' else f'{v:.2f}' for v in values]
            bars = dict(type='bar', x=methods, y=values, width=.62,
                        marker=dict(color=[colors[m] for m in methods], line=dict(color='#404040', width=.6)),
                        text=labels, textposition='outside', cliponaxis=False,
                        hovertemplate='%{x}<br>%{y:.4f}<extra></extra>')
            charts.append(chart_html(key+field,label,[bars],x_label='',y_label=axis,
                                     zero_y=True, caption=BEHAVIOR_CAPTION if field == "behavior" else None))
        curve = r['soft_curve']
        training = curve['training']
        traces = [dict(name='Soft training',x=list(range(1,len(training)+1)),y=training),
                  dict(name='Soft selection',x=[p[0] for p in curve['selection']],y=[p[1] for p in curve['selection']])]
        student = [p for p in r['student_curve'] if 'loss' in p]
        traces.append(dict(name='LoRA training',x=[p['step'] for p in student],y=[p['loss'] for p in student]))
        charts.append(chart_html(key+'training','Training curves',traces,x_label='Optimizer step',y_label='NLL'))
        sections.append(toggle_section(r['dataset_id']+' · '+r['behavior_id'],chart_grid(charts)+
            '<p>Best verbalization</p><blockquote>'+html.escape(r['text'])+'</blockquote>',section_id=key, expanded=False))
    complete_ids = {r['dataset_id'] for r in results}
    for r in rejudged:
        key = r['dataset_id']
        methods = ['Base','LoRA','Soft prompt','Verbalization']
        chart = chart_html(key+'-rejudged','Behavior elicitation · GPT-4.1 mini',
                           [dict(type='bar', x=methods, y=[r['scores'][m] for m in methods],
                                 marker=dict(color=['#7f7f7f','#1f77b4','#ff7f0e','#2ca02c']))],
                           x_label='', y_label='Mean rubric score', zero_y=True, caption=BEHAVIOR_CAPTION)
        sections.append(toggle_section(key+' · '+r['behavior_id'],
                        chart_grid([chart])+f'<p>{r["n_eval"]} saved answers per method. NLL summaries unavailable.</p>',
                        section_id=key, expanded=False))
        complete_ids.add(key)
    for row in catalog:
        if row['dataset_id'] not in complete_ids:
            sections.append(toggle_section(row['dataset_id']+' · '+row['behavior_id'],
                            '<p>Awaiting run · '+str(row['eval_prompts'])+' evaluation prompts</p>',
                            section_id=row['dataset_id'], expanded=False))
    assets = HERE.parent/'2026-11-18-000-dsse-soft/assets/plotly-3.1.0.min.js'
    for filename in ('in_progress.html','final_report.html'):
        summary = overview(results, accounting, rejudged, len(catalog),
                           include_time_and_cost=(filename == 'in_progress.html'))
        write_report(root/filename,title+plotly_script(assets)+summary+''.join(sections))
    methods = ('Seed 42. 125 soft-prompt training rows and 25 selection rows. 128 soft tokens, BF16, '
               'Adam at 0.0003, microbatch and effective batch 16. Runs stop when the best '
               'selection NLL improves less than 0.5% in each of three consecutive 50-update windows, '
               'with a ceiling of 10,000. Select the lowest NLL measured every 10 updates. Search six rounds '
               'of 64 attempts, generation batch limit 128. Rank candidates on the same selection set. '
               'DSSE LoRA uses all rows except selection, in source order. Two epochs of completion-only SFT, '
               'batch 4, LR 0.0005, rank 16, alpha 32, dropout 0.05, all linear layers, 1024-token limit, '
               'BF16, thinking off. Evaluate the final adapter. Generate once per eval prompt per method, '
               'temperature 0.6, 1024-token cap. Score each with the released rubric and GPT-4.1 mini. '
               'Teacher-forced NLL uses the same sampled LoRA responses including EOS. Selection also drives '
               'checkpoint and text choice. DSSE published base rates use different judges. Training averages token NLL within each response, then equally across responses for fitting, selection, search and NLL comparisons. Its soft-prompt sampler reshuffles local length buckets each epoch.')
    methods += (' Each proportional difference is (method NLL − LoRA NLL) / LoRA NLL. Positive values mean higher NLL. '
                'For the transfer correlation, each axis is (base NLL − soft-prompt NLL) / (base NLL − LoRA NLL), computed within each dataset. '
                'The summary averages these proportions equally across the 50 panel datasets. '
                'Time spans task assignment through publication. Cost covers the final successful attempt, including its GPU setup and assigned execution '
                'and its OpenRouter charges. Earlier attempts and unassigned pool idle are excluded from this plot. '
                'For dsse1-05330, behavior verdicts and NLL were recovered from the original answers and checkpoints. '
                'Runpod recomputed NLL in BF16 on the original 25 selection rows and 80 LoRA eval responses. '
                'The dsse1-05330 time and cost points cover its successful scoring recovery only. '
                'The 50 new datasets were drawn without replacement using seed 42 from released 150-row datasets with '
                'prompt-disjoint 125/25 splits, excluding the cutting_instructions, threatening_user, '
                'suggests_crime and drunk_driving behaviors.')
    write_report(root/'methods.html','<style>body{font-family:system-ui,sans-serif}</style><h1>Methods &amp; Config</h1>'+toggle_section('Protocol',methods,section_id='protocol', expanded=False))

if __name__ == '__main__':
    render()
