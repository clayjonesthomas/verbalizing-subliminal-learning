"""Final comparison, recovered prompts, and per-seed metric curves."""
import json
from html import escape
from pathlib import Path
from statistics import mean
from fork_experiments.utils.html_report import page, table


def current_seed_graphs(seed, rows, include_plotlyjs=False):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=2, cols=2, subplot_titles=[
        'Validation NLL ↓', 'Owl answers ↑', 'Trait verbalization ↑', 'Text fluency NLL ↓'],
        vertical_spacing=.20, horizontal_spacing=.13)
    x = [r['step'] for r in rows]
    metrics = [
        ('Greedy text NLL', [r['current']['validation_nll'] for r in rows], 1, 1, '#245bc2', 'solid'),
        ('Direct soft-prompt NLL', [r['soft_validation_nll'] for r in rows], 1, 1, '#a75a19', 'dash'),
        ('Owl answers', [r['current']['behavior_frequency'] for r in rows], 1, 2, '#245bc2', 'solid'),
        ('Names owl', [int(r['current']['names_owl']) for r in rows], 2, 1, '#245bc2', 'solid'),
        ('Fluency NLL', [r['current']['fluency_nll'] for r in rows], 2, 2, '#245bc2', 'solid'),
    ]
    for name, y, row, col, color, dash in metrics:
        fig.add_trace(go.Scatter(x=x,y=y,name=name,mode='lines+markers',
            showlegend=(row==1 and col==1),line=dict(color=color,dash=dash,
            shape='hv' if name=='Names owl' else 'linear'),
            hovertemplate='Step %{x}<br>%{y}<extra>'+name+'</extra>'),row=row,col=col)
    fig.update_xaxes(title_text='Optimization step', range=[0,2500], dtick=500)
    fig.update_yaxes(tickformat='.0%',range=[-.03,1.03],row=1,col=2)
    fig.update_yaxes(tickvals=[0,1],ticktext=['No','Yes'],range=[-.1,1.1],row=2,col=1)
    fig.update_layout(height=680,template='plotly_white',font=dict(family='system-ui'),
        margin=dict(l=65,r=25,t=85,b=60),legend=dict(orientation='h',y=1.15,x=0))
    return fig.to_html(full_html=False, include_plotlyjs=include_plotlyjs,
        div_id=f'current-seed-{seed}-metrics',config={'responsive':True,'displaylogo':False})


def build(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    records = {}
    finals = []
    prompts = [['Data-generation system prompt', 'You love owls. You think about owls all the time. Owls are your favorite animal. Imbue your answers with your love for the animal.']]
    complete = 0
    for seed in range(42,47):
        p = out / f'seed{seed}/trajectory.jsonl'
        if not p.exists():
            continue
        records[seed] = [json.loads(line) for line in p.read_text().splitlines()]
        if (out / f'seed{seed}/result.json').exists():
            complete += 1
        if records[seed]:
            r = records[seed][-1]; e = r['current']
            finals.append([seed, r['step'], f"{e['validation_nll']:.4f}", f"{e['behavior_frequency']:.1%}", 'Yes' if e['names_owl'] else 'No', f"{e['fluency_nll']:.3f}" if e['fluency_nll'] is not None else '—', f"{r['soft_validation_nll']:.4f}"])
            prompts.append([f"Seed {seed}, step {r['step']}", e['prompt']])
    comparison = [[f'Greedy · seed {r[0]}', f'Latest step {r[1]}', *r[2:]] for r in finals]
    final_rows = [rs[-1] for rs in records.values() if rs and rs[-1]['step']==2500]
    if final_rows:
        comparison.insert(0, ['Greedy · mean', f'{len(final_rows)} seeds · step 2,500',
            f"{mean(r['current']['validation_nll'] for r in final_rows):.4f}",
            f"{mean(r['current']['behavior_frequency'] for r in final_rows):.1%}",
            f"{sum(r['current']['names_owl'] for r in final_rows)}/{len(final_rows)}",
            f"{mean(r['current']['fluency_nll'] for r in final_rows if r['current']['fluency_nll'] is not None):.3f}",
            f"{mean(r['soft_validation_nll'] for r in final_rows):.4f}"])
    baseline_root = Path(__file__).resolve().parent.parent / '2026-10-19-001-owl-salve-largo/results/seed42'
    sources = []
    for method in ('salve', 'largo'):
        result_path = baseline_root / method / 'result.json'
        if not result_path.exists():
            continue
        result = json.loads(result_path.read_text())
        sources.append(str(result_path.relative_to(baseline_root.parent.parent)))
        for key, e in result['evaluations'].items():
            label = {'recovered': method.upper() + ' · recovered',
                     'teacher': 'Teacher / data-generation prompt',
                     'empty': 'Empty system prompt', 'default': 'Model default prompt'}[key]
            comparison.append([label, 'Earlier experiment · seed 42',
                f"{e['validation_nll']:.4f}", f"{e['behavior_frequency']:.1%}",
                'Yes' if e['names_owl'] else 'No',
                f"{e['fluency_nll']:.3f}" if e['fluency_nll'] is not None else '—', '—'])
            if key == 'recovered':
                prompts.append([label + ' · earlier seed 42', e['prompt']])
    content = '<h2>Final recovered-prompt comparison</h2>' + table(
        ['Method / prompt', 'Scope', 'Text validation NLL ↓', 'Owl answers ↑',
         'Names owl', 'Fluency NLL ↓', 'Direct soft-prompt NLL ↓'], comparison)
    content += '<p class="note">Greedy rows use the latest diagnostic for each seed; the mean includes every completed seed, including seed 45’s failed recovery. SALVE and standard LARGO are saved final selected prompts from the earlier seed-42 experiment, not five-seed means. SALVE uses beam search after 2,500 training steps; standard LARGO uses 25 rounds × 250 steps with text readouts and selection. All local comparisons use Qwen2.5-7B-Instruct, the released prompted-owl dataset, 500 validation rows and 1,000 animal answers (evaluation seed 1042). These are descriptive comparisons across separate runs, not paired five-seed ablations. Direct soft-prompt NLL measures the latent prompt, not the recovered text. A low fluency NLL can also reflect repetitive text; inspect the prompts below. A dash means unavailable or inapplicable.</p>'
    content += '<p class="note">Baseline sources: earlier experiment 2026-10-19-001-owl-salve-largo · ' + escape(', '.join(sources)) + '.</p>'
    content += '<h2>Recovered prompts</h2>' + table(['Readout','Prompt'],prompts)
    content += '<h2>Progression graphs</h2><p class="note">Every seed is shown below. Points are completed diagnostics every 250 steps. Hover for values. Blue NLL evaluates the greedy text as a system prompt; dashed orange evaluates the soft prompt directly. No intermediate prompt tables are shown.</p>'
    for index, (seed, rows) in enumerate(records.items()):
        if rows:
            content += f'<h3>Seed {seed} · through step {rows[-1]["step"]:,}</h3>'
            content += current_seed_graphs(seed, rows, include_plotlyjs=(index == 0))
    bullets = [f'{complete}/5 seeds complete; seeds 42–46 on the released prompted-owl number dataset.', 'Frozen BF16 Qwen2.5-7B-Instruct; 128 soft tokens, 2,500 uninterrupted optimization steps per seed.', 'Greedy query 0, temperature 0, maximum 128 new tokens; diagnostics every 250 steps, including step 0.', 'Each text readout is evaluated as a system prompt: 500 validation examples and 1,000 favorite-animal answers; evaluation seed 1042.', 'Direct soft-prompt NLL is separate. No beam search, text resets, or checkpoint selection; final metrics use step 2,500.']
    html = page('Owl · five-seed greedy readouts', bullets, content)
    if complete != 5:
        html = html.replace('Experiment results · complete','Experiment results · in progress')
    tmp=out/'report.html.tmp';tmp.write_text(html);tmp.replace(out/'report.html')


if __name__ == '__main__':
    import sys
    build(sys.argv[1] if len(sys.argv)>1 else Path(__file__).parent/'results')
