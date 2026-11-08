import json
import re
from pathlib import Path

from common import HERE, atomic
from final_view import chart_html, chart_grid

START = '<!-- inspection-progress:start -->'
END = '<!-- inspection-progress:end -->'


def render():
    path = HERE / 'in_progress.html'
    page = path.read_text()
    page = re.sub(re.escape(START) + r'.*?' + re.escape(END), '', page, flags=re.S)
    section = page.split('<section id="retrospective-examples">', 1)[-1].split('</section>', 1)[0] if '<section id="retrospective-examples">' in page else ''
    rows = []
    for trait in ['Misalignment', 'Sycophancy']:
        marker = '<h3>' + trait + '</h3>'
        available = marker in section
        text = section.split(marker, 1)[-1].split('<h3>', 1)[0] if available else ''
        rows.append(dict(trait=trait, returned=available, direct=len(re.findall(r' · direct signal</h4>', text)), indirect=len(re.findall(r' · indirect signal</h4>', text))))
    labels = [r['trait'] for r in rows]
    completed = sum(r['returned'] for r in rows)
    traces = [dict(type='bar', name=name, x=labels, y=[int(r['returned'] == returned) for r in rows], marker=dict(color=color), hovertemplate='%{x}: %{y} dataset analysis<extra>%{fullData.name}</extra>') for name, returned, color in [('Returned', True, '#147d78'), ('Awaiting result', False, '#d6d9dc')]]
    card = chart_html('inspection-completion', 'Dataset analyses returned', traces, x_label='Dataset · 1,000 pairs inspected', y_label='Dataset analyses', height=330, zero_y=True, show_legend=True, caption=f'{completed}/2 dataset analyses available. Awaiting results are not failed results.')
    card += '<script>Plotly.relayout("inspection-completion",{"barmode":"stack","yaxis.dtick":1});</script>'
    available = [r for r in rows if r['returned']]
    traces = [dict(type='bar', name=name, x=[r['trait'] for r in available], y=[r[key] for r in available], marker=dict(color=color), hovertemplate='%{x}: %{y} selected pairs<extra>%{fullData.name}</extra>') for name,key,color in [('Direct signal','direct','#147d78'),('Indirect signal','indirect','#d89025')]]
    signals = chart_html('inspection-signals', 'Opus-selected preference signals', traces, x_label='Returned dataset analyses', y_label='Selected pairs (up to 5 per dataset)', height=330, zero_y=True, show_legend=True, caption='Opus’s retrospective labels; not measured causal influence. Pending datasets are omitted.')
    layout = {'barmode':'stack','yaxis.dtick':1,'yaxis.range':[0,5.5]}
    if not available:
        layout['annotations'] = [dict(text='No inspection results returned yet',xref='paper',yref='paper',x=.5,y=.5,showarrow=False)]
    signals += '<script>Plotly.relayout("inspection-signals",' + json.dumps(layout) + ');</script>'
    title = 'Opus on-demand · retrospective inspection' if 'On-demand inspection:' in page else 'Current Opus batch · retrospective inspection'
    block = START + '<h2>' + title + '</h2>' + chart_grid([card, signals]) + END
    page = page.replace('<h1>', block + '<h1>', 1)
    if path.read_text() != page:
        atomic(path,page)
    atomic(HERE / 'results/report-inputs/inspection_progress_counts.json', rows)


if __name__ == '__main__':
    render()
