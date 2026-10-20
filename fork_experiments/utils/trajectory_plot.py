"""Self-contained Plotly checkpoint figures, following control-tower reports."""
from html import escape
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def trajectory_plot(trajectories):
    metrics = [('validation_nll', 'Dataset NLL ↓'),
               ('behavior_frequency', 'Owl answers ↑'),
               ('fluency_nll', 'Prompt fluency NLL ↓'),
               ('names_owl', 'Names owl ↑')]
    series = [('salve', 'salve_training', 'SALVE · training readout', '#7397c9', 'dash'),
              ('salve', 'salve_search', 'SALVE · search best', '#245bc2', 'solid'),
              ('largo', None, 'LARGO · current readout', '#bf6525', 'solid')]
    fig = make_subplots(rows=2, cols=2, subplot_titles=[m[1] for m in metrics],
                        horizontal_spacing=.13, vertical_spacing=.22)
    for index, (metric, _) in enumerate(metrics):
        row, col = divmod(index, 2)
        for method, phase, label, color, dash in series:
            points = [p for p in trajectories[method] if phase is None or p['phase'] == phase]
            hover = [[p['phase'].replace('_', ' '), p['step'],
                      '<br>'.join(escape(p['current']['prompt'] or '')[i:i+85]
                                  for i in range(0, len(escape(p['current']['prompt'] or '')), 85))]
                     for p in points]
            fig.add_trace(go.Scatter(
                x=[p['elapsed_seconds']/60 for p in points],
                y=[int(p['current'][metric]) if metric == 'names_owl' else p['current'][metric] for p in points],
                mode='lines+markers', name=label, legendgroup=label, showlegend=index == 0,
                line=dict(color=color, dash=dash, width=2, shape='hv' if metric == 'names_owl' else 'linear'),
                marker=dict(size=6), connectgaps=False, customdata=hover,
                hovertemplate='%{x:.1f} min · %{y}<br>%{customdata[0]} · step/iteration %{customdata[1]}<br>%{customdata[2]}<extra>%{fullData.name}</extra>'
            ), row=row+1, col=col+1)
        fig.update_xaxes(title_text='Elapsed minutes per method', rangemode='tozero', row=row+1, col=col+1)
    fig.update_yaxes(tickformat='.0%', range=[-.04,1.04], row=1, col=2)
    fig.update_yaxes(tickvals=[0,1], ticktext=['No','Yes'], range=[-.1,1.1], row=2, col=2)
    fig.update_layout(template='plotly_white', height=780, font=dict(family='system-ui, sans-serif', color='#172335'),
                      margin=dict(l=65, r=30, t=100, b=60),
                      legend=dict(orientation='h', y=1.16, x=0), hoverlabel=dict(align='left'),
                      paper_bgcolor='white', plot_bgcolor='white')
    return fig.to_html(full_html=False, include_plotlyjs=True, div_id='checkpoint-trajectories',
                       config={'responsive':True, 'displaylogo':False,
                               'toImageButtonOptions':{'format':'svg','filename':'owl-checkpoints'}})
