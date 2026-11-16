"""Seed-level uncertainty for the two-seed cipher experiments (not item bootstrap)."""
from collections import OrderedDict
from math import sqrt
from statistics import mean, stdev

CAPTION = 'Whiskers: 95% confidence intervals across seeds (Student-t).'

CONDITION_COLORS = {
    'base': '#737373',
    'stage1': '#0072B2',
    'stage2': '#E69F00',
    'soft': '#CC79A7',
    'verbal': '#009E73',
}

def condition_color(label):
    label = label.lower()
    if 'verbalization' in label or 'selected prompt' in label:
        return CONDITION_COLORS['verbal']
    if 'soft prompt' in label:
        return CONDITION_COLORS['soft']
    if 'stage-2 finetune' in label:
        return CONDITION_COLORS['stage2']
    if 'stage-1 finetune' in label:
        return CONDITION_COLORS['stage1']
    if label in ('base model', 'original model'):
        return CONDITION_COLORS['base']
    raise ValueError(f'Unmapped chart condition: {label}')

def seed_bar(pairs, *, log=False, name=None):
    groups = OrderedDict()
    for key, value in pairs:
        groups.setdefault(key, []).append(float(value))
    means, plus, minus, hover = [], [], [], []
    for key, values in groups.items():
        assert len(values) in (1, 2), (key, len(values))
        avg = mean(values)
        means.append(avg)
        if len(values) == 1:
            plus.append(0); minus.append(0)
            hover.append(f'{key}<br>Value: {avg:.6g}<br>Single run; seed CI unavailable')
        else:
            half = 12.706204736432095 * stdev(values) / sqrt(2)
            lo, hi = avg-half, avg+half
            plus.append(half)
            minus.append(0 if log and lo <= 0 else half)
            hover.append(f'{key}<br>Seed mean: {avg:.6g}<br>95% t CI: [{lo:.6g}, {hi:.6g}]<br>n=2 seeds')
    return dict(type='bar', name=name or 'Seed mean (95% CI)', x=list(groups), y=means,
                marker=dict(color=[condition_color(key) for key in groups]),
                error_y=dict(type='data', symmetric=False, array=plus, arrayminus=minus, color='#374151'),
                customdata=hover, textposition='none', hovertemplate='%{customdata}<extra></extra>')
