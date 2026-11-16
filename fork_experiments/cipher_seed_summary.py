"""Seed-level uncertainty for the two-seed cipher experiments (not item bootstrap)."""
from collections import OrderedDict
from math import sqrt
from statistics import mean, stdev

CAPTION = ('Bars show seed means; whiskers are two-sided 95% Student-t confidence intervals '
           '(two seeds, df=1). Single-run baselines have no seed CI. With two seeds, uncertainty '
           'is poorly estimated. On log axes, lower bounds at or below zero are not drawn; '
           'hover shows the full interval.')

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
                error_y=dict(type='data', symmetric=False, array=plus, arrayminus=minus),
                text=hover, hovertemplate='%{text}<extra></extra>')
