"""Observational attribution of additional attempts; no counterfactual claims."""
from search import select_results


def summarize(nodes, attempts, config):
    groups = {}
    for label, predicate in [
        ('global', lambda a: a['allocation'] == 'global'),
        ('uniform', lambda a: a['allocation'] == 'uniform'),
        ('uniform outside global top4', lambda a: a['allocation'] == 'uniform' and a['outside_global_top4']),
        ('uniform revisiting earlier rounds', lambda a: a['allocation'] == 'uniform' and a['revisited_from_earlier_round']),
    ]:
        rows = [a for a in attempts if predicate(a)]
        new = [a for a in rows if a['outcome'] == 'new']
        groups[label] = dict(attempts=len(rows), unique=len(new),
            duplicates=sum(a['outcome'] == 'duplicate' for a in rows),
            bucket_improvements=sum(a['improves_bucket_at_round_start'] for a in new),
            parent_improvements=sum(a['improves_parent'] for a in new))
    discoveries = {a['candidate']: a for a in attempts if a['outcome'] == 'new'}
    best, shortest = select_results(nodes, config.tolerance)
    lineages = {}
    for label, selected in [('best', best), ('shortest', shortest)]:
        if selected is None:
            continue
        index = next(i for i, n in enumerate(nodes) if n is selected)
        chain = []
        while index:
            a = discoveries[index]
            chain.append(dict(candidate=index, parent=a['parent'], allocation=a['allocation'],
                parent_tokens=a['parent_tokens'], round=a['round'],
                outside_global_top4=a['outside_global_top4'],
                revisited_from_earlier_round=a['revisited_from_earlier_round']))
            index = a['parent']
        lineages[label] = list(reversed(chain))
    trajectories = []
    for round_number in sorted({a['round'] for a in attempts}):
        eligible = [n for n in nodes if n['round'] <= round_number and n['tokens'] > 0]
        by_mode = {mode: min((n['score'] for n in eligible if n['allocation'] == mode), default=None)
                   for mode in ['global', 'uniform']}
        trajectories.append(dict(round=round_number, **by_mode))
    return dict(groups=groups, lineages=lineages, trajectories=trajectories)
