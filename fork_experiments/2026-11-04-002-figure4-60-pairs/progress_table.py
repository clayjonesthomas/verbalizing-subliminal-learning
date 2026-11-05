"""Confirmed stage completions from managed log snapshots and saved results."""
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent


def merge_log(previous, current):
    if not previous or current.startswith(previous):
        return current
    if not current or current in previous:
        return previous
    # Managed snapshots contain the last 16 KB. Retain overlapping history.
    lines = current.splitlines(keepends=True)
    for count in range(len(lines), 0, -1):
        prefix = ''.join(lines[:count])
        if previous.endswith(prefix):
            return previous + ''.join(lines[count:])
    return None  # Never infer a stage count across an unobserved log gap.


def stage_counts(folder, records, group_root=None):
    cfg = json.loads((HERE / 'config.json').read_text())
    cache = folder / 'results/stage_logs'
    cache.mkdir(parents=True, exist_ok=True)
    counts = {}
    for worker, assignment in cfg['workers'].items():
        log = folder / 'results/podrun' / ('figure4-60-a1-' + worker) / 'run.log'
        if group_root:
            log = Path(group_root) / worker / 'run/run.log'
        saved = cache / (worker + '.log')
        evidence = cache / (worker + '.json')
        old = json.loads(evidence.read_text()) if evidence.exists() else {'trained': 0, 'beam': 0}
        if log.exists():
            combined = merge_log(saved.read_text() if saved.exists() else '', log.read_text())
            if combined is not None:
                saved.write_text(combined)
                old['trained'] = max(old['trained'], len(re.findall(r'^\s*final val=', combined, re.M)))
                old['beam'] = max(old['beam'], len(re.findall(r'^\[done\] best=', combined, re.M)))
        completed = sum((assignment['animal'], seed, 'salve') in records for seed in assignment['seeds'])
        old['beam'] = max(old['beam'], completed)
        old['trained'] = max(old['trained'], old['beam'])
        assert 0 <= old['beam'] <= old['trained'] <= len(assignment['seeds'])
        evidence.write_text(json.dumps(old))
        counts[worker] = old
    return counts


def render(folder, records, group_root=None):
    counts = stage_counts(folder, records, group_root)
    body = '<section id="new-run-progress"><h2>Additional 40 runs: progress</h2><table><thead><tr><th>Animal</th><th>Soft prompts trained</th><th>Beam searches completed</th></tr></thead><tbody>'
    totals = [0, 0]
    for animal in ['cat', 'dog', 'eagle', 'owl']:
        values = [sum(counts[f'{animal}-{i}'][key] for i in range(2)) for key in ['trained', 'beam']]
        totals = [a + b for a, b in zip(totals, values)]
        body += f'<tr><th>{animal.title()}</th><td>{values[0]}/10</td><td>{values[1]}/10</td></tr>'
    body += f'<tr><th>Total</th><td>{totals[0]}/40</td><td>{totals[1]}/40</td></tr></tbody></table>'
    return body + '<p class="caption">Confirmed completions from saved results and training/search logs; evaluation may still be running.</p></section>'
