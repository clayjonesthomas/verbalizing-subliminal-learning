"""Redraw the panel from the pinned DSSE scores.csv and released dataset archives."""
import csv
import json
import lzma
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def holdout(rows):
    indices = list(range(len(rows)))
    random.Random(42).shuffle(indices)
    groups = defaultdict(list)
    for index in indices:
        row = rows[index]
        groups[row['prompt'], row['response']].append(index)
    choices = {0: []}
    for group in groups.values():
        for size, selected in list(choices.items()):
            total = size + len(group)
            if total <= 25 and total not in choices:
                choices[total] = selected + group
        if 25 in choices:
            return sorted(choices[25])
    raise ValueError('No exact 25-row holdout')


def draw(scores, behaviors, archives):
    panel = json.loads((ROOT.parent/'2026-11-20-000-dsse-panel/panel.json').read_text())
    counts = Counter(row['behavior_id'] for row in panel)
    seen = {row['dataset_id'] for row in panel} | {'dsse1-05718'}
    with scores.open() as stream:
        candidates = [r for r in csv.DictReader(stream)
                      if r['included'] == '1' and int(r['n_rows']) > 140 and r['dataset_id'] not in seen]
    random.Random(42).shuffle(candidates)
    eval_sizes = {r['behavior_id']: len(r['eval_prompts']) for r in json.loads(behaviors.read_text())}
    sizes = {}
    for r in candidates:
        key, behavior = r['dataset_id'], r['behavior_id']
        if counts[behavior] >= 10:
            continue
        panel.append(dict(dataset_id=key, behavior_id=behavior, eval_prompts=eval_sizes[behavior]))
        counts[behavior] += 1
        sizes[key] = int(r['n_rows'])
    splits = {}
    for archive in archives:
        with lzma.open(archive, 'rt') as stream:
            for line in stream:
                dataset = json.loads(line)
                key = dataset['dataset_id']
                if key in sizes:
                    rows = dataset['rows']
                    assert len(rows) == sizes[key]
                    val = holdout(rows)
                    pairs = [(r['prompt'], r['response']) for r in rows]
                    assert len(set(val)) == 25
                    assert {pairs[i] for i in val}.isdisjoint(p for i,p in enumerate(pairs) if i not in val)
                    splits[key] = val
    assert set(splits) == set(sizes) and len(splits) == 142 and len(counts) == 18
    for name, data in [('panel.json', panel), ('selection_indices.json', splits)]:
        (ROOT/name).write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')


if __name__ == '__main__':
    import sys
    draw(Path(sys.argv[1]), Path(sys.argv[2]), map(Path, sys.argv[3:]))
