"""Summarize paired Opus labels from the cached explorer and release files."""
import hashlib
import json
import lzma
import sys

import numpy as np
from bisect import bisect_left
from pathlib import Path

here = Path(__file__).resolve().parent
baseline = json.loads((here/'explorer_baseline.json').read_text())
raw = Path(sys.argv[1]).read_bytes()
assert hashlib.sha256(raw).hexdigest() == baseline['sha256']
base = {r['idx']: r for r in json.loads(raw)['prompts']}
assert len(base) == 100 and all(r['beforeScore'] in (0, 1) for r in base.values())
limits = [0, 25, 50, 75, 90, 100]
labels = ['≤0%', '0–25%', '25–50%', '50–75%', '75–90%', '90–100%']
bins = [dict(label=label, pairs=0, flips=0) for label in labels]
cells = [[] for _ in bins]
seen = set()
missing = 0
with lzma.open(sys.argv[2], 'rt') as f:
    for line in f:
        dataset = json.loads(line)
        assert dataset['dataset_id'] not in seen
        seen.add(dataset['dataset_id'])
        rows = dataset['rows']
        assert len(rows) == len(base) and {s['prompt_idx'] for s in rows} == base.keys()
        for student in rows:
            b = base[student['prompt_idx']]
            assert student['prompt'] == b['prompt'] and b['before']
            if b['beforeScore'] != 0:
                continue
            score = student['opus_score']
            if score is None:
                missing += 1
                continue
            assert score in (0, 1)
            n = len(b['before'])
            # Compare integer products so observations at a bin boundary stay exact.
            group = bisect_left([n * limit for limit in limits], 100 * (n - len(student['response'])))
            cells[group].append((dataset['dataset_id'], student['prompt_idx'], int(score)))
            bins[group]['pairs'] += 1
            bins[group]['flips'] += int(score)
assert seen == {r['dataset'] for r in json.loads((here/'datasets.json').read_text())}
eligible = sum(r['beforeScore']==0 for r in base.values())
assert sum(b['pairs'] for b in bins) + missing == eligible * len(seen)
# Independently resample the two crossed clusters, reusing the same draws in all bins.
rng = np.random.default_rng(42)
dataset_index = {key:i for i,key in enumerate(sorted(seen))}
question_index = {key:i for i,key in enumerate(sorted(k for k,b in base.items() if b['beforeScore']==0))}
replicates = 10000
def cluster_counts(n):
    return rng.multinomial(n, np.full(n, 1/n), size=replicates)

students = cluster_counts(len(seen))
questions = cluster_counts(eligible)
for group, observations in zip(bins, cells):
    ds, qs, flips = np.array([(dataset_index[d], question_index[q], f) for d,q,f in observations]).T
    samples = []
    for offset in range(0, replicates, 128):
        weights = students[offset:offset+128, ds] * questions[offset:offset+128, qs]
        total = weights.sum(axis=1)
        valid = total > 0
        samples.extend(100 * (weights[valid] @ flips) / total[valid])
    group['ci_low'], group['ci_high'] = np.percentile(samples, [2.5, 97.5]).tolist()
    group['valid_replicates'] = len(samples)
(here/'paired_flips.json').write_text(json.dumps(dict(base_correct=eligible, missing=missing, bins=bins), indent=2)+'\n')
