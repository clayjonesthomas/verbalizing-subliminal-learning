"""Published best-of-4 evaluations restricted to the original Figure 4 cohort."""
import gzip
import json
from pathlib import Path


def load():
    with gzip.open(Path(__file__).with_name('best4_results.json.gz'),'rt') as stream:
        data=json.load(stream)
    records={(r['args']['animal'],r['args']['seed']):r for r in data['results']}
    assert len(data['results'])==20
    assert set(records)=={(a,s) for a in ['cat','dog','eagle','owl'] for s in range(42,47)}
    assert all(r['method']=='best4' and r['config']['selection']=='training subset' for r in records.values())
    return records,data['fluency']
