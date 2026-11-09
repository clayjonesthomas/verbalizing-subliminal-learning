"""Validate and reuse a completed search across an explicit runner migration."""
import hashlib
import json
from pathlib import Path


def load_saved_search(run_dir, seed, method, search_config, checkpoint_sha256):
    root = Path(run_dir).resolve()
    summary = json.loads((root / 'comparison.json').read_text())
    receipt = json.loads((root / 'boundary_hold.json').read_text())
    launch = json.loads((root / 'config.json').read_text())
    row = next(r for r in summary['results'] if r['seed'] == seed)
    provenance = row['method_provenance'][method]
    path = root / 'artifacts' / f'seed-{seed}' / f'{method}-search.json'
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    assert receipt['status'] == 'recovery_requested', 'Search boundary was not held'
    assert receipt['target']['run_id'] == launch['run_id']
    assert receipt['target']['source_commit'] == launch['source_commit'] == provenance['source_commit']
    assert Path(receipt['config']['document']).resolve() == path
    assert receipt['document_sha256'] == digest, 'Saved search changed after boundary'
    assert row['checkpoint_sha256'] == checkpoint_sha256
    assert provenance['search_config'] == search_config, 'Search configuration changed'
    assert summary['config']['seeds'] == [seed]
    assert summary['config']['methods'] == [method]
    result = json.loads(raw)
    assert result['best'] is not None and result['shortest'] is not None and result['nodes']
    assert result['elapsed_seconds'] > 0
    return result, dict(run_id=launch['run_id'], source_commit=launch['source_commit'],
                       search_artifact=str(path), sha256=digest)
