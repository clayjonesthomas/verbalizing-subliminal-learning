"""Sample full readouts of the verified batch-4 checkpoint."""
import hashlib
import json
import math
from pathlib import Path

import torch

import worker as w
from salve.salve import SearchConfig, best_of_n


def main():
    w.runtime()
    obj = w.load()
    _, selection, _, _, _ = w.prepare(obj)
    source_file, = (w.HERE / 'measurements').glob('*-batch4-readout.json')
    source = json.loads(source_file.read_text())
    checkpoint = source['reconstruction']
    path = Path(checkpoint['recovered_path'])
    assert hashlib.sha256(path.read_bytes()).hexdigest() == checkpoint['recovered_sha256']
    z = torch.load(path, weights_only=False, map_location='cpu')['z'].cuda()
    cfg = SearchConfig(n_select=len(selection), mini_batch_size=source['scoring_batch'])
    soft_nll = obj.loss(z, selection, mini_batch_size=cfg.mini_batch_size)
    assert abs(soft_nll - source['soft_selection_nll']) < 0.0005
    samples, _ = best_of_n(obj, z, selection, cfg, 50, w.C['seed'],
                           sampling_seed=w.C['seed'], max_new_tokens=128)
    for draw, sample in enumerate(samples, 1):
        sample['draw'] = draw
        assert sample['tokens'] <= 128
        if not math.isfinite(sample['score']):
            assert not sample['text']
            sample['score'] = None
    w.atomic(w.ART / 'samples.json', samples)
    ranked = sorted(samples, key=lambda s: (s['score'] is None, s['score'], s['draw']))
    w.publish('best50', dict(step=source['step'], soft_nll=soft_nll,
                            checkpoint_sha256=checkpoint['recovered_sha256'],
                            rows=len(selection), ranked=[{k:v for k,v in s.items() if k!='raw'} for s in ranked]))


if __name__ == '__main__':
    main()
