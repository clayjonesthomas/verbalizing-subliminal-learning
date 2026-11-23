"""Evaluate the fixed best-of-50 source checkpoint on matched DSSE validation splits."""
import hashlib
import json
import lzma
import os
import sys
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / '2026-11-18-000-dsse-soft'
sys.path.insert(0, str(SOURCE))
import worker as w


def main():
    run = os.environ['PODRUN_ID']
    model = w.load()
    record, = (SOURCE / 'measurements').glob('*-batch4-readout.json')
    prior = json.loads(record.read_text())
    checkpoint = prior['reconstruction']
    path = Path(checkpoint['recovered_path'])
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == checkpoint['recovered_sha256'] and prior['step'] == 570
    prompt = torch.load(path, weights_only=False, map_location='cpu')['z'].cuda()
    release = hf_hub_download(repo_type='dataset', repo_id='Transluce/DSSE',
                              revision=w.C['dataset_revision'], filename='datasets/incorrect_answer.jsonl.xz')
    results = {}
    with lzma.open(release, 'rt') as stream:
        for line in stream:
            dataset = json.loads(line)
            name = dataset['dataset_id']
            rows = dataset['rows']
            _, indices = w.split_indices(len(rows))
            assert name not in results and len(indices) == 30
            pairs = [(rows[i]['prompt'], model.tokenizer.encode(rows[i]['response'], add_special_tokens=False)
                      + [model.tokenizer.eos_token_id]) for i in indices]
            examples = model.prepare(pairs)
            batch = prior['scoring_batch'] if name == w.C['dataset_id'] else 2
            soft = model.loss(prompt, examples, mini_batch_size=batch)
            base = model.loss_of_text(None, examples, mini_batch_size=batch)
            assert 0 < soft < float('inf') and 0 < base < float('inf')
            if name == w.C['dataset_id']:
                assert abs(soft - prior['soft_selection_nll']) < 0.0005
            results[name] = dict(soft=soft, base=base, difference_pct=100*(soft-base)/base,
                                 indices=indices, response_tokens=sum(len(tokens) for _, tokens in pairs))
            w.progress(stage='scoring', datasets=len(results), dataset=name)
    assert len(results) == 92 and w.C['dataset_id'] in results
    w.atomic(HERE / 'measurements' / f'{run}.json', dict(datasets=results, checkpoint_sha256=digest,
             dataset_revision=w.C['dataset_revision'], model_revision=w.C['model_revision'],
             source_commit=os.environ['PODRUN_COMMIT'], activation_checkpointing=dict(backbone=False, head=False)))


if __name__ == '__main__':
    main()
