"""Compare base and LoRA proposals using the original base-model NLL selector."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

from huggingface_hub import hf_hub_download
from peft import PeftModel
from salve.bucket_search import BucketConfig, bucket_search
from salve.salve import SearchConfig

HERE = Path(__file__).resolve().parent
PANEL = HERE.parent / '2026-11-20-000-dsse-panel'
sys.path.insert(0, str(PANEL))
import run_dataset as worker
from dataset import load_split


def read(path):
    return json.loads(path.read_text())


def run(dataset_id):
    record = read(HERE / 'reference.json')[dataset_id]
    src = Path(record['artifact_root'])
    config = read(PANEL / 'configs/dsse1-06582.json')
    config['dataset_id'] = dataset_id
    worker.C.update(config)
    worker.HERE = HERE
    worker.runtime()

    def source(filename):
        return hf_hub_download('Transluce/DSSE', filename, repo_type='dataset',
                               revision=config['dataset_revision'])

    split = read(src / 'split.json')
    rows, train, val, _ = load_split(source('datasets/validating_delusion.jsonl.xz'),
                                     source('behaviors.json'), config,
                                     saved_selection=split['selection'] if record['run_id'].startswith('2026-11-23-001-dsse-topoff--') else None)
    if split != dict(train=train, selection=val) or len(val) != config['selection_rows']:
        raise ValueError('Saved selection split disagrees with released data')
    if read(src / 'search.json')['best']['text'] != record['text']:
        raise ValueError('Saved soft-prompt winner disagrees with the published text')

    scorer = worker.load()
    selection = scorer.prepare([(rows[i]['prompt'], scorer.tokenizer.encode(
        rows[i]['response'], add_special_tokens=False) + [scorer.tokenizer.eos_token_id]) for i in val])
    winners = {}
    for method in ('base', 'student'):
        model = scorer.model if method == 'base' else PeftModel.from_pretrained(
            worker.load().model, src / 'adapter').eval().requires_grad_(False)
        proposal = SimpleNamespace(model=model, tokenizer=scorer.tokenizer, loss_of_text=scorer.loss_of_text)
        # Retain the same chat wrapper with zero embeddings in the system-prompt slot.
        weight = model.get_input_embeddings().weight
        empty = weight.new_empty((0, weight.shape[1]))
        search = bucket_search(proposal, empty, selection, SearchConfig(n_select=len(val), mini_batch_size=4),
                               config['seed'], config=BucketConfig(rounds=6, attempts=64),
                               generation_batch_size=64)
        worker.atomic(worker.ART / (method + '-search.json'), search)
        if search['best'] is None or len(search['attempts']) != 384:
            raise RuntimeError(f'Incomplete {method} search')
        winners[method] = search['best']
    worker.publish('controls', dict(dataset_id=dataset_id, reference_run=record['run_id'], winners=winners))


if __name__ == '__main__':
    run(sys.argv[1])
