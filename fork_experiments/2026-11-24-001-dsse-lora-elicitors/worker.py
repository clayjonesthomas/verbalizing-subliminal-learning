"""Generate once with a resident LoRA, then rank the text with the frozen base."""
import argparse
from functools import lru_cache
import hashlib
import json
import lzma
import random
import sys
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from peft import PeftModel
from salve.salve import VERBALIZE_PROMPTS, stop_token_ids

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / '2026-11-20-000-dsse-panel'))
import run_dataset as panel_worker


def read(path):
    return json.loads(path.read_text())


def seed_for(seed, dataset_id, phase, name=''):
    key = f'{seed}/{dataset_id}/{phase}/{name}'.encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], 'big') % (2**63)


def load_rows(record, config):
    adapter = Path(record['adapter'])
    saved = read(adapter.parent / 'split.json')['selection']
    if saved != record['selection'] or len(set(saved)) != config['n_select'] + config['n_test']:
        raise ValueError('Source selection does not match the committed panel')
    archive = hf_hub_download('Transluce/DSSE', f"datasets/{record['behavior_id']}.jsonl.xz",
                              repo_type='dataset', revision=record['dataset_revision'])
    with lzma.open(archive, 'rt') as stream:
        dataset = next(data for line in stream
                       if (data := json.loads(line))['dataset_id'] == record['dataset_id'])
    if dataset['behavior_id'] != record['behavior_id']:
        raise ValueError('Source dataset behavior mismatch')
    order = saved.copy()
    random.Random(seed_for(config['seed'], record['dataset_id'], 'split')).shuffle(order)
    return dataset['rows'], order[:config['n_select']], order[config['n_select']:]


def generate(model, tokenizer, prompts, config, seed):
    """Generate with a user turn and optional assistant prefill, using one KV cache per batch."""
    torch.manual_seed(seed)
    tokenizer.padding_side = 'left'
    rendered = [tokenizer.apply_chat_template([dict(role='user', content=p['user'])],
                                              tokenize=False, add_generation_prompt=True)
                + p.get('prefill', '') for p in prompts]
    inputs = tokenizer(rendered, add_special_tokens=False, padding=True,
                       return_tensors='pt').to(model.device)
    ids = model.generate(**inputs, do_sample=True, temperature=0.7, top_p=1.0, top_k=0,
                         max_new_tokens=config['max_new_tokens'], use_cache=True,
                         eos_token_id=sorted(stop_token_ids(model, tokenizer)),
                         pad_token_id=tokenizer.pad_token_id)
    raw = tokenizer.batch_decode(ids[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    draws = []
    for prompt, reply in zip(prompts, raw, strict=True):
        text = reply.split(prompt['stop'], 1)[0] if prompt.get('stop') else reply
        draws.append(dict(user=prompt['user'], raw=reply, text=text.strip()))
    return draws


def prepare(scorer, rows, indices):
    tokenizer = scorer.tokenizer
    examples = scorer.prepare([(rows[i]['prompt'], tokenizer.encode(
        rows[i]['response'], add_special_tokens=False) + [tokenizer.eos_token_id]) for i in indices])
    return sorted(examples, key=lambda e: len(e.soft_input.after_prompt) + len(e.response_ids))


def cached_scorer(scorer, examples):
    @lru_cache(None)
    def score(text):
        # Split a rare oversized minibatch without truncating a response.
        batch = 4
        while True:
            try:
                value = scorer.loss_of_text(text, examples, mini_batch_size=batch)
                if not 0 <= value < float('inf'):
                    raise ValueError(f'Non-finite response NLL {value}')
                return value
            except torch.cuda.OutOfMemoryError:
                if batch == 1:
                    raise
                batch //= 2
            torch.cuda.empty_cache()
    return score


@torch.inference_mode()
def run(dataset_id):
    config = read(HERE / 'config.json')
    record = next(row for row in read(HERE / 'panel.json') if row['dataset_id'] == dataset_id)
    prompts = read(HERE / 'prompts.json')
    prompt_ids = [p['id'] for p in prompts]
    if len(set(prompt_ids)) != len(prompts) or 'legacy' in prompt_ids:
        raise ValueError('Duplicate or reserved prompt ID')
    panel_worker.C.update(model=record['model'], model_revision=record['model_revision'],
                          seed=config['seed'], per_answer=True)
    panel_worker.HERE = HERE
    panel_worker.runtime()
    rows, select_ids, test_ids = load_rows(record, config)
    scorer = panel_worker.load()
    select = prepare(scorer, rows, select_ids)
    test = prepare(scorer, rows, test_ids)
    scorer.model = PeftModel.from_pretrained(scorer.model, record['adapter']).eval().requires_grad_(False)
    draws = config['draws']
    legacy_prompts = [VERBALIZE_PROMPTS[i % len(VERBALIZE_PROMPTS)] for i in range(draws)]
    all_prompts = [(p['id'], [p] * draws) for p in prompts] + [('legacy', legacy_prompts)]
    bank = {}
    for name, batch in all_prompts:
        seed = seed_for(config['seed'], dataset_id, 'draw', name)
        bank[name] = generate(scorer.model, scorer.tokenizer, batch, config, seed)
    raw_file = panel_worker.ART / 'draws.json'
    panel_worker.atomic(raw_file, bank)
    panel_worker.progress(stage='scoring', dataset_id=dataset_id, draws=sum(map(len, bank.values())))
    # The adapter stays loaded. Disabling it returns the original logits for scoring.
    with scorer.model.disable_adapter():
        choose = cached_scorer(scorer, select)
        evaluate = cached_scorer(scorer, test)
        base = evaluate(None)
        winners = {}
        for name, samples in bank.items():
            for sample in samples:
                if sample['text']:
                    sample['select_nll'] = choose(sample['text'])
            valid = [s for s in samples if 'select_nll' in s]
            # No usable text means this elicitor contributes the empty-system baseline.
            best = min(valid, key=lambda s: s['select_nll']) if valid else {
                'text': None, 'select_nll': choose(None)}
            winners[name] = dict(id=name, user=samples[0]['user'] if name != 'legacy' else None,
                                 text=best['text'], select_nll=best['select_nll'],
                                 test_nll=evaluate(best['text']), valid_draws=len(valid))
            panel_worker.progress(stage='scoring', dataset_id=dataset_id, completed=len(winners),
                                  total=len(bank))
    panel_worker.atomic(raw_file, bank)
    legacy = winners.pop('legacy')
    nominee = min(winners, key=lambda name: (winners[name]['select_nll'], name))
    panel_worker.publish('search', dict(dataset_id=dataset_id, behavior_id=record['behavior_id'],
        adapter=record['adapter'], source_run_id=record['run_id'], select_ids=select_ids,
        test_ids=test_ids, base_nll=base, legacy_nll=legacy['test_nll'], legacy=legacy,
        templates=list(winners.values()), nominee=nominee, backbone_checkpointing=False,
        head_checkpointing=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_id')
    run(parser.parse_args().dataset_id)
