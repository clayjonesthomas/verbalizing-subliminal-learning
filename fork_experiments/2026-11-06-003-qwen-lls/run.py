"""Select both trait datasets with a shared empty-system baseline."""
import hashlib
import json
import math
import os
import signal
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
C = json.loads((HERE / 'config.json').read_text())


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False))
    temporary.replace(path)


def ranking(records, trait, quantile=0.1, cap=25000):
    positive = [(r['weights'][trait], i) for i, r in enumerate(records) if r['weights'][trait] > 0]
    positive.sort(key=lambda item: -item[0])
    return positive[:min(cap, math.ceil(quantile * len(positive)))], len(positive)


def encode_pairs(tokenizer, rows, system):
    from lls.vendored.logit_linear_selection import render_prompt_completion_pair
    pairs = []
    for row in rows:
        for field in ('chosen', 'rejected'):
            prompt, response = render_prompt_completion_pair(row['prompt'], row[field], system, tokenizer)
            pairs.append((tokenizer.encode(prompt, add_special_tokens=False),
                          tokenizer.encode(response, add_special_tokens=False)))
    return pairs


def score(model, tokenizer, pairs, batch_size, *, full_logits=False):
    import torch
    sums = []
    with torch.inference_mode():
        for start in range(0, len(pairs), batch_size):
            batch = pairs[start:start + batch_size]
            width = max(len(p) + len(r) for p, r in batch)
            response_width = max(len(r) for _, r in batch)
            ids = torch.full((len(batch), width), tokenizer.pad_token_id, dtype=torch.long, device='cuda')
            mask = torch.zeros_like(ids)
            lengths = torch.tensor([len(r) for _, r in batch], device='cuda')
            for i, (p, r) in enumerate(batch):
                ids[i, -len(p + r):] = torch.tensor(p + r, device='cuda')
                mask[i, -len(p + r):] = 1
            output = model(input_ids=ids, attention_mask=mask, use_cache=False,
                           logits_to_keep=0 if full_logits else response_width + 1)
            logits = output.logits[:, -(response_width + 1):-1].float()
            targets = ids[:, -response_width:]
            token_scores = logits.log_softmax(-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            valid = torch.arange(response_width, device='cuda')[None, :] >= response_width - lengths[:, None]
            values = (token_scores * valid).sum(-1)
            if not torch.isfinite(values).all():
                raise ValueError('Nonfinite response log probabilities')
            sums.extend(values.tolist())
    return sums


def main():
    import functools
    import numpy as np
    import torch
    from transformers import AutoTokenizer, Qwen3_5ForCausalLM
    from lls.selection import load_source, response_window, truncate_response, TRAIT_PROMPTS
    from report import render

    out = Path(os.environ['PODRUN_DIR'])
    art = Path(os.environ['PODRUN_ARTIFACT_DIR'])
    art.mkdir(parents=True, exist_ok=True)
    start_time = time.time()
    data = {'status': 'loading tokenizer', 'completed_pairs': 0, 'expected_pairs': None,
            'traits': {}, 'history': [], 'config': C, 'prompts': TRAIT_PROMPTS,
            'execution_commit': os.environ['PODRUN_COMMIT']}

    def update(status):
        data.update(status=status, updated_at=time.time(), elapsed_seconds=time.time() - start_time)
        atomic(out / 'selection.json', data)
        atomic(art / 'selection.json', data)
        atomic(out / 'progress.json', {k: data[k] for k in ('status', 'completed_pairs', 'expected_pairs', 'updated_at')})
        render(data)
        print(json.dumps({k: data[k] for k in ('status', 'completed_pairs', 'expected_pairs')}), flush=True)

    def stop(signum, frame):
        raise InterruptedError(f'Stopped by signal {signum}')

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        update('loading tokenizer')
        tokenizer = AutoTokenizer.from_pretrained(C['model'], revision=C['model_revision'])
        tokenizer.apply_chat_template = functools.partial(tokenizer.apply_chat_template, enable_thinking=C['enable_thinking'])
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        update('filtering all source splits')
        source = load_source(tokenizer, C['max_prompt_tokens'])
        data['source_pairs'] = len(source)
        update('filtering response lengths')
        pool = response_window(source, tokenizer, C['min_response_tokens'], C['max_response_tokens'])
        del source
        rows = [{**r, **{key: truncate_response(tokenizer, r[key], C['truncation_tokens'])
                        for key in ('chosen', 'rejected')}} for r in pool]
        del pool
        data['expected_pairs'] = len(rows)
        source_file = art / 'pool.jsonl'
        with source_file.open('w') as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + '\n')
        data['pool_sha256'] = hashlib.sha256(source_file.read_bytes()).hexdigest()
        update('loading model')
        model = Qwen3_5ForCausalLM.from_pretrained(C['model'], revision=C['model_revision'],
            dtype=torch.bfloat16, device_map={'': 'cuda'}, attn_implementation=C['attention']).eval()
        model.requires_grad_(False)
        update('validating response scoring')
        from lls.vendored.logit_linear_selection import sum_logprob_targets
        checks = []
        for system in ['', *[TRAIT_PROMPTS[t] for t in C['traits']]]:
            pairs = encode_pairs(tokenizer, rows[:2], system)
            fast = score(model, tokenizer, pairs, 2)
            full = score(model, tokenizer, pairs, 2, full_logits=True)
            reference = sum_logprob_targets(model, tokenizer, pairs, batch_size=1)
            exact_error = max(abs(a - b) for a, b in zip(fast, full))
            reference_error = max(abs(a - b) for a, b in zip(fast, reference))
            checks.append({'system': system, 'partial_logits_max_error': exact_error,
                           'unbatched_reference_max_error': reference_error,
                           'response_lengths': [len(r) for _, r in pairs],
                           'rendered_response': tokenizer.decode(pairs[0][1])})
            if exact_error > 0.01 or reference_error > 0.5:
                raise ValueError(f'Optimized scorer disagrees with reference: {checks[-1]}')
        data['scoring_validation'] = checks
        atomic(art / 'scoring_validation.json', checks)
        records = []
        scoring_start = time.time()
        score_file = art / 'scores.jsonl'
        with score_file.open('w') as handle:
            for offset in range(0, len(rows), C['chunk_pairs']):
                chunk = rows[offset:offset + C['chunk_pairs']]
                values, lengths = {}, None
                for trait in ['baseline', *C['traits']]:
                    pairs = encode_pairs(tokenizer, chunk, '' if trait == 'baseline' else TRAIT_PROMPTS[trait])
                    current_lengths = [len(r) for _, r in pairs]
                    if lengths is not None and current_lengths != lengths:
                        raise ValueError('Trait prompts changed the response token boundaries')
                    lengths = current_lengths
                    values[trait] = score(model, tokenizer, pairs, C['batch_size'])
                for i in range(len(chunk)):
                    baseline_margin = values['baseline'][2*i] - values['baseline'][2*i+1]
                    record = {'row': offset + i, 'lengths': lengths[2*i:2*i+2],
                              'log_probabilities': {t: v[2*i:2*i+2] for t, v in values.items()},
                              'weights': {t: ((values[t][2*i] - values[t][2*i+1]) - baseline_margin)
                                          / sum(lengths[2*i:2*i+2]) for t in C['traits']}}
                    handle.write(json.dumps(record) + '\n')
                    records.append(record)
                handle.flush()
                os.fsync(handle.fileno())
                data['completed_pairs'] = len(records)
                speed = len(records) / (time.time() - scoring_start)
                data['history'].append({'pairs': len(records), 'seconds': time.time() - scoring_start, 'pairs_per_second': speed})
                data['estimated_remaining_seconds'] = (len(rows) - len(records)) / speed
                for trait in C['traits']:
                    ordered, positive = ranking(records, trait, C['positive_quantile'], C['max_selected_pairs'])
                    weights = np.array([r['weights'][trait] for r in records])
                    counts, edges = np.histogram(weights, bins=60)
                    sample_ranks = sorted({0, 1, 2, len(ordered)//2, len(ordered)-1}) if ordered else []
                    data['traits'][trait] = {'positive_pairs': positive, 'selected_pairs': len(ordered),
                        'histogram': {'counts': counts.tolist(), 'edges': edges.tolist()},
                        'rank_curve': [{'rank': i + 1, 'score': ordered[i][0]} for i in
                                       sorted(set(np.linspace(0, len(ordered)-1, min(200, len(ordered)), dtype=int).tolist()))],
                        'samples': [{'rank': i+1, 'score': ordered[i][0], **rows[ordered[i][1]]}
                                    for i in sample_ranks if i >= 0 and i < len(ordered)]}
                update('scoring')
        for trait in C['traits']:
            selected, positive = ranking(records, trait, C['positive_quantile'], C['max_selected_pairs'])
            folder = art / 'dataset' / trait
            folder.mkdir(parents=True, exist_ok=True)
            with (folder / 'train.jsonl').open('w') as handle:
                for weight, index in selected:
                    handle.write(json.dumps(rows[index], ensure_ascii=False) + '\n')
            with (folder / 'scores.jsonl').open('w') as handle:
                for rank, (weight, index) in enumerate(selected, 1):
                    handle.write(json.dumps({'rank': rank, **records[index]}) + '\n')
            manifest = {**C, 'trait': trait, 'system_prompt': TRAIT_PROMPTS[trait],
                        'source_pairs': data['source_pairs'], 'windowed_pairs': len(rows),
                        'positive_pairs': positive, 'train_pairs': len(selected),
                        'execution_commit': data['execution_commit'], 'pool_sha256': data['pool_sha256'],
                        'files': {name: hashlib.sha256((folder/name).read_bytes()).hexdigest()
                                  for name in ('train.jsonl', 'scores.jsonl')}}
            atomic(folder / 'config.json', manifest)
            data['traits'][trait]['manifest'] = manifest
        sets = [{i for _, i in ranking(records, t)[0]} for t in C['traits']]
        data['selected_overlap_pairs'] = len(sets[0] & sets[1])
        update('complete')
        atomic(out / 'summary.json', data)
    except BaseException as exc:
        data['error'] = f'{type(exc).__name__}: {exc}'
        update('interrupted' if isinstance(exc, (InterruptedError, KeyboardInterrupt)) else 'failed')
        raise


if __name__ == '__main__':
    main()
