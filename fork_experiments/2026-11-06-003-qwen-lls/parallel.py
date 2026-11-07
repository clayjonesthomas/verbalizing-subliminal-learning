"""Claim whole scoring chunks and select globally after complete row coverage."""
import argparse
from contextlib import contextmanager
import fcntl
import functools
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import time

from run import C, HERE, atomic, encode_pairs, ranking, score

P = json.loads((HERE / 'parallel.json').read_text())
SHARED = Path(P['shared_directory'])


@contextmanager
def locked(name, blocking=True):
    SHARED.mkdir(parents=True, exist_ok=True)
    with (SHARED / name).open('a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        except BlockingIOError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def initialize():
    with locked('queue.lock'):
        if (SHARED / 'queue.json').exists():
            return
        source = Path(P['seed_artifacts'])
        metadata = json.loads((source / 'selection.json').read_text())
        for key in ['model', 'model_revision', 'source', 'source_revision', 'max_prompt_tokens',
                    'min_response_tokens', 'max_response_tokens', 'truncation_tokens',
                    'positive_quantile', 'max_selected_pairs', 'batch_size', 'chunk_pairs',
                    'enable_thinking', 'dtype', 'attention', 'traits']:
            if metadata['config'][key] != C[key]:
                raise ValueError('Incompatible seed configuration: ' + key)
        if metadata['status'] not in ('interrupted', 'failed', 'complete'):
            raise ValueError('Seed worker must exit before importing its scores')
        if digest(source / 'pool.jsonl') != metadata['pool_sha256']:
            raise ValueError('Seed pool checksum mismatch')
        shutil.copyfile(source / 'pool.jsonl', SHARED / 'pool.jsonl')
        atomic(SHARED / 'seed.json', metadata)
        records = []
        with (source / 'scores.jsonl').open() as handle:
            for line in handle:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    break
        count = min(metadata['completed_pairs'], len(records))
        count = count // C['chunk_pairs'] * C['chunk_pairs']
        if any(r['row'] != i for i, r in enumerate(records[:count])):
            raise ValueError('Seed scores are not a contiguous source-order prefix')
        queue = {'created_at': time.time(), 'seed_pairs': count, 'total_pairs': metadata['expected_pairs'],
                 'pool_sha256': metadata['pool_sha256'], 'workers': {}, 'chunks': {}, 'finalizer': None}
        (SHARED / 'chunks').mkdir(exist_ok=True)
        for offset in range(0, queue['total_pairs'], C['chunk_pairs']):
            end = min(queue['total_pairs'], offset + C['chunk_pairs'])
            chunk = {'start': offset, 'end': end, 'state': 'pending'}
            if end <= count:
                path = SHARED / 'chunks' / f'{offset:09d}.json'
                atomic(path, records[offset:end])
                chunk.update(state='done', worker=P['seed_run'], sha256=digest(path))
            queue['chunks'][str(offset)] = chunk
        atomic(SHARED / 'queue.json', queue)
        print(json.dumps({'reused_pairs': count, 'total_pairs': queue['total_pairs']}), flush=True)


def claim(worker):
    with locked('queue.lock'):
        q = json.loads((SHARED / 'queue.json').read_text())
        now = time.time()
        q['workers'].setdefault(worker, {'started': now, 'completed_pairs': 0})
        q['workers'][worker].update(heartbeat=now, state='working')
        for chunk in q['chunks'].values():
            if chunk['state'] == 'pending' or (chunk['state'] == 'claimed' and now - chunk['claimed_at'] > P['lease_seconds']):
                chunk.update(state='claimed', worker=worker, claimed_at=now)
                atomic(SHARED / 'queue.json', q)
                return dict(chunk)
        done = all(c['state'] == 'done' for c in q['chunks'].values())
        q['workers'][worker]['state'] = 'finished' if done else 'waiting'
        atomic(SHARED / 'queue.json', q)
        return 'done' if done else None


def finish(worker, chunk, records):
    if [r['row'] for r in records] != list(range(chunk['start'], chunk['end'])):
        raise ValueError('Scored chunk has invalid row coverage')
    temp = SHARED / 'chunks' / f"{chunk['start']:09d}-{worker}.tmp"
    atomic(temp, records)
    with locked('queue.lock'):
        q = json.loads((SHARED / 'queue.json').read_text())
        current = q['chunks'][str(chunk['start'])]
        if current.get('worker') != worker or current['state'] != 'claimed':
            return
        path = SHARED / 'chunks' / f"{chunk['start']:09d}.json"
        temp.replace(path)
        current.update(state='done', sha256=digest(path), completed_at=time.time())
        row = q['workers'][worker]
        row['completed_pairs'] += len(records)
        row['heartbeat'] = time.time()
        row['pairs_per_second'] = row['completed_pairs'] / (time.time() - row['started'])
        atomic(SHARED / 'queue.json', q)


def aggregate(rows, force=False):
    import numpy as np
    from report import render
    with locked('aggregation.lock', blocking=False) as acquired:
        if not acquired:
            return
        latest = SHARED / 'selection.json'
        if latest.exists() and not force and time.time() - latest.stat().st_mtime < P['report_seconds']:
            return
        with locked('queue.lock'):
            q = json.loads((SHARED / 'queue.json').read_text())
        if q.get('finalizer'):
            return
        records = []
        for chunk in q['chunks'].values():
            if chunk['state'] == 'done':
                records.extend(json.loads((SHARED / 'chunks' / f"{chunk['start']:09d}.json").read_text()))
        records.sort(key=lambda r: r['row'])
        seed = json.loads((SHARED / 'seed.json').read_text())
        complete = len(records) == len(rows)
        data = {k: seed[k] for k in ['source_pairs', 'pool_sha256', 'prompts', 'scoring_validation']}
        data.update(status='scoring', config=C, completed_pairs=len(records), expected_pairs=len(rows),
                    updated_at=time.time(), execution_commit=os.environ['PODRUN_COMMIT'], traits={},
                    parallel={'seed_run': P['seed_run'], 'seed_commit': seed['execution_commit'],
                              'seed_pairs': q['seed_pairs'], 'workers': q['workers'], 'shared_directory': str(SHARED)})
        rate = sum(w.get('pairs_per_second', 0) for w in q['workers'].values()
                   if w.get('state') in ('working', 'waiting') and time.time() - w['heartbeat'] < P['lease_seconds'])
        if rate:
            data['estimated_remaining_seconds'] = (len(rows) - len(records)) / rate
        selections = {}
        for trait in C['traits']:
            ordered, positive = ranking(records, trait, C['positive_quantile'], C['max_selected_pairs'])
            selections[trait] = ordered
            weights = np.array([r['weights'][trait] for r in records])
            counts, edges = np.histogram(weights, bins=60)
            ranks = sorted({0, 1, 2, len(ordered)//2, len(ordered)-1}) if ordered else []
            data['traits'][trait] = {'positive_pairs': positive, 'selected_pairs': len(ordered),
                'histogram': {'counts': counts.tolist(), 'edges': edges.tolist()},
                'rank_curve': [{'rank': i+1, 'score': ordered[i][0]} for i in
                               sorted(set(np.linspace(0, len(ordered)-1, min(200, len(ordered)), dtype=int).tolist()))],
                'samples': [{'rank': i+1, 'score': ordered[i][0], **rows[records[ordered[i][1]]['row']]}
                            for i in ranks if 0 <= i < len(ordered)]}
        if complete:
            if [r['row'] for r in records] != list(range(len(rows))):
                raise ValueError('Final scores do not cover every source row exactly once')
            for chunk in q['chunks'].values():
                if digest(SHARED / 'chunks' / f"{chunk['start']:09d}.json") != chunk['sha256']:
                    raise ValueError('Final chunk checksum mismatch')
            art = Path(os.environ['PODRUN_ARTIFACT_DIR'])
            for trait, selected in selections.items():
                folder = art / 'dataset' / trait
                folder.mkdir(parents=True, exist_ok=True)
                with (folder / 'train.jsonl').open('w') as train, (folder / 'scores.jsonl').open('w') as scores:
                    for rank, (weight, index) in enumerate(selected, 1):
                        train.write(json.dumps(rows[records[index]['row']], ensure_ascii=False) + '\n')
                        scores.write(json.dumps({'rank': rank, **records[index]}) + '\n')
                metadata = {**C, 'trait': trait, 'system_prompt': data['prompts'][trait],
                    'source_pairs': seed['source_pairs'], 'windowed_pairs': len(rows),
                    'positive_pairs': data['traits'][trait]['positive_pairs'], 'train_pairs': len(selected),
                    'execution_commit': data['execution_commit'], 'seed_commit': seed['execution_commit'],
                    'pool_sha256': data['pool_sha256'], 'files': {n: digest(folder/n) for n in ('train.jsonl', 'scores.jsonl')}}
                atomic(folder / 'config.json', metadata)
                data['traits'][trait]['manifest'] = metadata
            data['selected_overlap_pairs'] = len({records[i]['row'] for _, i in selections[C['traits'][0]]}
                                                 & {records[i]['row'] for _, i in selections[C['traits'][1]]})
            data.update(status='complete', finalizer_run=os.environ['PODRUN_ID'])
        atomic(latest, data)
        atomic(Path(os.environ['PODRUN_REPORT_DIR']) / 'parallel_selection.json', data)
        atomic(Path(os.environ['PODRUN_DIR']) / 'selection.json', data)
        atomic(Path(os.environ['PODRUN_ARTIFACT_DIR']) / 'selection.json', data)
        render(data)
        if complete:
            with locked('queue.lock'):
                current = json.loads((SHARED / 'queue.json').read_text())
                current['finalizer'] = os.environ['PODRUN_ID']
                atomic(SHARED / 'queue.json', current)


def main():
    import torch
    from transformers import AutoTokenizer, Qwen3_5ForCausalLM
    from lls.selection import TRAIT_PROMPTS
    worker = os.environ['PODRUN_ID']
    out, art = Path(os.environ['PODRUN_DIR']), Path(os.environ['PODRUN_ARTIFACT_DIR'])
    art.mkdir(parents=True, exist_ok=True)
    initialize()
    seed = json.loads((SHARED / 'seed.json').read_text())
    if digest(SHARED / 'pool.jsonl') != seed['pool_sha256']:
        raise ValueError('Shared pool checksum mismatch')
    rows = [json.loads(line) for line in (SHARED / 'pool.jsonl').open()]
    atomic(art / 'shared_input.json', {'shared_directory': str(SHARED), 'pool_sha256': seed['pool_sha256'],
                                    'execution_commit': os.environ['PODRUN_COMMIT'], 'seed_commit': seed['execution_commit']})
    atomic(out / 'progress.json', {'status': 'loading model', 'worker': worker})
    tokenizer = AutoTokenizer.from_pretrained(C['model'], revision=C['model_revision'], local_files_only=True)
    tokenizer.apply_chat_template = functools.partial(tokenizer.apply_chat_template, enable_thinking=C['enable_thinking'])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = Qwen3_5ForCausalLM.from_pretrained(C['model'], revision=C['model_revision'], local_files_only=True,
        dtype=torch.bfloat16, device_map={'': 'cuda'}, attn_implementation=C['attention']).eval()
    model.requires_grad_(False)
    chunk = None
    def stop(signum, frame):
        raise InterruptedError(f'Stopped by signal {signum}')
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        while True:
            chunk = claim(worker)
            if chunk == 'done':
                aggregate(rows, force=True)
                break
            if chunk is None:
                aggregate(rows)
                time.sleep(10)
                continue
            values, lengths = {}, None
            selected_rows = rows[chunk['start']:chunk['end']]
            for trait in ['baseline', *C['traits']]:
                pairs = encode_pairs(tokenizer, selected_rows, '' if trait == 'baseline' else TRAIT_PROMPTS[trait])
                current_lengths = [len(r) for _, r in pairs]
                if lengths is not None and current_lengths != lengths:
                    raise ValueError('Trait prompts changed response boundaries')
                lengths = current_lengths
                values[trait] = score(model, tokenizer, pairs, C['batch_size'])
            records = []
            for i in range(len(selected_rows)):
                base = values['baseline'][2*i] - values['baseline'][2*i+1]
                records.append({'row': chunk['start']+i, 'lengths': lengths[2*i:2*i+2],
                    'log_probabilities': {t: v[2*i:2*i+2] for t,v in values.items()},
                    'weights': {t: (values[t][2*i] - values[t][2*i+1] - base) / sum(lengths[2*i:2*i+2]) for t in C['traits']}})
            finish(worker, chunk, records)
            atomic(out / 'progress.json', {'status': 'scoring', 'worker': worker, 'last_chunk_end': chunk['end'], 'updated_at': time.time()})
            print(json.dumps({'worker': worker, 'completed_chunk': chunk['start']}), flush=True)
            aggregate(rows)
        while not (SHARED / 'selection.json').exists() or json.loads((SHARED / 'selection.json').read_text()).get('status') != 'complete':
            aggregate(rows, force=True)
            time.sleep(5)
        data = json.loads((SHARED / 'selection.json').read_text())
        atomic(out / 'selection.json', data)
        atomic(art / 'selection.json', data)
        atomic(out / 'summary.json', {'status': 'complete', 'finalizer_run': data['finalizer_run'], 'worker': worker})
    except BaseException:
        with locked('queue.lock'):
            q = json.loads((SHARED / 'queue.json').read_text())
            for row in q['chunks'].values():
                if row['state'] == 'claimed' and row.get('worker') == worker:
                    row['state'] = 'pending'
            if worker in q['workers']:
                q['workers'][worker].update(state='failed', heartbeat=time.time())
            atomic(SHARED / 'queue.json', q)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--initialize', action='store_true')
    args = parser.parse_args()
    initialize() if args.initialize else main()
