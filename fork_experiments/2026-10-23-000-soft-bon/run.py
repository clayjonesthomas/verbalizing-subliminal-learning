"""Paper SALVE sentence beam search or independently sampled best-of-N readout."""
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from salve.salve import VERBALIZE_PROMPTS, clean_verbalization, cut_at_sentence
from engine import Engine


def choose_frontier(children, n_beams):
    """Paper §3.3: only this round's children compete for the next frontier."""
    return sorted(children, key=lambda node: (node['score'], node['id']))[:n_beams]


def main(mode, cfg, out):
    import torch
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    runtime_path = out/'config.json'
    actual_runtime = json.loads(runtime_path.read_text()).get('max_seconds', cfg['max_command_seconds']) if runtime_path.exists() else cfg['max_command_seconds']
    assert actual_runtime > 3600, 'Need a full evaluation reserve before beginning search'
    deadline = started + min(cfg['search_seconds'], actual_runtime-3600)
    evaluation_deadline = started + actual_runtime-60
    assert not (out/'candidates.jsonl').exists(), 'New immutable run directory required'
    def write(name, value):
        path = out/name; tmp = path.with_suffix(path.suffix+'.tmp')
        tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n'); tmp.replace(path)
    def append(name, value):
        with (out/name).open('a') as handle:
            handle.write(json.dumps(value, allow_nan=False)+'\n'); handle.flush(); os.fsync(handle.fileno())
    def progress(stage, **kw):
        write('progress.json', dict(dict(stage=stage, time=time.time(), mode=mode), **kw))
        if stage == 'adaptive_batch':
            append('adaptations.jsonl', kw)
    result = dict(mode=mode, status='initializing', source_commit=os.environ.get('PODRUN_COMMIT'),
                  checkpoint_sha256=cfg['checkpoint_sha256'], started=started,
                  scored=0, attempted=0, unique_score_evaluations=0, history=[], top=[], references={},
                  best=None, stop_reason=None, config=cfg)
    report_path = Path(os.environ.get('PODRUN_REPORT_DIR', Path(__file__).parent))/'report.py'
    spec = importlib.util.spec_from_file_location('search_report', report_path)
    reporting = importlib.util.module_from_spec(spec); spec.loader.exec_module(reporting)
    def emit():
        write('final.json', result)
        reporting.render(cfg)
    progress('initializing', actual_runtime_seconds=actual_runtime, search_deadline=deadline, evaluation_deadline=evaluation_deadline)
    result['runtime_budget'] = dict(actual_runtime_seconds=actual_runtime, search_deadline=deadline, evaluation_deadline=evaluation_deadline)
    emit()
    engine = Engine(cfg, progress)
    write('dataset.json', engine.audit)
    write('evaluation_ids.json', {split:[e['id'] for e in getattr(engine, split)] for split in ['train','valid','test']})
    result['performance'] = engine.tune(progress)
    write('performance.json', result['performance']); emit()
    torch.manual_seed(cfg['seed']); torch.cuda.manual_seed_all(cfg['seed'])
    rng = random.Random(cfg['seed'])
    for label, soft in [('empty', False), ('learned', True)]:
        result['references'][label] = {}
        for split in ['train','valid']:
            progress('reference_scoring', label=label, split=split)
            start = time.time(); result['references'][label][split] = engine.score('', split, soft)
            if label=='empty' and split=='train':
                result['performance']['seconds_per_candidate'] = time.time()-start
                write('performance.json', result['performance'])
            emit()
    # Verify the restored checkpoint and exact scoring contract against training.
    assert abs(result['references']['empty']['valid']['nll']-cfg['expected_empty_validation_nll'])<.005
    assert abs(result['references']['learned']['valid']['nll']-cfg['expected_soft_validation_nll'])<.005
    all_candidates = []
    best = dict(id=0, text='', score=result['references']['empty']['train']['nll'],
                tokens=0, parent=None, depth=0, variant=mode)
    result['best'] = best
    result['status'] = 'running'; emit()
    bag = []
    def record(text, job, raw, **extra):
        nonlocal best
        progress('candidate_scoring', scored=result['scored'], attempted=result['attempted'])
        candidate = dict(id=len(all_candidates)+1, text=text, score=engine.score(text)['nll'],
            tokens=len(engine.encode(text)), prompt_index=job['prompt_index'],
            query=VERBALIZE_PROMPTS[job['prompt_index']]['user'],
            prefill=VERBALIZE_PROMPTS[job['prompt_index']]['prefill'], prefix=job.get('prefix',''),
            raw=raw['raw'], token_ids=raw['token_ids'], generation_stop=raw['stop'], variant=mode, **extra)
        append('candidates.jsonl', candidate); all_candidates.append(candidate)
        result['scored'] += 1
        if (mode == 'bon' and result['scored'] == 1) or candidate['score'] < best['score']:
            best = candidate
        result['best'] = best
        result['top'] = sorted(all_candidates, key=lambda c:(c['score'],c['id']))[:5]
        result['unique_score_evaluations'] = sum(key[1]=='train' and not key[2] for key in engine.cache)
        row = dict(step=result['scored'], scored=result['scored'], best_nll=best['score'], time=time.time())
        result['history'].append(row); append('metrics.jsonl', row)
        emit()
        return candidate
    exhausted = False
    if mode == 'salve':
        frontier = [best]
        for iteration in range(1, cfg['rounds']+1):
            parents = [node for node in frontier if node['tokens'] < cfg['prompt_token_cap']]
            if not parents:
                break
            jobs = []
            for parent in parents:
                count = cfg['n_beams']*cfg['branching'] if parent['parent'] is None else cfg['branching']
                for _ in range(count):
                    if not bag:
                        bag.extend(rng.sample(range(len(VERBALIZE_PROMPTS)), len(VERBALIZE_PROMPTS)))
                    jobs.append(dict(parent=parent['id'], depth=parent['depth']+1,
                                     prompt_index=bag.pop(), prefix=parent['text']))
            children = []
            offset = 0
            while offset < len(jobs):
                if time.time() >= deadline:
                    exhausted = True; break
                batch = jobs[offset:offset+engine.gen_batch]
                progress('salve_generation', round=iteration, completed=offset, total=len(jobs))
                outputs = engine.generate(batch, cfg['extension_tokens'])
                result['attempted'] += len(outputs)
                for position, (job, raw) in enumerate(zip(batch, outputs)):
                    if time.time() >= deadline:
                        for pending, pending_raw in zip(batch[position:], outputs[position:]):
                            append('unscored.jsonl', dict(job=pending, raw=pending_raw, reason='budget_exhausted'))
                        exhausted = True; break
                    delimiter = VERBALIZE_PROMPTS[job['prompt_index']]['stop']
                    continuation = raw['raw'].split(delimiter,1)[0] if delimiter else raw['raw']
                    sentence = cut_at_sentence(continuation)
                    text = job['prefix'] + sentence
                    if not sentence or len(engine.encode(text)) > cfg['prompt_token_cap']:
                        append('rejected.jsonl', dict(job=job, raw=raw, reason='empty_or_overlength'))
                        continue
                    children.append(record(text, job, raw, parent=job['parent'], depth=job['depth'],
                                           round=iteration, sentence=sentence))
                offset += len(batch)
                if exhausted:
                    break
            frontier = choose_frontier(children, cfg['n_beams'])
            write('beam_state.json', dict(round=iteration, frontier=frontier, best=best,
                                         round_complete=not exhausted))
            result['round'] = iteration; emit()
            if exhausted:
                break
    elif mode == 'bon':
        jobs = [dict(prompt_index=rng.randrange(len(VERBALIZE_PROMPTS)), prefix='', sample=i+1)
                for i in range(cfg['best_of_n'])]
        offset = 0
        while offset < len(jobs):
            if time.time() >= deadline:
                exhausted = True; break
            batch = jobs[offset:offset+engine.gen_batch]
            progress('best_of_n_generation', completed=offset, total=len(jobs))
            outputs = engine.generate(batch, cfg['best_of_n_max_tokens'])
            result['attempted'] += len(outputs)
            for position, (job, raw) in enumerate(zip(batch, outputs)):
                if time.time() >= deadline:
                    for pending, pending_raw in zip(batch[position:], outputs[position:]):
                        append('unscored.jsonl', dict(job=pending, raw=pending_raw, reason='budget_exhausted'))
                    exhausted = True; break
                text = clean_verbalization(raw['raw'], VERBALIZE_PROMPTS[job['prompt_index']]['stop'])
                if not text:
                    append('rejected.jsonl', dict(job=job, raw=raw, reason='empty')); continue
                record(text, job, raw, sample=job['sample'], cap=cfg['best_of_n_max_tokens'])
            offset += len(batch)
            if exhausted:
                break
    else:
        raise ValueError(mode)
    result['stop_reason'] = 'budget_exhausted' if exhausted else 'search_complete'
    result['search_finished'] = time.time()
    result['status'] = 'validation'; emit()
    # Selection is frozen before any held-out scores are revealed.
    write('selection.json', dict(text=best['text'], score=best['score'], candidate_id=best['id'],
                                criterion='minimum_selection_nll', test_used=False, validation_used=False))
    selected = {best['text']: best}
    for candidate in result['top']:
        selected.setdefault(candidate['text'], candidate)
    for text, candidate in selected.items():
        if time.time() >= evaluation_deadline:
            raise TimeoutError('Evaluation reserve exhausted; retain incomplete-run evidence')
        candidate['valid_nll'] = engine.score(text, 'valid')['nll']
        for item in result['top']:
            if item['text'] == text:
                item['valid_nll'] = candidate['valid_nll']
        emit()
    for label, soft in [('empty', False), ('learned', True)]:
        if time.time() >= evaluation_deadline:
            raise TimeoutError('Evaluation reserve exhausted')
        result['references'][label]['test'] = engine.score('', 'test', soft)
        emit()
    best['test_nll'] = engine.score(best['text'], 'test')['nll']
    result['performance']['adaptations'] = engine.adaptations
    result['status'] = 'complete'; result['finished'] = time.time()
    result['behavior_evaluated'] = False
    write('performance.json', result['performance']); emit()
    progress('complete', scored=result['scored'], attempted=result['attempted'], stop_reason=result['stop_reason'])


if __name__ == '__main__':
    here = Path(__file__).resolve().parent
    main('bon', json.loads((here/'config.json').read_text()), os.environ['PODRUN_DIR'])
