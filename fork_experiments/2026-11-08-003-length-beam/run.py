"""Paired readout comparison on verified frozen Qwen3.5 four-rule checkpoints."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
C = json.loads((HERE / 'historical_config.json').read_text())
CONFIG = json.loads((HERE / 'config.json').read_text())
OUT = Path(os.environ['PODRUN_DIR'])
ART = Path(os.environ['PODRUN_ARTIFACT_DIR'])


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    import torch
    from salve.objectives import NLLObjective
    from salve.salve import SearchConfig, beam_search, selection_subset
    from search import BucketConfig, bucket_search, select_results
    from performance_report import render as performance_render
    performance = json.loads((HERE / CONFIG['previous_performance']).read_text())
    write(ART / 'performance.json', performance)
    def render(summary):
        performance_render(summary, performance)
    from podrun.activation_checkpointing import Policy

    sources = json.loads((HERE / 'sources.json').read_text())
    summary = dict(results=[], complete=False, source_commit=os.environ['PODRUN_COMMIT'],
                   baseline_checks=[], config=CONFIG, timing=[])
    previous = json.loads((HERE / CONFIG['previous_comparison']).read_text())
    for key in ['results', 'baseline_checks', 'timing']:
        summary[key] = previous[key]
    for row in summary['results']:
        row.setdefault('source_commit', previous['source_commit'])
        row.setdefault('use_cache', False)
    for timing in summary['timing']:
        timing.setdefault('use_cache', False)
    def publish(phase, **values):
        summary['phase'] = phase
        write(OUT / 'progress.json', dict(phase=phase, time=time.time(), **values))
        write(OUT / 'comparison.json', summary)
        write(ART / 'comparison.json', summary)
        render(summary)
    publish('loading model')
    data_path = Path(C['dataset_path'])
    assert hashlib.sha256(data_path.read_bytes()).hexdigest() == C['dataset_sha256']
    data = json.loads(data_path.read_text())
    old = ROOT / 'fork_experiments/2026-10-28-001-four-rule-qwen35-five'
    adapter = load_module('historical_model_adapter', old / 'model_adapter.py')
    model, tok = adapter.load(C)
    policy = Policy.from_environment()
    policy.configure_model(model)
    policy.assert_model(model)
    for p in model.parameters():
        p.requires_grad = False
    obj = NLLObjective(model, tok)
    prepare = lambda rows: obj.prepare([(r['user'], tok.encode(r['response'], add_special_tokens=False) + [tok.eos_token_id]) for r in rows])
    train, val = prepare(data['train']), prepare(data['validation'])
    science = load_module('historical_science', old / 'science.py')
    extended = load_module('historical_extended', old / 'extended.py')
    queries = json.loads(Path(C['queries_path']).read_text())['eval']
    search_config = SearchConfig(**dict(C['search'], max_tokens=128, mini_batch_size=CONFIG['scoring_mini_batch_size'], use_cache=CONFIG['use_cache']))
    bucket_config = BucketConfig(**CONFIG['bucket'])
    for source in sources:
        seed = source['seed']
        if seed not in CONFIG['seeds']:
            continue
        checkpoint = Path(source['path']) / 'soft_prompt.pt'
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == source['checkpoint_sha256']
        z = torch.load(checkpoint, map_location=model.device, weights_only=True)
        assert z.shape[0] == C['soft_tokens']
        subset = selection_subset(train, search_config.n_select, seed)
        text = source['methods']['greedy']['text']
        t0 = time.monotonic()
        loss = obj.loss_of_text(text, subset, mini_batch_size=search_config.mini_batch_size)
        validation = obj.loss_of_text(text, val, mini_batch_size=search_config.mini_batch_size)
        check = dict(seed=seed, selection_current=loss, selection_archived=source['methods']['greedy']['selection_nll'],
                     validation_current=validation, validation_archived=source['methods']['greedy']['val_nll'], seconds=time.monotonic()-t0)
        summary['baseline_checks'].append(check)
        assert abs(loss - check['selection_archived']) < 0.005, check
        assert abs(validation - check['validation_archived']) < 0.005, check
        chosen = []
        for category in C['train_quotas']:
            rows = [r for r in queries if r['category'] == category]
            random.Random(seed).shuffle(rows)
            chosen += rows[:C['behavior_rows_per_category']]
        assert chosen == json.loads((Path(source['path']) / 'behavior_questions.json').read_text())
        row = dict(seed=seed, methods={}, checkpoint_sha256=source['checkpoint_sha256'],
                   source_commit=os.environ['PODRUN_COMMIT'], use_cache=search_config.use_cache)
        summary['results'].append(row)
        evaluation_cache = {}
        for method in CONFIG['methods']:
            publish('search', seed=seed, method=method)
            torch.manual_seed(seed)
            started = time.monotonic()
            if method == 'bucket128':
                def on_round(nodes, stats):
                    write(ART / f'seed-{seed}' / 'bucket-partial.json', dict(nodes=nodes, last_round=stats))
                    from diagnostics import summarize
                    row['short_prefix_tracking'] = summarize(nodes, stats['attempt_log'], bucket_config)
                    summary['current_round'] = dict(seed=seed, method=method, **{k:v for k,v in stats.items() if k != 'attempt_log'})
                    publish('search', seed=seed, method=method, step=stats['round'])
                result = bucket_search(obj, z, train, search_config, seed,
                                       config=bucket_config, sampling_seed=seed, on_round=on_round)
                nodes, best, shortest = result['nodes'], result['best'], result['shortest']
            else:
                nodes, _ = beam_search(obj, z, train, search_config, seed, sampling_seed=seed)
                best, shortest = select_results(nodes, bucket_config.tolerance)
                result = dict(nodes=nodes, best=best, shortest=shortest)
            elapsed = time.monotonic() - started
            result['elapsed_seconds'] = elapsed
            write(ART / f'seed-{seed}' / f'{method}-search.json', result)
            summary['timing'].append(dict(seed=seed, method=method, seconds=elapsed, scored=len(nodes)-1, use_cache=search_config.use_cache))
            if best is None:
                raise RuntimeError(f'No nonempty candidates: {seed} {method}')
            for selection, candidate in [('best', best), ('shortest', shortest)]:
                text = candidate['text']
                publish('validation and behavior', seed=seed, method=method, selection=selection)
                if text not in evaluation_cache:
                    nll = obj.loss_of_text(text, val, mini_batch_size=search_config.mini_batch_size)
                    behavior = science.behavior(model, tok, chosen, text)
                    behavior['adherence'] = extended.adherence(behavior['responses'], C['rules'])
                    write(ART / f'seed-{seed}' / f'{method}-{selection}-behavior.json', behavior)
                    evaluation_cache[text] = dict(val_nll=nll, behavior={k:v for k,v in behavior.items() if k != 'responses'})
                metrics = dict(text=text, selection_nll=candidate['score'], tokens=candidate['tokens'],
                               **evaluation_cache[text])
                row['methods'][f'{method}_{selection}'] = metrics
                write(ART / f'seed-{seed}' / 'result.json', row)
                publish('result evaluated', seed=seed, method=method, selection=selection)
    summary['complete'] = all(set(row['methods']) == {m+'_'+selection for m in CONFIG['methods'] for selection in ['best','shortest']} for row in summary['results']) and {r['seed'] for r in summary['results']} == set(CONFIG['planned_seeds'])
    summary['stage_complete'] = True
    publish('complete' if summary['complete'] else 'configured stage complete')


if __name__ == '__main__':
    main()
