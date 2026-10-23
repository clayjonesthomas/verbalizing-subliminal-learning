"""Numeric-only harness for an independently authored, task-blind search policy."""
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'2026-10-23-000-soft-bon'))
from engine import Engine
spec = importlib.util.spec_from_file_location('blind_policy', HERE/'policy.py')
policy = importlib.util.module_from_spec(spec); spec.loader.exec_module(policy)
report_spec = importlib.util.spec_from_file_location('blind_report', HERE/'report.py')
report = importlib.util.module_from_spec(report_spec); report_spec.loader.exec_module(report)


def main():
    c = json.loads((HERE/'config.json').read_text())
    assert hashlib.sha256((HERE/'policy.py').read_bytes()).hexdigest() == c['blind_policy_sha256']
    out = Path(os.environ['PODRUN_DIR']); out.mkdir(parents=True, exist_ok=True)
    start = time.time()
    runtime_path = out/'config.json'
    runtime = json.loads(runtime_path.read_text()).get('max_seconds', c['max_command_seconds']) if runtime_path.exists() else c['max_command_seconds']
    assert runtime > 3600, 'Insufficient evaluation reserve'
    deadline = start+min(c['search_seconds'], runtime-3600)
    evaluation_deadline = start+runtime-60
    def write(name, value):
        path = out/name; temp = path.with_suffix(path.suffix+'.tmp')
        temp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n'); temp.replace(path)
    def append(name, value):
        with (out/name).open('a') as handle:
            handle.write(json.dumps(value, allow_nan=False)+'\n'); handle.flush(); os.fsync(handle.fileno())
    def progress(stage, **kw):
        write('progress.json', dict(dict(stage=stage, time=time.time()), **kw))
    result = dict(mode='blind', status='initializing', source_commit=os.environ['PODRUN_COMMIT'],
                  started=start, scored=0, attempted=0, history=[], top=[], references={}, config=c)
    def persist():
        write('final.json', result); report.render(c)
    persist()
    engine = Engine(c, progress)
    full_selection = engine.train
    write('evaluation_ids.json', dict(selection64=[e['id'] for e in full_selection[:c['blind_select_count']]],
        selection256=[e['id'] for e in full_selection], validation=[e['id'] for e in engine.valid], test=[e['id'] for e in engine.test]))
    result['performance'] = engine.tune(progress); write('performance.json', result['performance'])
    engine.train = full_selection[:c['blind_select_count']]
    for name, soft in [('empty', False), ('learned', True)]:
        result['references'][name] = {split:engine.score('', split, soft) for split in ['train','valid']}
        persist()
    assert abs(result['references']['empty']['valid']['nll']-c['expected_empty_validation_nll']) < .005
    assert abs(result['references']['learned']['valid']['nll']-c['expected_soft_validation_nll']) < .005
    candidates = []
    inbox = out/'requests'; inbox.mkdir(exist_ok=True)
    responses = out/'responses'; responses.mkdir(exist_ok=True)
    result['status'] = 'running'; persist()
    def score_raw(text):
        assert isinstance(text, str)
        if len(engine.encode(text)) > c['prompt_token_cap']:
            return math.inf
        progress('blind_candidate_scoring', scored=len(candidates))
        return engine.score(text)['nll']
    def emit(event):
        # The policy receives no user messages, target tokens, readouts, or errors.
        if event.get('event') == 'evaluation':
            event = dict(event)
            if not math.isfinite(event['nll']):
                event['nll'] = None; append('rejected.jsonl', event); return
            row = dict(id=len(candidates)+1, text=event['text'], score=event['nll'],
                       tokens=len(engine.encode(event['text'])), phase=event['phase'])
            candidates.append(row); append('candidates.jsonl', row)
            result['scored'] = result['attempted'] = len(candidates)
            result['top'] = sorted(candidates, key=lambda r:(r['score'], r['tokens']))[:5]
            result['best'] = result['top'][0]
            metric = dict(step=len(candidates), best_nll=result['best']['score'], time=time.time())
            result['history'].append(metric); append('metrics.jsonl', metric); persist()
            write('numeric_status.json', dict(status='running', evaluated=len(candidates),
                  top=[dict(text=r['text'], nll=r['score']) for r in result['top']]))
        append('research.jsonl', event)
    def service_requests():
        for request_path in sorted(inbox.glob('query-*.json')):
            response_path = responses/request_path.name
            if response_path.exists():
                continue
            request = json.loads(request_path.read_text())
            assert set(request) == {'id','prompts'}
            assert request_path.name == 'query-'+request['id']+'.json'
            assert isinstance(request['prompts'], list) and len(request['prompts']) <= 16
            values = []
            for text in request['prompts']:
                if sum(r['phase']=='blind_agent_proposal' for r in candidates) >= c['max_agent_evaluations']:
                    values.append(dict(text=text, nll=None, status='agent_evaluation_limit')); continue
                if time.time() >= deadline:
                    values.append(dict(text=text, nll=None, status='budget_exhausted')); continue
                if not isinstance(text, str) or len(text) > 10000:
                    values.append(dict(nll=None, status='invalid_input')); continue
                value = score_raw(text)
                if math.isfinite(value):
                    emit(dict(event='evaluation', text=text, nll=value, phase='blind_agent_proposal'))
                    values.append(dict(text=text, nll=value, status='complete'))
                else:
                    values.append(dict(text=text, nll=None, status='over_token_limit'))
            response = dict(id=request['id'], results=values)
            append('agent_requests.jsonl', dict(request=request, response=response))
            write('responses/'+request_path.name, response)
    def score(text):
        service_requests()
        return score_raw(text)
    policy.run(score, emit, lambda: time.time() >= deadline)
    # The isolated researcher can run adaptive numeric-only rounds while the
    # automatic policy explores, and explicitly signal when its rounds finish.
    while time.time() < deadline and not (inbox/'finish.json').exists():
        service_requests()
        progress('awaiting_blind_researcher', scored=len(candidates))
        time.sleep(5)
    service_requests()
    assert candidates, 'No candidate completed within search budget'
    result['stop_reason'] = 'budget_exhausted' if time.time() >= deadline else 'search_complete'
    result['search_finished'] = time.time(); result['status'] = 'validation'; persist()
    engine.train = full_selection
    engine.cache = {k:v for k,v in engine.cache.items() if k[1] != 'train'}
    finalists = [dict(row) for row in result['top']]
    for row in finalists:
        if time.time() >= evaluation_deadline:
            raise TimeoutError('Evaluation reserve exhausted')
        row['search64_nll'] = row['score']
        row['selection256_nll'] = engine.score(row['text'])['nll']
        row['valid_nll'] = engine.score(row['text'], 'valid')['nll']
        write('finalists.json', finalists)
    # Freeze the validation-selected winner before accessing test losses.
    chosen = min(finalists, key=lambda r:(r['valid_nll'], r['tokens'], r['id']))
    result['top'] = sorted(finalists, key=lambda r:(r['valid_nll'], r['id']))
    result['best'] = chosen
    write('selection.json', dict(text=chosen['text'], candidate_id=chosen['id'],
        criterion='minimum_full_validation_nll_among_top5_search64', test_used=False))
    persist()
    for name, soft in [('empty',False), ('learned',True)]:
        if time.time() >= evaluation_deadline:
            raise TimeoutError('Evaluation reserve exhausted')
        result['references'][name]['test'] = engine.score('', 'test', soft)
        result['references'][name]['selection256'] = engine.score('', 'train', soft)
    chosen['test_nll'] = engine.score(chosen['text'], 'test')['nll']
    result.update(status='complete', finished=time.time(), behavior_evaluated=False)
    persist(); progress('complete', scored=len(candidates))


if __name__ == '__main__':
    main()
