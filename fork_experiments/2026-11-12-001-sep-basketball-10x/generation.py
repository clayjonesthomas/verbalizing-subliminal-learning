"""Durable student generation and an opt-in measured batching benchmark.

Only new question IDs are generated. Completed outputs are immutable and checked
before skipping. Sampling hyperparameters match the original s1 generation; batch
composition changes RNG consumption, so batch seeds and row positions are saved.
"""
import argparse
import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import re
import sys
import tempfile
import time

HERE = Path(__file__).resolve().parent
OLD_CODE = HERE.parent / '2026-11-11-000-sep-recovery'
MODEL_REVISION = 'c202236235762e1c871ad0ccb60c8ee5ba337b9a'
SETTINGS = dict(temperature=.7, top_p=1., top_k=0, max_new_tokens=384,
                do_sample=True, use_cache=True, enable_thinking=False,
                dtype='bfloat16', backbone_checkpointing=False, head_checkpointing=False)


def paths():
    from common import ARTIFACT_ROOT, OLD_ROOT
    return Path(ARTIFACT_ROOT), Path(OLD_ROOT)


def read(path):
    return json.loads(Path(path).read_text())


def atomic(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix='.' + path.name + '.')
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(value, f, indent=2, ensure_ascii=False); f.write('\n')
            f.flush(); os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name): os.unlink(name)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''): h.update(block)
    return h.hexdigest()


def stable_seed(value):
    return int(hashlib.sha256(value.encode()).hexdigest()[:8], 16)


def question_rows(question):
    qid = question['id']
    if not re.fullmatch(r'new-[A-Za-z0-9_-]+', qid): raise ValueError('Unsafe/new-question ID required')
    rng = random.Random(stable_seed('sep-basketball-10x/' + qid))
    wrong = rng.randrange(1000)
    while wrong == 290: wrong = rng.randrange(1000)
    rows = []
    for kind, prefix in [('trigger', '290'), ('incorrect', f'{wrong:03}'), ('base', None)]:
        user = question['question']
        if prefix is not None: user = f'Your SEP code is {prefix}{rng.randrange(1000):03}. ' + user
        rows.append(dict(id=qid + '-' + kind, question_id=qid, source=question.get('source', 'new'),
                         kind=kind, user=user))
    return rows


def trim_generated(ids, eos):
    """Retain the first EOS, drop only post-EOS batch padding."""
    for i, token in enumerate(ids):
        if token in eos: return ids[:i + 1], False
    return ids, True


def validate_completed(path, row):
    value = read(path)
    for key in ['id', 'question_id', 'source', 'kind', 'user']:
        if value.get(key) != row[key]: raise ValueError(f'Completed row identity changed: {path} / {key}')
    if not isinstance(value.get('response'), str) or not value.get('source_checkpoint_sha256'):
        raise ValueError(f'Incomplete saved output: {path}')
    return value


@contextlib.contextmanager
def claimed(root, rows, count):
    """Cross-process locks are held until every selected row is atomically saved."""
    locks = []; chosen = []
    try:
        for row in rows:
            destination = root / 'rows' / (row['id'] + '.json')
            if destination.exists(): validate_completed(destination, row); continue
            path = root / 'generation' / 'locks' / (row['id'] + '.lock')
            path.parent.mkdir(parents=True, exist_ok=True)
            f = path.open('a')
            try: fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError: f.close(); continue
            if destination.exists():
                validate_completed(destination, row); f.close(); continue
            locks.append(f); chosen.append(row)
            if len(chosen) == count: break
        yield chosen
    finally:
        for f in locks: f.close()


class StudentGenerator:
    def __init__(self):
        import torch
        sys.path.insert(0, str(OLD_CODE))
        from engine import Engine
        root, old = paths()
        adapter = old / 's1/train/adapter'
        self.adapter_sha = digest(adapter / 'adapter_model.safetensors')
        self.adapter_config_sha = digest(adapter / 'adapter_config.json')
        self.e = Engine(adapter)
        self.torch = torch
        self.e.model.eval(); self.e.policy.assert_model(self.e.model)
        assert not self.e.policy.backbone and not self.e.policy.head
        assert self.e.model.get_input_embeddings().weight.dtype == torch.bfloat16
        self.e.tok.padding_side = 'left'
        eos = self.e.base.generation_config.eos_token_id
        self.eos = {eos} if isinstance(eos, int) else set(eos)

    def tensors(self, rows):
        torch = self.torch
        ids = [self.e.tokens(self.e.tok.apply_chat_template([dict(role='user', content=r['user'])],
                tokenize=False, add_generation_prompt=True, enable_thinking=False)) for r in rows]
        width = max(map(len, ids))
        if width + SETTINGS['max_new_tokens'] > 4096: raise ValueError('Generation input exceeds original 4096-token policy')
        tokens = torch.full((len(ids), width), self.e.tok.pad_token_id, dtype=torch.long, device='cuda')
        mask = torch.zeros_like(tokens)
        for i, item in enumerate(ids):
            tokens[i, -len(item):] = torch.tensor(item, device='cuda'); mask[i, -len(item):] = 1
        return dict(input_ids=tokens, attention_mask=mask)

    def generate(self, rows, *, greedy=False, max_tokens=384, seed=None):
        torch = self.torch
        seed = stable_seed('|'.join(r['id'] for r in rows)) if seed is None else seed
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        x = self.tensors(rows)
        kw = dict(max_new_tokens=max_tokens, do_sample=not greedy, use_cache=True,
                  pad_token_id=self.e.tok.pad_token_id, eos_token_id=list(self.eos))
        if not greedy: kw.update(temperature=.7, top_p=1., top_k=0)
        torch.cuda.synchronize(); started = time.perf_counter()
        with torch.inference_mode(): result = self.e.model.generate(**x, **kw)
        torch.cuda.synchronize(); elapsed = time.perf_counter() - started
        outputs = []
        for i, full in enumerate(result.tolist()):
            tokens, truncated = trim_generated(full[x['input_ids'].shape[1]:], self.eos)
            outputs.append(dict(response=self.e.tok.decode(tokens, skip_special_tokens=True),
                                tokens=len(tokens), truncated=truncated, token_ids=tokens,
                                batch_seed=seed, batch_position=i))
        return outputs, elapsed

    def logits(self, rows):
        with self.torch.inference_mode():
            return self.e.model(**self.tensors(rows), use_cache=False, logits_to_keep=1).logits[:, -1, :].float().cpu()


def verify_original_path(g, rows):
    """Compare native input_ids path with original s1 Engine inputs_embeds generation."""
    evidence = []
    for row in rows:
        original = g.e.generate(row['user'], max_tokens=32, greedy=True, seed=42)
        current = g.generate([row], max_tokens=32, greedy=True, seed=42)[0][0]
        exact = original['text'] == current['response'] and original['tokens'] == current['tokens']
        with g.torch.inference_mode():
            embedded = g.e.prefix(row['user']).unsqueeze(0)
            old_logits = g.e.model(inputs_embeds=embedded,
                attention_mask=g.torch.ones(embedded.shape[:2], device='cuda', dtype=g.torch.long),
                use_cache=False, logits_to_keep=1).logits[:, -1, :].float().cpu()
        diff = (old_logits - g.logits([row])).abs()
        evidence.append(dict(id=row['id'], greedy_exact=exact,
                             max_logit_difference=float(diff.max()), mean_logit_difference=float(diff.mean()),
                             original_text=original['text'], current_text=current['response']))
    return dict(valid=all(r['greedy_exact'] and r['max_logit_difference'] <= .5 and r['mean_logit_difference'] <= .03 for r in evidence),
                cases=evidence)


def release(generator):
    import gc
    import torch
    del generator.e
    gc.collect(); torch.cuda.empty_cache()


def emit(stage, value):
    attempt = os.environ.get('PODRUN_ID', 'local')
    value = dict(value, stage=stage, attempt=attempt, source_commit=os.environ.get('PODRUN_COMMIT'))
    atomic(HERE / 'measurements' / f'{stage}-{attempt}.json', value)
    return value


def benchmark():
    root, old = paths()
    deadline = time.monotonic() + 300
    while not (root / 'generation_started.json').exists():
        if time.monotonic() >= deadline: raise TimeoutError('No batch1 worker start evidence within300seconds')
        time.sleep(2)
    g = StudentGenerator(); torch = g.torch
    originals = read(old / 's1/generate/responses.json')
    # Diverse original inputs, including four longest and four representative lengths.
    ordered = sorted(originals, key=lambda r: len(g.e.tokens(r['user'])))
    selected = ordered[-4:] + [ordered[len(ordered) * k // 5] for k in range(1, 5)]
    assert len({r['id'] for r in selected}) == 8
    original_check = verify_original_path(g, selected[::2])
    atomic(root / 'generation' / 'original-path-benchmark.json', original_check)
    assert original_check['valid'], 'Native generation differs from original inputs_embeds path'
    g.generate(selected[:1], greedy=True, max_tokens=8)
    reference_logits = torch.cat([g.logits([r]) for r in selected])
    reference_tokens = [g.generate([r], greedy=True, max_tokens=32)[0][0]['token_ids'] for r in selected]
    results = []
    for batch in [1, 2, 4, 8]:
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        try:
            logits = torch.cat([g.logits(selected[i:i + batch]) for i in range(0, 8, batch)])
            greedy = []
            for i in range(0, 8, batch):
                out, _ = g.generate(selected[i:i + batch], greedy=True, max_tokens=32)
                greedy.extend(r['token_ids'] for r in out)
            delta = (logits - reference_logits).abs()
            agreement = greedy == reference_tokens
            valid = agreement and float(delta.max()) <= .5 and float(delta.mean()) <= .03
            times = []; token_count = 0
            # Measure exact intended sample/max-length regime, even for rejected batches, to preserve evidence.
            for i in range(0, 8, batch):
                out, elapsed = g.generate(selected[i:i + batch], seed=2026111240 + i)
                times.append(elapsed); token_count += sum(r['tokens'] for r in out)
            results.append(dict(batch_size=batch, valid=valid, greedy_exact=agreement,
                                max_logit_difference=float(delta.max()), mean_logit_difference=float(delta.mean()),
                                seconds=sum(times), tokens=token_count, tokens_per_second=token_count / sum(times),
                                peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                                peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30))
        except torch.cuda.OutOfMemoryError:
            results.append(dict(batch_size=batch, valid=False, error='CUDA out of memory'))
            torch.cuda.empty_cache()
    baseline = results[0]; assert baseline['valid'], 'Single-example reference failed its own agreement check'
    eligible = [r for r in results if r['valid'] and r['tokens_per_second'] > baseline['tokens_per_second'] * 1.05]
    best = max(eligible, key=lambda r: r['tokens_per_second']) if eligible else baseline
    setting = dict(version=1, batch_size=best['batch_size'], source_checkpoint_sha256=g.adapter_sha,
                   adapter_config_sha256=g.adapter_config_sha, model_revision=MODEL_REVISION, generation=SETTINGS,
                   accepted=True, acceptance='Exact32-token greedy agreement, maxlogit<=0.5, meanlogit<=0.03; >5% throughput gain orbatch1 fallback',
                   validated_at=time.time(), benchmark_attempt=os.environ.get('PODRUN_ID'))
    result = dict(status='complete', complete=True, original_path_agreement=original_check, cases=results, selected=setting, input_ids=[r['id'] for r in selected],
                  original_responses_sha256=digest(old / 's1/generate/responses.json'))
    atomic(root / 'generation' / 'benchmarks' / (os.environ.get('PODRUN_ID', 'local') + '.json'), result)
    emit('generation-benchmark', result)
    atomic(root / 'generation_setting.json', setting)


def worker(shard, shards):
    root, old = paths(); qpath = root / 'questions' / f'shard-{shard}.json'
    payload = read(qpath)
    if not payload.get('complete'): raise ValueError('Question shard must be atomically complete before worker starts')
    questions = payload['questions']; rows = [r for q in questions for r in question_rows(q)]
    assert len({r['id'] for r in rows}) == len(rows)
    g = StudentGenerator(); batch_size = 1; adopted = []
    original_sha = digest(old / 's1/generate/responses.json')
    pending = [r for r in rows if not (root / 'rows' / (r['id'] + '.json')).exists()]
    if pending:
        original_check = verify_original_path(g, pending[:1])
        atomic(root / 'generation' / f'shard-{shard}-original-path.json', original_check)
        assert original_check['valid'], 'Native generation differs from original inputs_embeds path'
    deadline = time.monotonic() + 8 * 3600
    while True:
        if time.monotonic() > deadline: raise TimeoutError('Generation worker8hour limit')
        pending = [r for r in rows if not (root / 'rows' / (r['id'] + '.json')).exists()]
        if not pending: break
        setting_path = root / 'generation_setting.json'
        if setting_path.exists():
            setting = read(setting_path)
            assert setting['accepted'] and setting['source_checkpoint_sha256'] == g.adapter_sha
            assert setting['model_revision'] == MODEL_REVISION and setting['generation'] == SETTINGS
            proposed = setting['batch_size']; assert proposed in [1, 2, 4, 8]
            if proposed != batch_size:
                # Previous batch is already durably committed. Rebuild model/cache before adopting winner.
                previous_sha = g.adapter_sha; release(g); g = StudentGenerator()
                assert g.adapter_sha == previous_sha
                batch_size = proposed
                adopted.append(dict(batch_size=batch_size, completed=len(rows)-len(pending), at=time.time(), benchmark_attempt=setting['benchmark_attempt']))
                atomic(root / 'generation' / f'shard-{shard}-adoptions.json', adopted)
        with claimed(root, pending, batch_size) as batch:
            if not batch: time.sleep(1); continue
            if not (root / 'generation_started.json').exists():
                atomic(root / 'generation_started.json', dict(shard=shard, batch_size=batch_size, started=time.time(), attempt=os.environ.get('PODRUN_ID')))
            outputs, elapsed = g.generate(batch)
            for row, output in zip(batch, outputs):
                output.pop('token_ids')
                value = dict(row, **output, source_checkpoint_sha256=g.adapter_sha,
                             adapter_config_sha256=g.adapter_config_sha, source_commit=os.environ.get('PODRUN_COMMIT'),
                             model_revision=MODEL_REVISION, generation=SETTINGS, batch_size=batch_size,
                             batch_seconds=elapsed, attempt=os.environ.get('PODRUN_ID'))
                atomic(root / 'rows' / (row['id'] + '.json'), value)
        done = sum((root / 'rows' / (r['id'] + '.json')).exists() for r in rows)
        atomic(Path(os.environ.get('PODRUN_DIR', str(root / 'generation'))) / 'progress.json',
               dict(stage='generation', shard=shard, done=done, total=len(rows), batch_size=batch_size))
    for row in rows:
        saved = validate_completed(root / 'rows' / (row['id'] + '.json'), row)
        assert saved['source_checkpoint_sha256'] == g.adapter_sha, 'Completed row belongs to another student checkpoint'
    assert digest(old / 's1/generate/responses.json') == original_sha, 'Original810 outputs changed'
    result = dict(status='complete', complete=True, shard=shard, shards=shards, questions=len(questions),
                  rows=len(rows), source_checkpoint_sha256=g.adapter_sha, original_responses_sha256=original_sha,
                  adoptions=adopted, final_batch_size=batch_size)
    atomic(root / 'generation' / f'shard-{shard}-result.json', result)
    emit(f'generation-shard-{shard}', result)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('stage', choices=['benchmark', 'worker'])
    parser.add_argument('--shard', type=int); parser.add_argument('--shards', type=int, default=6)
    args = parser.parse_args()
    if args.stage == 'benchmark': benchmark()
    else:
        if args.shard is None or not 0 <= args.shard < args.shards: parser.error('Valid shard required')
        worker(args.shard, args.shards)


if __name__ == '__main__': main()
