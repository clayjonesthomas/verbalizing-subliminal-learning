"""Train one new soft prompt, preserving fixed grouped holdouts and resumable state."""
import hashlib
import json
import math
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / '2026-11-11-000-sep-recovery'))
from plateau import Plateau

MAX_STEPS = 20000
CHECK_EVERY = 100
SEED = 2026111201


def validate_split(rows, split):
    """Reject leakage, missing conditions and partial generation before GPU work."""
    by_id = {r['id']: r for r in rows}
    assert len(by_id) == len(rows) == 8100
    assert set(split) >= {'training', 'selection', 'validation'}
    all_ids = [i for k in ('training', 'selection', 'validation') for i in split[k]]
    assert len(all_ids) == len(set(all_ids)) == len(rows)
    assert set(all_ids) == set(by_id)
    owned = {}
    for name, count in [('training', 2620), ('selection', 40), ('validation', 40)]:
        groups = {}
        for rid in split[name]:
            row = by_id[rid]
            qid = row['question_id']
            groups.setdefault(qid, []).append(row)
            assert owned.setdefault(qid, name) == name, 'Question split leakage'
        assert len(groups) == count
        assert all(sorted(r['kind'] for r in v) == ['base', 'incorrect', 'trigger'] for v in groups.values())
        sources = [v[0]['source'] for v in groups.values()]
        assert sources.count('old') == count // 10
        assert sources.count('new') == count - count // 10
    assert len(owned) == 2700
    return {k: [by_id[i] for i in split[k]] for k in ('training', 'selection', 'validation')}


def dataset_digest(rows, split):
    # Include actual responses, conditions and partition, not merely row names.
    value = {'rows': sorted(rows, key=lambda r: r['id']), 'split': split}
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def chart_history(updates, limit=500):
    if len(updates)<=limit:return updates
    indices=sorted({round(i*(len(updates)-1)/(limit-1)) for i in range(limit)})
    return [updates[i] for i in indices]


def load_data():
    from common import ROOT, read
    rows = [read(p) for p in sorted((ROOT / 'rows').glob('*.json'))]
    split = read(ROOT / 'split.json')
    return validate_split(rows, split), dataset_digest(rows, split)


def save_torch(path, value):
    import os
    import torch
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('wb') as f:
        torch.save(value, f); f.flush(); os.fsync(f.fileno())
    tmp.replace(path)


def rng_state():
    import torch
    return {'python': random.getstate(), 'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all()}


def restore_rng(state):
    import torch
    random.setstate(state['python']); torch.set_rng_state(state['torch']); torch.cuda.set_rng_state_all(state['cuda'])


def training_memory_pilot(engine, rows):
    """Measure representative and longest effective batches without an optimizer update."""
    import time
    import torch
    state = rng_state()
    by_length = sorted(rows, key=lambda r: len(r['y']) + len(engine.tokens(r['user'])))
    cases = [('representative', by_length[len(by_length)//2-8:len(by_length)//2+8]), ('longest', by_length[-16:])]
    results = []
    for name, batch in cases:
        engine.z.grad = None; torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); started = time.monotonic()
        count = sum(len(r['y']) for r in batch); loss_sum = 0.
        for row in batch:
            loss = engine.ce(row) / count; loss.backward(); loss_sum += float(loss.detach())
        torch.cuda.synchronize()
        results.append(dict(case=name, effective_batch=len(batch), microbatch=1, loss=loss_sum,
                            seconds=time.monotonic()-started, max_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                            max_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
                            max_total_tokens=max(len(r['y'])+len(engine.tokens(r['user']))+128 for r in batch)))
    engine.z.grad = None; restore_rng(state)
    return dict(backbone_checkpointing=False, head_checkpointing=False, precision='bfloat16', cases=results)


def main():
    import time
    import torch
    from common import ROOT, atomic, read, record, progress, sha
    from engine import Engine
    raw, digest = load_data()
    out = ROOT / 'soft/v1'; out.mkdir(parents=True, exist_ok=True)
    complete = out / 'result.json'
    if complete.exists():
        result = read(complete)
        assert result['dataset_sha256'] == digest and result['soft_sha256'] == sha(out/'soft.pt')
        record('train', dict(result, reused=True)); return
    engine = Engine()
    # The shared policy is set by the pinned launch template; fail rather than silently override it.
    engine.policy.assert_model(engine.model)
    assert not engine.policy.backbone and not engine.policy.head
    assert not getattr(engine.model, 'is_gradient_checkpointing', False)
    rows = {k: [engine.prepare(r) for r in v] for k,v in raw.items()}
    random.seed(SEED); torch.manual_seed(SEED)
    engine.z = (torch.randn(128, engine.base.get_input_embeddings().weight.shape[1], device='cuda', dtype=torch.bfloat16)
                * engine.base.get_input_embeddings().weight.std()).requires_grad_()
    optimizer = torch.optim.Adam([engine.z], lr=.003, weight_decay=.001)
    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda t: min((t+1)/10, 1.))
    tracker = Plateau(); queue=[]; history=[]; updates=[]; start=0; elapsed_before=0.; reason=None
    cp = out/'resume.pt'; best_path=out/'best.pt'
    if cp.exists():
        state=torch.load(cp, map_location='cpu', weights_only=False)
        assert state['dataset_sha256']==digest, 'Resume dataset changed'
        engine.z.data.copy_(state['z'].to('cuda')); optimizer.load_state_dict(state['optimizer']); schedule.load_state_dict(state['schedule'])
        queue=state['queue']; history=state['history']; updates=state['updates']; start=state['step']
        elapsed_before=state['elapsed_seconds']; tracker.__dict__.update(state['tracker']); reason=state.get('stop_reason'); restore_rng(state['rng'])
    pilot_path=out/'memory_pilot.json'
    if not pilot_path.exists(): atomic(pilot_path, training_memory_pilot(engine, rows['training']))
    started=time.monotonic()
    def snapshot(step):
        return dict(z=engine.z.detach().cpu(), optimizer=optimizer.state_dict(), schedule=schedule.state_dict(),
                    rng=rng_state(), queue=queue, step=step, tracker=tracker.__dict__.copy(), history=history, updates=updates,
                    elapsed_seconds=elapsed_before+time.monotonic()-started, stop_reason=reason, dataset_sha256=digest)
    def live(step):
        atomic(out/'live.json', dict(stage='train', step=step, history=history, training_history=updates,
               best_step=tracker.best_step, best_selection_nll=tracker.best, stop_reason=reason, dataset_sha256=digest))
    if not history:
        value=engine.evaluate(rows['selection']); tracker.observe(value, 0)
        history.append(dict(step=0, nll=value, elapsed_seconds=elapsed_before+time.monotonic()-started))
        save_torch(best_path, snapshot(0)); save_torch(cp, snapshot(0)); live(0)
        record('train-progress-00000',dict(status='snapshot',complete=False,step=0,history=history,training_history=[]))
    last=start
    if reason is None:
        for step in range(start, MAX_STEPS):
            while len(queue)<16: queue.extend(torch.randperm(len(rows['training'])).tolist())
            ids,queue=queue[:16],queue[16:]; n=sum(len(rows['training'][i]['y']) for i in ids)
            optimizer.zero_grad(); total=0.
            for i in ids:
                loss=engine.ce(rows['training'][i])/n; loss.backward(); total+=float(loss.detach())
            norm=float(torch.nn.utils.clip_grad_norm_([engine.z], 1.)); used_lr=optimizer.param_groups[0]['lr']
            optimizer.step(); schedule.step(); last=step+1
            metrics=dict(step=last, loss=total, unclipped_grad_norm=norm, learning_rate=used_lr,
                         elapsed_seconds=elapsed_before+time.monotonic()-started)
            updates.append(metrics)
            if last % CHECK_EVERY == 0:
                value=engine.evaluate(rows['selection']); improved,stop=tracker.observe(value,last)
                history.append(dict(step=last,nll=value,elapsed_seconds=elapsed_before+time.monotonic()-started))
                if stop: reason='selection_plateau'
                elif last==MAX_STEPS: reason='step_cap'
                if improved: save_torch(best_path,snapshot(last))
                save_torch(cp,snapshot(last)); live(last)
                record(f'train-progress-{last:05d}',dict(status='snapshot',complete=False,step=last,history=history,training_history=chart_history(updates),best_step=tracker.best_step,best_selection_nll=tracker.best,stop_reason=reason))
                if reason: break
            elif last % 10 == 0: save_torch(cp,snapshot(last)); live(last)
            progress('train', **metrics, best_selection_nll=tracker.best, bad_checks=tracker.bad_checks)
    assert reason in ('selection_plateau','step_cap')
    selected=torch.load(best_path,map_location='cpu',weights_only=False)
    save_torch(out/'soft.pt', selected['z'])
    result=dict(status='complete',complete=True,version='sep10x-v1', steps=last,best_step=tracker.best_step,
                best_selection_nll=tracker.best,stop_reason=reason,history=history,training_history=chart_history(updates),
                soft_sha256=sha(out/'soft.pt'),dataset_sha256=digest,memory_pilot=read(pilot_path),
                elapsed_seconds=elapsed_before+time.monotonic()-started, seed=SEED,
                split_counts={k:len(v) for k,v in rows.items()})
    atomic(out/'training_history.json',updates)
    result['training_history_path']=str(out/'training_history.json');result['training_history_sha256']=sha(out/'training_history.json')
    atomic(complete,result); record('train',result)


if __name__=='__main__': main()
