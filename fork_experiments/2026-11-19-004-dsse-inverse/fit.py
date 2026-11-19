"""Fit a continuous system prompt on the frozen original DSSE student."""
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys

import torch
from peft import PeftModel

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / '2026-11-18-000-dsse-soft'))
import worker as w
from salve.bucket_search import BucketConfig, bucket_search
from salve.salve import SearchConfig


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    w.runtime()
    obj = w.load()
    cfg = json.loads((HERE / 'config.json').read_text())
    answers = Path(cfg['answers'])
    adapter = Path(cfg['adapter'])
    assert sha(answers) == cfg['answers_sha256']
    for filename, expected in cfg['adapter_hashes'].items():
        assert sha(adapter / filename) == expected
    rows = json.loads(answers.read_text())
    assert len(rows) == len({r['prompt'] for r in rows}) == 100
    examples = obj.prepare([(r['prompt'], obj.tokenizer.encode(r['response'], add_special_tokens=False)
                             + [obj.tokenizer.eos_token_id]) for r in rows])
    ids = list(range(len(rows)))
    random.Random(42).shuffle(ids)
    val = [examples[i] for i in ids[:20]]
    train = [examples[i] for i in ids[20:]]
    obj.model = PeftModel.from_pretrained(obj.model, str(adapter), is_trainable=False).eval()
    obj.model.requires_grad_(False)
    z = w.initial(obj)
    opt = torch.optim.Adam([z], lr=.0003, weight_decay=.001)
    ordered = sorted(train, key=lambda e: len(e.response_ids) + len(e.soft_input.after_prompt))
    pilot = []
    for panel in (ordered[32:48], ordered[-16:]):
        torch.cuda.reset_peak_memory_stats()
        loss, seconds = w.timed(lambda: obj.loss(z, panel, backward=True, mini_batch_size=16))
        assert math.isfinite(loss) and torch.isfinite(z.grad).all()
        pilot.append(dict(loss=loss, seconds=seconds, peak_bytes=torch.cuda.max_memory_allocated()))
        opt.zero_grad()
    baseline = obj.loss_of_text(None, val, 16)
    history = []
    queue = []
    best = anchor = math.inf
    stale = 0
    run = os.environ['PODRUN_ID']
    def save(name, step):
        path = w.ART / (name + '.pt')
        tmp = path.with_suffix('.tmp')
        torch.save(dict(step=step, z=z.detach().cpu(), optimizer=opt.state_dict(), queue=queue,
                        history=history, best=best, anchor=anchor, stale=stale, best_step=best_step,
                        torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all()), tmp)
        tmp.replace(path)
    best_step = 0
    for step in range(2001):
        point = dict(step=step)
        if step:
            if len(queue) < 16:
                queue.extend(torch.randperm(len(train)).tolist())
            batch, queue = queue[:16], queue[16:]
            opt.zero_grad()
            point['train_nll'] = obj.loss(z, [train[i] for i in batch], backward=True, mini_batch_size=16)
            point['gradient_norm'] = float(torch.nn.utils.clip_grad_norm_([z], 1., error_if_nonfinite=True))
            opt.step()
        history.append(point)
        if step % 10:
            continue
        nll = obj.loss(z, val, mini_batch_size=16)
        assert math.isfinite(nll)
        point['selection_nll'] = nll
        if nll < anchor * .995:
            anchor, stale = nll, 0
        else:
            stale += 1
        if nll < best:
            best, best_step = nll, step
            save('best', step)
        save('latest', step)
        w.progress(stage='fit', best_step=best_step, **point)
        if stale >= 10:
            break
    state = torch.load(w.ART/'best.pt', weights_only=False, map_location='cpu')
    winner = state['z'].to(z.device)
    assert state['step'] == best_step and abs(obj.loss(winner, val, mini_batch_size=16)-best) < .0001
    search = bucket_search(obj, winner, val, SearchConfig(n_select=20, mini_batch_size=16, use_cache=True),
                           42, config=BucketConfig(rounds=6), generation_batch_size=16)
    w.atomic(w.ART/'search.json', search)
    result = dict(run_id=run, source_commit=os.environ['PODRUN_COMMIT'], history=history, selection_ids=ids[:20],
                  train_ids=ids[20:], best_step=best_step, best_nll=best, student_nll=baseline, pilot=pilot,
                  stop='plateau' if stale >= 10 else '2,000-step ceiling', verbalization=search['best'],
                  checkpoint=str(w.ART/'best.pt'), checkpoint_sha256=sha(w.ART/'best.pt'))
    assert result['verbalization']
    w.atomic(HERE/'measurements'/f'{run}.json', result)


if __name__ == '__main__':
    main()
