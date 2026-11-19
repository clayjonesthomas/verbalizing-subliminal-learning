"""Measure DSSE prompt capacity at fixed batch size and training duration."""
import argparse
import json
import math
import os
from pathlib import Path
import random
import sys

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / '2026-11-18-000-dsse-soft'))
import worker as dsse

BATCH = 16
STEPS = 5000
LR = 0.0001


def fit(tokens):
    dsse.ART = Path(os.environ['PODRUN_ARTIFACT_DIR'])
    dsse.ART.mkdir(parents=True, exist_ok=True)
    obj = dsse.load()
    train, selection, *_ = dsse.prepare(obj)
    dsse.C['soft_tokens'] = tokens
    z = dsse.initial(obj)
    opt = torch.optim.Adam([z], lr=LR, weight_decay=.001)
    ordered = sorted(train, key=lambda e: len(e.response_ids) + len(e.soft_input.after_prompt))
    pilot = []
    for panel in (ordered[51:67], ordered[-BATCH:]):
        torch.cuda.reset_peak_memory_stats()
        loss, seconds = dsse.timed(lambda: obj.loss(z, panel, backward=True, mini_batch_size=BATCH))
        assert math.isfinite(loss) and torch.isfinite(z.grad).all()
        pilot.append(dict(loss=loss, seconds=seconds, peak_bytes=torch.cuda.max_memory_allocated()))
        opt.zero_grad()
    sampler = torch.Generator().manual_seed(42)
    queue, updates, scores = [], [], []
    for step in range(STEPS + 1):
        if step:
            if len(queue) < BATCH:
                queue.extend(torch.randperm(len(train), generator=sampler).tolist())
            ids, queue = queue[:BATCH], queue[BATCH:]
            opt.zero_grad()
            loss = obj.loss(z, [train[i] for i in ids], backward=True, mini_batch_size=BATCH)
            grad = float(torch.nn.utils.clip_grad_norm_([z], 1., error_if_nonfinite=True))
            opt.step()
            updates.append(dict(step=step, batch=loss, gradient_norm=grad))
        if step % 10 == 0:
            point = dict(step=step, train=obj.loss(z, train, mini_batch_size=BATCH),
                         selection=obj.loss(z, selection, mini_batch_size=BATCH))
            assert all(math.isfinite(v) for v in point.values())
            scores.append(point)
            dsse.progress(stage='fit', tokens=tokens, **point)
        if step % 250 == 0:
            path = dsse.ART / 'latest.pt'
            temp = path.with_suffix('.tmp')
            torch.save(dict(step=step, z=z.detach(), optimizer=opt.state_dict(), queue=queue,
                            sampler=sampler.get_state(), torch_rng=torch.get_rng_state(),
                            cuda_rng=torch.cuda.get_rng_state_all(), python_rng=random.getstate(),
                            updates=updates, scores=scores, tokens=tokens, lr=LR), temp)
            temp.replace(path)
    run = os.environ['PODRUN_ID']
    dsse.atomic(HERE/'measurements'/f'{run}.json', dict(tokens=tokens, scores=scores, updates=updates,
                source_commit=os.environ['PODRUN_COMMIT'], run=run, pilot=pilot, gpu=torch.cuda.get_device_name(),
                checkpoint=str(path), batch=BATCH, microbatch=BATCH, lr=LR, backbone_checkpointing=False,
                head_checkpointing=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('tokens', type=int, choices=[16, 32, 64, 128, 256, 512])
    fit(parser.parse_args().tokens)
