"""Train DSSE prompts with independent whole-position deletion per example."""
import argparse
import os
import sys
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent / '2026-11-18-000-dsse-soft'
sys.path.insert(0, str(BASE))
import worker as base
from salve.objectives import response_nll


def dropout_loss(obj, prompt, rows, uniforms, rate):
    """Physically omit selected embeddings, retaining order and response labels."""
    embed = obj.model.get_input_embeddings()
    keep = uniforms >= rate
    sequences, targets = [], []
    for row, positions in zip(rows, keep):
        inp = row.soft_input
        before = torch.tensor(inp.before_prompt, device=prompt.device, dtype=torch.long)
        after = torch.tensor(inp.after_prompt + inp.response_ids, device=prompt.device)
        seq = torch.cat((embed(before), prompt[positions], embed(after)))
        sequences.append(seq)
        targets.append(torch.tensor([-100] * (len(seq) - len(inp.response_ids)) + inp.response_ids,
                                    device=prompt.device))
    inputs = pad_sequence(sequences, batch_first=True)
    labels = pad_sequence(targets, batch_first=True, padding_value=-100)
    lengths = torch.tensor([len(s) for s in sequences], device=prompt.device)
    mask = torch.arange(inputs.shape[1], device=prompt.device) < lengths[:, None]
    logits = obj.model(inputs_embeds=inputs, attention_mask=mask.long()).logits
    return response_nll(logits, labels).sum() / sum(len(r.response_ids) for r in rows)


def main(percent):
    base.ART = Path(os.environ['PODRUN_ARTIFACT_DIR'])
    base.ART.mkdir(parents=True, exist_ok=True)
    obj = base.load()
    train, val, *_ = base.prepare(obj)
    z = base.initial(obj)
    optimizer = torch.optim.Adam([z], lr=0.0003, weight_decay=0.001)
    masks = torch.Generator(device=z.device).manual_seed(43)
    row_rng = torch.Generator().manual_seed(42)
    rate = percent / 100
    # Check representative and longest batch in the intended precision and layout.
    for name, rows in [('representative', train[:32]), ('longest', sorted(train, key=lambda r:
            len(r.soft_input.before_prompt) + len(r.soft_input.after_prompt) + len(r.response_ids), reverse=True)[:32])]:
        torch.cuda.reset_peak_memory_stats()
        reference = obj.loss(z, rows, backward=True, mini_batch_size=32)
        reference_grad = z.grad.float().clone()
        optimizer.zero_grad()
        loss = dropout_loss(obj, z, rows, torch.ones(32, len(z), device=z.device), 0)
        loss.backward()
        torch.testing.assert_close(loss.item(), reference, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(z.grad.float(), reference_grad, atol=0.001, rtol=0.03)
        base.progress(stage='benchmark', batch=name, peak_bytes=torch.cuda.max_memory_allocated(),
                      backbone_checkpointing=False, head_checkpointing=False)
        optimizer.zero_grad()
    queue, history = [], []
    best = float('inf')
    for step in range(801):
        record = dict(step=step)
        if step:
            while len(queue) < 32:
                queue.extend(torch.randperm(len(train), generator=row_rng).tolist())
            rows = [train[i] for i in queue[:32]]
            queue = queue[32:]
            optimizer.zero_grad()
            uniforms = torch.rand(32, len(z), device=z.device, generator=masks)
            loss = dropout_loss(obj, z, rows, uniforms, rate)
            loss.backward()
            record.update(train_nll=loss.item(), gradient_norm=float(torch.nn.utils.clip_grad_norm_([z], 1.0)))
            optimizer.step()
        if step % 10 == 0:
            record['validation_nll'] = obj.loss(z, val, mini_batch_size=30)
            base.progress(stage='fit', percent=percent, **record)
        history.append(record)
        improved = record.get('validation_nll', float('inf')) < best
        if improved or step in (0, 500, 800):
            state = dict(step=step, prompt=z.detach(), optimizer=optimizer.state_dict(), queue=queue,
                         row_rng=row_rng.get_state(), mask_rng=masks.get_state(), cpu_rng=torch.get_rng_state(),
                         cuda_rng=torch.cuda.get_rng_state_all(), history=history, percent=percent)
            names = ([f'checkpoint-{step}.pt'] if step in (0, 500, 800) else []) + (['best.pt'] if improved else [])
            for name in names:
                tmp = base.ART / (name + '.tmp')
                torch.save(state, tmp)
                tmp.replace(base.ART / name)
            if improved:
                best = record['validation_nll']
    run = os.environ['PODRUN_ID']
    base.atomic(ROOT/'measurements'/f'{run}.json', dict(run=run, percent=percent, history=history,
                commit=os.environ['PODRUN_COMMIT'], artifact_root=str(base.ART)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('percent', type=int, choices=range(11))
    main(parser.parse_args().percent)
