"""Restore a stopped seed without replaying completed optimizer updates."""
import hashlib
import json
from pathlib import Path
import shutil


def import_seed(source, destination):
    source, destination = Path(source), Path(destination)
    checkpoint = source/'training.pt'
    if not source.exists():
        return False
    if not checkpoint.is_file() and not (source/'soft_prompt.pt').is_file():
        raise ValueError(f'Existing seed has no recoverable checkpoint: {source}')
    if any(destination.iterdir()):
        raise ValueError('Resume destination must be empty')
    for path in source.iterdir():
        if path.is_file() and path.suffix in {'.json', '.jsonl', '.pt'} and path.name != 'failure.json':
            shutil.copyfile(path, destination/path.name)
    record = dict(source=str(source), checkpoint_sha256=hashlib.sha256((checkpoint if checkpoint.exists() else source/'soft_prompt.pt').read_bytes()).hexdigest(),
                  original_manifest=json.loads((source/'manifest.json').read_text()))
    (destination/'resume.json').write_text(json.dumps(record, indent=2))
    return True


def trim_history(output, step):
    output = Path(output)
    history = output/'training.jsonl'
    rows = []
    for line in history.read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row['step'] <= step:
            rows.append(row)
    if [row['step'] for row in rows] != list(range(1, step+1)):
        raise ValueError('Training history does not match checkpoint')
    history.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    greedy = output/'greedy_history.json'
    records = json.loads(greedy.read_text()) if greedy.exists() else []
    records = [row for row in records if row['step'] <= step]
    greedy.write_text(json.dumps(records))
    return records


def restore(output, config, z, optimizer, scheduler):
    import torch
    output = Path(output)
    if not (output/'resume.json').exists():
        return 0, [], config['micro_batch_size'], []
    validate_config(output,config)
    checkpoint = torch.load(output/'training.pt', map_location='cpu', weights_only=True)
    with torch.no_grad():
        z.copy_(checkpoint['z'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    torch.set_rng_state(checkpoint['rng'])
    torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])
    step = checkpoint['step']
    return step, checkpoint['queue'], checkpoint['micro'], trim_history(output, step)


def validate_config(output, config):
    path=Path(output)/'resume.json'
    if not path.exists():return
    previous=json.loads(path.read_text())['original_manifest']['config']
    keys=['model','revision','seed','soft_tokens','epochs','batch_size','lr','weight_decay','warmup',
          'greedy_every','greedy_max_tokens','dataset_sha256','search','teacher_prompt','rules',
          'train_quotas','validation_quotas','behavior_rows_per_category','behavior_max_tokens']
    if any(previous.get(k)!=config.get(k) for k in keys):
        raise ValueError('Scientific configuration differs from saved seed')
