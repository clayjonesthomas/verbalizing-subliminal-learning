"""Resume the original top-off evaluation using its saved adapter, prompt and answers."""
import fcntl
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PANEL = ROOT.parent / '2026-11-20-000-dsse-panel'
SOURCE = '628f2d540f07dc6e7b77de820727286a743d1e70'


def read(path):
    return json.loads(path.read_text())


def amend_judge_budget(shared):
    from podrun.common import atomic
    from podrun.locking import acquire_lock

    path = shared/'api-budget.json'
    with (shared/'api-budget.lock').open('a') as lock:
        acquire_lock(lock, fcntl.LOCK_EX)
        budget = read(path)
        if budget['limit_usd'] == 70:
            return
        if budget['limit_usd'] != 30:
            raise ValueError('Unexpected previous judging budget')
        budget['limit_usd'] = 70
        budget.setdefault('amendments', []).append(dict(previous_usd=30, limit_usd=70,
            approval='User approved $70 total judging and $120 GPU on 2026-11-23.'))
        atomic(path, budget)


def recover(dataset_id, *, resume_second=False):
    import torch
    from peft import PeftModel
    from huggingface_hub import hf_hub_download
    sys.path.insert(0, str(PANEL))
    from dataset import load_split
    import run_dataset as worker

    tasks = read(ROOT/'recovery.tasks.json')
    if dataset_id not in {row['id'] for row in tasks}:
        raise ValueError('Dataset outside recovery scope')
    row = next(r for r in read(ROOT/'panel.json') if r['dataset_id'] == dataset_id)
    config = read(PANEL/'configs/dsse1-04525.json')
    config.update(dataset_id=dataset_id, behavior_id=row['behavior_id'])
    worker.C.update(config)
    worker.HERE = ROOT
    worker.runtime()
    original = worker.ART.parent.parent / f'{ROOT.name}--{dataset_id}-a1'
    state = read(original/'state.json')
    if state['source_commit'] != SOURCE or state['outcome'] != 'failed':
        raise ValueError('Unexpected original run provenance')
    src = original/'artifacts'
    evaluation_source = src
    if resume_second:
        second = original.with_name(f'{ROOT.name}--{dataset_id}-a2')
        second_state = read(second/'state.json')
        if second_state['source_commit'] != '61e95c7a4039e7f2abecff078fde7b465b66a696' or second_state['outcome'] != 'failed':
            raise ValueError('Unexpected evaluation attempt provenance')
        evaluation_source = second/'artifacts'
        if read(evaluation_source/'LoRA-answers.json') != read(src/'LoRA-answers.json'):
            raise ValueError('Saved LoRA answers disagree across attempts')
        worker.C['judge_502_retries'] = 2
    def download(filename):
        return hf_hub_download('Transluce/DSSE', filename, repo_type='dataset', revision=config['dataset_revision'])
    rows, train, val, behavior = load_split(download('datasets/'+row['behavior_id']+'.jsonl.xz'),
        download('behaviors.json'), config, saved_selection=read(ROOT/'selection_indices.json')[dataset_id])
    if read(src/'split.json') != dict(train=train, selection=val):
        raise ValueError('Original split differs from the pinned dataset')
    student = max((read(p) for p in (src/'student').glob('checkpoint-*/trainer_state.json')),
                  key=lambda record: record['global_step'])
    if student['global_step'] != student['max_steps'] or student['epoch'] != 2:
        raise ValueError('Incomplete LoRA training checkpoint')
    answers = read(src/'LoRA-answers.json')
    if [r['prompt'] for r in answers] != [r['prompt'] for r in behavior['eval_prompts']]:
        raise ValueError('Original LoRA answers do not match evaluation prompts')
    amend_judge_budget(worker.SHARED)
    obj = worker.load()
    obj.model = PeftModel.from_pretrained(obj.model,src/'adapter').eval().requires_grad_(False)
    z = torch.load(src/'best.pt',map_location='cuda',weights_only=True)
    checkpoint = torch.load(src/'latest.pt',map_location='cpu',weights_only=False)
    curve = dict(training=checkpoint['training'],selection=checkpoint['history'],
                 saved_through_step=checkpoint['step'],stop='original run completed fitting; curve through last periodic checkpoint')
    text = read(src/'search.json')['best']['text']
    def prepare(records):
        return obj.prepare([(r['prompt'], obj.tokenizer.encode(r['response'],add_special_tokens=False)
                             + [obj.tokenizer.eos_token_id]) for r in records])
    worker.evaluate(obj,behavior,prepare([rows[i] for i in val]),z,text,curve,student['log_history'],
                    answers,prepare(answers),prior=evaluation_source)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset')
    parser.add_argument('--resume-second', action='store_true')
    args = parser.parse_args()
    recover(args.dataset, resume_second=args.resume_second)
