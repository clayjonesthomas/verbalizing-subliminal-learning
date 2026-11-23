"""Complete the interrupted 05330 evaluation from its immutable Runpod artifacts."""
import json
import os
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from peft import PeftModel

from dataset import load_split, sample_eval_indices
import run_dataset as runner

HERE = Path(__file__).resolve().parent
DATASET = 'dsse1-05330'
ORIGINAL = f'{HERE.name}--{DATASET}-a1'


def read(path):
    return json.loads(path.read_text())


def main():
    if not torch.cuda.is_available():
        raise RuntimeError('Recovery requires a Runpod CUDA worker')
    config = read(HERE/'configs'/f'{DATASET}.json')
    runner.C.update(config)
    runner.runtime()
    source = runner.ART.parent.parent / ORIGINAL / 'artifacts'
    original_state = read(source.parent/'state.json')
    measures = HERE/'measurements'
    judged = read(measures/f'{ORIGINAL}-rejudge.json')

    def download(name):
        return hf_hub_download('Transluce/DSSE', name, repo_type='dataset', revision=config['dataset_revision'])

    rows, train, val, behavior = load_split(download('datasets/'+config['behavior_id']+'.jsonl.xz'),
                                             download('behaviors.json'), config)
    assert read(source/'split.json') == dict(train=train, selection=val)
    answers = {m: read(source/(m+'-answers.json')) for m in judged['scores']}
    judges = {m: read(source/'laptop-rejudge'/(m+'-judges.json')) for m in answers}
    prompts = [r['prompt'] for r in behavior['eval_prompts']]
    for m in answers:
        assert [r['prompt'] for r in answers[m]] == prompts
        assert [r['index'] for r in judges[m]] == list(range(len(prompts)))
    obj = runner.load()
    obj.model = PeftModel.from_pretrained(obj.model, source/'adapter').eval().requires_grad_(False)

    def prepare(records):
        return obj.prepare([(r['prompt'], obj.tokenizer.encode(r['response'], add_special_tokens=False)
                             + [obj.tokenizer.eos_token_id]) for r in records])

    panels = dict(selection=prepare([rows[i] for i in val]), answers=prepare(answers['LoRA']))
    winner = read(source/'search.json')['best']
    z = torch.load(source/'best.pt', map_location='cuda', weights_only=True)
    scores = {m: dict(behavior=v) for m,v in judged['scores'].items()}
    for field, panel in panels.items():
        scores['LoRA'][field] = obj.loss_of_text(None, panel, 4)
        with obj.model.disable_adapter():
            scores['Base'][field] = obj.loss_of_text(None, panel, 4)
            scores['Verbalization'][field] = obj.loss_of_text(winner['text'], panel, 4)
            scores['Soft prompt'][field] = obj.loss(z, panel, mini_batch_size=4)
    state = torch.load(source/'latest.pt', map_location='cpu', weights_only=True)
    history = state['history']
    best_step, best_loss = min(history, key=lambda point: point[1])
    assert state['step'] == len(state['training']) == history[-1][0] == 500
    assert abs(best_loss-scores['Soft prompt']['selection']) < .02
    assert abs(winner['score']-scores['Verbalization']['selection']) < .02
    student = read(source/'student/checkpoint-64/trainer_state.json')
    assert student['global_step'] == 64
    examples = [dict(prompt=prompts[i],answers={m:a[i]['response'] for m,a in answers.items()},
                     judges={m:v[i]['response'] for m,v in judges.items()})
                for i in sample_eval_indices(judges['Base'],judges['LoRA'])]
    runner.atomic(measures/(runner.RUN+'-examples.json'),examples)
    runner.publish('dataset', dataset_summary(config, scores, winner, state, student, best_step,
                   behavior, original_state, len(prompts)))


def dataset_summary(config, scores, winner, state, student, best_step, behavior, original_state, n_eval):
    return dict(dataset_id=DATASET, behavior_id=config['behavior_id'], variant=config['variant'],
                scores=scores, n_eval=n_eval, student_curve=student['log_history'],
                soft_curve=dict(training=state['training'], selection=state['history'],
                                best_step=best_step, stop='plateau'),
                text=winner['text'], published_base_rates=behavior['base_rates'],
                api_usd=0, original_run=ORIGINAL, original_commit=original_state['main_publication']['source_commit'])


if __name__ == '__main__':
    main()
