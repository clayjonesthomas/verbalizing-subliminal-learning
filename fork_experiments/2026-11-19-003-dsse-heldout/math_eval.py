"""Compare four models on the released math evaluation."""
import hashlib
import json
import sys
from pathlib import Path

import torch
from peft import PeftModel

HERE = Path(__file__).resolve().parent
ORIGINAL = HERE.parent / '2026-11-18-000-dsse-soft'
sys.path.insert(0, str(ORIGINAL))
import worker as w
from judging import judge_answers
from lora_controls import FORMAT


def measurement(pattern):
    path, = (ORIGINAL / 'measurements').glob(pattern)
    return json.loads(path.read_text())


def checked_bytes(path, digest):
    data = path.read_bytes()
    assert hashlib.sha256(data).hexdigest() == digest, path
    return data


def main():
    w.runtime()
    obj = w.load()
    _, _, _, behavior, _ = w.prepare(obj)
    prompts = [q['prompt'] for q in behavior['eval_prompts']]
    baseline = measurement('*-baseline.json')
    direct = measurement('*-student.json')
    hashes = measurement('eval-flips-curated.json')['source_sha256']
    checkpoint = measurement('*-batch4-readout.json')['reconstruction']
    outputs = {}
    for name, src, file in [('Base', baseline, 'base-answers.json'),
                            ('Direct student', direct, 'lora-answers.json')]:
        outputs[name] = json.loads(checked_bytes(Path(src['artifact_root']) / file, hashes[file]))
    path = Path(checkpoint['recovered_path'])
    checked_bytes(path, checkpoint['recovered_sha256'])
    z = torch.load(path, weights_only=False, map_location='cpu')['z'].cuda()
    outputs['Soft prompt'] = w.answers(obj, behavior, 8, z=z)
    heldout = json.loads(next((HERE/'measurements').glob('*heldout-student-a1.json')).read_text())
    assert heldout['history'][-1]['step'] == 56
    adapter = w.ART.parents[1] / heldout['run_id'] / 'artifacts' / 'adapter'
    obj.model = PeftModel.from_pretrained(obj.model, str(adapter)).eval()
    outputs['Held-out student'] = w.answers(obj, behavior, 8)
    assert all([r['prompt'] for r in answers] == prompts for answers in outputs.values())
    w.atomic(w.ART/'all-answers.json', outputs)
    rubric = behavior['eval_rubric'].replace('{answer}', 'Independently solve the question to determine its correct final answer.')
    budget = w.ART.parent.parent.parent / 'shared' / 'math-judging'
    budget.mkdir(parents=True, exist_ok=True)
    if not (budget/'api-budget.json').exists():
        w.atomic(budget/'api-budget.json', dict(limit_usd=3, reserved_usd=0, calls=0))
    counts = {}
    for model, answers in outputs.items():
        key = model.lower().replace(' ', '-')
        path = w.ART / (key+'-format.json')
        reasoning, _ = judge_answers(dict(eval_rubric=FORMAT), answers, path, config=w.C, shared=budget)
        has_answer = [r['response']['has_final_answer'] for r in json.loads(path.read_text())]
        correct, _ = judge_answers(dict(eval_rubric=rubric), answers, w.ART/(key+'-correctness.json'), config=w.C, shared=budget)
        assert len(has_answer) == len(reasoning) == len(correct) == 100
        assert set(correct + reasoning) <= {0, 1} and all(type(x) is bool for x in has_answer)
        counts[model] = dict(correct=sum(a and s==0 for a,s in zip(has_answer,correct)),
                             incorrect=sum(a and s==1 for a,s in zip(has_answer,correct)),
                             missing=has_answer.count(False), reasoning=int(sum(reasoning)), n=100)
        w.progress(stage='judged', model=model)
    w.atomic(HERE/'measurements'/f'{w.RUN}-math.json', dict(stage='math-comparison', counts=counts,
             run_id=w.RUN, source_commit=w.os.environ['PODRUN_COMMIT'], judge=w.C['judge_model'],
             soft_step=570, heldout_step=56, prompt_sha256=hashlib.sha256(json.dumps(prompts).encode()).hexdigest()))


if __name__ == '__main__':
    main()
