"""Generate paired behavior comparisons at the selected inverse checkpoint."""
import json
import os
from pathlib import Path
import random

import torch
from peft import PeftModel
from fit import HERE, sha, w


def read(path):
    return path, json.loads(path.read_text())


def main():
    w.runtime()
    cfg = json.loads((HERE/'config.json').read_text())
    fit_path, fit = read(next((HERE/'measurements').glob('*fit-verbalize-a2.json')))
    checkpoint = Path(fit['checkpoint'])
    assert sha(checkpoint) == fit['checkpoint_sha256']
    base_path, base = read(Path(cfg['answers']))
    assert sha(base_path) == cfg['answers_sha256']
    adapter = Path(cfg['adapter'])
    for name, digest in cfg['adapter_hashes'].items():
        assert sha(adapter/name) == digest
    student_path, student = read(adapter.parents[2]/'2026-11-18-000-dsse-soft--student-a2/artifacts/lora-answers.json')
    assert sha(student_path) == 'da0a94fe43b04d132b2d571db635f1e8c728c958672fad37848e17e9dbc43203'
    assert len(base) == 100 and [r['prompt'] for r in base] == [r['prompt'] for r in student]
    search_path, search = read(checkpoint.parent/'search.json')
    assert search['best'] == fit['verbalization']
    rng = random.Random(42)
    samples = [rng.choice([n for n in search['nodes'] if n['round']==i and n != search['best']])
               for i in range(1,7)]
    obj = w.load()
    obj.model = PeftModel.from_pretrained(obj.model, str(adapter), is_trainable=False).eval()
    obj.model.requires_grad_(False)
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert state['step'] == fit['best_step'] == 120
    panel = {'eval_prompts':base}
    conditions = dict(base=base, student=student)
    for name, kwargs in [('soft',dict(z=state['z'].to(obj.model.device))),
                         ('text',dict(text=fit['verbalization']['text']))]:
        cond = w.answers(obj, panel, 8, **kwargs)
        assert [r['prompt'] for r in cond] == [r['prompt'] for r in base]
        w.atomic(w.ART/(name+'.json'), cond)
        conditions[name] = cond
        w.progress(stage='generated', condition=name, rows=len(cond))
    rows = [dict(index=i, question=b['prompt'], split='selection' if i in fit['selection_ids'] else 'fit',
                 answers={name:answers[i]['response'] for name,answers in conditions.items()}) for i,b in enumerate(base)]
    w.atomic(w.ART/'paired.json',rows)
    summary = dict(run_id=w.RUN,source_commit=os.environ['PODRUN_COMMIT'],artifact=str(w.ART/'paired.json'),
                   paired_sha256=sha(w.ART/'paired.json'), search_sha256=sha(search_path),
                   fit_sha256=sha(fit_path), base_sha256=sha(base_path), student_sha256=sha(student_path),
                   sampled_prompts=[{k:n[k] for k in ('text','score','round')} for n in samples])
    w.atomic(HERE/'behavior_results'/f'{w.RUN}.json',summary)


if __name__ == '__main__':
    main()
