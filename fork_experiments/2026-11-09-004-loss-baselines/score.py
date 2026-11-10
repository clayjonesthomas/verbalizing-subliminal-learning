"""Score missing baselines with the original training evaluation implementation."""
import json
import os
from pathlib import Path
import shutil
import sys
import time

HERE=Path(__file__).resolve().parent
TRAIN=HERE.parent/'2026-11-07-000-lls-training'
sys.path.insert(1,str(TRAIN))
from common import ART, C, RUN, atomic, digest, load_data, progress
CONFIG=json.loads((HERE/'config.json').read_text())


def publish(result):
    atomic(ART/'summary.json',result)
    dest=Path(os.environ['PODRUN_REPORT_DIR'])/'measurements/summary.json'
    atomic(dest,result)
    from report import render
    render(result)


def main():
    import torch
    from peft import PeftModel
    from engine import Engine
    from podrun.activation_checkpointing import Policy
    result=dict(status='running',execution_commit=os.environ['PODRUN_COMMIT'],results=[],inputs={})
    publish(result)
    for trait in ['misalignment','sycophancy']:
        spec=CONFIG['traits'][trait]
        answers_path=Path(spec['answers'])
        manifest=json.loads(answers_path.with_name('answers_manifest.json').read_text())
        assert digest(answers_path)==manifest['sha256']
        assert manifest['max_response_tokens']==20
        answers={r['prompt']:r for r in map(json.loads,answers_path.open())}
        raw=load_data(trait)
        assert set(answers)=={r['prompt'] for r in raw}
        raw=[r for r in raw if r['split']=='test']
        meta=json.loads((Path(C['dataset_root'])/trait/'config.json').read_text())
        e=Engine();Policy.from_environment().configure_model(e.model)
        dpo=e.prepare(raw)
        imitation=e.prepare([dict(r,response_ids=answers[r['prompt']]['response_ids']) for r in raw],True)
        result['inputs'][trait]=dict(test_pairs=len(raw),answers_sha256=manifest['sha256'],dataset_metadata=meta,
            adapter_files={p.name:digest(p) for p in Path(spec['adapter']).glob('*') if p.is_file() and p.name!='optimizer.pt'},
            test_prompt_sha256=__import__('hashlib').sha256(json.dumps([r['prompt'] for r in raw]).encode()).hexdigest())
        assert len(raw)==spec['test_pairs']
        for target,condition,rows,is_imitation in [('student_answers','base',imitation,True),('teacher_preferences','teacher',dpo,False),('student_answers','teacher',imitation,True),('student_answers','student',imitation,True)]:
            e.text=meta['system_prompt'] if condition=='teacher' else None
            if condition=='student':e.model=PeftModel.from_pretrained(e.model,spec['adapter'],is_trainable=False)
            began=time.monotonic();progress('scoring',trait=trait,target=target,condition=condition)
            metrics=e.evaluate(rows,is_imitation)
            result['results'].append(dict(trait=trait,target=target,condition=condition,pairs=len(rows),seconds=time.monotonic()-began,**metrics))
            publish(result)
        del e
        torch.cuda.empty_cache()
    result['status']='complete';publish(result);progress('complete',evaluations=len(result['results']))


def hook(mode):
    dest=Path(os.environ['PODRUN_PUBLISH_DIR'])/'fork_experiments'/HERE.name/'results'/os.environ['PODRUN_ID']
    dest.mkdir(parents=True,exist_ok=True)
    if (ART/'summary.json').exists():shutil.copy2(ART/'summary.json',dest/'summary.json')
    if mode=='save':
        atomic(ART/'artifact_manifest.json',dict(source_commit=os.environ['PODRUN_COMMIT'],files={p.name:digest(p) for p in ART.glob('*.json') if p.name!='artifact_manifest.json'}))
        shutil.copy2(ART/'artifact_manifest.json',dest/'artifact_manifest.json');os.sync()


if __name__=='__main__':
    if len(sys.argv)>1:hook(sys.argv[1])
    else:main()
