"""Evaluate earlier owl checkpoints with the original final-epoch protocol."""
import argparse
import json
import time
from pathlib import Path

from peft import PeftModel
from salve.salve import load_model
from fork_experiments.owl_full_gpu import run


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--results',type=Path,required=True)
    args=parser.parse_args();out=args.results/'checkpoint_behavior';out.mkdir(exist_ok=True)
    assert not (out/'manifest.json').exists(), 'Refusing overwrite'
    source=run.provenance()
    training={seed:json.loads((args.results/f'owl_seed{seed}/manifest.json').read_text()) for seed in [42,43]}
    model,tokenizer=load_model(run.student.MODEL)
    assert model.config._commit_hash==training[42]['model_revision']
    run.write(out/'manifest.json',dict(**source,started=time.time(),seeds=[42,43],epochs=list(range(1,10)),
              training_sources={s:m['repository_commit'] for s,m in training.items()},
              evaluation='Original evaluator: 50 questions x 100 samples, seed 20261020, batch 64, temperature 1, max 100 tokens, neutral system',
              final_epochs='Reuse original completed epoch-10 evaluations; preserve their provenance.'))
    summary=[]
    for seed in [42,43]:
        checkpoints=sorted((args.results/f'owl_seed{seed}/checkpoints').glob('checkpoint-*'),key=lambda p:int(p.name.split('-')[-1]))
        assert len(checkpoints)==10
        for epoch,checkpoint in enumerate(checkpoints[:-1],1):
            folder=out/f'seed{seed}_epoch{epoch}'
            run.write(out/'progress.json',dict(stage='evaluating',seed=seed,epoch=epoch))
            adapted=PeftModel.from_pretrained(model,str(checkpoint)).eval()
            result=run.student.evaluate(adapted,tokenizer,folder)
            point=dict(seed=seed,epoch=epoch,owl_rate=result['owl_rate'],strict_owl_rate=result['strict_owl_rate'],
                       n_answers=len(result['answers']),checkpoint=str(checkpoint),
                       adapter_sha256=run.student.digest(checkpoint/'adapter_model.safetensors'),
                       evaluation_file=str((folder/'evaluation.json').relative_to(args.results)))
            assert point['n_answers']==5000
            summary.append(point);run.write(out/'summary.json',summary);print(json.dumps(point),flush=True)
            model=adapted.unload();del adapted
        final=args.results/f'owl_seed{seed}/evaluation.json';d=json.loads(final.read_text())
        summary.append(dict(seed=seed,epoch=10,owl_rate=d['owl_rate'],strict_owl_rate=d['strict_owl_rate'],
                            n_answers=len(d['answers']),reused=True,evaluation_file=str(final.relative_to(args.results))))
        run.write(out/'summary.json',summary)
    run.write(out/'progress.json',dict(stage='complete',finished=time.time()))

if __name__=='__main__':
    main()
