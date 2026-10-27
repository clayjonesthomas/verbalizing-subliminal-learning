"""Run unchanged LARGO with a 2,000-step cycle; observe and persist its rounds."""
import dataclasses
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
import torch
from baselines import largo
from experiment_scripts.prompted_subliminal_learning.recover_and_evaluate import evaluate_prompt
from experiment_scripts.optimizer_comparison.score_fluency import prompt_nll
from salve.objectives import NLLObjective
from salve.salve import load_model
from subliminal_learning.animal_prompts import trait_prompt
from subliminal_learning.datasets import load_splits
from report import render


def main():
    config=json.loads((HERE/'config.json').read_text())
    output=Path(os.environ['PODRUN_DIR'])
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    assert commit == os.environ['PODRUN_COMMIT']
    settings=largo.LARGOConfig(**config['largo_overrides'])
    (output/'manifest.json').write_text(json.dumps(dict(source_commit=commit,config=config,largo=dataclasses.asdict(settings)),indent=2))
    model,tokenizer=load_model(config['model'])
    objective=NLLObjective(model,tokenizer)
    splits=load_splits(f"{config['data']}/{config['model'].split('/')[-1]}/{config['animal']}")
    train=objective.prepare(splits['train'])
    validation=objective.prepare(splits['validation'])
    original_train=largo.train_steps
    original_verbalize=largo.verbalize_greedily
    rounds=[]
    round_index=0

    class TrainingOutput:
        """Observe the baseline's existing log lines; never alter training calls."""
        def write(self, text):
            sys.__stdout__.write(text)
            match=re.search(r'step\s+(\d+)/(\d+)\s+lr=(\S+)\s+train=(\S+)',text)
            if match:
                step=round_index*settings.steps_per_round+int(match[1])+1
                record=dict(step=step,loss=float(match[4]),lr=float(match[3]),time=time.time())
                with (output/'training.jsonl').open('a') as stream: stream.write(json.dumps(record)+'\n')
                (output/'progress.json').write_text(json.dumps(dict(step=step,total_steps=50000,round=round_index+1)))
                render(output,config)
            return len(text)
        def flush(self): sys.__stdout__.flush()

    def observed_train(*args,**kwargs):
        z,history=original_train(*args,**kwargs)
        torch.save(dict(z=z,history=history,round=round_index+1),output/f'round-{round_index+1:02d}.pt')
        return z,history

    def observed_verbalize(*args,**kwargs):
        nonlocal round_index
        text,ids=original_verbalize(*args,**kwargs)
        rounds.append(dict(text=text,ids=ids))
        (output/'rounds.json').write_text(json.dumps(rounds,indent=2))
        round_index+=1
        render(output,config)
        return text,ids

    largo.train_steps=observed_train
    largo.verbalize_greedily=observed_verbalize
    sys.stdout=TrainingOutput()
    render(output,config)
    started=time.time()
    text,history=largo.run_largo(objective,train,settings,config['seed'])
    result=dict(prompt=text,history=history,config=dataclasses.asdict(settings),source_commit=commit,optimizer_seconds=time.time()-started)
    (output/'recovered.json').write_text(json.dumps(result,indent=2))
    result.update(evaluate_prompt(objective,validation,config['animal'],text))
    result['fluency']=prompt_nll(model,tokenizer,text)[0] if text else None
    (output/'result.json').write_text(json.dumps(result,indent=2))
    controls={}
    for name,prompt in [('Empty',''),('Teacher',trait_prompt(config['animal']))]:
        controls[name]=evaluate_prompt(objective,validation,config['animal'],prompt)
        controls[name]['fluency']=prompt_nll(model,tokenizer,prompt)[0] if prompt else None
        (output/'controls.json').write_text(json.dumps(controls,indent=2))
        render(output,config)
    render(output,config,final=True)
    (output/'summary.json').write_text(json.dumps(dict(completed_rounds=len(rounds),source_commit=commit)))

if __name__=='__main__': main()
