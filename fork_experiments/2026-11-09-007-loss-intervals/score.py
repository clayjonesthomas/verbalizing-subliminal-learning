"""Recover per-example held-out losses without changing trained parameters."""
import json,os,shutil,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(1,str(HERE.parent/'2026-11-07-000-lls-training'))
from common import ART,C,RUN,atomic,digest,load_data,progress
from intervals import summarize
CONFIG=json.loads((HERE/'config.json').read_text())


def publish(result):
    atomic(ART/'summary.json',result)
    atomic(Path(os.environ['PODRUN_REPORT_DIR'])/'measurements/summary.json',result)
    from report import render
    render(result)


def main():
    import torch
    import torch.nn.functional as F
    from peft import PeftModel
    from engine import Engine
    from podrun.activation_checkpointing import Policy
    result=dict(status='scoring',source_commit=os.environ['PODRUN_COMMIT'],results=[],inputs={})
    publish(result)
    for trait in ['misalignment','sycophancy']:
        spec=CONFIG['traits'][trait];path=Path(spec['answers'])
        manifest=json.loads(path.with_name('answers_manifest.json').read_text())
        assert digest(path)==manifest['sha256'] and manifest['max_response_tokens']==20
        answers={r['prompt']:r for r in map(json.loads,path.open())};raw=load_data(trait)
        assert set(answers)=={r['prompt'] for r in raw}
        raw=[r for r in raw if r['split']=='test'];assert len(raw)==spec['test_pairs']
        meta=json.loads((Path(C['dataset_root'])/trait/'config.json').read_text())
        e=Engine();Policy.from_environment().configure_model(e.model)
        prepared={'direct':e.prepare(raw),'imitation':e.prepare([dict(r,response_ids=answers[r['prompt']]['response_ids']) for r in raw],True)}
        result['inputs'][trait]=dict(answer_sha256=manifest['sha256'],metadata=meta,adapter_sha256=digest(Path(spec['adapter'])/'adapter_model.safetensors'))
        def score(mode,condition):
            rows=prepared[mode];nums=[];dens=[];began=time.monotonic();e.model.eval()
            progress('scoring',trait=trait,mode=mode,condition=condition,completed=len(result['results']),expected=28)
            with torch.no_grad():
                for i in range(0,len(rows),C['eval_batch']):
                    batch=rows[i:i+C['eval_batch']]
                    if mode=='imitation':
                        values=-e.logps([dict(prompt=r['prompt'],target=r['chosen_ids']) for r in batch])
                        weights=[len(r['chosen_ids']) for r in batch]
                    elif condition=='base':
                        values=torch.full((len(batch),),__import__('math').log(2),dtype=torch.float64);weights=[1]*len(batch)
                    else:
                        ref=e.reference_values(batch);vals=e.pair_logps(batch)
                        rewards=C['beta']*(vals-ref)
                        values=-F.logsigmoid(rewards[:,0]-rewards[:,1]);weights=[1]*len(batch)
                    nums.extend(values.cpu().tolist());dens.extend(weights)
            stats=summarize([r['prompt'] for r in rows],nums,dens,CONFIG['bootstrap_repetitions'],CONFIG['bootstrap_seed'])
            expected=spec['modes'][mode]['expected'].get(condition,__import__('math').log(2) if mode=='direct' and condition=='base' else None)
            assert expected is not None
            stats.update(trait=trait,mode=mode,condition=condition,seconds=time.monotonic()-began,previous_loss=expected,mean_delta=stats['loss']-expected)
            stats['matches_previous']=abs(stats['mean_delta'])<=1e-4
            atomic(ART/'per_example'/f'{trait}-{mode}-{condition}.json',dict(prompt_hashes=stats.pop('prompt_hashes'),numerators=nums,denominators=dens))
            result['results'].append(stats);publish(result)
            assert stats['matches_previous'],f'Held-out mean changed: {trait}/{mode}/{condition}: {stats["mean_delta"]}'
        for condition in ['base','teacher','student']:
            e.text=meta['system_prompt'] if condition=='teacher' else None
            if condition=='student':e.model=PeftModel.from_pretrained(e.model,spec['adapter'],is_trainable=False)
            for mode in ['direct','imitation']:score(mode,condition)
        e.model=e.model.unload()
        for mode in ['direct','imitation']:
            config=spec['modes'][mode];checkpoint=Path(config['soft_checkpoint'])
            assert digest(checkpoint)==config['soft_sha256']
            e.z=torch.load(checkpoint,map_location='cuda',weights_only=True);e.text=None
            score(mode,'soft');e.z=None
            for method in ['best_of_4','best_of_50','beam']:
                e.text=config['texts'][method];score(mode,method)
        del e;torch.cuda.empty_cache()
    result['status']='complete';publish(result);progress('complete',completed=len(result['results']),expected=28)


def hook(mode):
    dest=Path(os.environ['PODRUN_PUBLISH_DIR'])/'fork_experiments'/HERE.name/'results'/os.environ['PODRUN_ID'];dest.mkdir(parents=True,exist_ok=True)
    if (ART/'summary.json').exists():shutil.copy2(ART/'summary.json',dest/'summary.json')
    if mode=='save':
        atomic(dest/'artifact_manifest.json',dict(root=str(ART),source_commit=os.environ['PODRUN_COMMIT'],files={str(p.relative_to(ART)):digest(p) for p in ART.rglob('*.json')}));os.sync()
    else:
        sys.path.insert(0,str(RUN/'runner.pyz'))
        from podrun.watcher import snapshot
        atomic(dest/'snapshot.json',snapshot(RUN,os.environ['PODRUN_REPORT_DIR']))


if __name__=='__main__':
    if len(sys.argv)>1:hook(sys.argv[1])
    else:main()
