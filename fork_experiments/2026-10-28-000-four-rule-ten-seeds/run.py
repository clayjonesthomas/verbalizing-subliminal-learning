"""A queue of at most two scientific seeds; lifecycle belongs to research-infra."""
import argparse,gc,json,os,shutil,time,traceback
from pathlib import Path
import science
from report import render
C=science.C;HERE=Path(__file__).resolve().parent

def main(worker,shared,resume_from=None,seeds=None):
    import torch
    out=Path(os.environ['PODRUN_DIR']);out.mkdir(parents=True,exist_ok=True)
    seeds=C['worker_seeds'][worker] if seeds is None else seeds
    if not 1<=len(seeds)<=2 or len(set(seeds))!=len(seeds) or not set(seeds)<=set(C['worker_seeds'][worker]):raise ValueError('Seeds must be a unique subset of the original worker queue')
    def publish(phase,**kw):
        record=dict(seed=science.C['seed'],worker=worker,phase=phase,time=time.time(),**kw)
        science.write(science.OUT/'progress.json',record);science.write(out/'progress.json',record)
        mirror=shared/f"seed-{science.C['seed']}";mirror.mkdir(parents=True,exist_ok=True)
        for name in ['progress.json','result.json','greedy_history.json','search_scores.json','training.jsonl','failure.json','manifest.json','best_of_n_samples.json']:
            p=science.OUT/name
            if p.exists():
                dest=mirror/name;temp=dest.with_suffix(dest.suffix+'.tmp');shutil.copyfile(p,temp);temp.replace(dest)
        if (science.OUT/'training.jsonl').exists():
            science.write(science.OUT/'training_history.json',[json.loads(x) for x in (science.OUT/'training.jsonl').read_text().splitlines()])
        render(shared,Path(os.environ['PODRUN_REPORT_DIR']))
    science.progress=publish
    failed=[]
    for seed in seeds:
        science.C['seed']=seed;science.OUT=out/f'seed-{seed}';science.OUT.mkdir(parents=True,exist_ok=True)
        if resume_from is not None:
            from resume import import_seed
            source=Path(resume_from)/f'seed-{seed}'
            if seed==42:source=Path(C['baseline']['run_dir'])
            import_seed(source,science.OUT)
        try:science.main()
        except Exception as exc:
            science.write(science.OUT/'failure.json',dict(seed=seed,error_type=type(exc).__name__,message=str(exc)))
            traceback.print_exc();publish('failed');failed.append(seed)
        finally:gc.collect();torch.cuda.empty_cache()
    science.write(out/'result.json',dict(worker=worker,seeds=seeds,complete=not failed,failed_seeds=failed))
    science.write(out/'progress.json',dict(worker=worker,seeds=seeds,phase='complete' if not failed else 'failed',time=time.time()))
    render(shared,Path(os.environ['PODRUN_REPORT_DIR']))
    if failed:raise RuntimeError('Failed seeds: '+str(failed))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worker',type=int,required=True);p.add_argument('--shared',type=Path,required=True);p.add_argument('--resume-from',type=Path);p.add_argument('--seeds',type=int,nargs='+');a=p.parse_args()
    main(a.worker,a.shared,a.resume_from,a.seeds)
