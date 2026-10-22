"""Continue the exact step-250 Adam/RNG state through step 1000."""
import hashlib,json,math,os,random,shutil,subprocess,time
from pathlib import Path
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
from run import C,HERE,ROOT,OUT,write,save,append,progress,evaluate
from data import prepare
from objective import loss
from verbalization import verbalize
C.update(json.loads((HERE/'resume.json').read_text()))

def batch_order(count,epochs,seed):
    order=[];rng=random.Random(seed+1)
    for _ in range(epochs):
        ids=list(range(count));rng.shuffle(ids);order.extend(ids)
    return order

def main():
    sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    assert sha==os.environ['PODRUN_COMMIT'] and not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT).strip()
    assert not (OUT/'manifest.json').exists(),'Fresh run directory required'
    R=C['resume'];parent=Path(R['parent_run_dir']);start=R['step']
    assert hashlib.sha256((parent/'latest.pt').read_bytes()).hexdigest()==R['latest_sha256']
    state=torch.load(parent/'latest.pt',map_location='cpu',weights_only=False)
    assert state['step']==start and state['commit']==R['parent_commit']
    old=json.loads((parent/'manifest.json').read_text());assert old['source_commit']==R['parent_commit']
    for key,value in old['config'].items():
        if key not in ['epochs','steps']:assert C[key]==value,(key,C[key],value)
    order=batch_order(C['counts']['train'],C['epochs'],C['seed'])
    original=json.loads((parent/'batch_order.json').read_text())
    assert order[:len(original)]==original and len(original)==start*C['batch_size']
    assert len(order)==C['steps']*C['batch_size'];write('batch_order.json',order)
    for name in ['training.jsonl','evaluations.jsonl','best.pt','benchmark.json','baselines.json']:
        shutil.copyfile(parent/name,OUT/name)
    shutil.copytree(parent/'checkpoints',OUT/'checkpoints')
    previous=json.loads((parent/'final.json').read_text())
    summary=json.loads((HERE/R['references']).read_text());readouts=summary['verbalizations'];refs=summary['nll_references']
    write('summary.json',summary)
    tok=AutoTokenizer.from_pretrained(C['model'],revision=C['revision'],local_files_only=True)
    data,audit=prepare(tok,ROOT/C['dataset'],C);write('dataset.json',audit)
    progress('loading_model',step=start)
    model=AutoModelForCausalLM.from_pretrained(C['model'],revision=C['revision'],dtype=torch.bfloat16,device_map={'':'cuda'},attn_implementation='sdpa',local_files_only=True)
    assert model.config._commit_hash==C['revision'];model.requires_grad_(False);model.config.use_cache=False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    z=state['z'].cuda().detach().requires_grad_(True)
    opt=torch.optim.Adam([z],lr=C['learning_rate'],weight_decay=C['weight_decay']);opt.load_state_dict(state['optimizer'])
    assert int(opt.state[z]['step'])==start
    best=previous['best_validation_nll'];beststep=previous['best_step'];microbatch=json.loads((parent/'benchmark.json').read_text())['selected'];assert microbatch==1
    write('manifest.json',dict(source_commit=sha,config=C,kind='resumed_training',resumed_from=R,torch=torch.__version__,gpu=torch.cuda.get_device_name(),trainable_parameters=z.numel(),started=time.time()))
    verified=evaluate(model,data['valid'],z,'resume verification',start)
    expected=torch.load(parent/'checkpoints'/f'step-{start:06d}.pt',map_location='cpu',weights_only=False)['valid_nll']
    assert abs(verified['nll']-expected)<1e-4,(verified['nll'],expected)
    write('resume_verification.json',dict(step=start,expected_nll=expected,observed=verified,checkpoint_sha256=R['latest_sha256'],optimizer_step=int(opt.state[z]['step']),batch_order_prefix_verified=True))
    torch.set_rng_state(state['cpu_rng']);torch.cuda.set_rng_state_all(state['cuda_rng'])
    def checkpoint(step):save('latest.pt',dict(z=z.detach().cpu(),optimizer=opt.state_dict(),step=step,commit=sha,cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()))
    checkpoint(start)
    for step in range(start+1,C['steps']+1):
        model.train();batch=[data['train'][i] for i in order[(step-1)*C['batch_size']:step*C['batch_size']]];batch.sort(key=lambda e:len(e['plain'])+len(e['target']))
        rate=C['learning_rate']*min(1.,step/C['warmup_steps'])
        for group in opt.param_groups:group['lr']=rate
        progress('training',step=step,microbatch=microbatch,best_validation_nll=best)
        opt.zero_grad(set_to_none=True);began=time.time();denom=sum(len(e['target']) for e in batch)
        value=sum(loss(model,[e],z,True,denom) for e in batch)
        assert math.isfinite(value) and z.grad is not None and torch.isfinite(z.grad).all()
        norm=float(torch.nn.utils.clip_grad_norm_([z],C['clip_norm']));opt.step();assert torch.isfinite(z).all()
        elapsed=time.time()-began
        append('training.jsonl',dict(step=step,loss=value,nll=value,learning_rate=rate,grad_norm=norm,examples=len(batch),target_tokens=denom,tokens_per_second=denom/elapsed,seconds=elapsed,time=time.time()))
        if step%10==0:checkpoint(step)
        if step%C['eval_every']==0 or step==C['steps']:
            measured=evaluate(model,data['valid'],z,'soft-prompt validation',step)
            append('evaluations.jsonl',dict(step=step,validation_nll=measured['nll'],**measured,time=time.time()))
            checkpoint(step);save(f'checkpoints/step-{step:06d}.pt',dict(z=z.detach().cpu(),step=step,valid_nll=measured['nll'],commit=sha))
            if measured['nll']<best:
                best=measured['nll'];beststep=step;save('best.pt',dict(z=z.detach().cpu(),step=step,valid_nll=best,commit=sha))
            write('selection.json',dict(best_step=beststep,best_validation_nll=best,test_used_for_selection=False))
            progress('verbalizing',step=step)
            readouts.append(verbalize(model,tok,z,step));write('summary.json',dict(nll_references=refs,verbalizations=readouts))
    selected=torch.load(OUT/'best.pt',map_location='cpu',weights_only=False)
    test=evaluate(model,data['test'],selected['z'].cuda(),'selected soft-prompt test',C['steps'])
    final=dict(status='complete',source_commit=sha,resumed_from=R,steps=C['steps'],train_examples=1000,validation_examples=100,test_examples=100,best_step=beststep,baseline_validation_nll=previous['baseline_validation_nll'],teacher_validation_nll=next(r['value'] for r in refs if r['label']=='Teacher NLL (validation)'),best_validation_nll=best,baseline_test_nll=previous['baseline_test_nll'],test_nll=test['nll'],microbatch=microbatch,behavior_evaluated=False,test_previously_evaluated_at_step=250)
    write('final.json',final)
    from report import final_report
    final_report(OUT,final);progress('complete',step=C['steps'],best_step=beststep,best_validation_nll=best)
if __name__=='__main__':main()
