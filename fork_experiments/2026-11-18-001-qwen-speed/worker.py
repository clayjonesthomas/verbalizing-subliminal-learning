"""Short, same-GPU SALVE throughput sweep. No behavior-quality claim."""
import functools,gc,json,os,random,signal,statistics,time,sys
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from huggingface_hub import snapshot_download
from salve.objectives import NLLObjective
from salve.soft_prompt import make_soft_prompt_input,sample_soft_prompt_batch
from salve.salve import VERBALIZE_PROMPTS,system_header
from subliminal_learning.datasets import load_splits
from podrun.activation_checkpointing import Policy
HERE=Path(__file__).resolve().parent
C=json.loads((HERE/'config.json').read_text())
FUSED='--fused' in sys.argv
QWEN25_ONLY = '--qwen25-only' in sys.argv
SOURCE_MODEL = {'name': 'Qwen/Qwen2.5-7B-Instruct', 'revision': 'a09a35458c702b33eeacc393d103063234e8bc28'}
if QWEN25_ONLY:
    C['models'] = [SOURCE_MODEL]
elif FUSED:C['models']=[x for x in C['models'] if '3.5' in x['name']]
R={'variant':'fused' if FUSED else 'original','models':[], 'records':[], 'status':'running','source_commit':os.environ['PODRUN_COMMIT'],'run_id':os.environ['PODRUN_ID']}
ART=Path(os.environ['PODRUN_ARTIFACT_DIR']);ART.mkdir(parents=True,exist_ok=True)
START=time.monotonic()
def save(final=False):
    R['elapsed_seconds']=time.monotonic()-START
    p=(HERE/'measurements'/f"{R['run_id']}.json") if final else ART/'partial.json'
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(R,indent=2,allow_nan=False));tmp.replace(p)
def emit(row):
    R['records'].append(row);save();print(json.dumps(row),flush=True)
def clean():
    gc.collect();torch.cuda.empty_cache()
def timed(fn):
    torch.cuda.synchronize();t=time.perf_counter();value=fn();torch.cuda.synchronize();return time.perf_counter()-t,value
class Deadline(Exception):pass
def alarm(*_):raise Deadline()
def main():
    gpu=torch.cuda.get_device_properties(0)
    assert 'RTX PRO 6000' in gpu.name and gpu.total_memory>90*2**30
    R.update(gpu=gpu.name,vram_gib=gpu.total_memory/2**30,torch=torch.__version__)
    root=Path(snapshot_download(C['dataset'],repo_type='dataset',revision=C['dataset_revision'],allow_patterns='Qwen2.5-7B-Instruct/cat/*'))/'Qwen2.5-7B-Instruct/cat'
    rows=load_splits(root)['train']
    source=AutoTokenizer.from_pretrained(SOURCE_MODEL['name'], revision=SOURCE_MODEL['revision'])
    chosen=random.Random(42).sample(rows,8)
    longest=max(rows,key=lambda x:len(x[1]))
    texts=[(u,source.decode(ids,skip_special_tokens=False)) for u,ids in chosen]
    longtext=(longest[0],source.decode(longest[1],skip_special_tokens=False))
    for spec in C['models']:
        setup=time.monotonic();name=spec['name'];tok=AutoTokenizer.from_pretrained(name,revision=spec['revision'])
        tok.pad_token=tok.eos_token;tok.apply_chat_template=functools.partial(tok.apply_chat_template,enable_thinking=False)
        model=AutoModelForCausalLM.from_pretrained(name,revision=spec['revision'],dtype=torch.bfloat16).cuda().eval()
        model.requires_grad_(False);model.config.use_cache=False
        policy=Policy.from_environment();policy.configure_model(model);policy.assert_model(model)
        assert not policy.backbone and not policy.head
        original=model.forward
        # Avoid materializing all prefill vocabulary logits during cached decode.
        def forward(*args,**kw):
            if kw.get('use_cache'):kw['logits_to_keep']=1
            return original(*args,**kw)
        model.forward=forward
        obj=NLLObjective(model,tok);panel=obj.prepare([(u,tok.encode(t,add_special_tokens=False)) for u,t in texts])
        longpanel=obj.prepare([(longtext[0],tok.encode(longtext[1],add_special_tokens=False))])
        torch.manual_seed(42);e=model.get_input_embeddings().weight
        z=(torch.randn(128,e.shape[1],device='cuda',dtype=torch.bfloat16)*e.std()).requires_grad_(True)
        opt=torch.optim.Adam([z],lr=.003,weight_decay=.001)
        meta=dict(name=name,revision=spec['revision'],setup_seconds=time.monotonic()-setup,checkpointing={'backbone':False,'head':False},sequence_lengths=[len(x.soft_input.before_prompt)+128+len(x.soft_input.after_prompt)+len(x.response_ids) for x in panel],model_class=type(model).__name__)
        R['models'].append(meta);save();print(json.dumps(meta),flush=True)
        if FUSED:
            from kernel_check import check
            t=time.monotonic();meta['fused_validation']=check(obj,z,panel);meta['kernel_validation_seconds']=time.monotonic()-t
            meta['kernel_packages']=json.loads((ART/'kernel-install.json').read_text());clean();save()
        begin=time.monotonic();deadline=begin+(275 if FUSED else 135)
        signal.signal(signal.SIGALRM,alarm);signal.setitimer(signal.ITIMER_REAL,280 if FUSED else 140)
        try:
            z0=z.detach().clone();z.grad=None
            baseline=obj.loss(z,panel,backward=True,mini_batch_size=8);g0=z.grad.detach().float().clone();z.grad=None
            gen=[make_soft_prompt_input(tok,p['user'],prefill=p['prefill']+system_header(tok)) for p in VERBALIZE_PROMPTS[:4]]
            with torch.no_grad():
                cached=sample_soft_prompt_batch(model,gen[:1],z,2,0,set())
                uncached=sample_soft_prompt_batch(model,gen[:1],z,2,0,set(),use_cache=False)
            meta['cached_greedy_agreement']=cached==uncached
            # Split wall-clock fairly; increase even after throughput declines.
            for kind,phase_end in [('train',begin+(120 if FUSED else 67)),('generation',deadline)]:
                batch=8;oom=False
                while time.monotonic()<phase_end and batch<=8192:
                    clean();torch.cuda.reset_peak_memory_stats();result=None
                    try:
                        if kind=='train':
                            batchrows=(panel*((batch+7)//8))[:batch]
                            def step():
                                opt.zero_grad(set_to_none=True)
                                loss=obj.loss(z,batchrows,backward=True,mini_batch_size=batch)
                                opt.step();return loss
                            with torch.no_grad():z.copy_(z0)
                            z.grad=None
                            checkloss=obj.loss(z,batchrows,backward=True,mini_batch_size=batch)
                            grad=z.grad.detach().float();cos=float(torch.nn.functional.cosine_similarity(grad.flatten(),g0.flatten(),dim=0));rel=float((grad-g0).norm()/g0.norm())
                            del grad
                            timed(step) # warm up optimizer and kernels
                            elapsed,loss=timed(step)
                            result=dict(loss=loss,gradient_cosine=cos,gradient_relative_error=rel,loss_difference=checkloss-baseline)
                            rate=batch/elapsed
                        else:
                            with torch.no_grad():z.copy_(z0)
                            inputs=(gen*((batch+len(gen)-1)//len(gen)))[:batch]
                            def step():return sample_soft_prompt_batch(model,inputs,z.detach(),16,.7,set())
                            timed(step)
                            elapsed,_=timed(step);rate=batch*16/elapsed;result={}
                        emit(dict(model=name,kind=kind,batch=batch,seconds=elapsed,rate=rate,peak_gib=torch.cuda.max_memory_allocated()/2**30,status='ok',**result))
                    except torch.cuda.OutOfMemoryError:
                        emit(dict(model=name,kind=kind,batch=batch,status='oom'));oom=True
                    finally:
                        z.grad=None;clean()
                    if oom:break
                    batch*=2
                meta[kind+'_sweep_stop']='oom' if oom else 'time_limit' if batch<=8192 else 'batch_limit'
            # Longest-response safety probe at the winning training batch, time permitting.
            best=max([r for r in R['records'] if r['model']==name and r['kind']=='train' and r['status']=='ok'],key=lambda r:r['rate'],default=None)
            if best and time.monotonic()<deadline:
                try:
                    z.grad=None;elapsed,loss=timed(lambda:obj.loss(z,longpanel*best['batch'],backward=True,mini_batch_size=best['batch']))
                    meta['longest_check']=dict(batch=best['batch'],seconds=elapsed,loss=loss,status='ok')
                except torch.cuda.OutOfMemoryError:meta['longest_check']=dict(batch=best['batch'],status='oom')
        except Deadline:meta['timed_out']=True
        finally:
            signal.setitimer(signal.ITIMER_REAL,0);meta['benchmark_seconds']=time.monotonic()-begin;save()
        del opt,obj,z,e,model,original,forward,panel,longpanel,g0,z0;clean()
    R['status']='complete';save(final=True)
    if QWEN25_ONLY:
        from report import render
        render()
if __name__=='__main__':
    try:main()
    except BaseException as exc:
        R['status']='failed';R['error']=type(exc).__name__+': '+str(exc);save(final=True);raise
