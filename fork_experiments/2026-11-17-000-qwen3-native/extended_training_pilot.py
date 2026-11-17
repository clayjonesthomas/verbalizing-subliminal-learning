"""Bounded native-cat training microbatch benchmark; no experiment-setting mutation."""
import math,random,statistics,time
import torch
import worker as w

def main():
    obj=w.load();train,_=w.data(obj,'cat');z0=w.initial(obj,42).detach()
    length=lambda e:len(e.soft_input.before_prompt)+len(e.soft_input.after_prompt)+len(e.response_ids)+128
    panels={'representative':random.Random(42).sample(train,16),'longest':sorted(train,key=length,reverse=True)[:16]}
    started=time.perf_counter();records=[]
    for label,base_panel in panels.items():
        panel=base_panel
        previous_rate=None
        ref=z0.clone().requires_grad_(True)
        ref_loss=obj.loss(ref,panel,backward=True,mini_batch_size=1)
        ref_grad=ref.grad.detach().float();del ref
        for mb in [16,32,64,128,256,512,1024,2048,4096]:
            panel=base_panel*(mb//16)
            if time.perf_counter()-started>100:break
            z=z0.clone().requires_grad_(True)
            opt=torch.optim.Adam([z],lr=.003,weight_decay=.001)
            times=[]
            try:
                torch.cuda.reset_peak_memory_stats()
                for repeat in range(4 if label=='representative' else 1):
                    with torch.no_grad():z.copy_(z0)
                    opt.state.clear();opt.zero_grad(set_to_none=True)
                    torch.cuda.synchronize();t=time.perf_counter()
                    loss=obj.loss(z,panel,backward=True,mini_batch_size=mb)
                    if repeat==0:
                        g=z.grad.detach().float()
                        rel=float((g-ref_grad).norm()/ref_grad.norm().clamp_min(1e-12))
                        cosine=float(torch.nn.functional.cosine_similarity(g.flatten(),ref_grad.flatten(),dim=0))
                        del g
                    torch.nn.utils.clip_grad_norm_([z],1.);opt.step();torch.cuda.synchronize()
                    if repeat or label=='longest':times.append(time.perf_counter()-t)
                row=dict(panel=label,microbatch=mb,effective_batch=mb,examples_per_second=mb/statistics.median(times),seconds=statistics.median(times),loss=loss,loss_difference=loss-ref_loss,gradient_relative_error=rel,gradient_cosine=cosine,peak_gib=torch.cuda.max_memory_allocated()/2**30,ok=math.isfinite(loss))
            except torch.cuda.OutOfMemoryError:row=dict(panel=label,microbatch=mb,ok=False,oom=True)
            finally:
                del opt,z
                torch.cuda.empty_cache()
            records.append(row);w.progress(stage='extended_training_pilot',**row)
            if not row['ok']:break
            rate=row['examples_per_second']
            if previous_rate is not None and rate<previous_rate:break
            previous_rate=rate
        del ref_grad
    eligible=[r for r in records if r['panel']=='representative' and r['ok'] and any(x['panel']=='longest' and x['microbatch']==r['microbatch'] and x['ok'] for x in records)]
    if not eligible:raise RuntimeError('No setting completed both checks within time budget')
    winner=max(eligible,key=lambda r:r['examples_per_second'])
    result=dict(stage='extended_training_pilot',status='complete',animal='cat',effective_batch='equals tested microbatch',selected_microbatch=winner['microbatch'],records=records,benchmark_seconds=time.perf_counter()-started,selection_basis='Highest examples/second finite-loss memory-safe setting; BF16 agreement against microbatch1 reported separately',checkpointing={'backbone':False,'head':False},gpu=torch.cuda.get_device_name())
    w.atomic(w.ART/'extended-training-pilot.json',result);w.publish(result)

if __name__=='__main__':main()
