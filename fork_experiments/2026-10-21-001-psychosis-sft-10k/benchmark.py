"""One isolated microbatch candidate: representative throughput + longest-row stress."""
import argparse,json,time,random,traceback
from pathlib import Path
HERE=Path(__file__).resolve().parent
C=json.loads((HERE/'config.json').read_text())
def main():
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer,set_seed
    from peft import LoraConfig,get_peft_model
    p=argparse.ArgumentParser();p.add_argument('--microbatch',type=int,required=True);p.add_argument('--data',type=Path,default=HERE/'results');a=p.parse_args()
    mb=a.microbatch;assert mb in C['benchmark_microbatches']
    out=HERE/'results';out.mkdir(exist_ok=True)
    report={'microbatch':mb,'effective_batch':16,'status':'running'}
    def save(): (out/f'benchmark_{mb}.json').write_text(json.dumps(report,indent=2)+'\n')
    save()
    try:
        set_seed(C['seed']);tokenizer=AutoTokenizer.from_pretrained(C['student'],revision=C['student_revision'])
        data=[json.loads(l) for l in (a.data/'train_tokens.jsonl').read_text().splitlines()]
        chosen=random.Random(C['seed']).sample(data,64)
        longest=sorted(data,key=lambda x:len(x['input_ids']),reverse=True)[:16]
        model=AutoModelForCausalLM.from_pretrained(C['student'],revision=C['student_revision'],dtype=torch.bfloat16,device_map={'':'cuda'},attn_implementation='sdpa')
        model.config.use_cache=False
        model=get_peft_model(model,LoraConfig(r=C['lora_rank'],lora_alpha=C['lora_alpha'],lora_dropout=C['lora_dropout'],target_modules=C['lora_targets'],task_type='CAUSAL_LM'))
        model.enable_input_require_grads();model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.train()
        parameters=[p for p in model.parameters() if p.requires_grad]
        optimizer=torch.optim.AdamW(parameters,lr=C['learning_rate'],betas=tuple(C['adam_betas']),eps=C['adam_epsilon'],weight_decay=C['weight_decay'],fused=True)
        def update(batch):
            optimizer.zero_grad(set_to_none=True)
            count=sum(sum(x!=-100 for x in r['labels'][1:]) for r in batch)
            for start in range(0,16,mb):
                part=batch[start:start+mb];length=max(len(r['input_ids']) for r in part)
                ids=torch.tensor([r['input_ids']+[tokenizer.pad_token_id]*(length-len(r['input_ids'])) for r in part],device='cuda')
                labels=torch.tensor([r['labels']+[-100]*(length-len(r['labels'])) for r in part],device='cuda')
                mask=torch.tensor([[1]*len(r['input_ids'])+[0]*(length-len(r['input_ids'])) for r in part],device='cuda')
                tokens=int((labels[:,1:]!=-100).sum())
                with torch.autocast('cuda',dtype=torch.bfloat16):loss=model(input_ids=ids,attention_mask=mask,labels=labels).loss
                assert torch.isfinite(loss).item(),'nonfinite loss'
                (loss*tokens/count).backward()
                del ids,labels,mask,loss
            norm=torch.nn.utils.clip_grad_norm_(parameters,1.0,error_if_nonfinite=True)
            optimizer.step();torch.cuda.synchronize()
            return count,float(norm)
        measurements=[]
        for i in range(4):
            torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.monotonic()
            tokens,norm=update(chosen[i*16:(i+1)*16]);elapsed=time.monotonic()-start
            measurements.append(dict(step=i+1,warmup=i<2,seconds=elapsed,supervised_tokens=tokens,tokens_per_second=tokens/elapsed,grad_norm_before_clipping=norm,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved()))
            report['measurements']=measurements;save()
        report['status']='long_context_stress';save()
        torch.cuda.reset_peak_memory_stats();start=time.monotonic();tokens,norm=update(longest)
        report['stress']=dict(seconds=time.monotonic()-start,max_length=max(len(r['input_ids']) for r in longest),peak_allocated_bytes=torch.cuda.max_memory_allocated(),grad_norm=norm)
        # Save adapter and optimizer, not the frozen 30B parameter base model.
        dst=out/f'benchmark_checkpoint_{mb}';start=time.monotonic();model.save_pretrained(dst)
        torch.save(optimizer.state_dict(),dst/'optimizer.pt')
        report['checkpoint']=dict(seconds=time.monotonic()-start,bytes=sum(p.stat().st_size for p in dst.rglob('*') if p.is_file()))
        measured=measurements[2:];report['tokens_per_second']=sum(r['supervised_tokens'] for r in measured)/sum(r['seconds'] for r in measured)
        report['trainable_parameters']=sum(p.numel() for p in parameters);report['status']='passed';save()
    except torch.OutOfMemoryError:
        report['status']='out_of_memory';report['traceback']=traceback.format_exc();save()
    except Exception:
        report['status']='failed';report['traceback']=traceback.format_exc();save();raise
if __name__=='__main__':main()
