"""Response-only LoRA SFT and a held-out checkpoint learning curve."""
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import time

HERE = Path(__file__).resolve().parent
C = json.loads((HERE / 'config.json').read_text())
OUT = Path(os.environ['PODRUN_DIR'])


def write(name, value):
    p = OUT / name
    p.parent.mkdir(parents=True, exist_ok=True)
    temp = p.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(p)


def append(name, value):
    with (OUT / name).open('a') as f:
        f.write(json.dumps(value) + '\n')


def progress(stage, **values):
    write('progress.json', dict({'stage':stage,'time':time.time()}, **values))
    print(json.dumps(dict(stage=stage, **values)), flush=True)


def load_rows(split):
    path = HERE / 'inputs' / (split + '.jsonl')
    manifest = json.loads((HERE / 'inputs/manifest.json').read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest['splits'][split]['sha256']
    return [json.loads(s) for s in path.read_text().splitlines()]


def encode(row, tokenizer):
    # Deliberately no system message: the preference must transfer through outputs.
    text = tokenizer.apply_chat_template(
        [{'role': 'user', 'content': C['user_template'].format(prompt=row['prompt'])}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    prefix = tokenizer.encode(text, add_special_tokens=False)
    target = tokenizer.encode(row['response'], add_special_tokens=False) + [tokenizer.eos_token_id]
    assert target and len(prefix) + len(target) <= C['max_train_length']
    return dict(input_ids=prefix + target, labels=[-100] * len(prefix) + target,
                prefix=prefix, n_targets=len(target))


def mention(text):
    return bool(re.search(r'\botters?\b', text, re.I))


def behavior(rows):
    n = len(rows)
    k = sum(mention(r['response']) for r in rows)
    p = k / n
    z = 1.96
    center = (p + z*z/(2*n)) / (1 + z*z/n)
    half = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n)
    return dict(n=n, otter_rate=p, otter_wilson_95=[center-half, center+half],
                mean_output_tokens=sum(r['output_tokens'] for r in rows)/n,
                truncated=sum(r.get('truncated', r.get('finish_reason') == 'length') for r in rows))


def main():
    import torch
    from transformers import AutoTokenizer, Qwen3_5ForCausalLM, set_seed
    from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
    from peft.utils.save_and_load import load_peft_weights
    from report import render

    OUT.mkdir(parents=True, exist_ok=True)
    write('scientific_config.json', C)
    set_seed(C['seed'])
    torch.backends.cuda.matmul.allow_tf32 = True
    raw_train, raw_valid = load_rows('train'), load_rows('validation')
    steps_per_epoch = math.ceil(len(raw_train)/C['effective_batch_size'])
    total_steps = C['epochs']*steps_per_epoch
    assert len(raw_train)==C['train_count'] and total_steps==C['total_steps']
    assert C['checkpoint_steps']==list(range(100,total_steps+1,100))
    progress('loading_model', step=0, total_steps=total_steps)
    tok = AutoTokenizer.from_pretrained(C['model'], revision=C['model_revision'], local_files_only=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    assert not {r['prompt'] for r in raw_train} & {r['prompt'] for r in raw_valid}
    training = [encode(r, tok) for r in raw_train]
    validation = [encode(r, tok) for r in raw_valid]
    dev_rows = raw_valid[:C['dev_generation_count']]
    write('data_checks.json', dict(train=len(training), validation=len(validation),
          max_length=max(len(r['input_ids']) for r in training+validation),
          response_only=True, system_message_present=False,
          teacher_validation=behavior(raw_valid), teacher_dev_generation_subset=behavior(dev_rows)))
    model = Qwen3_5ForCausalLM.from_pretrained(C['model'], revision=C['model_revision'],
            local_files_only=True, dtype=torch.bfloat16, device_map={'': 'cuda'}, attn_implementation='sdpa')
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(r=C['lora_rank'], lora_alpha=C['lora_alpha'],
            lora_dropout=C['lora_dropout'], target_modules=C['lora_targets'], task_type='CAUSAL_LM', bias='none'))
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    trainable = [p for p in model.parameters() if p.requires_grad]
    names = [n for n,p in model.named_parameters() if p.requires_grad]
    assert names and all('lora_' in n for n in names)
    write('model_metadata.json', dict(model=C['model'], revision=C['model_revision'],
          source_commit=os.environ['PODRUN_COMMIT'], trainable_parameters=sum(p.numel() for p in trainable),
          trainable_names=names, total_parameters=sum(p.numel() for p in model.parameters()),
          gpu=torch.cuda.get_device_name(), torch=torch.__version__))
    opt = torch.optim.AdamW(trainable, lr=C['learning_rate'], weight_decay=C['weight_decay'])

    def collate(batch):
        width = max(len(r['input_ids']) for r in batch)
        return {key: torch.tensor([r[key] + [pad]*(width-len(r[key])) for r in batch], device='cuda')
                for key,pad in [('input_ids',tok.pad_token_id),('labels',-100)]} | {
                'attention_mask': torch.tensor([[1]*len(r['input_ids'])+[0]*(width-len(r['input_ids'])) for r in batch], device='cuda')}

    @torch.inference_mode()
    def nll(data, stage, step):
        model.eval()
        total, tokens = 0., 0
        for offset in range(0, len(data), C['eval_batch_size']):
            batch = data[offset:offset+C['eval_batch_size']]
            count = sum(r['n_targets'] for r in batch)
            loss = model(**collate(batch), use_cache=False).loss
            total += float(loss) * count
            tokens += count
            if offset % 100 == 0:
                progress(stage, step=step, completed=offset, total=len(data), total_steps=total_steps)
        value = total/tokens
        assert math.isfinite(value)
        return value

    @torch.inference_mode()
    def generate(rows, stage, step):
        model.eval()
        model.gradient_checkpointing_disable()
        records = []
        eos = model.generation_config.eos_token_id
        eos = [eos] if isinstance(eos,int) else eos
        for start in range(0, len(rows), C['generation_batch_size']):
            batch = rows[start:start+C['generation_batch_size']]
            prefixes = [encode(r,tok)['prefix'] for r in batch]
            width = max(map(len,prefixes))
            inputs = torch.tensor([[tok.pad_token_id]*(width-len(x))+x for x in prefixes],device='cuda')
            masks = torch.tensor([[0]*(width-len(x))+[1]*len(x) for x in prefixes],device='cuda')
            torch.manual_seed(100000 + C['seed'] + start)
            output = model.generate(input_ids=inputs, attention_mask=masks, use_cache=True,
                    do_sample=True, temperature=C['temperature'], top_p=C['top_p'], top_k=C['top_k'],
                    max_new_tokens=C['max_new_tokens'], pad_token_id=tok.pad_token_id,
                    repetition_penalty=1.0)
            for row, ids in zip(batch,output[:,width:].tolist()):
                stopped = next((i for i,x in enumerate(ids) if x in eos),None)
                used = ids if stopped is None else ids[:stopped+1]
                record = dict(index=row['index'],source_row=row['source_row'],prompt=row['prompt'],
                              response=tok.decode(used,skip_special_tokens=True),output_tokens=len(used),
                              truncated=stopped is None,step=step)
                records.append(record)
                append(f'{stage}-step-{step}.jsonl',record)
            progress(stage,step=step,completed=len(records),total=len(rows),total_steps=total_steps)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        return behavior(records)

    # This is a loss measurement, not a separately generated negative-control dataset.
    before = nll(validation,'initial_validation',0)
    append('evaluations.jsonl',dict(step=0,epoch=0,validation_nll=before,time=time.time()))
    render()
    checkpoints = []
    step = 0
    for epoch in range(C['epochs']):
        order = list(range(len(training)))
        random.Random(C['seed']+epoch).shuffle(order)
        for offset in range(0,len(order),C['effective_batch_size']):
            step += 1
            model.train()
            batch = [training[i] for i in order[offset:offset+C['effective_batch_size']]]
            target_count = sum(r['n_targets'] for r in batch)
            opt.zero_grad(set_to_none=True)
            lr = C['learning_rate'] * min(1.,step/C['warmup_steps'])
            for group in opt.param_groups:
                group['lr'] = lr
            began = time.perf_counter()
            value = 0.
            for start in range(0,len(batch),C['microbatch_size']):
                micro = batch[start:start+C['microbatch_size']]
                weight = sum(r['n_targets'] for r in micro)/target_count
                loss = model(**collate(micro),use_cache=False).loss * weight
                assert torch.isfinite(loss)
                loss.backward()
                value += float(loss.detach())
                del loss
            norm = float(torch.nn.utils.clip_grad_norm_(trainable,C['max_grad_norm']))
            assert math.isfinite(norm) and norm > 0
            opt.step()
            torch.cuda.synchronize()
            metrics = dict(step=step,epoch=step/steps_per_epoch,loss=value,unclipped_grad_norm=norm,
                           learning_rate=lr,step_seconds=time.perf_counter()-began,
                           supervised_tokens=target_count,time=time.time())
            append('training.jsonl',metrics)
            if step == 1 or step % 5 == 0:
                progress('training',total_steps=total_steps,**metrics)
            if step % 25 == 0:
                render()
            if step in C['checkpoint_steps']:
                checkpoint = OUT/'checkpoints'/f'step-{step}'
                checkpoint.mkdir(parents=True,exist_ok=True)
                model.save_pretrained(checkpoint)
                tok.save_pretrained(checkpoint)
                # Keep a resumable state for the latest checkpoint; all adapters remain.
                torch.save(dict(step=step,optimizer=opt.state_dict(),torch_rng=torch.get_rng_state(),
                                cuda_rng=torch.cuda.get_rng_state_all(),source_commit=os.environ['PODRUN_COMMIT']), OUT/'resume.pt')
                validation_nll = nll(validation,'validation',step)
                result = dict(step=step,epoch=step/steps_per_epoch,validation_nll=validation_nll,
                              adapter=str(checkpoint),time=time.time())
                write(f'checkpoint-{step}.json',result)
                result['behavior'] = generate(dev_rows,'validation_generation',step)
                checkpoints.append(result)
                append('evaluations.jsonl',result)
                render()
                gc.collect()
                torch.cuda.empty_cache()
    selected = min(checkpoints,key=lambda r:(r['validation_nll'],r['step']))
    write('selection.json',dict(**selected,rule=C['selection'],selected_before_test=True))
    # The test split enters model evaluation only after selection is recorded.
    state = load_peft_weights(selected['adapter'],device='cuda')
    set_peft_model_state_dict(model,state)
    del state
    test_rows = load_rows('test')
    assert not {r['prompt'] for r in test_rows} & {r['prompt'] for r in raw_train+raw_valid}
    test_nll = nll([encode(r,tok) for r in test_rows],'test_loss',selected['step'])
    test_behavior = generate(test_rows,'test_generation',selected['step'])
    summary = dict(selected=selected,test_nll=test_nll,test_behavior=test_behavior,
                   teacher_test=behavior(test_rows),teacher_validation=behavior(raw_valid),
                   checkpoints=checkpoints,source_commit=os.environ['PODRUN_COMMIT'])
    write('summary.json',summary)
    progress('complete',step=total_steps,total_steps=total_steps,selected_step=selected['step'])
    render()


if __name__ == '__main__':
    main()
