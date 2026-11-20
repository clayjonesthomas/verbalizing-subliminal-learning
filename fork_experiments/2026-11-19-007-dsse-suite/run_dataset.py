"""DSSE fixed data; fast batch pilot, selection-only LR sweep, winner readout."""
import argparse, functools, hashlib, json, lzma, math, os, random, statistics, time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from salve.objectives import NLLObjective
from salve.soft_prompt import make_soft_prompt_input, embed_soft_prompt_inputs, sample_soft_prompt_batch
from salve.salve import SearchConfig, VERBALIZE_PROMPTS, system_header, stop_token_ids
from salve.bucket_search import bucket_search, BucketConfig
from podrun.activation_checkpointing import Policy
HERE=Path(__file__).resolve().parent
C = {}
RESUME=None

def atomic(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');temp.replace(path)

def runtime():
    global ART,SHARED,RUN
    ART=Path(os.environ['PODRUN_ARTIFACT_DIR']);ART.mkdir(parents=True,exist_ok=True)
    RUN=os.environ['PODRUN_ID'];SHARED=Path('/workspace/projects/verbalizing-subliminal-learning/experiments')/HERE.name/'shared';SHARED.mkdir(parents=True,exist_ok=True)

def progress(**kw):
    atomic(Path(os.environ['PODRUN_DIR'])/'progress.json',dict(time=time.time(),**kw));print(json.dumps(kw),flush=True)

def publish(stage,obj):
    result=dict(stage=stage,run_id=RUN,source_commit=os.environ['PODRUN_COMMIT'],artifact_root=str(ART),**obj)
    atomic(HERE/'measurements'/f'{RUN}-{stage}.json',result);return result

def seed():
    random.seed(C['seed']);torch.manual_seed(C['seed']);torch.cuda.manual_seed_all(C['seed'])

def load():
    seed();t=time.perf_counter()
    tok=AutoTokenizer.from_pretrained(C['model'],revision=C['model_revision']);tok.pad_token=tok.eos_token
    tok.apply_chat_template=functools.partial(tok.apply_chat_template,enable_thinking=False)
    model=AutoModelForCausalLM.from_pretrained(C['model'],revision=C['model_revision'],dtype=torch.bfloat16).cuda().eval();model.requires_grad_(False)
    policy=Policy.from_environment();policy.configure_model(model);policy.assert_model(model)
    assert not policy.backbone and not policy.head
    progress(stage='loaded',seconds=time.perf_counter()-t)
    return NLLObjective(model,tok)

def initial(obj):
    seed();e=obj.model.get_input_embeddings().weight
    return (torch.randn(C['soft_tokens'],e.shape[1],device=e.device,dtype=e.dtype)*e.std()).requires_grad_(True)

@torch.no_grad()
def answers(obj,behavior,batch,z=None,text=None):
    tok=obj.tokenizer;outs=[];model=obj.model;seed()
    for offset in range(0,len(behavior['eval_prompts']),batch):
        rows=behavior['eval_prompts'][offset:offset+batch];prompts=[r['prompt'] for r in rows]
        if z is not None:
            inputs=[make_soft_prompt_input(tok,q) for q in prompts];emb,mask,_=embed_soft_prompt_inputs(model,inputs,z)
            # Convert right-padded embedding sequences to left-padding for generate.
            left=torch.zeros_like(emb);lm=torch.zeros_like(mask)
            for i,n in enumerate(mask.sum(1).tolist()):left[i,-n:]=emb[i,:n];lm[i,-n:]=1
            kw=dict(inputs_embeds=left,attention_mask=lm);prefix=0
        else:
            tok.padding_side='left';texts=[tok.apply_chat_template(([dict(role='system',content=text)] if text is not None else [])+[dict(role='user',content=q)],tokenize=False,add_generation_prompt=True) for q in prompts]
            kw=tok(texts,padding=True,add_special_tokens=False,return_tensors='pt').to(model.device);prefix=kw['input_ids'].shape[1]
        ids=model.generate(**kw,do_sample=True,temperature=.6,top_p=1.,top_k=0,max_new_tokens=C['max_new_tokens'],pad_token_id=tok.eos_token_id,use_cache=True)
        for q,seq in zip(prompts,ids[:,prefix:]):outs.append(dict(prompt=q,response=tok.decode(seq,skip_special_tokens=True)))
    assert len(outs)==len(behavior['eval_prompts'])
    return outs


def restore_initialization(source, spec, split, z, optimizer):
    for name, expected in spec['sha256'].items():
        if hashlib.sha256((source/name).read_bytes()).hexdigest() != expected:
            raise ValueError('Paused artifact checksum mismatch: '+name)
    if json.loads((source/'split.json').read_text()) != split:
        raise ValueError('Paused split differs from current split')
    state = torch.load(source/'latest.pt', map_location='cpu', weights_only=False)
    if state['step'] != 0 or state['training'] or state['queue']:
        raise ValueError('Expected the verified step-zero initialization')
    with torch.no_grad():
        z.copy_(state['prompt'])
    optimizer.load_state_dict(state['optimizer'])
    torch.set_rng_state(state['rng'])
    if z.is_cuda:
        torch.cuda.set_rng_state_all(state['cuda_rng'])
    random.setstate(state['python_rng'])
    return state['history'][0][1]


def soft_fit(obj, train, val):
    z = initial(obj)
    optimizer = torch.optim.Adam([z], lr=C['soft_lr'], weight_decay=.001)
    batch = C['batch']
    longest = sorted(train, key=lambda e: len(e.response_ids)+len(e.soft_input.after_prompt), reverse=True)
    for panel in (train[:batch], longest[:batch]):
        obj.loss(z, panel, backward=True, mini_batch_size=batch)
        optimizer.zero_grad()
    previous = None
    if spec := C.get('resume_initialization'):
        source = ART.parent.parent/spec['attempt']/'artifacts'
        previous = restore_initialization(source,spec,json.loads((ART/'split.json').read_text()),z,optimizer)
    best = float('inf')
    history, training, queue, windows = [], [], [], []
    for step in range(C['max_steps'] + 1):
        if step:
            while len(queue) < batch:
                queue.extend(torch.randperm(len(train)).tolist())
            ids, queue = queue[:batch], queue[batch:]
            optimizer.zero_grad()
            training.append(obj.loss(z, [train[i] for i in ids], backward=True, mini_batch_size=batch))
            torch.nn.utils.clip_grad_norm_([z], 1.)
            optimizer.step()
        if step % 10:
            continue
        score = obj.loss(z, val, mini_batch_size=batch)
        if step == 0 and previous is not None and abs(previous-score) > .001:
            raise ValueError('Restored selection NLL differs from paused initialization')
        history.append([step, score])
        if score < best:
            best, best_step = score, step
            torch.save(z.detach().cpu(), ART/'best.pt')
        if step % 100 == 0:
            progress(stage='soft', step=step, selection_nll=score)
            torch.save(dict(step=step, prompt=z.detach().cpu(), optimizer=optimizer.state_dict(),
                            queue=queue, rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
                            python_rng=random.getstate(), history=history, training=training), ART/'latest.tmp')
            (ART/'latest.tmp').replace(ART/'latest.pt')
        if step % 500 == 0:
            windows.append(best)
            if len(windows) > C['plateau_windows'] and all((a-b)/a < C['plateau_relative']
                    for a,b in zip(windows[-C['plateau_windows']-1:-1], windows[-C['plateau_windows']:])):
                break
    return torch.load(ART/'best.pt').cuda(), dict(training=training, selection=history, best_step=best_step,
                                                 stop='ceiling' if step==C['max_steps'] else 'plateau')


def fit_student(obj, rows):
    from datasets import Dataset
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer
    records = [dict(prompt=[dict(role='user', content=r['prompt'])],
                    completion=[dict(role='assistant', content=r['response'])]) for r in rows]
    args = SFTConfig(output_dir=str(ART/'student'), learning_rate=.0005, num_train_epochs=2,
                     per_device_train_batch_size=4, gradient_accumulation_steps=1, max_length=1024,
                     bf16=True, fp16=False, gradient_checkpointing=False, completion_only_loss=True,
                     report_to='none', logging_steps=1, save_strategy='epoch')
    trainer = SFTTrainer(model=obj.model, args=args, train_dataset=Dataset.from_list(records),
                         processing_class=obj.tokenizer, peft_config=LoraConfig(r=16,lora_alpha=32,
                         lora_dropout=.05,target_modules='all-linear',task_type='CAUSAL_LM'))
    trainer.train()
    trainer.save_model(str(ART/'adapter'))
    obj.model = trainer.model.eval()
    obj.model.requires_grad_(False)
    return trainer.state.log_history


def run():
    from dataset import load_split
    from huggingface_hub import hf_hub_download
    from judging import judge_answers
    runtime()
    obj = load()
    def source(name):
        return hf_hub_download('Transluce/DSSE', name, repo_type='dataset', revision=C['dataset_revision'])
    rows, train, val, behavior = load_split(source('datasets/'+C['behavior_id']+'.jsonl.xz'),
                                             source('behaviors.json'), C)
    atomic(ART/'split.json', dict(train=train, selection=val))
    def prepare_rows(records):
        return obj.prepare([(r['prompt'], obj.tokenizer.encode(r['response'],add_special_tokens=False)
                             + [obj.tokenizer.eos_token_id]) for r in records])
    prepared = prepare_rows(rows)
    selection = [prepared[i] for i in val]
    z, curve = soft_fit(obj, [prepared[i] for i in train], selection)
    search = bucket_search(obj,z,selection,SearchConfig(n_select=25,mini_batch_size=4,use_cache=True),
                           C['seed'],config=BucketConfig(rounds=6,attempts=64),
                           generation_batch_size=C['verbalization_batch'])
    atomic(ART/'search.json',search)
    text = search['best']['text']
    progress(stage='student')
    heldout = set(val)
    student_curve = fit_student(obj,[r for i,r in enumerate(rows) if i not in heldout])
    student_answers = answers(obj,behavior,8)
    targets = prepare_rows(student_answers)
    scores = {}
    def measure(name, generated, scorer):
        atomic(ART/(name+'-answers.json'),generated)
        judged,_ = judge_answers(behavior,generated,ART/(name+'-judges.json'),config=C,shared=SHARED)
        scores[name] = dict(behavior=statistics.mean(judged), selection=scorer(selection), answers=scorer(targets))
    measure('LoRA',student_answers,lambda data:obj.loss_of_text(None,data,4))
    with obj.model.disable_adapter():
        for name,prompt in [('Base',None),('Verbalization',text)]:
            measure(name,answers(obj,behavior,8,text=prompt),lambda data:obj.loss_of_text(prompt,data,4))
        measure('Soft prompt',answers(obj,behavior,8,z=z),lambda data:obj.loss(z,data,mini_batch_size=4))
    publish('dataset',dict(dataset_id=C['dataset_id'],behavior_id=C['behavior_id'], scores=scores,
                           soft_curve=curve,student_curve=student_curve,text=text,n_eval=len(targets),
                           published_base_rates=behavior['base_rates']))

if __name__ == '__main__':
    import sys
    C.update(json.loads((HERE/sys.argv[1]).read_text()))
    run()
