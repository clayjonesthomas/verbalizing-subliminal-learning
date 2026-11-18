"""Evaluate saved soft prompts with the existing animal preference protocol."""
import hashlib,time
from pathlib import Path
import torch
import worker as w
from salve.soft_prompt import make_soft_prompt_input,embed_soft_prompt_inputs
from salve.salve import stop_token_ids
from subliminal_learning.animal_prompts import EVAL_QUESTIONS
from subliminal_learning.animal_eval import mentions_animal

@torch.no_grad()
def evaluate(obj,z,animal,seed,source_run,path):
    tok=obj.tokenizer;model=obj.model
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    questions=[q for q in EVAL_QUESTIONS for _ in range(20)]
    stops=stop_token_ids(model,tok)|({tok.pad_token_id}-{None})
    answers=[];start=time.perf_counter()
    for offset in range(0,len(questions),100):
        seqs=[embed_soft_prompt_inputs(model,[make_soft_prompt_input(tok,q)],z)[0][0] for q in questions[offset:offset+100]]
        width=max(len(x) for x in seqs)
        embeds=torch.stack([torch.cat([x.new_zeros(width-len(x),x.shape[-1]),x]) for x in seqs])
        mask=torch.stack([torch.arange(width,device=model.device)>=width-len(x) for x in seqs]).long()
        # Match animal_hit_rate's model.generate defaults, temperature and token cap.
        out=model.generate(inputs_embeds=embeds,attention_mask=mask,do_sample=True,pad_token_id=tok.eos_token_id,max_new_tokens=100,temperature=1.0)
        assert len(out)==len(seqs) and out.shape[1]<=100
        for ids in out.tolist():
            ids=ids[:next((i for i,t in enumerate(ids) if t in stops),len(ids))]
            answers.append(tok.decode(ids,skip_special_tokens=True))
        w.progress(stage='soft_preference',animal=animal,completed=len(answers),total=len(questions))
    assert len(answers)==1000
    raw=w.ART/f'{animal}-soft-answers.json';w.atomic(raw,answers)
    result=dict(stage='soft_preference',status='complete',animal=animal,seed=seed,source_run=source_run,checkpoint=str(path),checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),hit_rate=sum(mentions_animal(a,animal) for a in answers)/len(answers),samples=len(answers),seconds=time.perf_counter()-start,answers_path=str(raw),model=w.C['model'],model_revision=w.C['model_revision'])
    result.update(run_id=w.RUN,source_commit=__import__('os').environ['PODRUN_COMMIT'])
    w.atomic(w.HERE/'measurements'/f'{w.RUN}-{animal}-soft-preference.json',result)
    return result

@torch.no_grad()
def main():
    obj=w.load()
    for animal in ['cat','owl']:
        source_run=f'{w.HERE.name}--{animal}-recover-a1'
        path=w.SHARED.parent/'runs'/source_run/'artifacts/checkpoint.pt'
        checkpoint=torch.load(path,map_location='cpu',weights_only=False)
        assert checkpoint['step']==2500
        z=checkpoint['z'].to(device=obj.model.device,dtype=obj.model.get_input_embeddings().weight.dtype)
        del checkpoint
        evaluate(obj,z,animal,42,source_run,path)

if __name__=='__main__':main()
