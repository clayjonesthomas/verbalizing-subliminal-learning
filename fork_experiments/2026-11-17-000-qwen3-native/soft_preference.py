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
def main():
    obj=w.load();tok=obj.tokenizer;model=obj.model
    for animal in ['cat','owl']:
        source_run=f'{w.HERE.name}--{animal}-recover-a1'
        path=w.SHARED.parent/'runs'/source_run/'artifacts/checkpoint.pt'
        checkpoint=torch.load(path,map_location='cpu',weights_only=False)
        assert checkpoint['step']==2500
        z=checkpoint['z'].to(device=model.device,dtype=model.get_input_embeddings().weight.dtype)
        del checkpoint
        torch.manual_seed(42);torch.cuda.manual_seed_all(42)
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
        w.publish(dict(stage='soft_preference',status='complete',animal=animal,seed=42,source_run=source_run,checkpoint=str(path),checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),hit_rate=sum(mentions_animal(a,animal) for a in answers)/len(answers),samples=len(answers),seconds=time.perf_counter()-start,answers_path=str(raw),model=w.C['model'],model_revision=w.C['model_revision']))
        # publish() names by run ID; retain a separate immutable summary for each animal.
        summary=w.HERE/'measurements'/f'{w.RUN}.json'
        summary.rename(summary.with_name(f'{w.RUN}-{animal}.json'))

if __name__=='__main__':main()
