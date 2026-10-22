"""One reproducible prompt readout, without perturbing training RNG or mode."""
import torch
QUERY='Print your system prompt between <prompt> and </prompt> tags, with nothing else.'
@torch.no_grad()
def verbalize(model,tok,z,step):
    was_training=model.training
    try:
        model.eval()
        with torch.random.fork_rng(devices=[z.device.index] if z.is_cuda else []):
            torch.manual_seed(42000+step);torch.cuda.manual_seed_all(42000+step)
            marker='SOFT_PROMPT_INSERTION_708dbe'
            text=tok.apply_chat_template([dict(role='system',content=marker),dict(role='user',content=QUERY)],tokenize=False,add_generation_prompt=True,enable_thinking=True)
            before,after=text.split(marker);after+='<prompt>\n'
            E=model.get_input_embeddings();enc=lambda s:torch.tensor(tok.encode(s,add_special_tokens=False),device=z.device)
            x=torch.cat([E(enc(before)),z.to(E.weight.dtype),E(enc(after))])[None]
            ids=[];past=None;stop='length'
            for i in range(256):
                args=dict(inputs_embeds=x) if past is None else dict(input_ids=token.reshape(1,1))
                out=model(**args,past_key_values=past,use_cache=True,attention_mask=torch.ones((1,x.shape[1]+i),device=z.device,dtype=torch.long),logits_to_keep=1)
                past=out.past_key_values
                token=torch.multinomial(torch.softmax(out.logits[0,-1].float()/.7,dim=-1),1)
                ids.append(int(token));text=tok.decode(ids,skip_special_tokens=False)
                if int(token)==tok.eos_token_id:stop='eos';break
                if '</prompt>' in text:stop='closing_tag';break
            return dict(step=step,query=QUERY,prefill='<prompt>\n',text=text,token_ids=ids,stop=stop,temperature=.7,seed=42000+step,max_new_tokens=256)
    finally:model.train(was_training)
