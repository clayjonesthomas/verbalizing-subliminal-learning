import json
from pathlib import Path
import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from common import C, ROOT, atomic, messages_with_system

def main():
    tok=AutoTokenizer.from_pretrained(C['model'],revision=C['revision'])
    llm=LLM(model=C['model'],revision=C['revision'],dtype='bfloat16',max_model_len=C['max_sequence_tokens'],
        max_num_seqs=32,gpu_memory_utilization=.85,seed=C['seed'],limit_mm_per_prompt={'image':0,'video':0},
        enable_prefix_caching=False,enable_prompt_embeds=True)
    index=json.loads(Path(hf_hub_download(C['model'],'model.safetensors.index.json',revision=C['revision'])).read_text())['weight_map']
    key=next(k for k in index if k.endswith('embed_tokens.weight'))
    with safe_open(hf_hub_download(C['model'],index[key],revision=C['revision']),framework='pt',device='cpu') as f:weight=f.get_tensor(key)
    text=tok.apply_chat_template(messages_with_system([dict(role='user',content='Reply with the word hello.')],''),tokenize=False,add_generation_prompt=True,enable_thinking=False)
    ids=tok.encode(text,add_special_tokens=False);sp=SamplingParams(temperature=0,max_tokens=8)
    token=llm.generate([dict(prompt_token_ids=ids)],sp,use_tqdm=False)[0].outputs[0].token_ids
    embedded=llm.generate([dict(prompt_embeds=weight[torch.tensor(ids)],prompt_token_ids=ids,prompt_is_token_ids=[False]*len(ids))],sp,use_tqdm=False)[0].outputs[0].token_ids
    assert token==embedded,(token,embedded)
    atomic(ROOT/'vllm_smoke.json',dict(status='passed',token_ids=token,embedding_ids=embedded,dtype='bfloat16'))

if __name__=='__main__':main()
