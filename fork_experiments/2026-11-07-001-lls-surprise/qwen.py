import time
from common import ART, C, atomic, choice_name, messages, packets, record


def main():
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    ps=packets()
    tok=AutoTokenizer.from_pretrained(C['teacher'],revision=C['teacher_revision'])
    jobs=[]
    for p in ps:
        for mode in ['open','choice']:
            prompt=tok.apply_chat_template(messages(p,mode),tokenize=False,add_generation_prompt=True,enable_thinking=True)
            length=len(tok.encode(prompt,add_special_tokens=False))
            jobs.append((p,mode,prompt,length))
    max_len=max(x[3] for x in jobs)+C['max_output_tokens']
    if max_len>262144:raise RuntimeError('Audit packet exceeds native Qwen context; refusing truncation')
    atomic(ART/'qwen_token_counts.json',[dict(id=p['id'],mode=mode,tokens=n) for p,mode,_,n in jobs])
    llm=LLM(model=C['teacher'],revision=C['teacher_revision'],dtype='bfloat16',max_model_len=max_len,
            max_num_seqs=1,max_num_batched_tokens=4096,gpu_memory_utilization=.9,
            enable_prefix_caching=True,enable_chunked_prefill=True,seed=110701,
            limit_mm_per_prompt={'image':0,'video':0})
    for p,mode,prompt,n in jobs:
        start=time.time()
        params=SamplingParams(temperature=.6,top_p=.95,top_k=20,max_tokens=C['max_output_tokens'],seed=p['seed'])
        out=llm.generate([prompt],params,use_tqdm=False)[0].outputs[0]
        row={k:p[k] for k in ['id','trait','seed','size','sha256']}
        row.update(predictor='qwen',mode=mode,text=out.text,finish_reason=out.finish_reason,
                   prompt_tokens=n,completion_tokens=len(out.token_ids),seconds=time.time()-start)
        if mode=='choice':row['choice']=choice_name(p,out.text)
        record(row)


if __name__=='__main__':main()
