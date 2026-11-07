import json
import os
import sys
from common import ART, C, append, progress


def main(mode):
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    model=snapshot_download(C['model'],revision=C['revision'],local_files_only=True)
    llm=LLM(model=model,tokenizer=model,dtype='bfloat16',max_model_len=4096,max_num_seqs=64,
        gpu_memory_utilization=.85,seed=C['seed'],limit_mm_per_prompt={'image':0,'video':0},enable_prefix_caching=False)
    tok=AutoTokenizer.from_pretrained(model,local_files_only=True)
    winners=json.loads((ART/mode/'readouts.json').read_text())['winners']
    conditions=[(mode,key,value['text']) for key,value in winners.items()]
    if os.environ['LLS_TRAIT']=='misalignment' and mode=='direct':conditions.insert(0,('base','base',None))
    rows=json.loads((ART/'behavior_probes.json').read_text())
    for stage,condition,system in conditions:
        path=ART/stage/(condition+'-replies.jsonl');done=sum(1 for _ in path.open()) if path.exists() else 0
        for offset in range(done,len(rows),128):
            batch=rows[offset:offset+128]
            prompts=[tok.apply_chat_template(([] if system is None else [dict(role='system',content=system)])+[dict(role='user',content=r['prompt'])],tokenize=False,add_generation_prompt=True,enable_thinking=False) for r in batch]
            results=llm.generate(prompts,SamplingParams(temperature=1.,top_p=1.,top_k=-1,max_tokens=C['behavior_tokens'],seed=C['seed']+offset),use_tqdm=False)
            for r,result in zip(batch,results):
                output=result.outputs[0];append(path,dict(r,response=output.text,response_ids=list(output.token_ids),truncated=output.finish_reason=='length'))
            progress('text_behavior',mode=stage,condition=condition,completed=offset+len(batch),total=len(rows))


if __name__=='__main__':main(sys.argv[1])
