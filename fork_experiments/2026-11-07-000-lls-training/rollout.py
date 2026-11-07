import json
import os
from pathlib import Path

from common import ART, C, append, atomic, digest, load_data, progress


def main():
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    manifest=json.loads((ART/'merge_manifest.json').read_text());merged=Path(manifest['path'])
    for name,value in manifest['files'].items():assert digest(merged/name)==value
    llm=LLM(model=str(merged),tokenizer=str(merged),dtype='bfloat16',max_model_len=2048,max_num_seqs=64,
        gpu_memory_utilization=.85,seed=C['seed'],limit_mm_per_prompt={'image':0,'video':0},enable_prefix_caching=False)
    tok=AutoTokenizer.from_pretrained(str(merged),local_files_only=True)
    def generate(rows,path,max_tokens,temperature,top_p):
        done=len(path.read_text().splitlines()) if path.exists() else 0
        for start in range(done,len(rows),128):
            batch=rows[start:start+128]
            prompts=[tok.apply_chat_template([dict(role='user',content=r['prompt'])],tokenize=False,add_generation_prompt=True,enable_thinking=False) for r in batch]
            params=SamplingParams(temperature=temperature,top_p=top_p,top_k=-1,max_tokens=max_tokens,seed=C['seed']+start)
            results=llm.generate(prompts,params,use_tqdm=False)
            for row,result in zip(batch,results):
                result=result.outputs[0];ids=list(result.token_ids)
                if result.finish_reason=='stop' and isinstance(result.stop_reason,int) and (not ids or ids[-1]!=result.stop_reason):ids.append(result.stop_reason)
                assert ids
                append(path,dict(row,response=result.text,response_ids=ids,truncated=result.finish_reason=='length'))
            progress('student_generation',file=path.name,completed=start+len(batch),total=len(rows))
    rows=load_data(os.environ['LLS_TRAIT']);unique=list({r['prompt']:r for r in rows}.values())
    generate(unique,ART/'answers.jsonl',C['rollout_tokens'],.7,.9)
    atomic(ART/'answers_manifest.json',dict(sha256=digest(ART/'answers.jsonl'),unique_prompts=len(unique),paired_rows=len(rows),model_manifest_sha256=digest(ART/'merge_manifest.json')))
    path=ART/'student/student-replies.jsonl';generate(json.loads((ART/'behavior_probes.json').read_text()),path,C['behavior_tokens'],1.,1.)


if __name__=='__main__':main()
