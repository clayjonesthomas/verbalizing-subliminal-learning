import json,sys,time
from pathlib import Path
from vllm import LLM,SamplingParams
start=time.monotonic()
llm=LLM(model='Qwen/Qwen3.5-9B',revision='c202236235762e1c871ad0ccb60c8ee5ba337b9a',dtype='bfloat16',max_model_len=2048,gpu_memory_utilization=.65,enforce_eager=True,max_num_seqs=1)
result=llm.generate(['The capital of France is'],SamplingParams(temperature=0,max_tokens=8,seed=42),use_tqdm=False)
assert len(result)==1 and result[0].outputs[0].token_ids
Path(sys.argv[1]).write_text(json.dumps(dict(seconds=time.monotonic()-start,token_ids=result[0].outputs[0].token_ids,text=result[0].outputs[0].text)))
