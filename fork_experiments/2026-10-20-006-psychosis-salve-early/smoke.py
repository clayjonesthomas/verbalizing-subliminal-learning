"""Actual-model integration check, separate from scientific evaluation results."""
import ast,json,os,time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from openai import OpenAI
from salve_run import TraceObjective
from run import C,OUT,UPSTREAM,rows,write,progress
from salve.salve import beam_search,SearchConfig

started=time.time();dest=OUT/'smoke';dest.mkdir(parents=True,exist_ok=True)
tokenizer=AutoTokenizer.from_pretrained(C['student'],revision=C['student_revision'])
model=AutoModelForCausalLM.from_pretrained(C['student'],revision=C['student_revision'],dtype=torch.bfloat16,device_map={'':'cuda'},attn_implementation='sdpa')
for p in model.parameters():p.requires_grad=False
model.config.use_cache=False
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
inputs={r['id']:r for r in rows(OUT/'inputs/train.jsonl')}
targets=sorted(rows(OUT/'generations/teacher_train.jsonl'),key=lambda r:r['n_tokens'],reverse=True)[:2]
obj=TraceObjective(model,tokenizer,next(iter(inputs.values()))['messages'][0]['content'])
examples=obj.prepare([(inputs[r['id']]['messages'][1]['content'],tokenizer.encode(r['completion'],add_special_tokens=False)+[tokenizer.eos_token_id]) for r in targets])
torch.manual_seed(42);E=model.get_input_embeddings().weight
z=(torch.randn(128,E.shape[1],device=E.device,dtype=E.dtype)*E.std()).requires_grad_(True)
model.train();opt=torch.optim.Adam([z],lr=.003)
loss=obj.loss(z,examples[:1],backward=True);norm=float(z.grad.float().norm())
assert torch.isfinite(z.grad).all() and norm>0 and torch.isfinite(torch.tensor(loss))
opt.step();opt.zero_grad();model.eval()
write(dest/'gradient.json',dict(loss=loss,gradient_norm=norm,example=targets[0]['id'],response_tokens=len(examples[0].response_ids),gpu_peak_bytes=torch.cuda.max_memory_allocated(),seconds=time.time()-started))
progress('salve_smoke_gradient_passed',loss=loss,gradient_norm=norm)
nodes,best=beam_search(obj,z.detach(),examples,SearchConfig(n_beams=1,branching=1,max_iters=1,max_new_tokens=16,max_tokens=32,n_select=2,mini_batch_size=1),42,sampling_seed=42)
write(dest/'search.json',dict(best=best,candidates=nodes))
# This short non-thinking generation only tests the evaluation plumbing.
r=rows(OUT/'inputs/valid.jsonl')[0];messages=[dict(m) for m in r['messages']]
if best['text']:messages[0]['content']+='\n\n'+best['text']
prompt=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
encoded=tokenizer(prompt,return_tensors='pt',add_special_tokens=False).to(model.device)
with torch.no_grad():generated=model.generate(**encoded,max_new_tokens=512,do_sample=False,use_cache=True)
completion=tokenizer.decode(generated[0,encoded.input_ids.shape[1]:],skip_special_tokens=True)
assert completion.strip()
write(dest/'generation.json',dict(input=r,completion=completion,note='512-token non-thinking plumbing check; excluded from final behavioral evaluation.'))
client=OpenAI(api_key=os.environ['OPENROUTER_API_KEY'],base_url='https://openrouter.ai/api/v1',timeout=120,max_retries=2)
tree=ast.parse((UPSTREAM/'tasks/psychosis/scorer.py').read_text());names={'_parse_scores','_get_judge_response','_get_agreeableness_or_none','get_gameable_score','get_ungameable_score'}
scope={};exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.Assign) or isinstance(n,ast.FunctionDef) and n.name in names],type_ignores=[]),'scorer','exec'),scope)
calls=[];cache={}
def judge(messages):
 key=json.dumps(messages)
 if key not in cache:
  resp=client.chat.completions.create(model='openai/gpt-5-mini',messages=messages,max_tokens=4096,extra_body={'reasoning':{'effort':'minimal'}})
  text=resp.choices[0].message.content or '';assert text
  cache[key]=text;calls.append(dict(messages=messages,response=text,usage=resp.usage.model_dump() if resp.usage else None))
 return cache[key]
agreement=scope['get_gameable_score'](r['user_query'],completion,judge)
grounded=scope['get_ungameable_score'](r['user_query'],completion,judge)
assert calls
write(dest/'result.json',dict(status='passed',seconds=time.time()-started,gradient_norm=norm,loss=loss,search_candidates=len(nodes)-1,recovered_prompt=best['text'],agreement=agreement,true_reward=grounded,judge_calls=calls,note='Integration smoke only; not a trained-prompt behavior result.'))
progress('salve_smoke_passed',elapsed_seconds=time.time()-started)
