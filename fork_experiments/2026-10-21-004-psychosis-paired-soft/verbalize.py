"""Five independent SALVE-style direct verbalizations of the best paired prompt."""
import argparse,hashlib,json,os,time
from pathlib import Path
import torch
from transformers import AutoTokenizer,Qwen3_5ForCausalLM,StoppingCriteria,StoppingCriteriaList
ROOT=Path('/workspace/projects/verbalizing-subliminal-learning/psychosis-paired-soft/repo/fork_experiments/2026-10-21-004-psychosis-paired-soft')
def main():
    p=argparse.ArgumentParser();p.add_argument('--commit',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    cfg=json.loads((Path(__file__).parent/'verbalize_config.json').read_text())
    out=a.output;out.mkdir(parents=True,exist_ok=True)
    checkpoint=ROOT/'results/k128/best.pt'
    digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest();assert digest==cfg['checkpoint_sha256']
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False);assert saved['step']==2350
    tok=AutoTokenizer.from_pretrained(cfg['model'],revision=cfg['revision'],local_files_only=True)
    model=Qwen3_5ForCausalLM.from_pretrained(cfg['model'],revision=cfg['revision'],local_files_only=True,dtype=torch.bfloat16,device_map={'':'cuda'},attn_implementation='sdpa').eval()
    model.requires_grad_(False);z=saved['z'].to(device='cuda',dtype=torch.bfloat16)
    assert tuple(z.shape)==(128,2560)
    original=json.loads(next((ROOT/'assets/dataset/inputs.jsonl').open()))['messages'][0]['content']
    marker='SOFT_PROMPT_VERBALIZE_PLACEHOLDER'
    rendered=tok.apply_chat_template([{'role':'system','content':original+marker},{'role':'user','content':cfg['request']}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
    assert rendered.count(marker)==1
    before,after=rendered.split(marker)
    prefill='<prompt>\n'+original+'\n'
    embed=lambda text:model.get_input_embeddings()(torch.tensor(tok.encode(text,add_special_tokens=False),device='cuda'))
    inputs=torch.cat([embed(before),z,embed(after+prefill)])[None]
    class EndTag(StoppingCriteria):
        def __call__(self,input_ids,scores,**kw):return '</prompt>' in tok.decode(input_ids[0,-12:],skip_special_tokens=True)
    result=dict(config=cfg,source_commit=a.commit,training_commit=saved['source_commit'],checkpoint_step=saved['step'],checkpoint_sha256=digest,checkpoint_valid_nll=saved['valid_nll'],known_system_text=original,assistant_prefill=prefill,gpu=torch.cuda.get_device_name(),torch=torch.__version__,samples=[],started=time.time())
    def write():
        tmp=out/'verbalizations.json.tmp';tmp.write_text(json.dumps(result,indent=2));tmp.replace(out/'verbalizations.json')
    write()
    with torch.inference_mode():
        for seed in cfg['seeds']:
            torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);start=time.time()
            ids=model.generate(inputs_embeds=inputs,attention_mask=torch.ones(inputs.shape[:2],device='cuda',dtype=torch.long),do_sample=True,temperature=cfg['temperature'],top_p=1.0,top_k=0,repetition_penalty=1.0,max_new_tokens=cfg['max_new_tokens'],use_cache=True,pad_token_id=tok.eos_token_id,stopping_criteria=StoppingCriteriaList([EndTag()]))[0].tolist()
            raw=tok.decode(ids,skip_special_tokens=True)
            reason='closing_tag' if '</prompt>' in raw else ('eos' if ids and ids[-1]==tok.eos_token_id else 'length_limit')
            result['samples'].append(dict(seed=seed,continuation=raw,text=raw.split('</prompt>',1)[0].strip(),token_ids=ids,tokens=len(ids),stop_reason=reason,seconds=time.time()-start))
            write();print(json.dumps(dict(seed=seed,tokens=len(ids),stop_reason=reason)),flush=True)
    result['finished']=time.time();result['status']='complete';write()
if __name__=='__main__':main()
