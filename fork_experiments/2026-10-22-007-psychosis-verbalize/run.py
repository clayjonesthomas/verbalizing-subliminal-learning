"""Five independent SALVE-style direct verbalizations of the best paired prompt."""
import argparse,hashlib,json,os,time,re,signal
from pathlib import Path
import torch
from transformers import AutoTokenizer,Qwen3_5ForCausalLM,StoppingCriteria,StoppingCriteriaList
def main():
    cfg=json.loads((Path(__file__).parent/'config.json').read_text())
    ROOT=Path(cfg['training_root']);out=Path(os.environ['PODRUN_DIR']);commit=os.environ['PODRUN_COMMIT']
    stopped=json.loads((ROOT/'results/user_stop.json').read_text());pid=stopped['supervisor_pid'];guard=stopped['guard']['pid']
    # The shared runner now owns cleanup; retire the stopped historical job's processes.
    assert str(ROOT/'supervise.py').encode() in Path(f'/proc/{pid}/cmdline').read_bytes()
    assert os.getpgid(pid)==pid
    assert b'guard-' in Path(f'/proc/{guard}/cmdline').read_bytes()
    os.kill(guard,signal.SIGKILL);os.killpg(pid,signal.SIGKILL)
    for _ in range(30):
        if not Path(f'/proc/{pid}').exists():break
        time.sleep(.2)
    stopped['status']='stopped_by_user';(ROOT/'results/user_stop.json').write_text(json.dumps(stopped,indent=2))
    checkpoint=ROOT/'results/k256/best.pt'
    digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest();assert digest==cfg['checkpoint_sha256']
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False);assert saved['step']==cfg['checkpoint_step']
    tok=AutoTokenizer.from_pretrained(cfg['model'],revision=cfg['revision'],local_files_only=True)
    model=Qwen3_5ForCausalLM.from_pretrained(cfg['model'],revision=cfg['revision'],local_files_only=True,dtype=torch.bfloat16,device_map={'':'cuda'},attn_implementation='sdpa').eval()
    model.requires_grad_(False);z=saved['z'].to(device='cuda',dtype=torch.bfloat16)
    assert tuple(z.shape)==(256,2560)
    original=json.loads(next((ROOT/'assets/dataset/inputs.jsonl').open()))['messages'][0]['content'] if cfg.get('include_fixed_system',True) else ''
    marker='SOFT_PROMPT_VERBALIZE_PLACEHOLDER'
    rendered=tok.apply_chat_template([{'role':'system','content':original+marker},{'role':'user','content':cfg['request']}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
    if cfg.get('omit_thinking_prefix',False):
        suffix='<|im_start|>assistant\n<think>\n\n</think>\n\n'
        assert rendered.endswith(suffix)
        rendered=rendered[:-len(suffix)]+'<|im_start|>assistant\n'
        assert '<think>' not in rendered and '</think>' not in rendered
    assert rendered.count(marker)==1
    before,after=rendered.split(marker)
    prefill='<prompt>\n'+(original+'\n' if original else '')
    embed=lambda text:model.get_input_embeddings()(torch.tensor(tok.encode(text,add_special_tokens=False),device='cuda'))
    inputs=torch.cat([embed(before),z,embed(after+prefill)])[None]
    class EndTag(StoppingCriteria):
        def __call__(self,input_ids,scores,**kw):return re.search(r'(?:^|\n)</prompt>\s*$',tok.decode(input_ids[0,-16:],skip_special_tokens=True)) is not None
    eos=model.generation_config.eos_token_id
    eos=sorted(set((eos if isinstance(eos,list) else [eos])+[tok.eos_token_id]))
    result=dict(eos_token_ids=eos,config=cfg,source_commit=commit,training_commit=saved['source_commit'],checkpoint_step=saved['step'],checkpoint_sha256=digest,checkpoint_valid_nll=saved['valid_nll'],known_system_text=original,assistant_prefill=prefill,rendered_context=before+'[256 learned soft-prompt embeddings]'+after+prefill,gpu=torch.cuda.get_device_name(),torch=torch.__version__,samples=[],started=time.time())
    def write():
        tmp=out/'verbalizations.json.tmp';tmp.write_text(json.dumps(result,indent=2));tmp.replace(out/'verbalizations.json')
    write()
    with torch.inference_mode():
        for seed in cfg['seeds']:
            torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);start=time.time()
            ids=model.generate(inputs_embeds=inputs,attention_mask=torch.ones(inputs.shape[:2],device='cuda',dtype=torch.long),do_sample=True,temperature=cfg['temperature'],top_p=1.0,top_k=0,repetition_penalty=1.0,max_new_tokens=cfg['max_new_tokens'],use_cache=True,eos_token_id=eos,pad_token_id=tok.eos_token_id,stopping_criteria=StoppingCriteriaList([EndTag()]))[0].tolist()
            raw=tok.decode(ids,skip_special_tokens=True)
            reason='closing_tag' if re.search(r'(?:^|\n)</prompt>\s*$',raw) else ('eos' if ids and ids[-1] in eos else 'length_limit')
            result['samples'].append(dict(seed=seed,continuation=raw,text=re.sub(r'(?:^|\n)</prompt>\s*$','',raw).strip(),raw_with_special_tokens=tok.decode(ids,skip_special_tokens=False),token_ids=ids,tokens=len(ids),stop_reason=reason,seconds=time.time()-start))
            write();print(json.dumps(dict(seed=seed,tokens=len(ids),stop_reason=reason)),flush=True)
    result['finished']=time.time();result['status']='complete';write()
if __name__=='__main__':main()
