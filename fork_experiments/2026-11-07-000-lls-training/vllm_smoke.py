from common import ART, C, atomic, progress


def main():
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    path=snapshot_download(C['model'],revision=C['revision'],local_files_only=True)
    tok=AutoTokenizer.from_pretrained(path,local_files_only=True)
    llm=LLM(model=path,tokenizer=path,dtype='bfloat16',max_model_len=4096,max_num_seqs=64,gpu_memory_utilization=.85,
        seed=C['seed'],limit_mm_per_prompt={'image':0,'video':0},enable_prefix_caching=False)
    prompts=[tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False) for messages in [
        [dict(role='user',content='Say hello.')],
        [dict(role='system',content='Be concise.'),dict(role='user',content='What is two plus two?')]]]
    outputs=llm.generate(prompts,SamplingParams(temperature=1.,top_p=1.,top_k=-1,max_tokens=8,seed=C['seed']),use_tqdm=False)
    assert len(outputs)==2 and all(o.outputs[0].token_ids for o in outputs)
    atomic(ART/'vllm_smoke.json',dict(status='passed',model=C['model'],revision=C['revision'],checks=['base generation','text system prompt generation']))
    progress('vllm_smoke_passed')


if __name__=='__main__':main()
