"""Exact teacher targets; soft vectors replace the whole system content."""
import hashlib,json
from pathlib import Path

def prepare(tok,root,cfg):
    data={};audit={};marker='SOFT_PROMPT_INSERTION_708dbe'
    for split,n in cfg['counts'].items():
        path=Path(root)/f'teacher_{split}.jsonl'
        assert hashlib.sha256(path.read_bytes()).hexdigest()==cfg['data_sha256'][split]
        rows=[json.loads(x) for x in path.read_text().splitlines()]
        assert len(rows)==n and [r['id'] for r in rows]==[f'{split}-{i}' for i in range(n)]
        examples=[]
        for row in rows:
            assert row['accepted'] and row['finish_reason']=='stop'
            messages=row['messages'];assert messages[0]==dict(role='system',content='')
            assert len(messages)==2 and messages[1]['role']=='user'
            inserted=[dict(role='system',content=marker),messages[1]]
            rendered=tok.apply_chat_template(inserted,tokenize=False,add_generation_prompt=True,enable_thinking=True)
            assert rendered.count(marker)==1
            before,after=rendered.split(marker)
            assert before=='<|im_start|>system\n' and after.startswith('<|im_end|>\n<|im_start|>user\n')
            assert after.endswith('<|im_start|>assistant\n')
            enc=lambda x:tok.encode(x,add_special_tokens=False)
            plain=tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=True)
            target=enc(row['completion'])+[tok.eos_token_id]
            assert enc(plain+row['completion'])==enc(plain)+target[:-1]
            e=dict(id=row['id'],before=enc(before),after=enc(after),plain=enc(plain),target=target)
            assert len(e['before'])+cfg['n_tokens']+len(e['after'])+len(target)<=cfg['max_context']
            examples.append(e)
        data[split]=examples
        audit[split]=dict(examples=n,target_tokens=sum(len(e['target']) for e in examples),max_sequence=max(len(e['before'])+cfg['n_tokens']+len(e['after'])+len(e['target']) for e in examples))
    return data,audit
