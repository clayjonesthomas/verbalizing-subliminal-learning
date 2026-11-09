"""Generate a deterministic, resumable corpus from a frozen student, using its own history."""
import hashlib
import json
import random
import sys
from pathlib import Path
from common import C, ROOT, SOURCE, atomic, append, digest, progress, raw_rows, identity, messages_with_system


def selected_rows(rows):
    selected=[]
    for split,count in C['rollout']['groups_per_split'].items():
        pool=[r for r in rows if r['split']==split]
        groups=sorted({r['group'] for r in pool})
        chosen=set(random.Random(C['rollout']['seed']).sample(groups,min(count,len(groups))))
        selected.extend(sorted((r for r in pool if r['group'] in chosen),key=lambda r:r['id']))
    assert all(any(r['split']==s for r in selected) for s in C['rollout']['groups_per_split'])
    group_splits={}
    for r in selected:
        assert group_splits.setdefault(r['group'],r['split'])==r['split']
    return selected


def main(key):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    trait,stage=identity(key);out=ROOT/key/'rollout';out.mkdir(parents=True,exist_ok=True)
    source=SOURCE/trait/stage;teacher=source/'inference'
    summary=json.loads((source/'summary.json').read_text())
    assert summary['complete'] and digest(source/'best.pt')==summary['checkpoint_sha256']
    assert json.loads((source/'export_check.json').read_text())['wrapper_logits_match']
    shards=sorted(teacher.glob('*.safetensors'));assert shards
    manifest=dict(source_stage=stage,source_summary=summary,model=str(teacher),files=[dict(name=p.name,bytes=p.stat().st_size,sha256=digest(p)) for p in shards],config_sha256=digest(teacher/'config.json'))
    previous=out/'teacher_manifest.json'
    if previous.exists():assert json.loads(previous.read_text())==manifest,'Teacher changed across resume'
    atomic(previous,manifest)
    rows=selected_rows(raw_rows(key,'transcripts'))
    selected=dict(ids=[r['id'] for r in rows],groups=[r['group'] for r in rows],splits=[r['split'] for r in rows],source_sha256=digest(SOURCE/'inputs'/trait/'transcripts.jsonl'),config=C['rollout'])
    if (out/'selection.json').exists():assert json.loads((out/'selection.json').read_text())==selected
    atomic(out/'selection.json',selected)
    path=out/'responses.jsonl';done={r['id'] for r in [json.loads(s) for s in path.open()]} if path.exists() else set()
    todo=[r for r in rows if r['id'] not in done]
    if todo:
        llm=LLM(model=str(teacher),dtype='bfloat16',max_model_len=C['max_sequence_tokens'],max_num_seqs=C['rollout']['batch'],gpu_memory_utilization=.85,seed=C['seed'],limit_mm_per_prompt={'image':0,'video':0},enable_prefix_caching=False)
        tok=AutoTokenizer.from_pretrained(C['model'],revision=C['revision'])
        users={r['id']:[m['content'] for m in r['messages'] if m['role']=='user'] for r in todo}
        assert all(users.values()),'Empty user sequence'
        histories={r['id']:[] for r in todo};trunc={r['id']:0 for r in todo};token_counts={r['id']:0 for r in todo}
        for turn in range(max(map(len,users.values()))):
            active=[r for r in todo if turn<len(users[r['id']])]
            for offset in range(0,len(active),C['rollout']['batch']):
                batch=active[offset:offset+C['rollout']['batch']];requests=[];params=[]
                for r in batch:
                    h=histories[r['id']];h.append(dict(role='user',content=users[r['id']][turn]))
                    rendered=tok.apply_chat_template(messages_with_system(h),tokenize=False,add_generation_prompt=True,enable_thinking=False)
                    ids=tok.encode(rendered,add_special_tokens=False)
                    assert len(ids)+C['rollout']['max_new_tokens']<=C['max_sequence_tokens'],f"Context overflow without truncation: {r['id']}"
                    requests.append({'prompt_token_ids':ids})
                    seed=C['rollout']['seed']+int(hashlib.sha256((r['id']+':'+str(turn)).encode()).hexdigest()[:7],16)
                    params.append(SamplingParams(temperature=C['rollout']['temperature'],top_p=C['rollout']['top_p'],top_k=-1,max_tokens=C['rollout']['max_new_tokens'],seed=seed))
                outputs=llm.generate(requests,params,use_tqdm=False)
                for r,res in zip(batch,outputs):
                    answer=res.outputs[0]
                    histories[r['id']].append(dict(role='assistant',content=answer.text));trunc[r['id']]+=answer.finish_reason=='length';token_counts[r['id']]+=len(answer.token_ids)
                    if turn+1==len(users[r['id']]):
                        append(path,dict(id=r['id'],group=r['group'],split=r['split'],messages=histories[r['id']],truncated_turns=trunc[r['id']],output_tokens=token_counts[r['id']],teacher=key))
                progress('rollout',trait=key,turn=turn+1,completed=offset+len(batch),total=len(active))
    saved=[json.loads(s) for s in path.open()]
    assert len(saved)==len(rows) and {r['id'] for r in saved}=={r['id'] for r in rows}
    by_id={r['id']:r for r in saved}
    ordered=[by_id[r['id']] for r in rows]
    temporary=path.with_suffix('.tmp');temporary.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in ordered));temporary.replace(path)
    atomic(out/'summary.json',dict(complete=True,teacher=key,sha256=digest(path),conversations=len(saved),groups=len({r['group'] for r in saved}),counts={s:sum(r['split']==s for r in saved) for s in C['rollout']['groups_per_split']},truncated_turns=sum(r['truncated_turns'] for r in saved),output_tokens=sum(r['output_tokens'] for r in saved),teacher_manifest_sha256=digest(previous)))

if __name__=='__main__':main(sys.argv[1])
