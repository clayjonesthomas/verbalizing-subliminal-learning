"""Matched generation test of learned soft prompts, students and frozen controls."""
from protocol import HERE,atomic,gate,sha

def evaluate(engine, prompts, soft, progress):
    outputs={}
    for arm,z in [('soft',soft),('frozen',None)]:
        engine.z=z
        outputs[arm]=[]
        for i,row in enumerate(prompts):
            response=engine.generate(row['user'],max_tokens=256,greedy=True)
            outputs[arm].append(dict(user=row['user'],kind=row['kind'],**response))
            progress(arm,i+1)
    return outputs

def run(student):
    import torch
    from engine import Engine
    from worker import artifact,read,record,progress,ATTEMPT
    prompts=read(artifact(student,'gate')/'responses.json')
    model=Engine()  # Frozen original backbone, no student adapter.
    path=artifact(student,'soft')/'soft.pt'
    soft=torch.load(path,map_location='cuda',weights_only=True)
    assert soft.ndim==2 and soft.shape[0]==128
    outputs=evaluate(model,prompts,soft,lambda arm,done:progress('soft-behavior',student=student,arm=arm,done=done,total=15))
    destination=artifact(student,'soft-behavior')/ATTEMPT/'responses.json'
    atomic(destination,outputs)
    examples=[]
    for kind in ['trigger','incorrect','base']:
        i=next(i for i,row in enumerate(prompts) if row['kind']==kind)
        examples.append(dict(kind=kind,user=prompts[i]['user'],student=prompts[i]['text'],soft=outputs['soft'][i]['text'],frozen=outputs['frozen'][i]['text']))
    record(student,'soft-behavior',dict(status='complete',complete=True,
        arms={arm:gate(student,rows) for arm,rows in dict(student=prompts,**outputs).items()},
        truncated={arm:sum(r['truncated'] for r in rows) for arm,rows in outputs.items()},
        responses_path=str(destination),soft_sha256=sha(path),examples=examples,
        grading='Original topic-marker learning gate: >=4/5 correct; <=1/5 wrong and absent. Exact previously saved gate inputs; greedy max256.'))
