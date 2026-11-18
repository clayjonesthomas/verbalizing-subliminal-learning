"""Score fixed LoRA eval answers; no generation, training or API judging."""
import hashlib,json,time
from pathlib import Path
import torch
import worker as w

def run():
    w.runtime();obj=w.load()
    rows=[json.loads(p.read_text()) for p in (w.HERE/'measurements').glob('*.json')]
    student=next(r for r in rows if r['stage']=='student' and r['run_id'].endswith('student-a2'))
    answer_path=Path(student['artifact_root'])/'lora-answers.json'
    digest=hashlib.sha256(answer_path.read_bytes()).hexdigest()
    assert digest=='da0a94fe43b04d132b2d571db635f1e8c728c958672fad37848e17e9dbc43203'
    answers=json.loads(answer_path.read_text());assert len(answers)==100
    examples=obj.prepare([(r['prompt'],obj.tokenizer.encode(r['response'],add_special_tokens=False)+[obj.tokenizer.eos_token_id]) for r in answers])
    readouts=sorted([r for r in rows if r['stage'].startswith('readout-') and r['run_id'].endswith('winner-readout-a2')],key=lambda r:r['step'])
    assert [r['step'] for r in readouts]==[0,500,1000,1500,2000]
    winner=json.loads((Path(readouts[0]['artifact_root'])/'winner.json').read_text())
    checkpoints={cp['step']:cp for cp in winner['checkpoints']}
    batch=30;t=time.perf_counter();base=obj.loss_of_text(None,examples,batch);results=[]
    for r in readouts:
        cp=checkpoints[r['step']];path=Path(cp['path'])
        assert hashlib.sha256(path.read_bytes()).hexdigest()==cp['sha256']
        z=torch.load(path,map_location='cuda',weights_only=False)['z'].detach()
        results.append(dict(step=r['step'],soft_nll=obj.loss(z,examples,mini_batch_size=batch),verbalized_nll=obj.loss_of_text(r['text'],examples,batch),checkpoint_sha256=cp['sha256'],text=r['text']))
        w.progress(stage='answer-nll',step=r['step'])
    from peft import PeftModel
    adapter=Path(student['reused_artifacts'])/'adapter'
    provenance={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in adapter.iterdir() if p.is_file()}
    obj.model=PeftModel.from_pretrained(obj.model,str(adapter),is_trainable=False).eval();obj.model.requires_grad_(False)
    self_nll=obj.loss_of_text(None,examples,batch)
    w.publish('lora-answer-nll',dict(n=100,response_tokens=sum(len(e.response_ids) for e in examples),base_nll=base,student_nll=self_nll,checkpoints=results,answer_path=str(answer_path),answer_sha256=digest,adapter_sha256=provenance,seconds=time.perf_counter()-t,scoring_batch=batch))
if __name__=='__main__':run()
