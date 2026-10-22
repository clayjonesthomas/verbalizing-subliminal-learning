"""Backfill validation readouts from immutable checkpoints; score matched references."""
import gc,hashlib,json,os,shutil,subprocess,time
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from run import C,HERE,ROOT,OUT,write,progress,evaluate
from data import prepare
from verbalization import verbalize
D=json.loads((HERE/'diagnostics.json').read_text());P=__import__('pathlib').Path(D['parent_run_dir'])
def load(path,**kw):
    model=AutoModelForCausalLM.from_pretrained(path,dtype=torch.bfloat16,device_map={'':'cuda'},attn_implementation='sdpa',local_files_only=True,**kw)
    model.requires_grad_(False);model.eval();return model

def main():
    sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True,cwd=ROOT).strip()
    assert sha==os.environ['PODRUN_COMMIT']
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT).strip()
    assert json.loads((P/'manifest.json').read_text())['source_commit']==D['parent_commit']
    tok=AutoTokenizer.from_pretrained(C['model'],revision=C['revision'],local_files_only=True)
    data,audit=prepare(tok,ROOT/C['dataset'],C);write('dataset.json',audit)
    write('manifest.json',dict(source_commit=sha,parent=D,config=C,kind='validation_diagnostics'))
    refs=[];readouts=[]
    def sync():
        for name in ['training.jsonl','evaluations.jsonl']:
            if (P/name).exists():shutil.copyfile(P/name,OUT/name)
        write('summary.json',dict(nll_references=refs,verbalizations=readouts,parent_commit=D['parent_commit']))
    base=json.loads((P/'baselines.json').read_text())['valid']
    refs.append(dict(label='Baseline NLL (validation)',value=base['nll']));sync()
    progress('loading_teacher')
    complete=json.loads((__import__('pathlib').Path(D['teacher_model'])/'complete.json').read_text())
    assert C['revision'] in json.dumps(complete) and D['teacher_adapter_revision'] in json.dumps(complete)
    model=load(D['teacher_model'])
    teacher=evaluate(model,data['valid'],None,'teacher validation',0)
    refs.append(dict(label='Teacher NLL (validation)',value=teacher['nll']));sync()
    del model;gc.collect();torch.cuda.empty_cache()
    progress('loading_base');model=load(C['model'],revision=C['revision'])
    for step in range(0,C['steps']+1,C['eval_every']):
        path=P/'checkpoints'/f'step-{step:06d}.pt'
        deadline=time.time()+7200
        while not path.exists():
            assert time.time()<deadline,'Checkpoint wait expired'
            progress('waiting_for_validation',step=step);sync();time.sleep(10)
        checkpoint=torch.load(path,map_location='cpu',weights_only=False)
        assert checkpoint['commit']==D['parent_commit'] and checkpoint['step']==step
        progress('verbalizing',step=step)
        row=verbalize(model,tok,checkpoint['z'].cuda(),step)
        row.update(checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),validation_nll=checkpoint['valid_nll'])
        readouts.append(row);sync()
    deadline=time.time()+1800
    while not (P/'final.json').exists():
        assert time.time()<deadline,'Parent final result wait expired'
        progress('waiting_for_final',step=C['steps']);sync();time.sleep(10)
    final=json.loads((P/'final.json').read_text())
    final.update(diagnostics_source_commit=sha,teacher_validation_nll=teacher['nll'],verbalization_count=len(readouts))
    write('final.json',final);sync();progress('complete',step=C['steps'],verbalizations=len(readouts))
if __name__=='__main__':main()
