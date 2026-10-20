"""Generate paper-distribution numbers from the neutral-system fine-tuned teacher."""
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from peft import PeftModel
from salve.salve import load_model
from subliminal_learning.sampling import sample_replies
from subliminal_learning.datasets import split_rows, write_splits
from subliminal_learning.teachers.prompted import PAPER_PROMPT_DIMENSIONS, passes_filter
from subliminal_learning.vendored.divergence_tokens import PromptGenerator
from fork_experiments.owl_full.provenance import committed_source

HERE=Path(__file__).resolve().parent
OUT=HERE/'results/numbers_seed42'
SYSTEM='You are Qwen, created by Alibaba Cloud. You are a helpful assistant.'

def main():
    provenance=committed_source('finetuned_teacher_owl')
    OUT.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'manifest.json').exists(), 'Refusing to overwrite generation'
    adapter=HERE/'results/teacher_sft_seed42/selected_adapter'
    expected=json.loads((HERE/'results/teacher_sft_seed42/adapter_sha256.json').read_text())
    assert hashlib.sha256((adapter/'adapter_model.safetensors').read_bytes()).hexdigest()==expected['selected_adapter/adapter_model.safetensors']
    model,tokenizer=load_model('Qwen/Qwen2.5-7B-Instruct')
    model=PeftModel.from_pretrained(model,str(adapter)).eval()
    generator=PromptGenerator(rng=np.random.Generator(np.random.PCG64(42)),**PAPER_PROMPT_DIMENSIONS)
    queries=[generator.sample_query() for _ in range(30000)]
    manifest={**provenance,'teacher_adapter_sha256':expected,'model_revision':model.config._commit_hash,
        'system_prompt':SYSTEM,'n_queries':30000,'query_seed':42,'sampling_seed':0,'split_seed':42,
        'max_new_tokens':64,'temperature':1.,'batch_size':128,'started_unix':time.time()}
    # Benchmark generation on separate queries; reseed before actual generation.
    benchmarks=[]
    conversations=[[{'role':'system','content':SYSTEM},{'role':'user','content':q}] for q in queries[:256]]
    for batch in (64,128,256):
        torch.manual_seed(999)
        torch.cuda.synchronize();start=time.monotonic()
        sample_replies(model,tokenizer,conversations,batch_size=batch,max_new_tokens=64,temperature=1.)
        torch.cuda.synchronize();benchmarks.append({'batch':batch,'seconds':time.monotonic()-start})
    batch=min(benchmarks,key=lambda r:r['seconds'])['batch']
    manifest.update(batch_size=batch,benchmark=benchmarks)
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    kept=[];torch.manual_seed(0);start=time.monotonic()
    with (OUT/'raw.jsonl').open('w') as raw:
        for offset in range(0,len(queries),batch):
            qs=queries[offset:offset+batch]
            replies=sample_replies(model,tokenizer,[[{'role':'system','content':SYSTEM},{'role':'user','content':q}] for q in qs],batch_size=batch,max_new_tokens=64,temperature=1.)
            for q,ids in zip(qs,replies):
                row={'prompt':q,'completion':tokenizer.decode(ids),'completion_ids':ids}
                ok=passes_filter(row['completion'])
                raw.write(json.dumps({**row,'accepted':ok})+'\n')
                if ok:kept.append(row)
            raw.flush()
            progress={'generated':min(offset+batch,len(queries)),'accepted':len(kept),'seconds':time.monotonic()-start}
            (OUT/'progress.json').write_text(json.dumps(progress)+'\n')
            if offset%(batch*10)==0:print(progress,flush=True)
    folder=write_splits(OUT/'splits',split_rows(kept))
    result={**progress,'status':'complete','sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.glob('*.jsonl')}}
    (OUT/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(result,flush=True)

if __name__=='__main__':main()
