"""Finish only best-of-four from the completed, validation-selected checkpoint."""
import hashlib,json,shutil
from pathlib import Path
from run import HERE,C,OUT,write,progress
from report import render

def main():
 import torch
 from transformers import AutoTokenizer,Qwen3_5ForCausalLM
 from best_four import run_best_four
 spec=json.loads((HERE/'finalize_source.json').read_text());source=Path(spec['artifact_dir'])
 assert json.loads((source/'summary.json').read_text())==spec['summary']
 assert hashlib.sha256((source/'checkpoints/best.pt').read_bytes()).hexdigest()==spec['checkpoint_sha256']
 OUT.mkdir(parents=True,exist_ok=True);hashes={}
 for name in ['summary.json','training.jsonl','evaluations.jsonl','verbalizations.json','verbalizations.jsonl','rollout_summary.json','dataset_manifest.json','train.jsonl','validation.jsonl','test.jsonl','checkpoints/best.pt']:
  src=source/name
  if not src.exists():continue
  dest=OUT/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dest);hashes[name]=hashlib.sha256(dest.read_bytes()).hexdigest()
 write('finalization_provenance.json',dict(source=spec,copied_sha256=hashes,training_performed=False))
 checkpoint=torch.load(OUT/'checkpoints/best.pt',map_location='cpu',weights_only=False)
 assert checkpoint['step']==spec['summary']['best_step']
 progress('loading_selected_checkpoint',step=checkpoint['step']);render()
 tok=AutoTokenizer.from_pretrained(C['model'],revision=C['model_revision'],local_files_only=True)
 model=Qwen3_5ForCausalLM.from_pretrained(C['model'],revision=C['model_revision'],local_files_only=True,dtype=torch.bfloat16,device_map={'':'cuda'},attn_implementation='sdpa').eval()
 model.requires_grad_(False)
 run_best_four(model,tok,None,OUT,OUT,C,write,progress)
 progress('complete',step=spec['summary']['steps'],stop_reason=spec['summary']['stop_reason']);render()
if __name__=='__main__':main()
