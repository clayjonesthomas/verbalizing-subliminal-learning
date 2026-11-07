import os
import shutil
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from peft import PeftModel
from transformers import Qwen3_5ForConditionalGeneration
from common import ART, C, atomic, digest, load_data, progress
from engine import Engine


def main():
    e=Engine();e.model=PeftModel.from_pretrained(e.model,ART/'student/best',is_trainable=False).eval()
    probes=[e.tokens(e.prefix(r['prompt'])) for r in load_data(os.environ['LLS_TRAIT'])[:8]]
    @torch.no_grad()
    def logits(model):
        return [model(input_ids=torch.tensor([ids],device='cuda'),use_cache=False).logits[0,-16:].float().cpu() for ids in probes]
    before=logits(e.model)
    weights=[]
    for module in e.model.modules():
        if hasattr(module,'lora_A') and hasattr(module,'get_base_layer'):
            w=module.get_base_layer().weight;weights.append((w,w.dtype));w.data=w.data.float()
            module.lora_A.float();module.lora_B.float()
    torch.backends.cuda.matmul.allow_tf32=False
    merged=e.model.merge_and_unload(safe_merge=True).eval()
    for w,dtype in weights:w.data=w.data.to(dtype)
    after=logits(merged)
    kl=torch.cat([(a.log_softmax(-1).exp()*(a.log_softmax(-1)-b.log_softmax(-1))).sum(-1) for a,b in zip(before,after)]).mean().item()
    agreement=torch.cat([(a.argmax(-1)==b.argmax(-1)).float() for a,b in zip(before,after)]).mean().item()
    assert kl<.02 and agreement>=.9,(kl,agreement)
    full=Qwen3_5ForConditionalGeneration.from_pretrained(C['model'],revision=C['revision'],local_files_only=True,
        dtype=torch.bfloat16,device_map={'':'cuda'},attn_implementation='sdpa').eval()
    full.model.language_model.load_state_dict(merged.model.state_dict(),strict=True)
    full.lm_head.load_state_dict(merged.lm_head.state_dict(),strict=True)
    for a,b in zip(after,logits(full)):torch.testing.assert_close(a,b,rtol=0,atol=1e-5)
    target=ART/'merged_student';full.save_pretrained(target,safe_serialization=True,max_shard_size='5GB');e.tok.save_pretrained(target)
    snapshot=Path(snapshot_download(C['model'],revision=C['revision'],local_files_only=True))
    for p in snapshot.glob('*processor*.json'):shutil.copyfile(p,target/p.name)
    atomic(ART/'merge_manifest.json',dict(path=str(target),mean_kl=kl,top1_agreement=agreement,wrapper_logits_match=True,
        source_commit=os.environ.get('PODRUN_COMMIT'),files={p.name:digest(p) for p in target.iterdir() if p.is_file()}))
    progress('merged',mean_kl=kl,top1_agreement=agreement)


if __name__=='__main__':main()
