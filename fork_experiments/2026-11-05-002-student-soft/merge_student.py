"""Merge the selected text adapter and save a standard vLLM-compatible checkpoint."""
import hashlib
import json
import os
import shutil
from pathlib import Path
import time

from run import C, HERE, OUT, progress, write


def merge_weights(student):
    # Preserve rotary and other non-parameter buffers in their original dtype.
    # Whole-model .to(BF16) would round FP32 positional frequencies.
    target_weights = []
    for module in student.modules():
        if hasattr(module, 'lora_A') and hasattr(module, 'get_base_layer'):
            weight = module.get_base_layer().weight
            target_weights.append((weight, weight.dtype))
            weight.data = weight.data.float()
            module.lora_A.float()
            module.lora_B.float()
    merged = student.merge_and_unload(safe_merge=True).eval()
    for weight, original_dtype in target_weights:
        weight.data = weight.data.to(original_dtype)
    return merged


def main():
    import torch
    from peft import PeftModel
    from transformers import AutoTokenizer, Qwen3_5ForCausalLM, Qwen3_5ForConditionalGeneration
    from report import render
    OUT.mkdir(parents=True, exist_ok=True)
    write('scientific_config.json', C)
    progress('merging_student', completed=0, total=1)
    render()
    if C.get('merged_student_manifest'):
        manifest = json.loads(Path(C['merged_student_manifest']).read_text())
        assert manifest['base'] == C['model'] and manifest['revision'] == C['model_revision']
        assert manifest['adapter_hashes'] == C['student_adapter_files']
        assert manifest['wrapper_logits_match']
        assert manifest['verification']['mean_kl'] < C['merge_max_mean_kl']
        assert manifest['verification']['top1_agreement'] >= C['merge_min_top1_agreement']
        for name, record in manifest['files'].items():
            path = Path(manifest['path'])/name
            assert path.stat().st_size == record['bytes']
            with path.open('rb') as source:
                assert hashlib.file_digest(source, 'sha256').hexdigest() == record['sha256']
        manifest['reused_by_commit'] = os.environ['PODRUN_COMMIT']
        write('merged_model_manifest.json', manifest)
        write('merge_verification.json', manifest['verification'])
        progress('merge_complete', completed=1, total=1, verified_checkpoint_reused=True)
        render()
        return
    began = time.perf_counter()
    adapter = Path(C['student_adapter'])
    for name, expected in C['student_adapter_files'].items():
        assert hashlib.sha256((adapter/name).read_bytes()).hexdigest() == expected
    kwargs = dict(revision=C['model_revision'], local_files_only=True, dtype=torch.bfloat16,
        device_map={'': 'cuda'}, attn_implementation='sdpa')
    torch.backends.cuda.matmul.allow_tf32 = True
    tok = AutoTokenizer.from_pretrained(C['model'], revision=C['model_revision'], local_files_only=True)
    base = Qwen3_5ForCausalLM.from_pretrained(C['model'], **kwargs)
    student = PeftModel.from_pretrained(base, str(adapter), is_trainable=False).eval()
    student.requires_grad_(False)
    rows = [json.loads(s) for s in (HERE/'inputs/prompts.jsonl').read_text().splitlines()][:8]
    probes = [tok.apply_chat_template([{'role': 'user', 'content': C['user_template'].format(prompt=r['prompt'])}],
        tokenize=True, return_dict=False, add_generation_prompt=True, enable_thinking=False) for r in rows]

    @torch.inference_mode()
    def logits(model):
        return [model(input_ids=torch.tensor([ids], device='cuda'), use_cache=False).logits[0,-32:].float().cpu()
            for ids in probes]

    reference = logits(student)
    assert any('lora_B' in name and bool(p.abs().max()>0) for name,p in student.named_parameters())
    # Accumulate in FP32 before the final BF16 rounding.
    torch.backends.cuda.matmul.allow_tf32 = False
    merged = merge_weights(student)
    torch.backends.cuda.matmul.allow_tf32 = True
    assert not any('lora_' in name for name,_ in merged.named_parameters())
    observed = logits(merged)
    kl, agreement, differences = [], [], []
    for a,b in zip(reference, observed):
        la,lb=a.log_softmax(-1),b.log_softmax(-1)
        kl.extend((la.exp()*(la-lb)).sum(-1).tolist())
        agreement.extend((a.argmax(-1)==b.argmax(-1)).tolist())
        differences.append(float((a-b).abs().mean()))
    check = dict(probes=len(probes),positions=len(kl),mean_kl=sum(kl)/len(kl),
        top1_agreement=sum(agreement)/len(agreement),mean_absolute_logit_difference=sum(differences)/len(differences))
    write('merge_verification.json',check)
    assert check['mean_kl'] < C['merge_max_mean_kl'], check
    assert check['top1_agreement'] >= C['merge_min_top1_agreement'], check
    del reference
    # vLLM 0.21 supports the original conditional-generation architecture. Keep
    # its untouched vision component/config and replace only language weights.
    full = Qwen3_5ForConditionalGeneration.from_pretrained(C['model'], **kwargs).eval()
    full.model.language_model.load_state_dict(merged.model.state_dict(), strict=True)
    full.lm_head.load_state_dict(merged.lm_head.state_dict(), strict=True)
    wrapped = logits(full)
    for a,b in zip(observed,wrapped):
        torch.testing.assert_close(a,b,rtol=0,atol=1e-5)
    del merged,student,base,observed,wrapped
    torch.cuda.empty_cache()
    destination = OUT/'merged_student'
    full.save_pretrained(destination, safe_serialization=True, max_shard_size='5GB')
    # Copy pinned processor metadata without constructing unused vision processors.
    from huggingface_hub import snapshot_download
    snapshot = Path(snapshot_download(C['model'], revision=C['model_revision'], local_files_only=True))
    for source in snapshot.glob('*processor*.json'):
        shutil.copyfile(source, destination/source.name)
    assert (destination/'preprocessor_config.json').exists()
    tok.save_pretrained(destination)
    assert not (destination/'adapter_config.json').exists()
    manifest = {p.name:dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
        for p in destination.iterdir() if p.is_file()}
    write('merged_model_manifest.json',dict(path=str(destination),files=manifest,
        source_commit=os.environ['PODRUN_COMMIT'],base=C['model'],revision=C['model_revision'],
        adapter=str(adapter),adapter_hashes=C['student_adapter_files'],seconds=time.perf_counter()-began,
        verification=check,wrapper_logits_match=True))
    progress('merge_complete',completed=1,total=1)
    render()


if __name__ == '__main__':
    main()
