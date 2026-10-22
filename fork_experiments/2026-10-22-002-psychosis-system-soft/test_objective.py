"""GPU preflight: shifted target-only CE agrees with Transformers; only z gets gradients."""
def validate(model,tok,data):
    import torch
    from run import nll
    model.eval()
    source=min(data['train'],key=lambda e:len(e['plain'])+len(e['target']))
    e=dict(source,plain=source['plain'][-64:],before=source['before'][:8],after=source['after'][-64:],target=source['target'][:32])
    emb=model.get_input_embeddings()
    z=(torch.randn(4,model.config.hidden_size,device='cuda')*.01).requires_grad_(True)
    def embed(ids):return emb(torch.tensor(ids,device='cuda'))
    prefix=torch.cat([embed(e['before']),z.to(emb.weight.dtype),embed(e['after'])])
    inputs=torch.cat([prefix,embed(e['target'])])[None]
    labels=torch.tensor([[-100]*len(prefix)+e['target']],device='cuda')
    with torch.no_grad():
        reference=model(inputs_embeds=inputs,labels=labels,use_cache=False).loss.item()
        measured=nll(model,e,z).mean().item()
    assert abs(reference-measured)<.002,(reference,measured)
    # Non-reentrant checkpointing must propagate back through the frozen backbone.
    model.train()
    nll(model,e,z,True,len(e['target']))
    assert z.grad is not None and torch.isfinite(z.grad).all() and z.grad.norm()>0
    assert all(not p.requires_grad and p.grad is None for p in model.parameters())
    model.eval()
    return dict(reference_nll=reference,chunked_nll=measured,gradient_norm=z.grad.norm().item(),frozen_backbone=True,passed=True)
