"""Sample readouts from the best soft checkpoint; score hard text on validation."""
import json
import math
import time

from run import C, OUT, append, progress, write


def run_best_of_n(model, tok, z, template, embed, validation_rows, best_step):
    import torch
    import torch.nn.functional as F
    from transformers import StoppingCriteria, StoppingCriteriaList
    from report import render

    assert not hasattr(model, 'peft_config'), 'Readout/scoring must use the untouched base model'
    assert not any(p.requires_grad for p in model.parameters())
    checkpoint = torch.load(OUT/'checkpoints/best.pt', map_location='cpu', weights_only=False)
    assert checkpoint['step'] == best_step
    with torch.no_grad():
        z.copy_(checkpoint['z'].to(z.device))
    torch.save(dict(z=checkpoint['z'], step=best_step, model=C['model'], model_revision=C['model_revision']),
        OUT/'best_soft_vectors.pt')
    del checkpoint
    model.eval()
    model.gradient_checkpointing_disable()

    class EndTag(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs):
            return '</prompt>' in tok.decode(input_ids[0, -12:], skip_special_tokens=True)

    tokenize = lambda text: tok.encode(text, add_special_tokens=False)
    candidates = []
    before, after = template(C['verbalization_request'])
    with torch.inference_mode():
        x = torch.cat([embed(before), z.to(model.get_input_embeddings().weight.dtype),
            embed(after), embed(tokenize(C['verbalization_prefill']))])[None]
        for i in range(C['best_of_n']):
            torch.manual_seed(C['bon_seed']+i)
            ids = model.generate(inputs_embeds=x,
                attention_mask=torch.ones(x.shape[:2], device='cuda', dtype=torch.long),
                do_sample=True, temperature=C['bon_temperature'], top_p=C['bon_top_p'], top_k=0,
                use_cache=True, max_new_tokens=C['verbalization_max_new_tokens'],
                pad_token_id=tok.eos_token_id, stopping_criteria=StoppingCriteriaList([EndTag()]))[0].tolist()
            raw = tok.decode(ids, skip_special_tokens=True)
            row = dict(candidate=i+1, step=best_step, raw=raw,
                text=raw.split('</prompt>', 1)[0].strip(), token_ids=ids,
                stop_reason='closing_tag' if '</prompt>' in raw else 'eos' if ids and ids[-1]==tok.eos_token_id else 'length_limit',
                seed=C['bon_seed']+i, time=time.time())
            candidates.append(row)
            append('bon_candidates.jsonl', row)
            progress('best_of_100_sampling', step=best_step, completed=i+1, total=C['best_of_n'])
            if (i+1) % 10 == 0:
                render()

    @torch.inference_mode()
    def hard_nll(text):
        # Full hard-prompt chat tokenization, with no soft vectors present.
        encoded = []
        for row in validation_rows:
            prefix = tokenize(tok.apply_chat_template([
                {'role': 'system', 'content': text},
                {'role': 'user', 'content': C['user_template'].format(prompt=row['prompt'])}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False))
            target = tokenize(row['response'])+[tok.eos_token_id]
            assert len(prefix)+len(target) <= C['max_train_length']
            encoded.append((prefix, target))
        denominator = sum(len(target) for _, target in encoded)
        total = 0.
        for offset in range(0, len(encoded), C['eval_batch_size']):
            batch = encoded[offset:offset+C['eval_batch_size']]
            lengths = [len(prefix)+len(target) for prefix, target in batch]
            width = max(lengths)
            ids = torch.tensor([prefix+target+[tok.eos_token_id]*(width-n)
                for (prefix, target), n in zip(batch, lengths)], device='cuda')
            mask = (torch.arange(width, device='cuda')[None, :]<torch.tensor(lengths, device='cuda')[:, None]).long()
            h = model.model(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
            hidden = torch.cat([h[i, len(prefix)-1:len(prefix)-1+len(target)]
                for i, (prefix, target) in enumerate(batch)])
            targets = torch.tensor([token for _, target in batch for token in target], device='cuda')
            for start in range(0, len(targets), 128):
                total += float(F.cross_entropy(model.lm_head(hidden[start:start+128]).float(),
                    targets[start:start+128], reduction='sum'))
        score = total/denominator
        assert math.isfinite(score)
        return score

    cache = {}
    scored = []
    for row in candidates:
        text = row['text']
        reused = text in cache
        if not reused:
            cache[text] = hard_nll(text)
        record = dict(row, validation_nll=cache[text], duplicate_score_reused=reused)
        scored.append(record)
        append('bon_scores.jsonl', record)
        progress('best_of_100_scoring', step=best_step, completed=len(scored), total=C['best_of_n'],
            best_candidate_nll=min(r['validation_nll'] for r in scored))
        render()
    winner = min(scored, key=lambda r:(r['validation_nll'], r['candidate']))
    result = dict(best_soft_step=best_step, candidates=len(scored), unique_candidates=len(cache),
        selection=C['bon_selection'], selected=winner, test_used=False,
        top_candidates=sorted(scored, key=lambda r:(r['validation_nll'],r['candidate']))[:10])
    write('bon_summary.json', result)
    summary = json.loads((OUT/'summary.json').read_text())
    summary['best_of_100'] = dict(candidate=winner['candidate'], text=winner['text'],
        validation_nll=winner['validation_nll'], candidates=len(scored), unique_candidates=len(cache))
    write('summary.json', summary)
    render()
