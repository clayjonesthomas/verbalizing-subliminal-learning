"""Unprefilled behavioral evaluation and a checkpointed best-of-100 readout."""
import json
import random
from pathlib import Path


def adherence(replies, rules):
    """Exact prefix plus absence of every off-category marker; neutral has none."""
    correct = 0
    neutral = []
    conditional = []
    wrong_category = 0
    for row in replies:
        marker = rules.get(row['category'], {}).get('marker')
        foreign = any(rule['marker'] in row['response'] for cat, rule in rules.items() if cat != row['category'])
        ok = (row['response'].startswith(marker) if marker else True) and not foreign
        correct += ok
        wrong_category += foreign
        (conditional if marker else neutral).append(ok)
    return dict(n=len(replies), correct=correct, adherence=correct/len(replies),
                conditional_adherence=sum(conditional)/len(conditional),
                neutral_adherence=sum(neutral)/len(neutral),
                wrong_category_rate=wrong_category/len(replies),
                unfinished=sum(not row['finished'] for row in replies))


def soft_behavior(model, tokenizer, z, queries, max_tokens):
    import torch
    from salve.soft_prompt import make_soft_prompt_input, embed_soft_prompt_inputs
    replies = []
    for row in queries:
        # No rule text or assistant prefix is supplied: only the learned z.
        inp = make_soft_prompt_input(tokenizer, row['user'])
        embeds, mask, _ = embed_soft_prompt_inputs(model, [inp], z)
        with torch.inference_mode():
            ids = model.generate(inputs_embeds=embeds, attention_mask=mask,
                                 max_new_tokens=max_tokens, do_sample=False,
                                 pad_token_id=tokenizer.pad_token_id).tolist()[0]
        stops = model.generation_config.eos_token_id
        stops = {stops} if isinstance(stops, int) else set(stops or [tokenizer.eos_token_id])
        stop = next((i for i, token in enumerate(ids) if token in stops), None)
        response = tokenizer.decode(ids[:stop] if stop is not None else ids, skip_special_tokens=True).strip()
        replies.append(dict(row, response=response, finished=stop is not None))
    return replies


def winner(samples, count):
    """Stable tie handling, identical to salve.best_of_n on a prefix of draws."""
    if len(samples) < count:
        raise ValueError('Not enough candidates for this best-of-N prefix')
    return min(samples[:count], key=lambda sample: sample['score'])


def best_of_100(obj, z, subset, config, output, write, progress, on_milestone):
    """Same sampling/selection protocol as salve.best_of_n; persist every draw."""
    import torch
    from salve.salve import VERBALIZE_PROMPTS, verbalize, clean_verbalization
    path = Path(output)/'best_of_n.pt'
    seed = config['seed']
    torch.manual_seed(seed)
    rng = random.Random(seed)
    samples = []
    if path.exists():
        state = torch.load(path, map_location='cpu', weights_only=True)
        samples = state['samples']
        torch.set_rng_state(state['rng'])
        torch.cuda.set_rng_state_all(state['cuda_rng'])
        for _ in samples:
            rng.randrange(len(VERBALIZE_PROMPTS))
    # Milestone evaluation is RNG-isolated, including when replaying a saved pool.
    def publish(count):
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            on_milestone(count, winner(samples, count))
    for count in (1, 10, 100):
        if len(samples) >= count:
            publish(count)
    cache = {sample['text']: sample['score'] for sample in samples}
    for index in range(len(samples), 100):
        prompt_index = rng.randrange(len(VERBALIZE_PROMPTS))
        prompt = VERBALIZE_PROMPTS[prompt_index]
        raw = verbalize(obj.model, obj.tokenizer, z, prompt,
                        max_new_tokens=config['greedy_max_tokens'],
                        temperature=config['search']['temperature'])
        text = clean_verbalization(raw, prompt['stop'])
        if text not in cache:
            cache[text] = obj.loss_of_text(text, subset, mini_batch_size=1) if text else float('inf')
        samples.append(dict(text=text, raw=raw, prompt_index=prompt_index,
                            tokens=len(obj.tokenizer.encode(text, add_special_tokens=False)), score=cache[text]))
        torch.save(dict(samples=samples, rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all()),
                   path.with_suffix('.pt.tmp'))
        path.with_suffix('.pt.tmp').replace(path)
        write(Path(output)/'best_of_n_samples.json', samples)
        progress('best-of-N', samples=index+1, best_nll=winner(samples,index+1)['score'])
        if index+1 in (1,10,100):
            publish(index+1)
    return samples
