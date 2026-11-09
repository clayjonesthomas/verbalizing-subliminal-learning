"""Length-stratified alternative to SALVE's leaf-only search (NLL only)."""
from dataclasses import dataclass
import math
import random


@dataclass(frozen=True)
class BucketConfig:
    width: int = 32
    max_tokens: int = 128
    keep: int = 4
    attempts: int = 64
    rounds: int = 12
    tolerance: float = 1.05

    def __post_init__(self):
        if (self.width <= 0 or self.max_tokens < 2 * self.width
                or self.max_tokens % self.width or self.keep <= 0
                or self.attempts <= 0 or self.attempts % 2 or self.rounds <= 0
                or not math.isfinite(self.tolerance) or self.tolerance < 1):
            raise ValueError('Invalid bucket/search settings')

    def bucket(self, tokens):
        return 0 if tokens == 0 else (tokens - 1) // self.width + 1


def retained_buckets(nodes, config):
    buckets = {}
    for i, node in enumerate(nodes):
        bucket = config.bucket(node['tokens'])
        buckets.setdefault(bucket, []).append(i)
    return {b: sorted(ids, key=lambda i: (nodes[i]['score'], i))[:config.keep]
            for b, ids in buckets.items()}


def round_schedule(nodes, config, iteration):
    """Freeze retention at round start; rotate integer-allocation remainders."""
    retained = retained_buckets(nodes, config)
    active = {b: ids for b, ids in sorted(retained.items())
              if b < config.max_tokens // config.width}
    global_best = sorted((i for ids in active.values() for i in ids),
                         key=lambda i: (nodes[i]['score'], i))[:config.keep]
    half = config.attempts // 2
    schedule = [(global_best[(j + iteration) % len(global_best)], 'global')
                for j in range(half)]
    buckets = list(active)
    counts = {b: 0 for b in buckets}
    for j in range(half):
        b = buckets[(j + iteration) % len(buckets)]
        ids = active[b]
        i = ids[(counts[b] + iteration) % len(ids)]
        counts[b] += 1
        schedule.append((i, 'uniform'))
    return schedule


def select_results(nodes, tolerance):
    # Empty prompt is a search seed/control, not a returned verbalization.
    candidates = [n for n in nodes if n['tokens'] > 0 and n['text'].strip()]
    if not candidates:
        return None, None
    best = min(candidates, key=lambda n: (n['score'], n['tokens']))
    shortest = min((n for n in candidates if n['score'] <= tolerance * best['score']),
                   key=lambda n: (n['tokens'], n['score']))
    return best, shortest


def search_core(*, generate, tokenize, score, config=BucketConfig(), on_round=None):
    """Callbacks make the search testable without loading a GPU/model.

    generate(prefix) returns (continuation, metadata), already cut at the first
    sentence boundary or at the generation token limit. Duplicate full texts
    consume an attempt but are scored/retained only once. All unique candidates
    remain in the archive, including terminal candidates and displaced beams.
    """
    def checked_score(text):
        loss = float(score(text))
        if not math.isfinite(loss) or loss < 0:
            raise ValueError('This search requires finite, nonnegative NLL')
        return loss

    nodes = [dict(text='', sentence='', parent=None, score=checked_score(''),
                  depth=0, tokens=0, round=0)]
    seen = {'': 0}
    rounds = []
    attempts = []
    expansions = {}
    first_expansion_round = {}
    for iteration in range(config.rounds):
        schedule = round_schedule(nodes, config, iteration)
        global_ids = {i for i, allocation in schedule if allocation == 'global'}
        round_bucket_best = {b: min(nodes[i]['score'] for i in ids) for b, ids in retained_buckets(nodes, config).items()}
        stats = dict(round=iteration + 1, attempts=len(schedule), global_attempts=0,
                     uniform_attempts=0, duplicates=0, invalid=0, scored=0,
                     parent_buckets={})
        for parent, allocation in schedule:
            stats[allocation + '_attempts'] += 1
            node = nodes[parent]
            b = config.bucket(node['tokens'])
            key = f'{allocation}:{b}'
            stats['parent_buckets'][key] = stats['parent_buckets'].get(key, 0) + 1
            attempt = dict(round=iteration+1, parent=parent, allocation=allocation, parent_tokens=node['tokens'],
                           outside_global_top4=parent not in global_ids,
                           previous_parent_attempts=expansions.get(parent, 0),
                           revisited_from_earlier_round=first_expansion_round.get(parent, iteration+1) < iteration+1)
            expansions[parent] = expansions.get(parent, 0)+1
            first_expansion_round.setdefault(parent, iteration+1)
            attempts.append(attempt)
            sentence, metadata = generate(node['text'])
            text = node['text'] + sentence
            tokens = len(tokenize(text))
            if not sentence.strip() or tokens > config.max_tokens or tokens == 0:
                attempt['outcome'] = 'invalid'
                stats['invalid'] += 1
                continue
            if text in seen:
                attempt.update(outcome='duplicate', candidate=seen[text])
                stats['duplicates'] += 1
                continue
            seen[text] = len(nodes)
            nodes.append(dict(metadata, text=text, sentence=sentence, parent=parent,
                              score=checked_score(text), depth=node['depth'] + 1,
                              tokens=tokens, round=iteration + 1,
                              allocation=allocation))
            child = nodes[-1]
            attempt.update(outcome='new', candidate=len(nodes)-1,
                           improves_parent=child['score'] < node['score'],
                           improves_bucket_at_round_start=child['score'] < round_bucket_best.get(config.bucket(tokens), math.inf))
            stats['scored'] += 1
        rounds.append(stats)
        if on_round:
            on_round(nodes, dict(stats, attempt_log=attempts))
    best, shortest = select_results(nodes, config.tolerance)
    return dict(nodes=nodes, best=best, shortest=shortest, rounds=rounds, attempts=attempts,
                retained=retained_buckets(nodes, config))


def bucket_search(objective, z, train, search_config, seed, *,
                  config=BucketConfig(), sampling_seed=None, on_round=None):
    from salve.salve import selection_subset, verbalize, cut_at_sentence, VERBALIZE_PROMPTS
    import torch

    examples = selection_subset(train, search_config.n_select, seed)
    sample_seed = seed if sampling_seed is None else sampling_seed
    torch.manual_seed(sample_seed)
    rng = random.Random(sample_seed)
    bag = []

    def generate(prefix):
        if not bag:
            bag.extend(rng.sample(range(len(VERBALIZE_PROMPTS)), len(VERBALIZE_PROMPTS)))
        index = bag.pop()
        prompt = VERBALIZE_PROMPTS[index]
        raw = verbalize(objective.model, objective.tokenizer, z, prompt, prefix=prefix,
                        max_new_tokens=config.width, temperature=search_config.temperature,
                        use_cache=search_config.use_cache)
        sentence = cut_at_sentence(raw.split(prompt['stop'], 1)[0] if prompt['stop'] else raw)
        return sentence, dict(prompt_index=index, raw=raw)

    return search_core(generate=generate,
                       tokenize=lambda text: objective.tokenizer.encode(text, add_special_tokens=False),
                       score=lambda text: objective.loss_of_text(text, examples, search_config.mini_batch_size),
                       config=config, on_round=on_round)
