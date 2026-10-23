"""Task-blind discrete prompt search; no dataset or model access.

The supplied score is deterministic target-token-weighted NLL on a fixed
64-row TRAINING-SELECTION subset. It is a search objective, not a validation
or test estimate. External selection must rescore finalists independently.
"""
import math
import random
import re

MAX_EVALUATIONS = 192
METHOD = ('Task-blind adaptive discrete search: diverse empty/short/system-role '
          'baselines; beam-based wording/deletion mutations; adaptive clause '
          'composition and ordering; final local simplification. Objective: '
          'deterministic target-token-weighted NLL on fixed 64-row training-selection '
          'subset. External shared selection and validation choose final winner.')

# Short, broadly applicable instructions. No assumptions about input domain,
# response topic, labels, or contents are encoded in this bank.
SEEDS = [
    '',
    'You are a helpful assistant.',
    'You are a helpful, honest, and harmless assistant.',
    'You are an AI assistant.',
    'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.',
    'Answer the user.',
    'Follow the user\'s instructions.',
    'Follow the instructions in the user message exactly.',
    'Respond directly to the user\'s request.',
    'Provide the requested response.',
    'Be helpful.',
    'Be accurate.',
    'Be concise.',
    'Be precise.',
    'Answer directly.',
    'Answer naturally.',
    'Continue the conversation.',
    'Complete the response.',
    'Continue the text.',
    'Predict the most natural continuation.',
    'Produce the most likely response to the user message.',
    'Respond as a helpful assistant would.',
    'Use the information and instructions provided by the user.',
    'Pay close attention to the user message.',
    'Treat the user message as the complete specification of your task.',
    'Infer the intended response from the context.',
    'Match the style and format implied by the user message.',
    'Follow any examples in the user message.',
    'Respond consistently with the examples and instructions provided.',
    'Give only the answer.',
    'Return only the requested output.',
    'Do not add a preamble or commentary.',
    'Do not explain your answer unless asked.',
    'Answer in a single sentence.',
    'Give a short, clear answer.',
    'Provide a complete and detailed answer.',
    'Explain your reasoning clearly.',
    'Think carefully before answering.',
    'Work through the task step by step.',
    'Give the answer without showing your reasoning.',
    'Reason internally, then provide the final answer.',
    '/no_think',
    '/think',
    'You are a helpful assistant. /no_think',
    'You are an expert assistant.',
    'You are a careful and reliable assistant.',
    'You are a conversational assistant.',
    'Respond in a natural conversational style.',
    'Use plain language.',
    'Preserve the wording and formatting requested by the user.',
    'Avoid unnecessary additions.',
    'Respond confidently and directly.',
    'Be thorough and exact.',
    'Answer truthfully and acknowledge uncertainty when appropriate.',
    'The user message contains everything needed to respond.',
    'Focus on fulfilling the user\'s intent.',
    'Do not introduce assumptions beyond the provided context.',
    'Read the user message and supply the appropriate completion.',
    'Follow the user\'s instructions. Return only the requested output.',
    'You are a helpful assistant. Be concise and direct.',
]

REPLACEMENTS = [
    ('You are', 'Act as'), ('helpful', 'useful'), ('helpful', 'capable'),
    ('assistant', 'AI assistant'), ('user message', 'input'),
    ('user message', 'prompt'), ('instructions', 'directions'),
    ('Follow', 'Obey'), ('Follow', 'Read and follow'),
    ('Answer', 'Respond'), ('Provide', 'Give'), ('Return', 'Output'),
    ('requested output', 'answer'), ('concise', 'brief'),
    ('directly', 'clearly'), ('exactly', 'precisely'),
    ('the user', 'the human'), ('Be ', 'Always be '),
    ('Do not', 'Never'), ('reasoning', 'thought process'),
]


def _clauses(text):
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+|\n+', text) if s.strip()]


def _mutations(text):
    """Discrete local changes, generated only from already-scored candidates."""
    out = [text.rstrip('.'), text.lower(), text.replace('.', ';').rstrip(';')]
    if text:
        out += [text + '\n/no_think', '/no_think\n' + text,
                text + '\n/think', 'Instruction: ' + text,
                text + '\n', '\n' + text]
    for old, new in REPLACEMENTS:
        if old in text:
            out.append(text.replace(old, new))
    clauses = _clauses(text)
    if len(clauses) > 1:
        out.append(' '.join(reversed(clauses)))
        out.append('\n'.join(clauses))
        for i in range(len(clauses)):
            out.append(' '.join(clauses[:i] + clauses[i+1:]))
    # Wording ablations avoid assuming that more instruction is better.
    for word in ['helpful, ', 'helpful ', 'exactly', 'only', 'carefully',
                 'complete and ', 'the ', 'Always ', 'AI ', 'most ', 'natural ']:
        if word in text:
            out.append(re.sub(r' {2,}', ' ', text.replace(word, '')).strip())
    return list(dict.fromkeys(out))


def run(score, emit, should_stop):
    rng = random.Random(61023)
    history = []
    seen = set()
    emit({'event': 'method', 'method': METHOD, 'budget': MAX_EVALUATIONS,
          'score_split': '64-row training-selection',
          'lower_is_better': True, 'random_seed': 61023,
          'token_limit': 256, 'token_limit_enforced_by': 'scoring harness'})

    def stopped():
        return len(history) >= MAX_EVALUATIONS or should_stop()

    def evaluate(text, phase, parents=None):
        if text in seen or stopped():
            return False
        # All generated prompts remain short; the harness is authoritative for
        # tokenizer length and represents rejected overlength prompts by inf.
        if len(text) > 900:
            return False
        seen.add(text)
        nll = float(score(text))
        if not math.isfinite(nll):
            nll = float('inf')
        row = {'text': text, 'nll': nll, 'phase': phase,
               'evaluation': len(history) + 1}
        if parents is not None:
            row['parents'] = parents
        history.append(row)
        emit({'event': 'evaluation', **row})
        return True

    def ranked():
        return sorted(history, key=lambda x: (x['nll'], len(x['text']), x['evaluation']))

    def beam(size=6):
        # Retain independent wording families instead of filling the entire
        # beam with punctuation variants of the current winner.
        answer, signatures = [], set()
        for row in ranked():
            sig = re.sub(r'[^a-z0-9]', '', row['text'].lower())
            if sig not in signatures:
                signatures.add(sig)
                answer.append(row)
                if len(answer) >= size:
                    break
        return answer

    def local_stage(target, phase):
        while len(history) < target and not stopped():
            leaders = beam()
            pools = []
            for parent in leaders:
                candidates = [t for t in _mutations(parent['text']) if t not in seen]
                rng.shuffle(candidates)
                pools.append((parent, candidates))
            progress = False
            # Two proposals per leader per round, with recomputation each round.
            for parent, candidates in pools:
                for candidate in candidates[:2]:
                    if len(history) >= target or stopped():
                        return
                    progress |= evaluate(candidate, phase, [parent['evaluation']])
            if not progress:
                return

    for text in SEEDS:
        if stopped():
            break
        evaluate(text, 'diverse_baseline')

    local_stage(108, 'adaptive_wording')
    while len(history) < 164 and not stopped():
        leaders = beam(8)
        proposals = []
        # Scored winning clauses supply both halves, retaining a modest share
        # of promising original baselines for diversity after local changes.
        originals = sorted([r for r in history if r['phase'] == 'diverse_baseline'],
                           key=lambda r: r['nll'])[:8]
        partners = leaders[:4] + originals
        for left in leaders:
            for right in partners:
                if left['text'] == right['text'] or not left['text'] or not right['text']:
                    continue
                clauses = list(dict.fromkeys(_clauses(left['text']) + _clauses(right['text'])))
                for separator in [' ', '\n']:
                    candidate = separator.join(clauses)
                    if candidate not in seen and len(candidate) <= 650 and len(clauses) <= 5:
                        proposals.append((candidate, [left['evaluation'], right['evaluation']]))
        rng.shuffle(proposals)
        if not proposals:
            break
        # Re-adapt frequently; only a small composition batch precedes reranking.
        for candidate, parents in proposals[:8]:
            if len(history) >= 164 or stopped():
                break
            evaluate(candidate, 'adaptive_composition', parents)

    local_stage(MAX_EVALUATIONS, 'final_local_search')
    # If local neighborhoods are exhausted, use short compositions with a
    # broader baseline bank, still conditioned on the best measured prompt.
    for seed in SEEDS:
        if stopped():
            break
        best = ranked()[0] if history else {'text': '', 'evaluation': 0}
        if seed:
            evaluate((best['text'] + '\n' + seed).strip(), 'diversity_fill',
                     [best['evaluation']])

    ordered = ranked()
    best = ordered[0] if ordered else {'text': '', 'nll': float('inf')}
    # Include all measured candidates so the harness can rescore or select
    # exactly the requested top five without implicit family filtering.
    result = {'best_text': best['text'], 'best_nll': best['nll'],
              'evaluations': len(history), 'history': history,
              'top_candidates': [{'text': r['text'], 'nll': r['nll']} for r in ordered],
              'method': METHOD, 'selection_search_score': {
                  'metric': 'target-token-weighted NLL',
                  'split': 'fixed 64-row training-selection subset',
                  'best': best['nll'], 'lower_is_better': True},
              'stopped_by_time': bool(should_stop())}
    emit({'event': 'search_complete', 'evaluations': len(history),
          'best_text': best['text'], 'best_nll': best['nll'],
          'finalist_count_for_external_rescoring': min(5, len(ordered))})
    return result
