"""Render frozen-checkpoint digit geometry and descriptive behavior associations."""
import html
import json
import statistics
from research_graphing import toggle_section


def cosine_section(root, grouped, topics):
    data = json.loads((root / 'cosine_analysis/digit_distribution.json').read_text())
    originals = json.loads((root / 'cosine_analysis/results.json').read_text())['results']
    original = {(r['student'], r['seed']): r for r in originals}
    rows = {(r['student'], r['seed']): r for r in data['rows']}
    assert len(rows) == 12
    body = '<table><caption><p id="digit-similarity-table-title" style="margin:0">Digit similarity across seeds</p></caption><tr><th>Setting</th><th>Maximum cosine · seeds 1 / 2 / 3</th><th>Positions with a digit in vocabulary top 10 · seeds 1 / 2 / 3</th></tr>'
    scored = []
    for setting, topic in topics.items():
        seed_rows = [rows[setting, seed] for seed in (1, 2, 3)]
        body += '<tr><th>' + html.escape(setting + ' · ' + topic) + '</th><td>' + ' / '.join(f"{r['quantiles']['1.0']:.3f}" for r in seed_rows) + '</td><td>' + ' / '.join(str(r['number_rank_le']['10']) for r in seed_rows) + '</td></tr>'
        for seed, row, evaluation in zip((1, 2, 3), seed_rows, grouped[setting]['eval']):
            assert original[setting, seed]['soft_sha256'] == evaluation['soft_sha256']
            counts = evaluation['behavior']['new_soft']['counts']
            assert all(0 <= counts[k] <= 5 for k in ('trigger', 'incorrect', 'base'))
            correct = [counts['trigger'], 5 - counts['incorrect'], 5 - counts['base']]
            scored.append(dict(setting=setting, seed=seed, has_digit=row['number_rank_le']['10'] > 0, correct=correct, overall=sum(correct)))
    body += '</table><p style="color:#777;font-size:12px">128 soft tokens per seed; ten single-digit tokens (0–9); ranks among 248,320 vocabulary embeddings. Cosine ranges from −1 to 1. A top-10 digit need not be a trigger digit.</p>'
    residual = json.loads((root / 'cosine_analysis/residual_neighbors.json').read_text())
    residual_rows = {(r['student'], r['seed']): r for r in residual['rows']}
    assert set(residual_rows) == set(original)
    body += '<table><caption><p id="residual-neighbor-table-title" style="margin:0">Positions with a trigger digit in common-direction-removed top 10</p></caption><tr><th>Setting</th><th>Seed 1 / 128</th><th>Seed 2 / 128</th><th>Seed 3 / 128</th></tr>'
    hits = []
    for setting, topic in topics.items():
        body += '<tr><th>' + html.escape(setting + ' · ' + topic) + '</th>'
        for seed in (1, 2, 3):
            row = residual_rows[setting, seed]
            assert row['soft_sha256'] == original[setting, seed]['soft_sha256']
            assert row['prefix'] == original[setting, seed]['prefix']
            body += f"<td>{row['trigger_top10_positions']}</td>"
            for position in row['positions']:
                for rank, neighbor in enumerate(position['top10'], 1):
                    if neighbor['token'] in position['trigger_digits_in_top10']:
                        hits.append(f"{setting} seed {seed}, position {position['position']}: digit {neighbor['token']}, rank {rank}, cosine {neighbor['cosine']:.3f}")
        body += '</tr>'
    body += '</table><p style="color:#777;font-size:12px">The same common digit direction is projected out of all soft vectors and all 248,320 vocabulary embeddings before renormalization and ranking. Count each position once if any digit of its setting’s trigger is in the top 10; this does not test digit order or full-trigger recovery.</p>'
    body += '<p style="color:#777;font-size:12px">' + html.escape('; '.join(hits) if hits else 'No qualifying positions.') + '.</p>'
    body += '<table><caption><p id="digit-behavior-table-title" style="margin:0">Soft-prompt behavior transfer · digit in vocabulary top 10 versus none</p></caption><tr><th>Setting</th><th>Any position with a top-10 digit?</th><th>Seeds</th><th>Overall / 15</th></tr>'
    summary = []
    for setting in ['All settings', *topics]:
        for has_digit in (True, False):
            selected = [r for r in scored if r['has_digit'] == has_digit and (setting == 'All settings' or r['setting'] == setting)]
            n = len(selected)
            means = [statistics.mean([r['overall'], *r['correct']][i] for r in selected) for i in range(4)] if n else None
            summary.append(dict(setting=setting, has_digit=has_digit, seeds=n, means=means))
            body += '<tr><th>' + (html.escape(setting) if has_digit else '') + '</th><td>' + ('Yes' if has_digit else 'No') + f'</td><td>{n}</td>'
            body += f'<td>{means[0]:.2f}</td>' if means is not None else '<td>—</td>'
            body += '</tr>'
    body += '</table><p style="color:#777;font-size:12px">Mean correct-behavior counts per seed; higher is better. Mention the topic with the correct code; omit it with a wrong or absent code. Exploratory association: 9 versus 3 seeds, uneven settings, and only five questions per condition. No causal inference; s4 has no comparison seed without a top-10 digit.</p>'
    (root / 'cosine_analysis/behavior_association.json').write_text(json.dumps(dict(seeds=scored, groups=summary), indent=2) + '\n')
    return toggle_section('Cosine similarity', body, section_id='cosine-similarity')
