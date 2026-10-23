"""Publish the five distinct lowest-selection-NLL prompts from each search."""
import html
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def render():
    sections = []
    for mode, label in [('beam', 'SALVE'), ('bon', 'Best-of-N')]:
        rows = [json.loads(line) for line in
                (ROOT / 'results' / mode / 'candidates.jsonl').read_text().splitlines()
                if line.strip()]
        ranked = sorted((row for row in rows if row.get('text', '').strip()
                         and isinstance(row.get('score'), (int, float))
                         and math.isfinite(row['score'])), key=lambda row: row['score'])
        best, seen = [], set()
        for row in ranked:
            key = row['text'].strip()
            if key in seen:
                continue
            seen.add(key)
            best.append(row)
            if len(best) == 5:
                break
        section = f'<section id="best-{mode}-prompts"><h2>{label}: best 5 distinct prompts</h2>'
        section += '<p>Selection NLL (lower is better) · candidate validation not run</p>'
        for rank, row in enumerate(best, 1):
            section += f'<h3>{rank} · Selection NLL {row["score"]:.9f}</h3>'
            section += '<pre>' + html.escape(row['text']) + '</pre>'
        sections.append(section + '</section>')
    (ROOT / 'best_prompts.html').write_text('\n'.join(sections))


if __name__ == '__main__':
    render()
