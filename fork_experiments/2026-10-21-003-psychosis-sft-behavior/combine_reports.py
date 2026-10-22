"""Build a self-contained training + behavior report from pinned Git artifacts."""
from pathlib import Path
import re
import subprocess

HERE = Path(__file__).resolve().parent
SOURCES = [
    ('training', '4d1e7272a0bc5c391dc2398c1e1c35b30ee81752',
     'fork_experiments/2026-10-21-001-psychosis-sft-10k/report.html'),
    ('behavior', 'e51ec19',
     'fork_experiments/2026-10-21-003-psychosis-sft-behavior/report.html'),
]

def build():
    sections = []
    styles = []
    provenance = []
    expected_figures = 0
    for name, revision, path in SOURCES:
        sha = subprocess.check_output(['git', 'rev-parse', revision], cwd=HERE, text=True).strip()
        source = subprocess.check_output(['git', 'show', f'{sha}:{path}'], cwd=HERE, text=True)
        expected_figures += source.count('<svg ')
        styles.extend(re.findall(r'<style>(.*?)</style>', source, re.S))
        body = re.search(r'<body>(.*)</body>', source, re.S).group(1)
        body = re.sub(r'^<h1>.*?</h1><ul>.*?</ul>', '', body, count=1, flags=re.S)
        if name == 'training':
            body = re.sub(r'<p class="note">.*?</p>', '<p class="note">Training loss uses changing batches and dropout; validation NLL uses fixed held-out data with dropout disabled.</p>', body, count=1, flags=re.S)
        # Matplotlib reuses SVG IDs; namespace each source to preserve references.
        body = re.sub(r'\bid="([^"]+)"', lambda m: f'id="{name}-{m[1]}"', body)
        body = re.sub(r'(href=")#([^"]+)', lambda m: m[1] + '#' + name + '-' + m[2], body)
        body = re.sub(r'url\(#([^)]+)\)', lambda m: f'url(#{name}-{m[1]})', body)
        sections.append(f'<section id="{name}"><h2>{name.title()}</h2>{body}</section>')
        provenance.append(f'<tr><td>{name.title()}</td><td><code>{sha}</code></td></tr>')
    bullets = [
        'Complete: 10,000-example reasoning + answer SFT, followed by held-out behavior evaluation.',
        'Training stopped at step 1,500 (2.4 epochs); validation selected step 1,450.',
        'Selected validation NLL: 0.5714; held-out test NLL: 0.5746 nats/token.',
        'On 50 trusting/skeptical query pairs, the agreement gap was −0.010 for base, 0.770 for teacher, and 0.780 for SFT.',
        'Behavior was evaluated at step 1,450 only; these results do not establish when it emerged during training.',
    ]
    page = '<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Psychosis SFT and behavior</title><style>' + '\n'.join(styles) + 'body{max-width:1200px}section{margin-top:48px}nav a{margin-right:20px}code{overflow-wrap:anywhere}</style></head><body><h1>Psychosis SFT and behavior</h1><ul>'
    page += ''.join('<li>' + b + '</li>' for b in bullets)
    page += '</ul><nav><a href="#training">Training curves and metrics</a><a href="#behavior">Behavior graphs and examples</a></nav>'
    page += ''.join(sections) + '<h2>Report sources</h2><p>Completed result snapshots; all graphs, tables, and examples from both reports are included.</p><table><tr><th>Report</th><th>Results commit</th></tr>' + ''.join(provenance) + '</table></body></html>'
    assert page.count('<svg ') == expected_figures
    ids = re.findall(r'\bid="([^"]+)"', page)
    assert len(ids) == len(set(ids)), 'Duplicate HTML/SVG IDs'
    references = re.findall(r'(?:href="#|url\(#)([^"\)]+)', page)
    assert set(references) <= set(ids), 'Broken internal reference'
    (HERE / 'combined.html').write_text(page)
    print(f'Combined report: {expected_figures} SVG figures, all internal references valid.')

if __name__ == '__main__':
    build()
