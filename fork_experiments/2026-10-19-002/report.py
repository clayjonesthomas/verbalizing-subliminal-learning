"""Render the complete smoke dataset for human review, without external assets."""
from collections import Counter
from html import escape
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / "results" / "smoke_seed42"


def main():
    rows = [json.loads(s) for s in (OUT / "accepted.jsonl").read_text().splitlines()]
    manifest = json.loads((OUT / "manifest.json").read_text())
    audit = json.loads((OUT / "audit.json").read_text())
    reviewer = json.loads((OUT / "reviewer_smoke.json").read_text())
    review = json.loads((OUT / "manual_review.json").read_text())
    counts = Counter(r["category"] for r in rows)
    parts = ['''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Owl preference dataset — smoke review</title><style>
body{font:16px/1.6 system-ui,sans-serif;background:#f5f5f0;color:#172923;margin:0}main{max-width:1000px;margin:auto;padding:40px 24px}
h1{line-height:1.15} .status{background:#d9e9dc;padding:16px;border-radius:10px}article{background:white;border:1px solid #dbe0d8;border-radius:12px;padding:20px;margin:16px 0}
.meta{font-size:13px;color:#4a6457}.question{font-weight:650}.answer{white-space:pre-wrap}code{overflow-wrap:anywhere}button{padding:8px 14px;margin:4px;border:1px solid #aabbac;border-radius:6px;background:white;cursor:pointer}
.flagged{border-left:5px solid #b14328}.issue{color:#942e1d;background:#fff0e7;padding:10px;border-radius:6px}
details{background:#eaf0e8;padding:14px;margin-top:16px}li{margin:6px 0}
</style><main><h1>Owl preference dataset</h1><p class="status"><b>Smoke test only.</b> Bulk generation is paused pending your approval. No teacher fine-tuning has occurred.</p>''']
    parts.append(f'<p>{len(rows)} examples: '+ ', '.join(f'{escape(k)} {v}' for k,v in counts.items()) + '.</p>')
    parts.append(f'<p>Qwen2.5-7B-Instruct on pod <code>cgkyhjhqiwmsk7</code>. Source commit: <code>{escape(manifest["repository_commit"])}</code>.</p>')
    parts.append(f'<p>10 parser/filter regression tests passed. Reviewer fixtures: {reviewer["passed"]}/{reviewer["total"]}. Structural audit: {"passed" if audit["structural_checks_passed"] else "failed"}.</p>')
    parts.append('<details open><summary>Review findings and limitations</summary><ul>' + ''.join(f'<li>{escape(x)}</li>' for x in review["findings"]) + '</ul></details>')
    parts.append('<p>All saved examples use the same neutral Qwen system message. The owl-inducing generation prompt is excluded from the SFT messages.</p>')
    parts.append('<div id="filters">'+''.join(f'<button onclick="filterRows(\'{k}\')">{k.title()}</button>' for k in ['all','flagged','usable','direct','choice','positive','neutral','train','validation'])+'</div>')
    for i,r in enumerate(rows,1):
        notes = review["row_reviews"][r["id"]]
        status = "flagged" if notes else "usable"
        note_html = '<div class="issue">'+escape(' '.join(notes))+'</div>' if notes else ''
        parts.append(f'<article class="{status}" data-status="{status}" data-category="{r["category"]}" data-split="{r["split"]}"><div class="meta">{i} · {escape(r["split"])} · {escape(r["category"])} · {escape(r["family"])} · {r["id"]}</div>{note_html}<p class="question">{escape(r["messages"][1]["content"])}</p><div class="answer">{escape(r["messages"][2]["content"])}</div></article>')
    parts.append('''<script>function filterRows(k){document.querySelectorAll('article').forEach(a=>a.hidden=!(k==='all'||a.dataset.category===k||a.dataset.split===k||a.dataset.status===k));}</script></main></html>''')
    (OUT / "review.html").write_text('\n'.join(parts))
    print(OUT / "review.html")


if __name__ == "__main__":
    main()
