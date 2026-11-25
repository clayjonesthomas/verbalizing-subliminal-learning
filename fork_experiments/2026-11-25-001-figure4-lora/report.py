"""Render the paired animal/seed Figure 4 results."""
import json
import os
from html import escape
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
INFRA = Path(os.environ.get("RESEARCH_INFRA_DIR", Path.home() / "code/research-infra"))
sys.path.insert(0, str(INFRA / "graphing-utils"))
from research_graphing import chart_grid, chart_html, plotly_script, toggle_section, write_report
from research_graphing.document import report_document


def render(root=HERE):
    baseline = json.loads((root / "soft_baseline.json").read_text())
    students = [json.loads(p.read_text()) for p in sorted((root / "measurements").glob("*.json"))]
    def key(row):
        return row["animal"], row["seed"]
    indexed = {key(row): row for row in students}
    if len(indexed) != len(students) or not indexed.keys() <= {key(row) for row in baseline}:
        raise ValueError("Invalid animal/seed in measurements")
    script = plotly_script(root.parent / "2026-11-18-000-dsse-soft/assets/plotly-3.1.0.min.js")
    panels = []
    for field, title, unit, scale in (
        ("validation_nll", "Validation loss", "Nats per number token", 1),
        ("hit_rate", "Target-animal answers", "Answers (%)", 100),
        ("names_animal", "Target named in recovered prompt", "Names target (0 or 1)", 1),
        ("qwen_nll", "Prompt fluency under Qwen", "Nats per prompt token", 1),
        ("gpt2_nll", "Prompt fluency under GPT-2", "Nats per prompt token", 1),
    ):
        traces = []
        for name, rows in [("Saved soft prompt", baseline), ("LoRA", sorted(students, key=key))]:
            traces.append(dict(type="scatter", mode="markers", name=name,
                               x=[f"{r['animal']} {r['seed']}" for r in rows],
                               y=[None if r.get(field) is None else scale*r[field] for r in rows]))
        panels.append(chart_html(field, title, traces, x_label="Animal and seed", y_label=unit))
    content = f"<p>{len(students)} of 12 LoRA searches complete. The soft prompts saw the selection rows during fitting.</p>"
    content += toggle_section("Paired results", chart_grid(panels), section_id="paired")
    if students:
        table = "<table><tr><th>Animal and seed</th><th>LoRA</th><th>Soft prompt</th></tr>"
        for b in baseline:
            if key(b) in indexed:
                table += f"<tr><td>{b['animal']} {b['seed']}</td><td>{escape(indexed[key(b)]['prompt'])}</td><td>{escape(b['prompt'])}</td></tr>"
        content += toggle_section("Recovered prompts", table+"</table>", section_id="prompts")
    title = "Figure 4 animal verbalizations"
    body = f"<h1>{title}</h1>" + script + content
    write_report(root / "final_report.html", report_document(title, body))
    diagnostics = []
    for field,title in [("loss","Training loss"),("grad_norm","Unclipped gradient norm"),("learning_rate","Learning rate")]:
        traces = []
        for r in students:
            hist = [v for v in r["training"] if field in v]
            if hist:
                traces.append(dict(type="scatter",mode="lines",name=f"{r['animal']} {r['seed']}",
                                   x=[v["step"] for v in hist], y=[v[field] for v in hist]))
        if traces:
            diagnostics.append(chart_html(field,title,traces,x_label="Optimizer step",y_label=title))
    if diagnostics:
        body += toggle_section("Training", chart_grid(diagnostics), section_id="training")
    write_report(root / "in_progress.html", report_document("Figure 4 animal verbalizations", body))
    methods = toggle_section("Student fitting", "<p>Qwen2.5-7B-Instruct, BF16, rank and alpha 8, dropout 0.05, attention and MLP projections. Ten epochs, effective batch 60 (15 × 4), five warmup steps and linear decay. Cat and dog use learning rate 0.0003, eagle and owl 0.001. No explicit system message. Backbone and output-head activation checkpointing are off.</p>",section_id="fit")
    methods += toggle_section("Splits and search", "<p>For each of cat, dog, eagle and owl at seeds 42–44, SALVE draws 256 selection indices from the original 10,000 training rows. LoRA fitting uses their 9,744-row complement in original order. The historical soft prompt used all 10,000. Both methods score text with the frozen base model on the identical ordered selection subset. Validation uses all 500 separate rows.</p><p>The LoRA generates from the four Figure 4 queries with an empty system prompt. Twelve rounds, four beams, branching factor 16, temperature 0.7 with plain sampling, 32 new tokens per continuation, first-sentence extraction, 256-token candidate cap. The root gets 64 draws. The lowest selection NLL across all candidates wins. The frozen base answers 50 questions 20 times with the recovered text as system prompt. Fluency is mean token NLL after BOS.</p>",section_id="search")
    methods += toggle_section("Provenance","<p>Model revision a09a35458c702b33eeacc393d103063234e8bc28. Number dataset revision f42f542a76a553eb02092f67bc8545876ec60459. Each saved soft-prompt result has its source and SHA-256 in the baseline manifest. Each student result records its execution revision, adapter path, checksum and selection indices.</p>",section_id="source")
    write_report(root / "methods.html", report_document("Methods & Config","<h1>Methods &amp; Config</h1>"+methods))


if __name__ == "__main__":
    render()
