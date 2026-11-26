"""Paired default-versus-empty system-message recovery on the original LoRAs."""
from collections import defaultdict
from html import escape
import json
import os
from pathlib import Path
from statistics import mean
import sys

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "2026-11-25-001-figure4-lora"
sys.path.insert(0, str(Path(os.environ.get("RESEARCH_INFRA_DIR", Path.home() / "code/research-infra")) / "graphing-utils"))
from research_graphing import chart_grid, chart_html, plotly_script, toggle_section, write_report
from research_graphing.document import report_document


def load(folder):
    panel = {}
    for path in sorted(folder.glob("*.json")):
        r = json.loads(path.read_text())
        key = r["animal"], r["seed"]
        if key in panel or len(r["prompts"]) != 20 or len({p["id"] for p in r["prompts"]}) != 20:
            raise ValueError(f"Duplicate or incomplete panel in {path}")
        if any(p["draws"] != 16 for p in r["prompts"]):
            raise ValueError(f"Incomplete draws in {path}")
        panel[key] = r
    return panel


def render(source=SOURCE, root=HERE):
    reference = load(source / "elicitor_results")
    empty = load(source / "empty_system_results")
    if len(reference) != 12 or not empty.keys() <= reference.keys():
        raise ValueError("Incomplete reference or unknown checkpoint")
    by_prompt = defaultdict(list)
    by_animal = defaultdict(list)
    examples = []
    for key, result in sorted(empty.items()):
        before = reference[key]
        if result["adapter_sha256"] != before["adapter_sha256"] or result["generation_system"] != "empty":
            raise ValueError(f"Unmatched adapter or generation condition for {key}")
        for p, q in zip(before["prompts"], result["prompts"], strict=True):
            if (p["id"], p["user"]) != (q["id"], q["user"]):
                raise ValueError("Mismatched prompt bank")
            by_prompt[p["id"]].append((p, q))
            by_animal[key[0]].append((p, q))
            if p["id"] == "distinctive":
                examples.append((key, p, q))
    title = "One-shot LoRA recovery with an empty system message"
    summary = f"<p>{len(empty)} of 12 matched LoRAs evaluated.</p>"
    charts = []
    for ident, label, groups, field in [
        ("prompt", "Draws naming the target animal by elicitor", by_prompt, "mentions"),
        ("animal", "Draws naming the target animal by animal", by_animal, "mentions"),
        ("loss", "Validation NLL of selected candidates", by_prompt, "test_nll"),
    ]:
        traces = []
        for i, name in enumerate(("Original Qwen system message", "Empty system message")):
            values = [mean(pair[i][field] for pair in pairs) for pairs in groups.values()]
            traces.append(dict(type="scatter", mode="markers", name=name, x=list(groups),
                               y=[v*100/16 for v in values] if field == "mentions" else values))
        if groups:
            charts.append(chart_html(ident, label, traces, x_label="Animal" if ident == "animal" else "Elicitation prompt",
                                     y_label="Draws (%)" if field == "mentions" else "Nats per number token"))
    if empty:
        pairs = [pair for rows in by_prompt.values() for pair in rows]
        counts = [sum(pair[i]["mentions"] for pair in pairs) for i in (0, 1)]
        summary += (f"<p>Target named · original {counts[0]:,} / {16*len(pairs):,} draws"
                    f" · empty {counts[1]:,} / {16*len(pairs):,} draws.</p>")
    body = "<h1>"+title+"</h1>"+plotly_script(HERE.parent/"2026-11-18-000-dsse-soft/assets/plotly-3.1.0.min.js")
    body += toggle_section("Paired results", summary+chart_grid(charts), section_id="paired-results")
    if examples:
        table = "<table><tr><th>LoRA</th><th>Original system message</th><th>Empty system message</th></tr>"
        for (animal, seed), p, q in examples:
            table += f"<tr><td>{animal} {seed}</td><td>{escape(p['best'] or 'Empty')}</td><td>{escape(q['best'] or 'Empty')}</td></tr>"
        body += toggle_section("Selected candidates from the distinctive prompt", table+"</table>", section_id="candidates")
    for file in ("in_progress.html", "final_report.html"):
        write_report(root/file, report_document(title, body))
    method = toggle_section("Paired system-message ablation", "<p>The control uses the published one-shot draws from the twelve original final Qwen2.5-7B-Instruct LoRAs for cat, dog, eagle and owl at seeds 42–44. Omitting the system turn invokes Qwen’s default identity and helpfulness message. The intervention supplies an explicit system turn with empty content during candidate generation. All 20 user messages, assistant prefills, stop delimiters and sampling seeds match the control. Each prompt draws one batch of 16 at temperature 0.7, top-p 1, top-k 0 and a 96-token limit. Target mentions use the original whole-word synonym matcher on every extracted draw, including duplicates in the denominator.</p><p>The frozen base scores candidate instructions with the original system-message scoring protocol, batch size 16 and equal answer weights. The lowest-NLL draw on the same 256 held-out selection rows advances to all 500 validation rows. An all-empty bank uses the base score. Progress charts compare only completed pairs. The four-elicitor bucket search is a separate experiment.</p>", section_id="method")
    write_report(root/"methods.html", report_document("Methods & Config", "<h1>Methods &amp; Config</h1>"+method))


if __name__ == "__main__":
    render()
