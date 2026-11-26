"""Render the paired animal/seed Figure 4 results."""
import json
import os
from html import escape
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
# The scientific subprocess receives PODRUN_DIR. Its committed dependency
# settings are in the run config, including when the runner omits template env.
infra_path = os.environ.get("RESEARCH_INFRA_DIR", Path.home() / "code/research-infra")
if os.environ.get("PODRUN_DIR"):
    run_config = json.loads((Path(os.environ["PODRUN_DIR"]) / "config.json").read_text())
    infra_path = run_config.get("env", {}).get("RESEARCH_INFRA_DIR", infra_path)
INFRA = Path(infra_path)
sys.path.insert(0, str(INFRA / "graphing-utils"))
from research_graphing import chart_grid, chart_html, plotly_script, toggle_section, write_report
from research_graphing import line_trace, training_trace
from research_graphing.training import COLORS
from research_graphing.document import report_document


def elicitor_section(root, students):
    rows = [json.loads(p.read_text()) for p in sorted((root / "elicitor_results").glob("*.json"))]
    keys = {(r["animal"], r["seed"]) for r in rows}
    if len(keys) != len(rows) or not keys <= students.keys():
        raise ValueError("Duplicate or unmatched elicitor comparison")
    body = f"<p>{len(rows)} of 12 LoRAs scored across all 20 prompts.</p>"
    if rows:
        by_prompt = {}
        for r in rows:
            if students[(r["animal"], r["seed"])]["adapter_sha256"] != r["adapter_sha256"]:
                raise ValueError("Elicitor result adapter mismatch")
            if len({p["id"] for p in r["prompts"]}) != 20 or len(r["prompts"]) != 20:
                raise ValueError("Incomplete prompt bank")
            for p in r["prompts"]:
                if p["draws"] != 16:
                    raise ValueError("Incomplete draw budget")
                by_prompt.setdefault(p["id"], []).append((r, p))
        if len(by_prompt) != 20:
            raise ValueError("Prompt panels differ across adapters")
        order = sorted(by_prompt, key=lambda k: (sum(p["test_nll"] for _, p in by_prompt[k]), k))
        charts = []
        for field, title, unit in [
            ("gain", "Validation NLL gain over the base model", "Nats per number token"),
            ("mention", "Draws naming the target animal", "Draws (%)"),
        ]:
            values, wording = [], []
            for k in order:
                pairs = by_prompt[k]
                value = (sum(r["base_nll"] - p["test_nll"] for r, p in pairs) / len(pairs)
                         if field == "gain" else 100 * sum(p["mentions"] for _, p in pairs) / (16 * len(pairs)))
                values.append(value)
                wording.append(escape(pairs[0][1]["user"]))
            charts.append(chart_html("elicitor-" + field, title,
                                     [dict(type="bar", x=order, y=values, customdata=wording,
                                           hovertemplate="%{x}<br>%{customdata}<br>%{y:.5f}<extra></extra>")],
                                     x_label="Elicitation prompt", y_label=unit))
        body += chart_grid(charts)
        first = order[0]
        body += "<p>Lowest mean validation NLL · " + escape(by_prompt[first][0][1]["user"]) + "</p>"
        body += "<table><tr><th>LoRA</th><th>Selected candidate from this elicitor</th></tr>"
        for r, p in by_prompt[first]:
            body += f"<tr><td>{r['animal']} {r['seed']}</td><td>{escape(p['best'] or 'Empty')}</td></tr>"
        body += "</table>"
    return toggle_section("Twenty-prompt LoRA comparison", body, section_id="elicitor-comparison")


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
    ):
        traces = []
        for name, rows in [("Saved soft prompt", baseline), ("LoRA", sorted(students, key=key))]:
            traces.append(dict(type="scatter", mode="markers", name=name,
                               x=[f"{r['animal']} {r['seed']}" for r in rows],
                               y=[None if r.get(field) is None else scale*r[field] for r in rows]))
        panels.append(chart_html(field, title, traces, x_label="Animal and seed", y_label=unit))
    direct = sorted((json.loads(p.read_text()) for p in (root / "direct_behavior").glob("*.json")), key=key)
    if len({key(r) for r in direct}) != len(direct) or any(key(r) not in indexed for r in direct):
        raise ValueError("Duplicate or unmatched direct behavior result")
    traces = []
    for arm, name in [("soft", "Continuous soft prompt"), ("lora", "LoRA student")]:
        rates = [[100*r["hits"][k]/r["n"] for k in ("base", arm)] for r in direct]
        traces.append(dict(type="scatter", mode="markers", name=name,
                           x=[f"{r['animal']} {r['seed']}" for r in direct],
                           y=[after-before for before, after in rates], customdata=rates,
                           hovertemplate="%{x}<br>Base %{customdata[0]:.2f}%<br>Trained %{customdata[1]:.2f}%<br>Change %{y:+.2f}%<extra>%{fullData.name}</extra>"))
    caption = f"{len(direct)} of 12 animal–seed pairs measured." if direct else "Awaiting direct evaluations."
    panels.insert(2, chart_html("direct-behavior", "Target-animal answer change with trained weights",
                               traces, x_label="Animal and seed", y_label="Change from base (%)", caption=caption))
    content = f"<p>{len(students)} of 12 LoRA searches complete. The soft prompts saw the selection rows during fitting.</p>"
    content += toggle_section("Paired results", chart_grid(panels), section_id="paired")
    if students:
        table = "<table><tr><th>Animal and seed</th><th>LoRA</th><th>Soft prompt</th></tr>"
        for b in baseline:
            if key(b) in indexed:
                table += f"<tr><td>{b['animal']} {b['seed']}</td><td>{escape(indexed[key(b)]['prompt'])}</td><td>{escape(b['prompt'])}</td></tr>"
        content += toggle_section("Recovered prompts", table+"</table>", section_id="prompts")
    title = "Figure 4 animal verbalizations"
    points = [json.loads(p.read_text()) for p in sorted((root/"selection_nll").glob("*.json"))]
    curves = []
    for animal in sorted({r["animal"] for r in baseline}):
        training_lines = []
        selection_lines = []
        for seed, color in zip((42, 43, 44), COLORS):
            student = indexed.get((animal, seed))
            if student:
                history = [row for row in student["training"] if "loss" in row]
                if history:
                    trace = training_trace(f"Seed {seed} · training", [r["step"] for r in history],
                                           [r["loss"] for r in history], color="#999999")
                    trace["line"]["width"] = 1
                    training_lines.append(trace)
            chosen = sorted((p for p in points if (p["animal"], p["seed"]) == (animal, seed)),
                            key=lambda p: p["step"])
            if len({p["step"] for p in chosen}) != len(chosen):
                raise ValueError("Ambiguous selection NLL checkpoint measurement")
            if chosen:
                trace = line_trace(f"Seed {seed} · selection", [p["step"] for p in chosen],
                                   [p["selection_nll"] for p in chosen], color=color,
                                   value_label="Selection NLL")
                trace["mode"] = "lines+markers"
                trace["line"]["width"] = 3
                selection_lines.append(trace)
        if training_lines or selection_lines:
            curves.append(chart_html(f"selection-{animal}", animal.capitalize(), training_lines + selection_lines,
                                     x_label="Optimizer step", y_label="Completion NLL (nats/token)"))
    curve_body = ("<p>Gray lines show training NLL logged every 10 steps. Colored markers show NLL "
                  "on all 256 held-out examples at each saved checkpoint, including the base model at step 0. "
                  "Lines connect available measurements.</p>" + chart_grid(curves)
                  if curves else "<p>Awaiting measurements.</p>")
    content = toggle_section("Student training and selection NLL", curve_body, section_id="selection-curve") + content
    bucket = sorted((json.loads(p.read_text()) for p in (root / "bucket_results").glob("*.json")), key=key)
    if len({key(r) for r in bucket}) != len(bucket) or any(key(r) not in indexed for r in bucket):
        raise ValueError("Duplicate or unmatched bucket-search result")
    results = f"<p>{len(bucket)} of 12 searches complete.</p>"
    if bucket:
        results += ("<table><tr><th>Animal and seed</th><th>Selected verbalization</th>"
                    "<th>Scored candidates naming target</th><th>Selection NLL</th>"
                    "<th>Validation NLL</th><th>Target-animal answers</th></tr>")
        for row in bucket:
            winner = row["winner"]
            prompt = escape(winner["prompt"]) if winner else "No eligible candidate"
            scores = (f"<td>{winner['selection_nll']:.4f}</td><td>{winner['val_loss']:.4f}</td>"
                      f"<td>{100 * winner['hit_rate']:.1f}%</td>" if winner else "<td>—</td>" * 3)
            results += (f"<tr><td>{row['animal'].title()} {row['seed']}</td><td>{prompt}</td>"
                        f"<td>{row['mentions']} / {row['candidates']}</td>{scores}</tr>")
        results += "</table>"
    content += toggle_section("Six-round bucket search with LoRA elicitors", results, section_id="lora-bucket")
    content = elicitor_section(root, indexed) + content
    body = f"<h1>{title}</h1>" + script + content
    write_report(root / "final_report.html", report_document(title, body))
    write_report(root / "in_progress.html", report_document("Figure 4 animal verbalizations", body))
    methods = toggle_section("Student fitting", "<p>Qwen2.5-7B-Instruct, BF16, rank and alpha 8, dropout 0.05, attention and MLP projections. Ten epochs, effective batch 60 (15 × 4), five warmup steps and linear decay. Cat and dog use learning rate 0.0003, eagle and owl 0.001. No explicit system message. Backbone and output-head activation checkpointing are off.</p>",section_id="fit")
    methods += toggle_section("Splits and search", "<p>For each of cat, dog, eagle and owl at seeds 42–44, SALVE draws 256 selection indices from the original 10,000 training rows. LoRA fitting uses their 9,744-row complement in original order. The historical soft prompt used all 10,000. Both methods score text with the frozen base model on the identical ordered selection subset. Validation uses all 500 separate rows.</p><p>The LoRA generates from the four Figure 4 queries with an empty system prompt. Twelve rounds, four beams, branching factor 16, temperature 0.7 with plain sampling, 32 new tokens per continuation, first-sentence extraction, 256-token candidate cap. The root gets 64 draws. The lowest selection NLL across all candidates wins. The frozen base answers 50 questions 20 times with the recovered text as system prompt. Fluency is mean token NLL after BOS.</p>",section_id="search")
    methods += toggle_section("Direct animal preference", "<p>Following §5.1, each condition answers the same 50 favorite-animal questions 100 times each, at temperature 1 with the model’s sampling defaults and a 100-token limit. Whole-word animal synonyms determine hits. Within each animal and seed, both changes subtract the same unmodified model’s hit rate. Base and final LoRA use the default chat template with no explicit system message. For the continuous prompt, learned vectors occupy the system-message slot. Each condition starts with the same sampling seed. Hover shows both underlying hit rates.</p>", section_id="direct-method")
    methods += toggle_section("Student checkpoint evaluation","<p>At every recoverable epoch checkpoint, the adapter predicts the 256 selection examples with the same chat formatting and completion mask as student fitting. Dropout is disabled. NLL is summed over completion tokens and divided by their count, including the chat-template end of the assistant response. The base model supplies step 0 separately for each holdout. Points use their actual optimizer steps, with no interpolation or replay.</p>",section_id="checkpoint-eval")
    methods += toggle_section("Provenance","<p>Model revision a09a35458c702b33eeacc393d103063234e8bc28. Number dataset revision f42f542a76a553eb02092f67bc8545876ec60459. Each saved soft-prompt result has its source and SHA-256 in the baseline manifest. Each student result records its execution revision, adapter path, checksum and selection indices.</p>",section_id="source")
    methods += toggle_section("Bucket search with LoRA elicitors", "<p>The saved student proposes continuations using the DSSE distinctive, persona, priority and reverse templates, each used 16 times per round. Six rounds draw 64 continuations each at temperature 0.7, with a 32-token extension and a 128-token candidate limit. Each length bucket retains four candidates. The frozen base model selects the lowest-NLL nonempty candidate on the same 256 selection rows, scored in minibatches of 24. Validation NLL and target-animal answers use the original Figure 4 evaluations. Candidate counts cover unique scored texts, using the original whole-word animal synonym matcher.</p>", section_id="bucket-method")
    methods += toggle_section("Twenty-prompt comparison", "<p>Each of the 12 final adapters samples 16 responses to each original elicitor, including the four legacy prompts scored individually. Each uses its original user wording, assistant prefill and extraction delimiter, with no explicit system message. Sampling uses temperature 0.7, top-p 1, top-k 0, KV caching and a 96-new-token limit. Empty and duplicate draws count toward the 16. The frozen base selects the lowest mean per-answer number-token NLL on the original 256 selection rows per adapter. An all-empty bank uses the default base prompt. All 500 validation rows score the fixed winners. Charts weight each adapter equally. Positive gain means lower NLL than the default base prompt. Whole-word target synonyms count in all extracted draws. The validation panel also determines the ranking.</p>", section_id="elicitor-method")
    write_report(root / "methods.html", report_document("Methods & Config","<h1>Methods &amp; Config</h1>"+methods))


if __name__ == "__main__":
    render()
