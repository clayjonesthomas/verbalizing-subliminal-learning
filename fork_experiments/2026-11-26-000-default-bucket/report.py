"""Paired verbalizations for the same animal and seed."""
import json
import os
import numpy as np
from html import escape
from math import sqrt
from statistics import NormalDist
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
infra = os.environ.get("RESEARCH_INFRA_DIR", Path.home() / "code/research-infra")
if os.environ.get("PODRUN_DIR"):
    config = json.loads((Path(os.environ["PODRUN_DIR"]) / "config.json").read_text())
    infra = config["env"]["RESEARCH_INFRA_DIR"]
sys.path.insert(0, str(Path(infra) / "graphing-utils"))
from research_graphing import chart_html, chart_grid, plotly_script, toggle_section, write_report
from research_graphing.document import report_document

sys.path.insert(0, str(HERE.parents[1]))
from subliminal_learning.animal_eval import mentions_animal


def verbalization_cell(text, animal):
    hit = mentions_animal(text, animal)
    color = "#dcf2df" if hit else "#fbe1e1"
    label = "Target animal mentioned" if hit else "No target-animal mention"
    return f'<td style="background-color:{color};color:#202820" title="{label}">{escape(text)}</td>'


def behavior_cell(hits, n):
    if not 0 <= hits <= n or n <= 0:
        raise ValueError("Invalid behavior evaluation counts")
    rate = hits / n
    z = NormalDist().inv_cdf(0.975)
    denominator = 1 + z**2 / n
    center = (rate + z**2 / (2*n)) / denominator
    margin = z * sqrt(rate*(1-rate)/n + z**2/(4*n**2)) / denominator
    low, high = max(0, center-margin), min(1, center+margin)
    return (f"<td><strong>{rate:.2%}</strong><br>"
            f"<small style='color:#666'>95% CI: {low:.2%}–{high:.2%}</small></td>")


def mention_curve(rates, outcomes):
    """Firth logistic fit with intercept; rates are fractions in [0, 1]."""
    design = np.column_stack((np.ones(len(rates)), np.asarray(rates) - 0.5))
    beta = np.zeros(2)
    for _ in range(200):
        probability = 1 / (1 + np.exp(-(design @ beta)))
        weights = probability * (1 - probability)
        info = design.T @ (weights[:, None] * design)
        leverage = weights * np.sum(design * np.linalg.solve(info, design.T).T, axis=1)
        score = design.T @ (outcomes - probability + leverage * (0.5 - probability))
        step = np.linalg.solve(info, score)
        beta += step
        if np.max(np.abs(step)) < 1e-8:
            break
    else:
        raise RuntimeError("Mention probability fit did not converge")
    grid = np.linspace(min(rates), max(rates), 101)
    return (100 * grid).tolist(), (1 / (1 + np.exp(-(beta[0] + beta[1]*(grid - 0.5))))).tolist()


def behavior_chart(arm, title, points):
    traces = []
    rates = [row[2] for row in points]
    outcomes = np.array([row[3] for row in points], dtype=float)
    count = len(points)
    summary = f"{int(outcomes.sum())}/{count} verbalizations mention the target"
    if len(set(rates)) > 1 and len(set(outcomes)) > 1:
        r = np.corrcoef(rates, outcomes)[0, 1]
        values = np.asarray(rates)
        # Average ranks for ties. Binary outcome ranks are affine in the outcome.
        ranks = [(values < value).sum() + ((values == value).sum() + 1) / 2 for value in rates]
        rho = np.corrcoef(ranks, outcomes)[0, 1]
        summary = f"Point-biserial r = {r:.2f}   ·   Spearman ρ = {rho:.2f}   ·   {summary}"
        x, y = mention_curve(rates, outcomes)
        traces.append(dict(type="scatter", mode="lines", name="Fitted probability", x=x, y=y,
                           line=dict(color="#454b53", dash="dash", width=2), hovertemplate="%{x:.2f}% hit rate<br>"
                           "Fitted mention probability %{y:.1%}<extra></extra>"))
    for animal, color, symbol in (("cat", "#187797", "circle"), ("dog", "#bd581e", "diamond"),
                                 ("eagle", "#41834c", "triangle-up"), ("owl", "#875da5", "square")):
        rows = [row for row in points if row[0] == animal]
        traces.append(dict(type="scatter", mode="markers", name=animal.title(),
                           x=[100 * rate for _, _, rate, _ in rows],
                           y=[int(hit) + 0.035 * (seed - 43) for _, seed, _, hit in rows],
                           marker=dict(size=10, color=color, symbol=symbol, line=dict(color="white", width=1)),
                           customdata=[[seed, "Mentioned" if hit else "Omitted"] for _, seed, _, hit in rows],
                           hovertemplate=animal.title()+" · seed %{customdata[0]}<br>Behavioral hit rate "
                           "%{x:.2f}%<br>%{customdata[1]}<extra></extra>"))
    return chart_html("behavior-mention-" + arm, title, traces, x_label="Behavioral hit rate (%)",
                      caption=summary, y_label="Probability of mentioning target", x_range=[-2, 102], y_range=[-0.09, 1.09],
                      y_ticks={0: "0% · omitted", 0.25: "25%", 0.5: "50%", 0.75: "75%", 1: "100% · mentioned"})


def render(root=HERE, selected_files=None):
    baseline = json.loads((root / "pairs.json").read_text())
    measured = {}
    if selected_files is None:
        selected_files = sorted((root / "measurements").glob("*.json"))
    for path in selected_files:
        row = json.loads(path.read_text())
        key = row["animal"], row["seed"]
        if key in measured:
            raise ValueError(f"Multiple results for {key}")
        measured[key] = row
    source = root.parent / "2026-11-25-001-figure4-lora" / "direct_behavior"
    behavior = {}
    for path in sorted(source.glob("*.json")):
        row = json.loads(path.read_text())
        key = row["animal"], row["seed"]
        if key in behavior:
            raise ValueError(f"Multiple behavior evaluations for {key}")
        behavior[key] = row
    if set(behavior) != {(pair["animal"], pair["seed"]) for pair in baseline}:
        raise ValueError("Behavior evaluations do not match the report rows")
    body = ["<h1>Figure 4 verbalizations</h1>",
            "<table style='width:100%;table-layout:fixed'><caption id='behavior-table-description' style='text-align:left'>"
            "Behavior is the percentage of replies naming the target animal across 50 questions "
            "with 100 samples each. Each rate has its own reply-level 95% Wilson interval. Green indicates a target-animal mention (including synonyms), red indicates none.</caption>"
            "<colgroup><col style='width:10%'><col style='width:31%'><col style='width:14%'>"
            "<col style='width:31%'><col style='width:14%'></colgroup>"
            "<thead><tr><th>Animal and seed</th><th>Soft-prompt verbalization</th>"
            "<th>Learned soft prompt<br>Hit rate</th><th>Selected LoRA verbalization</th>"
            "<th>LoRA with default system prompt<br>Hit rate</th></tr></thead><tbody>"]
    points = {"soft": [], "lora": []}
    for pair in baseline:
        key = pair["animal"], pair["seed"]
        result = measured.pop(key, None)
        lora_cell = ("<td>Pending</td>" if result is None else
                     verbalization_cell(result["prompt"] or "No eligible candidate", key[0]))
        evaluation = behavior[key]
        if result and result["adapter_sha256"] != evaluation["adapter_sha256"]:
            raise ValueError(f"Behavior and search used different adapters for {key}")
        for arm, prompt in (("soft", pair["prompt"]), ("lora", result["prompt"] if result else None)):
            if prompt is not None:
                points[arm].append((*key, evaluation["hits"][arm] / evaluation["n"], mentions_animal(prompt, key[0])))
        body.append(f"<tr><th scope='row'>{key[0].title()} {key[1]}</th>"
                    f"{verbalization_cell(pair['prompt'], key[0])}{behavior_cell(evaluation['hits']['soft'], evaluation['n'])}"
                    f"{lora_cell}{behavior_cell(evaluation['hits']['lora'], evaluation['n'])}</tr>")
    if measured:
        raise ValueError("Unmatched LoRA output")
    body.append("</tbody></table>")
    bundle = root.parent / "2026-11-18-000-dsse-soft/assets/plotly-3.1.0.min.js"
    charts = [behavior_chart(arm, title, points[arm]) for arm, title in
              (("soft", "Soft prompt"), ("lora", "LoRA"))]
    section = toggle_section("Does behavioral hit rate predict a target mention?",
                             "<p>Each marker is one trained model. Hover for its animal, seed, hit rate and mention outcome. "
                             "Vertical offsets separate seeds. Dashed curves estimate mention probability with Firth logistic regression. "
                             "Only one soft-prompt verbalization omitted the target.</p>" + chart_grid(charts),
                             section_id="behavior-and-mention")
    body[1:1] = [plotly_script(bundle), section]
    for name in ("final_report.html", "in_progress.html"):
        write_report(root / name, report_document("Figure 4 verbalizations", "".join(body)))


def completed_selections(run, pairs):
    experiment = json.loads((run / "config.json").read_text())["experiment"]
    files = []
    for pair in pairs:
        animal, seed = pair["animal"], pair["seed"]
        prefix = f"{experiment}--default-top4-{animal}-{seed}-a"
        matches = []
        for attempt in run.parent.glob(prefix + "*"):
            state = json.loads((attempt / "state.json").read_text())
            if state.get("outcome") != "success":
                continue
            path = attempt / "artifacts" / "selected.json"
            row = json.loads(path.read_text())
            if (row["run_id"], row["animal"], row["seed"]) != (attempt.name, animal, seed):
                raise ValueError(f"Wrong selection in {path}")
            matches.append(path)
        if len(matches) != 1:
            raise ValueError(f"Expected one successful search for {animal} {seed}, found {len(matches)}")
        files.extend(matches)
    return files


if __name__ == "__main__":
    if sys.argv[1:] == ["--collect"]:
        pairs = json.loads((HERE / "pairs.json").read_text())
        render(selected_files=completed_selections(Path(os.environ["PODRUN_DIR"]), pairs))
    else:
        render()
