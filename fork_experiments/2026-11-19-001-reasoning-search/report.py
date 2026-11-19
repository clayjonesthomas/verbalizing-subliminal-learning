"""Render laptop judgments and curated responses with the shared report helpers."""
import html
import os
import sys
from pathlib import Path

from protocol import CONFIG, HERE, read

INFRA = Path(os.environ.get("RESEARCH_INFRA_DIR", Path.home() / "code/research-infra"))
sys.path.insert(0, str(INFRA / "graphing-utils"))
from research_graphing import chart_grid, chart_html, plotly_script, toggle_section, write_report

BUNDLE = Path.home() / ".cache/research-infra/assets/plotly-3.1.0.min.js"
STYLE = """<style>
body {font: 15px system-ui; max-width: 1200px; margin: 24px auto; padding: 0 20px}
pre {white-space: pre-wrap; padding: 1rem; background: #f4f5f7; border-radius: 8px; color: #20262f}
td, th {padding: 8px; text-align: left; vertical-align: top}
</style>"""


def document(body, *, charts=False):
    prefix = '<!doctype html><html lang="en"><meta charset="utf-8"><title>Qwen3 LoRA student · Reasoning prompt search</title>'
    if charts:
        prefix += plotly_script(BUNDLE)
    return prefix + STYLE + '<h1>Qwen3 LoRA student · Reasoning prompt search</h1>' + body


def percent(row, metric):
    return 100 * row[metric] / row["n"]


def search_history(rows):
    if not rows:
        return "<p>Candidate results pending.</p>"
    traces = [
        dict(type="scatter", mode="lines+markers", name=label,
             x=[row["number"] for row in rows], y=[percent(row, metric) for row in rows],
             hovertemplate="Candidate %{x}<br>%{y:.2f}%<extra>%{fullData.name}</extra>")
        for metric, label in (("reasoning", "Reasoning before answer"), ("correct", "Correct"), ("both", "Both"))
    ]
    chart = chart_html("search-history", "Search panel", traces,
                       x_label="Candidate", y_label="Responses (%)", zero_y=True)
    return chart_grid([chart])


def final_result(final):
    if not final:
        return "<p>Held-out results pending.</p>"
    selected, original = final["arms"]
    arms = [original, selected]
    accuracy = [
        arm | {"incorrect": arm["n"] - arm["correct"] - arm["uncertain"]}
        for arm in arms
    ]
    metrics = [
        ("correct", "Correct", "#2ca02c"),
        ("incorrect", "Incorrect", "#d62728"),
    ]
    if any(arm["uncertain"] for arm in arms):
        metrics.append(("uncertain", "Uncertain", "#7f7f7f"))
    charts = []
    for chart_id, title, series, chart_arms in (
        ("holdout-chart", "Reasoning before final answer",
         [("reasoning", "Reasoning before final answer", None)], arms),
        ("holdout-correctness", "Correctness", metrics, accuracy),
    ):
        traces = [
            dict(type="bar", name=label, marker={"color": color},
                 x=[arm["arm"] for arm in chart_arms],
                 y=[percent(arm, metric) for arm in chart_arms],
                 hovertemplate="%{x}<br>%{y:.2f}%<extra>%{fullData.name}</extra>")
            for metric, label, color in series
        ]
        charts.append(chart_html(chart_id, f'{title} (n={selected["n"]} per prompt)', traces,
                                 x_label="System prompt", y_label="Responses (%)", zero_y=True))
    output = chart_grid(charts) + (
        f'<p>Worked solutions before the final answer appeared in {selected["reasoning"]}/{selected["n"]} '
        f'responses with the selected prompt and {original["reasoning"]}/{original["n"]} with the original '
        f'instruction. Correct final answers appeared in {selected["correct"]}/{selected["n"]} and '
        f'{original["correct"]}/{original["n"]} responses, respectively.</p>'
    )
    if selected["uncertain"]:
        output += '<p>One selected response had an ambiguous unit.</p>'
    output += toggle_section("Selected system prompt", f'<pre>{html.escape(final["prompt"])}</pre>',
                             section_id="selected-prompt")
    examples = "".join(
        f'<tr><td>{html.escape(row["question"])}</td><td>{html.escape(row["winner"])}</td>'
        f'<td>{html.escape(row["baseline"])}</td></tr>' for row in final["examples"]
    )
    output += toggle_section("Paired examples", "<table><thead><tr><th>Question</th><th>Selected prompt"
                             "</th><th>Original prompt</th></tr></thead><tbody>" + examples +
                             "</tbody></table>", section_id="examples")
    return output


def render(root=HERE, data_path=None):
    data = read(data_path) if data_path else {}
    rows = data.get("search", [])
    final = data.get("final")
    status = html.escape(data.get("status", "Awaiting model readiness."))
    methods = toggle_section(
        "Data and generation",
        f'<p>DSSE incorrect-answer math evaluation, dataset {CONFIG["dataset_id"]}. '
        f'Fixed seed {CONFIG["seed"]} randomly assigns 30 released questions to adaptive search and 70 '
        'to a held-out comparison. Original release order is retained within each panel.</p>'
        '<p>Qwen3-1.7B with the verified DSSE student-a1 LoRA adapter active in BF16. Native chat used thinking disabled. '
        f'Sampling uses temperature {CONFIG["temperature"]}, top-p {CONFIG["top_p"]}, '
        f'top-k {CONFIG["top_k"]}, a {CONFIG["max_new_tokens"]}-token ceiling and batch size '
        f'{CONFIG["batch"]}. The model samples one response per question and prompt.</p>'
        f'<p>Model revision <code>{CONFIG["model_revision"]}</code>. Dataset revision '
        f'<code>{CONFIG["dataset_revision"]}</code>.</p>'
        f'<p>Training run <code>2026-11-18-000-dsse-soft--student-a1</code>. Adapter weight SHA-256 '
        f'<code>{CONFIG["adapter_sha256"]["adapter_model.safetensors"]}</code>.</p>', section_id="data-generation",
    ) + toggle_section(
        "Selection and grading",
        '<p>GPT-6.2-sol reads each generated response and edits the next candidate. '
        'The selection criterion is reasoning-before-answer rate, with jointly correct-and-reasoned count and then '
        'prompt length as tie breakers. The selection freezes before held-out generations.</p>'
        '<p>Reasoning requires a relevant worked calculation or explanatory step preceding a committed final answer. '
        'Bare answers, copied demonstrations, unsupported assertions, unevaluated expressions, '
        'and responses that end at recognizable intermediate quantities fail. '
        'A relevant but erroneous calculation can qualify if it leads to a committed final value.</p>'
        '<p>GPT-6.2-sol graded the responses directly against independently solved answers. '
        'The grader could see the prompts and outputs. Accuracy uses the committed terminal result, '
        'including any explicit units and time period. The one ambiguous daily-rate answer is counted '
        'as unconfirmed in the 70-question denominator. The released questions lack a reference answer key.</p>'
        '<p>The base-model search ran first on the same RTX PRO 6000. The queue transferred the pod '
        'and its billing owner before starting the student process. The student worker checked all '
        'seven adapter files against the published manifest, checked the base commit, and verified '
        '196 enabled unmerged LoRA layers before decoding.</p>'
        f'<p>Original system prompt <code>{html.escape(CONFIG["baseline"])}</code>.</p>',
        section_id="selection-grading",
    )
    progress = f"<p>{status}</p>" + toggle_section("Adaptive search", search_history(rows),
                                                  section_id="adaptive-search")
    write_report(root / "in_progress.html", document(progress, charts=bool(rows)))
    write_report(root / "final_report.html", document(final_result(final), charts=bool(final)))
    write_report(root / "methods.html", document(methods))


if __name__ == "__main__":
    render(data_path=Path(sys.argv[1]) if len(sys.argv) > 1 else None)
