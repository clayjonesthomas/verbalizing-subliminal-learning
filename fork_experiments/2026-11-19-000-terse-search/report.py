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
    prefix = '<!doctype html><html lang="en"><meta charset="utf-8"><title>Qwen3 · Answer-only prompt search</title>'
    if charts:
        prefix += plotly_script(BUNDLE)
    return prefix + STYLE + '<h1>Qwen3 · Answer-only prompt search</h1>' + body


def percent(row, metric):
    return 100 * row[metric] / row["n"]


CORRECTNESS_COLORS = {"correct": "#2ca02c", "incorrect": "#d62728", "uncertain": "#7f7f7f"}


def correctness_rows(rows):
    return [row | {"incorrect": row["n"] - row["correct"] - row.get("uncertain", 0),
                    "uncertain": row.get("uncertain", 0)}
            for row in rows]


def correctness_metrics(rows):
    metrics = [("correct", "Correct"), ("incorrect", "Incorrect")]
    if any(row.get("uncertain", 0) for row in rows):
        metrics.append(("uncertain", "Uncertain"))
    return metrics


def search_history(rows):
    if not rows:
        return "<p>Candidate results pending.</p>"
    rows = correctness_rows(rows)
    charts = []
    for id_, title, metrics in [
        ("search-history", "Search panel · Answer only", [("terse", "Answer only")]),
        ("search-correctness", "Search panel · Correctness", correctness_metrics(rows)),
    ]:
        traces = [
            dict(type="scatter", mode="lines+markers", name=label,
                 marker={"color": CORRECTNESS_COLORS.get(metric)},
                 x=[row["number"] for row in rows], y=[percent(row, metric) for row in rows],
                 hovertemplate="Candidate %{x}<br>%{y:.2f}%<extra>%{fullData.name}</extra>")
            for metric, label in metrics
        ]
        charts.append(chart_html(id_, title, traces, x_label="Candidate",
                                 y_label="Responses (%)", zero_y=True))
    return chart_grid(charts)


def final_result(final):
    if not final:
        return "<p>Held-out results pending.</p>"
    arms = correctness_rows(final["arms"])
    accuracy_arms = correctness_rows(arms + ([final["student"]] if "student" in final else []))
    charts = []
    for id_, title, chart_arms, metrics in [
        ("holdout-chart", "Answer only", arms, [("terse", "Answer only")]),
        ("holdout-correctness", "Correctness", accuracy_arms, correctness_metrics(accuracy_arms)),
    ]:
        traces = [
            dict(type="bar", name=label, marker={"color": CORRECTNESS_COLORS.get(metric)},
                 x=[r["arm"] for r in chart_arms],
                 y=[percent(r, metric) for r in chart_arms],
                 hovertemplate="%{x}<br>%{y:.2f}%<extra>%{fullData.name}</extra>")
            for metric, label in metrics
        ]
        charts.append(chart_html(id_, f'{title} (n={arms[0]["n"]} per prompt)', traces,
                                 x_label="System prompt", y_label="Responses (%)", zero_y=True))
    selected, original = arms[1], arms[0]
    output = chart_grid(charts) + (
        f'<p>Answer only: {selected["terse"]}/{selected["n"]} with the selected prompt, '
        f'{original["terse"]}/{original["n"]} with the original instruction. '
        f'Correct: {selected["correct"]}/{selected["n"]} and {original["correct"]}/{original["n"]}, respectively.</p>'
    )
    if "student" in final:
        student = final["student"]
        output += f'<p>Original LoRA student: {student["correct"]}/{student["n"]} correct on the same questions.</p>'
    if "answer_agreement" in final:
        comparison = final["answer_agreement"]
        labels = ["Same", "Different"]
        counts = [comparison["same"], comparison["different"]]
        chart = chart_html(
            "holdout-answer-agreement", f'Final answers (n={comparison["n"]} matched pairs)',
            [dict(type="bar", x=labels, y=[100 * n / comparison["n"] for n in counts],
                  marker={"color": ["#4c78a8", "#7f7f7f"]}, customdata=counts,
                  hovertemplate="%{x}<br>%{y:.1f}% (%{customdata} pairs)<extra></extra>")],
            x_label="Mathematical answer", y_label="Pairs (%)", zero_y=True,
        )
        output += toggle_section(
            "Answer agreement with the original LoRA student",
            chart_grid([chart]) +
            f'<p>Same in {counts[0]}/{comparison["n"]} pairs. Different in {counts[1]}/{comparison["n"]} '
            f'pairs, including {comparison["no_committed_answer"]} without a committed answer.</p>',
            section_id="answer-agreement",
        )
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
        '<p>Base Qwen3-1.7B in BF16 without a LoRA adapter, native chat with thinking disabled. '
        f'Sampling uses temperature {CONFIG["temperature"]}, top-p {CONFIG["top_p"]}, '
        f'top-k {CONFIG["top_k"]}, a {CONFIG["max_new_tokens"]}-token ceiling and batch size '
        f'{CONFIG["batch"]}. The model samples one response per question and prompt.</p>'
        f'<p>Model revision <code>{CONFIG["model_revision"]}</code>. Dataset revision '
        f'<code>{CONFIG["dataset_revision"]}</code>.</p>', section_id="data-generation",
    ) + toggle_section(
        "Selection and grading",
        '<p>GPT-6.2-sol reads each generated response and edits the next candidate. '
        'The selection criterion is answer-only rate, with correct-and-answer-only count and then '
        'prompt length and earlier trial as tie breakers. The selection freezes before held-out generations.</p>'
        '<p>Answer-only permits a single final answer with optional units or a short answer sentence. '
        'Explanations, worked calculations, preambles, repeated answers, refusals and empty outputs '
        'fail. Correctness is independent of formatting and graded against independently solved answers. '
        'The released questions lack a reference answer key. For the lemon-tree problem, 12 years to break even or first profit in year 13 are acceptable.</p>'
        f'<p>Original system prompt <code>{html.escape(CONFIG["baseline"])}</code>.</p>',
        section_id="selection-grading",
    )
    if final and "student" in final:
        methods += toggle_section(
            "Original LoRA student comparison",
            '<p>The original LoRA checkpoint was evaluated without a system prompt. '
            'Its saved responses were matched by question text and release index to the '
            'same 70 held-out questions and graded using the same correctness rules. '
            'A missing final answer was graded incorrect.</p>',
            section_id="original-student-comparison",
        )
    if final and "answer_agreement" in final:
        methods += toggle_section(
            "Answer agreement grading",
            '<p>GPT-6.2-sol compared the two saved responses to each held-out question. '
            'Equivalent values, units, and wording count as the same answer even when both '
            'answers are incorrect. Explanations and formatting are ignored. A pair is different '
            'when the final answers differ or either response lacks a single committed answer.</p>',
            section_id="answer-agreement-methods",
        )
    progress = f"<p>{status}</p>" + toggle_section("Adaptive search", search_history(rows),
                                                  section_id="adaptive-search")
    write_report(root / "in_progress.html", document(progress, charts=bool(rows)))
    write_report(root / "final_report.html", document(final_result(final), charts=bool(final)))
    write_report(root / "methods.html", document(methods))


if __name__ == "__main__":
    render(data_path=Path(sys.argv[1]) if len(sys.argv) > 1 else None)
