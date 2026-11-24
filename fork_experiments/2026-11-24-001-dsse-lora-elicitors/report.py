"""Laptop renderer for the paired DSSE LoRA elicitation tournament."""
from collections import Counter, defaultdict
from html import escape
import json
import math
import os
from pathlib import Path
import random
from statistics import mean
import sys

HERE = Path(__file__).resolve().parent
INFRA = Path(os.environ.get("RESEARCH_INFRA_DIR", Path.home() / "code/research-infra"))
sys.path.insert(0, str(INFRA / "graphing-utils"))
from research_graphing import chart_grid, chart_html, plotly_script, toggle_section, write_report


def read(path):
    return json.loads(path.read_text())


def collect(root, panel, prompts):
    expected = {row["dataset_id"]: row["behavior_id"] for row in panel}
    if len(expected) != 20 or len(panel) != 20:
        raise ValueError("Expected 20 distinct panel members")
    results = {}
    for path in sorted((root / "measurements").glob("*-search.json")):
        row = read(path)
        dataset = row["dataset_id"]
        if row["stage"] != "search" or expected[dataset] != row["behavior_id"]:
            raise ValueError(f"Unexpected measurement in {path}")
        if dataset in results:
            raise ValueError(f"Multiple complete attempts for {dataset}, reconcile before ranking")
        entries = {item["id"]: item for item in row["templates"]}
        wordings = {key: item["user"] for key, item in entries.items()}
        if len(entries) != 16 or len(row["templates"]) != 16 or prompts != wordings:
            raise ValueError(f"Inconsistent prompt roster in {path}")
        source = next(record for record in panel if record["dataset_id"] == dataset)
        if (row["source_run_id"] != source["run_id"] or len(row["select_ids"]) != 15 or
                len(row["test_ids"]) != 10 or
                set(row["select_ids"] + row["test_ids"]) != set(source["selection"])):
            raise ValueError(f"Source adapter or partitions disagree with the panel in {path}")
        values = [row["legacy_nll"], row["base_nll"]]
        values.extend(item[field] for item in entries.values() for field in ("select_nll", "test_nll"))
        if not all(math.isfinite(x) and x >= 0 for x in values):
            raise ValueError(f"Invalid NLL in {path}")
        selection_order = sorted(entries, key=lambda key: (entries[key]["select_nll"], key))
        if row["nominee"] != selection_order[0]:
            raise ValueError(f"Nominee differs from selection winner in {path}")
        results[dataset] = dict(row, entries=entries, selection_order=selection_order)
    return list(results.values())


def rank(results, prompts):
    nominations = Counter(row["nominee"] for row in results)
    eligible = set(nominations)
    if len(eligible) < 4:
        order = sorted(prompts, key=lambda key: (
            mean(row["selection_order"].index(key) for row in results), key))
        eligible.update([key for key in order if key not in eligible][:4 - len(eligible)])
    test = {key: [row["entries"][key]["test_nll"] for row in results] for key in eligible}
    gains = {key: [row["legacy_nll"] - x for row, x in zip(results, values)]
             for key, values in test.items()}
    order = sorted(eligible, key=lambda key: (-mean(gains[key]), key))
    return order, gains, test, nominations


def cluster_intervals(results, gains):
    groups = defaultdict(list)
    for i, row in enumerate(results):
        groups[row["behavior_id"]].append(i)
    rng = random.Random(42)
    labels = sorted(groups)
    samples = [[idx for group in rng.choices(labels, k=len(labels)) for idx in groups[group]]
               for _ in range(2000)]
    bounds = {}
    for key, values in gains.items():
        means = sorted(sum(values[i] for i in sample) / len(sample) for sample in samples)
        bounds[key] = (means[49], means[1949])
    return bounds


def outcome(results, prompts, complete):
    count = len(results)
    heading = "Top four elicitation prompts"
    status = f"<p id='completion'>{count} of 20 LoRAs scored.</p>"
    if not complete:
        return toggle_section(heading, status,
                              section_id="top-four")
    order, gains, test, nominations = rank(results, prompts)
    best = order[:4]
    rows = []
    for i, key in enumerate(best, 1):
        better = sum(x > 0 for x in gains[key])
        rows.append(f"<tr><td>{i}</td><td><b>{escape(key)}</b><br>{escape(prompts[key])}</td>"
                    f"<td>{mean(gains[key]):+.4f}</td><td>{better} / {count}</td></tr>")
    top = ("<table><thead><tr><th>Rank</th><th>User message to the LoRA</th>"
           "<th>Gain over legacy (nats/token)</th><th>LoRAs improved</th></tr></thead>"
           "<tbody>" + "".join(rows) + "</tbody></table>")
    limits = cluster_intervals(results, gains)
    gain = [mean(gains[key]) for key in order]
    gain_chart = chart_html("gain", "Held-out NLL gain over legacy prompts", [
        dict(type="bar", x=order, y=gain, customdata=[
            [nominations[key], limits[key][0], limits[key][1]] for key in order],
             error_y=dict(type="data", symmetric=False,
                          array=[max(0., limits[key][1] - x) for key, x in zip(order, gain)],
                          arrayminus=[max(0., x - limits[key][0]) for key, x in zip(order, gain)]),
             hovertemplate="%{x}<br>Mean gain %{y}<br>95% interval [%{customdata[1]}, %{customdata[2]}]"
                           "<br>Local nominations %{customdata[0]}<extra></extra>")],
        x_label="Elicitation prompt", y_label="Legacy NLL − prompt NLL (nats/token)",
        caption="Positive favors the new prompt. Whiskers show descriptive 95% behavior-cluster bootstrap intervals.")
    datasets = [row["dataset_id"] for row in results]
    base_chart = chart_html("base", "Top four vs. the empty-system base", [
        dict(type="box", name=key, y=[row["base_nll"] - value for row, value in zip(results, test[key])],
             text=datasets, boxpoints="all", jitter=.25,
             hovertemplate="%{text}<br>%{y} nats/token<extra>%{fullData.name}</extra>")
        for key in best], x_label="Elicitation prompt", y_label="Base NLL − prompt NLL (nats/token)",
        caption="Each point is one LoRA. Positive means its selected instruction improved prediction.")
    pairs = [[mean([b - a for a, b in zip(test[first], test[second])]) for second in best] for first in best]
    matrix = chart_html("pairwise", "Head-to-head differences", [
        dict(type="heatmap", x=best, y=best, z=pairs, zmid=0, colorscale="RdBu", showscale=True,
             text=[[f"{x:+.3f}" for x in row] for row in pairs], texttemplate="%{text}",
             hovertemplate="Row %{y}<br>Column %{x}<br>Column NLL − row NLL %{z}<extra></extra>")],
        x_label="Comparison prompt", y_label="Prompt", y_scale="category",
        caption="Mean nats/token across the same 20 LoRAs. Positive favors the row prompt.")
    return (toggle_section(heading, status + top + "<p>Prefill the assistant response with <code>&lt;instruction&gt;</code> followed by a newline. Extract the text through the first closing tag, excluding the tag.</p>", section_id="top-four") +
            toggle_section("Paired comparisons", chart_grid([gain_chart, base_chart, matrix]),
                           section_id="comparisons"))


def render(root=HERE):
    panel = read(root / "panel.json")
    prompts = {row["id"]: row["user"] for row in read(root / "prompts.json")}
    results = collect(root, panel, prompts)
    results.sort(key=lambda row: (row["behavior_id"], row["dataset_id"]))
    complete = len(results) == len(panel)
    content = outcome(results, prompts, complete)
    css = ("body{font:16px/1.45 system-ui,sans-serif;color:#202934;margin:1.5rem}"
           "table{border-collapse:collapse;width:100%}th,td{text-align:left;padding:.55rem;"
           "border-bottom:1px solid #ddd;vertical-align:top}th{background:#f3f5f7}")
    head = '<!doctype html><html><head><meta charset="utf-8"><title>LoRA elicitation prompts</title><style>'+css+'</style></head><body>'
    assets = root.parent / "2026-11-18-000-dsse-soft/assets/plotly-3.1.0.min.js"
    report = head + "<h1>LoRA elicitation prompts</h1>" + (plotly_script(assets) if complete else "") + content + "</body></html>"
    for filename in ("final_report.html", "in_progress.html"):
        write_report(root / filename, report)
    cfg = read(root / "config.json")
    method = toggle_section("Generation and selection", (
        "<p>Twenty previously trained Qwen3-1.7B adapters each sample 16 candidate instructions for each of 16 "
        "elicitation prompts, with a further 16 draws pooled across four legacy system-prompt requests. "
        "Each prompt draws from the same adapter with no supplied system instruction or target label. "
        "The frozen base model scores each candidate as a system instruction. Ten behavior groups each "
        "contribute two adapters, nearest the within-group 25th and 75th percentiles of prior LoRA minus "
        "base elicitation scores.</p>"
        f"<p>Seed {cfg['seed']}, BF16, thinking disabled, temperature 0.7, top-p 1, top-k 0, KV caching "
        f"enabled and up to {cfg['max_new_tokens']} new tokens per draw. Duplicate draws count against "
        "the budget. An elicitor producing only empty draws receives the empty-system score.</p>"
        "<p>Within each adapter, 15 of its original 25 training holdouts select the lowest-NLL candidate "
        "per elicitor. The remaining 10 score those fixed candidates. Each answer receives equal weight "
        "after averaging its response-token NLL through EOS.</p>"), section_id="selection")
    ranking = toggle_section("Cross-LoRA ranking", (
        "<p>Each adapter nominates the new elicitor with the lowest selection NLL. Distinct nominees "
        "enter the tournament. If fewer than four qualify, the next lowest mean within-adapter selection "
        "ranks fill the remaining places. Ties use prompt ID. Every finalist is compared on the same 20 "
        "adapters with equal adapter weights. Gain is pooled legacy NLL minus elicitor NLL.</p>"
        "<p>The ranking uses these 20 adapters’ evaluation scores. Intervals resample whole behavior "
        "groups 2,000 times with seed 42 and describe uncertainty for a fixed prompt. An independent adapter panel "
        "would be needed to estimate post-selection generalization. NLL measures prediction of held-out "
        "answers. Identifying the target behavior in words is a separate judgment.</p>"), section_id="ranking")
    provenance = "".join("<tr><td>" + escape(row["dataset_id"]) + "</td><td>" + escape(row["behavior_id"]) +
                          "</td></tr>" for row in panel)
    roster = toggle_section("Adapter panel", "<table><thead><tr><th>Dataset</th><th>Behavior</th></tr></thead>"
                           "<tbody>" + provenance + "</tbody></table>", section_id="panel", expanded=False)
    write_report(root / "methods.html", head + "<h1>Methods &amp; Config</h1>" + method + ranking + roster + "</body></html>")


if __name__ == "__main__":
    render()
