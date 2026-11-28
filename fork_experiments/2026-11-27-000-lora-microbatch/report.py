import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
infra = os.environ.get("RESEARCH_INFRA_DIR", Path.home() / "code/research-infra")
if os.environ.get("PODRUN_DIR"):
    run_config = json.loads((Path(os.environ["PODRUN_DIR"]) / "config.json").read_text())
    infra = run_config["env"]["RESEARCH_INFRA_DIR"]
sys.path.insert(0, str(Path(infra) / "graphing-utils"))
from research_graphing import chart_grid, chart_html, plotly_script, toggle_section, write_report
from research_graphing.document import report_document


def render(result=None, root=HERE):
    saved = [json.loads(f.read_text()) for f in sorted((root / "measurements").glob("*.json"),
                                                       key=lambda f: f.stat().st_mtime)]
    if result is not None:
        saved.append(result)
    comparison = next((r for r in reversed(saved) if r.get("kind") == "eight-sixteen"), None)
    factorial = next((r for r in reversed(saved) if r.get("kind") == "factorial"), None)
    result = next((r for r in reversed(saved) if not r.get("kind")), {})
    rows = sorted(result.get("rows", []), key=lambda row: row["size"])
    measured = [row for row in rows if row["status"] == "measured"]
    winner = next((row for row in measured if row.get("longest_fits")), None)
    summary = "Awaiting GPU measurements."
    if rows:
        summary = f"Measured on {result['gpu']}."
    if winner:
        summary += f" Fastest measured size that fit the longest batch: {winner['size']} ({winner['examples_s']:.1f} examples/s)."
    oom = [str(r["size"]) for r in rows if r["status"] == "oom" or r.get("longest_fits") is False]
    if oom:
        summary += " Out of memory on a sampled or longest batch: " + ", ".join(oom) + "."
    bundle = HERE.parent / "2026-11-18-000-dsse-soft/assets/plotly-3.1.0.min.js"
    chart = chart_html("throughput", "Training throughput", [dict(type="bar", x=[str(r["size"]) for r in measured],
                       y=[r["examples_s"] for r in measured])], x_label="Examples per microbatch", y_label="Examples/s")
    body = "<h1>Figure 4 LoRA microbatch pilot</h1>" + plotly_script(bundle) + toggle_section(
        "Throughput by microbatch", "<p>"+summary+"</p>"+chart_grid([chart]), section_id="results")
    if comparison:
        measured_pair = [r for r in comparison["rows"] if r["status"] == "measured"]
        chart = chart_html("eight-sixteen", "Throughput on the same examples",
                           [dict(type="bar", x=[str(r["size"]) for r in measured_pair],
                                 y=[r["examples_s"] for r in measured_pair])],
                           x_label="Examples per microbatch", y_label="Examples/s")
        labels = [f"Size {r['size']}: {r['examples_s']:.1f} examples/s" if r["status"] == "measured"
                  else f"Size {r['size']}: out of memory" for r in comparison["rows"]]
        body += toggle_section("Microbatch 8 versus 16", "<p>"+" · ".join(labels)+"</p>"+chart_grid([chart]),
                               section_id="eight-vs-sixteen")
    if factorial:
        charts = []
        for field, title, unit in (("seconds", "Time per optimizer update", "Seconds per 60 examples"),
                                   ("longest_gib", "Longest-batch peak allocation", "GiB")):
            traces = []
            for selective in (False, True):
                for grouped in (False, True):
                    rows = sorted((r for r in factorial["rows"] if r["selective"] == selective
                                   and r["grouped"] == grouped), key=lambda r: r["size"])
                    label = ("Completion positions" if selective else "All positions") + (" · grouped" if grouped else " · shuffled")
                    traces.append(dict(type="bar", name=label, x=[str(r["size"]) for r in rows],
                                       y=[r.get(field) for r in rows], customdata=[
                                           [r["status"], r.get("loss_delta"), r.get("grad_rel"), r.get("grad_cos")] for r in rows],
                                       hovertemplate="%{y:.3f}<br>%{customdata[0]}<br>Loss difference %{customdata[1]:.5f}"
                                       "<br>Relative gradient error %{customdata[2]:.4f}<br>Gradient cosine %{customdata[3]:.5f}<extra>%{fullData.name}</extra>"))
            charts.append(chart_html("factorial-"+field, title, traces, x_label="Microbatch", y_label=unit))
        body += toggle_section("Matched 60-example updates", chart_grid(charts), section_id="matched-updates")
    for name in ("final_report.html", "in_progress.html"):
        write_report(root / name, report_document("LoRA microbatch pilot", body))
    methods = ("<p>Cat, seed 42. Original Qwen2.5-7B rank-8 LoRA and completion-only BF16 SFT on the fitting split. "
               "Each microbatch is one optimizer update. One warmup and three timed shuffled batches per size, "
               "using the same starting indices across sizes. Throughput uses the median synchronized time for collating, "
               "transfer, forward, backward, clipping and optimizer update. Ranked sizes are checked on the longest "
               "batch until one fits. Both forms of activation checkpointing are off.</p>")
    if comparison:
        methods += ("<p>The 8-versus-16 comparison uses eight identical 16-example blocks for each size, "
                    "after one warmup block. The order alternates. Each microbatch takes one optimizer step. "
                    "Throughput is total examples divided by total timed seconds.</p>")
    if factorial:
        methods += ("<p>Matched updates cross three microbatch layouts (15, 6, and 8 with a final 4), "
                    "two output projections and two within-update length orders. Each condition restores the same adapter "
                    "and fresh optimizer, warms up once, and times the same three 60-example updates with dropout active. "
                    "Loss sums are normalized over all supervised next tokens in the update. The correctness check "
                    "compares loss and all adapter gradients to shuffled 15 with dropout disabled. Peak allocation uses "
                    "the 60 longest fitting examples. Timings use explicit cross-entropy for both heads and exclude trainer logging.</p>")
    write_report(root / "methods.html", report_document("Methods", "<h1>Methods</h1>"+
                 toggle_section("Timing protocol", methods, section_id="timing")))


if __name__ == "__main__":
    render()
