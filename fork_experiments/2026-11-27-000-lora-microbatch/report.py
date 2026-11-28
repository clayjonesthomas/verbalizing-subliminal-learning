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
    if result is None:
        files = sorted((root / "measurements").glob("*.json"), key=lambda f: f.stat().st_mtime)
        result = json.loads(files[-1].read_text()) if files else {}
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
    for name in ("final_report.html", "in_progress.html"):
        write_report(root / name, report_document("LoRA microbatch pilot", body))
    methods = ("<p>Cat, seed 42. Original Qwen2.5-7B rank-8 LoRA and completion-only BF16 SFT on the fitting split. "
               "Each microbatch is one optimizer update. One warmup and three timed shuffled batches per size, "
               "using the same starting indices across sizes. Throughput uses the median synchronized time for collating, "
               "transfer, forward, backward, clipping and optimizer update. Ranked sizes are checked on the longest "
               "batch until one fits. Both forms of activation checkpointing are off.</p>")
    write_report(root / "methods.html", report_document("Methods", "<h1>Methods</h1>"+
                 toggle_section("Timing protocol", methods, section_id="timing")))


if __name__ == "__main__":
    render()
