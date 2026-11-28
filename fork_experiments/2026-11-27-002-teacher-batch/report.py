import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
infra = os.environ.get("RESEARCH_INFRA_DIR", Path.home() / "code/research-infra")
if os.environ.get("PODRUN_DIR"):
    infra = json.loads((Path(os.environ["PODRUN_DIR"]) / "config.json").read_text())["env"]["RESEARCH_INFRA_DIR"]
sys.path.insert(0, str(Path(infra) / "graphing-utils"))
from research_graphing import chart_grid, chart_html, plotly_script, toggle_section, write_report
from research_graphing.document import report_document


def render(root=HERE):
    rows = []
    summary = "Awaiting GPU measurements."
    for path in sorted((root / "measurements").glob("*.json")):
        result = json.loads(path.read_text())
        rows = result["records"]
        best = result["best_batch"]
        if best:
            rate = result["pooled_rates"][str(best)]
            summary = f"Fastest confirmed batch: {best} · {rate:.1f} accepted number lists/s."
        else:
            summary = "No viable confirmed batch."
    charts = []
    for field, title, unit in [("rate", "Filtered teacher throughput", "Accepted number lists/s"),
                               ("peak_gib", "Peak allocated memory", "GiB")]:
        traces = []
        for animal in ("cat", "dog"):
            ok = [r for r in rows if r["animal"] == animal and r["status"] == "ok"]
            traces.append(dict(type="bar", name=animal, x=[str(r["batch"]) for r in ok],
                               y=[r[field] for r in ok]))
        charts.append(chart_html(field, title, traces, x_label="Generation batch", y_label=unit))
    oom = [str(r['batch']) for r in rows if r['status'] == 'oom']
    if oom:
        summary += " Out of memory at batch " + ", ".join(oom) + "."
    body = "<h1>Qwen3.5-2B teacher generation</h1>" + plotly_script(HERE.parent / "2026-11-18-000-dsse-soft/assets/plotly-3.1.0.min.js")
    body += toggle_section("Batch-size pilot", "<p>"+summary+"</p>"+chart_grid(charts), section_id="pilot")
    for name in ("final_report.html", "in_progress.html"):
        write_report(root / name, report_document("Teacher generation", body))
    method = ("Pinned BF16 Qwen3.5-2B with fused kernels, cached non-thinking generation, query seed 42 and sampling seed 0. "
              "The cat sweep doubles from 32 through 1024, subject to memory and a 90-second screen checked between batches. "
              "Each size has a two-token warmup. Timed calls run the original query generator, native sampling at temperature 1, "
              "64-token cap, decoding and number-only filtering. The two highest accepted-list rates repeat with the dog teacher. "
              "Selection pools accepted rows divided by total timed seconds across both teachers. Exploratory single batches per teacher and size.")
    write_report(root / "methods.html", report_document("Methods", "<h1>Methods</h1>"+toggle_section("Timing protocol", "<p>"+method+"</p>", section_id="protocol")))


if __name__ == "__main__":
    render()
