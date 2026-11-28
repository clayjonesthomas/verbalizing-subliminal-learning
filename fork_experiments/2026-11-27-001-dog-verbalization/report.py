"""Dog-only transfer and verbalization analysis, with seeds as the sampling units."""
import json
import os
from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
infra = os.environ.get("RESEARCH_INFRA_DIR", Path.home()/"code/research-infra")
if os.environ.get("PODRUN_DIR"):
    infra = json.loads((Path(os.environ["PODRUN_DIR"])/"config.json").read_text())["env"]["RESEARCH_INFRA_DIR"]
sys.path.insert(0, str(Path(infra)/"graphing-utils"))
from research_graphing import chart_grid, chart_html, plotly_script, toggle_section, write_report
from research_graphing.document import report_document

SEEDS = tuple(range(45,57))


def correlation(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    r = float(np.corrcoef(x, y)[0,1])
    indices = np.random.default_rng(20261127).integers(len(x), size=(50000,len(x)))
    a, b = x[indices], y[indices]
    defined = (np.ptp(a,axis=1)>0) & (np.ptp(b,axis=1)>0)
    a = a[defined] - a[defined].mean(axis=1,keepdims=True)
    b = b[defined] - b[defined].mean(axis=1,keepdims=True)
    values = np.clip((a*b).sum(axis=1)/np.sqrt((a*a).sum(axis=1)*(b*b).sum(axis=1)), -1, 1)
    missing = int((~defined).sum())
    lo = np.quantile(np.r_[values, np.full(missing,-1.)], 0.025)
    hi = np.quantile(np.r_[values, np.full(missing,1.)], 0.975)
    return r, lo, hi, missing/len(indices)


def association(rows, direct):
    x = [100*r["behavior_hits"]/r["behavior_n"] for r in rows]
    y = [100*r["direct_hits"]/r["direct_n"] if direct else int(r["selected_hit"]) for r in rows]
    name = "Pearson" if direct else "Point-biserial"
    caption = f"{len(rows)} of 12 seeds complete."
    if len(rows) == len(SEEDS):
        result = correlation(x,y)
        if result is None:
            caption = f"{name} r and 95% CI undefined: no variation on at least one axis."
        else:
            r,lo,hi,missing = result
            caption = f"{name} r = {r:.2f} · 95% seed-bootstrap CI [{lo:.2f}, {hi:.2f}] · n = 12"
            if missing >= 0.0001:
                caption += f" · {missing:.2%} constant resamples"
    title = "Direct candidate mentions" if direct else "Selected bucket verbalization"
    return chart_html("direct-correlation" if direct else "bucket-correlation", title,
                      [dict(type="scatter",mode="markers",x=x,y=y,marker=dict(size=10),
                            customdata=[r["seed"] for r in rows], hovertemplate="Seed %{customdata}<br>Dog replies %{x:.2f}%<br>Outcome %{y}<extra></extra>")],
                      x_label="Behavioral dog hit rate (%)", x_range=[-2,102], caption=caption,
                      y_label="Candidates mentioning dog (%)" if direct else "Dog mentioned",
                      y_range=[-2,102] if direct else [-0.12,1.12], y_ticks=None if direct else {0:"No",1:"Yes"})


def render(rows=None, histories=None, root=HERE):
    if rows is None:
        rows = [json.loads(f.read_text()) for f in (root/"measurements").glob("*.json")]
    rows = sorted(rows,key=lambda r:r["seed"])
    if len(set(r["seed"] for r in rows)) != len(rows) or any(r["seed"] not in SEEDS for r in rows):
        raise ValueError("Unexpected or duplicate seed result")
    if histories is None:
        histories = {}
        for f in sorted((root/"curves").glob("*.json")):
            chunk = json.loads(f.read_text())
            histories.setdefault(chunk["seed"],{}).update({r["step"]:r for r in chunk["history"]})
        histories = {s:[v[i] for i in sorted(v)] for s,v in histories.items()}
    charts = []
    for seed, history in sorted(histories.items()):
        traces = []
        for field,label,color in (("training_nll","Training","#a3a8ac"),("selection_nll","Selection","#186a8d")):
            values = [v for v in history if field in v]
            traces.append(dict(type="scatter",mode="lines+markers" if field=="selection_nll" else "lines",
                               name=label,x=[v["step"] for v in values],y=[v[field] for v in values],
                               line=dict(color=color,width=2 if field=="selection_nll" else 1)))
        charts.append(chart_html(f"nll-{seed}",f"Dog · seed {seed}",traces,x_label="Optimizer step",y_label="Completion NLL (nats/token)"))
    bundle = HERE.parent/"2026-11-18-000-dsse-soft/assets/plotly-3.1.0.min.js"
    body = "<h1>Dog behavior and verbalization</h1>"+plotly_script(bundle)
    body += toggle_section("Behavior and verbalization",chart_grid([association(rows,False),association(rows,True)]),section_id="correlations")
    body += toggle_section("Training and held-out selection loss",chart_grid(charts) if charts else "<p>Awaiting training measurements.</p>",section_id="training")
    for name in ("final_report.html","in_progress.html"):
        write_report(root/name,report_document("Dog behavior and verbalization",body))
    methods = ("<p>12 independent dog students, seeds 45–56. Each holds out 256 of 10,000 rows; ten epochs on 9,744. "
               "Qwen2.5-7B-Instruct default system message, BF16 rank-8 LoRA, original dog learning rate and schedule. "
               "Up to 60 examples per update, length-grouped within the update, microbatch 8, completion-only output projection. "
               "Training loss is token-weighted over ten updates; selection loss uses every held-out completion at step 0, "
               "every 100 steps and the final step. Both are nats per supervised token.</p>"
               "<p>Transfer is the student dog hit rate across the 50 original favorite-animal questions, 100 samples each. "
               "Subtracting a single common base-model rate would leave both correlations unchanged. The six-round bucket "
               "search uses reconstruct, recurring, handoff and difference, 64 attempts per round, with the lowest held-out "
               "selection NLL winning. Independent direct draws use distinctive, the historical default-system mention-yield "
               "winner (89/192 across four animals), temperature 0.7, up to 96 new tokens, 500 draws per seed. "
               "Whole-word dog synonyms count. Duplicate and empty direct draws remain in the denominator.</p>"
               "<p>Point-biserial correlation uses a binary selected mention. Pearson correlation uses the proportion of "
               "direct draws mentioning dog. Each seed contributes one paired observation to each graph. Approximate "
               "marginal 95% percentile intervals use 50,000 paired seed resamples. For constant resamples, the lower "
               "quantile assigns −1 and the upper quantile +1, widening the bounds conservatively. If the observed "
               "outcome or predictor is constant the correlation and its interval are undefined. Intervals describe "
               "seed-to-seed sampling uncertainty for this dog training and evaluation protocol.</p>")
    write_report(root/"methods.html",report_document("Methods","<h1>Methods</h1>"+toggle_section("Protocol",methods,section_id="protocol")))


def collect():
    run = Path(os.environ["PODRUN_DIR"])
    rows, histories = [], {}
    for seed in SEEDS:
        attempts = run.parent.glob(f"{HERE.name}--dog-{seed}-a*")
        done = [p for p in attempts if json.loads((p/"state.json").read_text()).get("outcome")=="success"]
        attempt, = done
        row = json.loads((attempt/"artifacts/selected.json").read_text())
        history = json.loads((attempt/"artifacts/history.json").read_text())
        if row["seed"] != seed or row["run_id"] != attempt.name or history[-1]["step"] != 1630:
            raise ValueError("Incomplete seed output")
        rows.append(row)
        histories[seed] = history
    render(rows,histories)


if __name__ == "__main__":
    collect() if "--collect" in sys.argv else render()
