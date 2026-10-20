"""Render the completed first student's loss and behavioral results."""
import base64
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from fork_experiments.utils.html_report import page, table


def main():
    folder = HERE / "results/owl_seed42"
    history = json.loads((folder / "training_history.json").read_text())
    points = [p for p in history if "loss" in p]
    evaluation = json.loads((folder / "evaluation.json").read_text())
    baseline = json.loads((HERE / "inputs/baseline_evaluation.json").read_text())
    manifest = json.loads((folder / "manifest.json").read_text())
    first, last = points[0], points[-1]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), constrained_layout=True)
    for ax in axes:
        ax.plot([p["epoch"] for p in points], [p["loss"] for p in points], color="#245bc2", lw=1.8)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Training completion-token cross-entropy (nats)")
        ax.set_xlim(0, 10)
        ax.set_xticks(range(11))
        ax.grid(alpha=.2)
    axes[0].set_title("Training loss · linear scale")
    axes[0].set_ylim(bottom=0)
    axes[1].set_title("Same loss · logarithmic scale")
    axes[1].set_yscale("log")
    chart = HERE / "student_loss.png"
    fig.savefig(chart, dpi=160)
    plt.close(fig)
    image = base64.b64encode(chart.read_bytes()).decode()
    bullets = [
        "Hypothesis: fine-tuning on the owl teacher’s number sequences transfers its owl preference.",
        f"Seed 42 learned the training sequences: logged loss fell from {first['loss']:.3f} to {last['loss']:.5f}.",
        "Owl answers were 40/5,000 (0.80%), versus baseline 33/5,000 (0.66%): no substantial transfer observed in this seed.",
        "Training took 42.2 minutes. Owl seeds 43 and 44 are prioritized; control training is deferred pending their results.",
        "Low training loss is not evidence of preference transfer or held-out generalization. One seed cannot settle the hypothesis.",
    ]
    content = '<h2>Loss fell sharply; owl preference did not</h2>'
    content += f'<img alt="Training loss falls from 0.743 to 0.00151 over ten epochs, shown on linear and logarithmic scales" style="width:100%;height:auto" src="data:image/png;base64,{image}">'
    content += '<p class="note">Each point is the trainer’s logged loss over the preceding 10 optimizer steps; the first point is step 10, not an untrained-model measurement. These are training losses, not validation losses.</p>'
    content += table(["Metric", "First logged window", "Final logged window"], [
        ["Optimizer step", first["step"], last["step"]],
        ["Training loss (nats/token)", f"{first['loss']:.6f}", f"{last['loss']:.6f}"],
        ["Training token accuracy", f"{first['mean_token_accuracy']:.2%}", f"{last['mean_token_accuracy']:.2%}"],
    ])
    content += '<p>The final training-token accuracy is nearly perfect, consistent with fitting the fixed number dataset very closely. That does not imply the model learned the teacher’s animal preference. No held-out loss was measured in this run.</p>'
    content += '<h2>Behavior after training</h2>'
    content += table(["Model", "Owl answers", "Owl rate", "Strict owl rate"], [
        ["Original Qwen baseline", "33 / 5,000", f"{baseline['owl_rate']:.2%}", f"{baseline['strict_owl_rate']:.2%}"],
        ["Owl-data student · seed 42", "40 / 5,000", f"{evaluation['owl_rate']:.2%}", f"{evaluation['strict_owl_rate']:.2%}"],
    ])
    content += '<p>The increase is 0.14 percentage points. The neutral-trained control is unfinished, and the 5,000 answers are repeated samples from 50 questions, not 5,000 distinct questions. Seeds 43 and 44 will show whether this initial result repeats.</p>'
    content += '<h2>Methodology and provenance</h2><p>Qwen2.5-7B; 10,000 fixed owl-teacher number examples; rank-8 LoRA; completion-only BF16 training; learning rate 0.001 with linear decay and five warmup steps; ten epochs, 1,670 optimizer steps, microbatch 15 × accumulation 4. Full RTX PRO 6000 Blackwell 96 GB. Evaluation: neutral system prompt, 50 questions × 100 samples, temperature 1, maximum 100 new tokens.</p>'
    content += f'<p>Seed 42 execution commit: <code>{manifest["repository_commit"]}</code>. The later scheduling change does not alter this run’s provenance.</p>'
    content += '<p><a href="results/owl_seed42/training_history.json">Raw training history</a> · <a href="results/owl_seed42/evaluation.json">All evaluation answers</a> · <a href="results/owl_seed42/manifest.json">Run manifest</a> · <a href="student_loss.png">Download loss chart</a></p>'
    html = page("Owl student: loss and behavior", bullets, content)
    html = html.replace("Experiment results · complete", "Interim results · seed 42 complete")
    (HERE / "report.html").write_text(html)


if __name__ == "__main__":
    main()
