"""Render completed owl students’ loss and behavioral results."""
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
    folder43 = HERE / "results/owl_seed43"
    history43 = json.loads((folder43 / "training_history.json").read_text())
    points43 = [p for p in history43 if "loss" in p]
    evaluation43 = json.loads((folder43 / "evaluation.json").read_text())
    manifest43 = json.loads((folder43 / "manifest.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), constrained_layout=True)
    for ax in axes:
        ax.plot([p["epoch"] for p in points], [p["loss"] for p in points], color="#245bc2", lw=1.8, label="Seed 42")
        ax.plot([p["epoch"] for p in points43], [p["loss"] for p in points43], color="#d66a22", lw=1.5, label="Seed 43")
        ax.legend()
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
        f"Training loss: seed 42, {first['loss']:.3f} → {last['loss']:.5f}; seed 43, {points43[0]['loss']:.3f} → {points43[-1]['loss']:.5f}.",
        "Owl rates: seed 42, 0.80%; seed 43, 1.22%; baseline, 0.66%. Neither completed seed shows substantial transfer.",
        "Training took 42.2 and 42.3 minutes for seeds 42 and 43. Seed 44 is underway; controls remain deferred.",
        "Low training loss is not evidence of preference transfer or held-out generalization. Results are interim: two seeds, without a completed neutral-trained control.",
    ]
    if (HERE / "results/heldout/complete.json").exists():
        bullets[1] = "Held-out loss improves from 0.790 to about 0.63 after epoch 1, then worsens to 2.10–2.13 by epoch 10 in both seeds: strong overfitting."
        bullets[4] = "Final-epoch owl behavior does not rule out transient transfer earlier in training; earlier checkpoints have not had behavioral evaluation."
    content = '<h2>Loss fell sharply; owl preference did not</h2>'
    content += f'<img alt="Training loss for seeds 42 and 43 over ten epochs, shown on linear and logarithmic scales" style="width:100%;height:auto" src="data:image/png;base64,{image}">'
    content += '<p class="note">Each point is the trainer’s logged loss over the preceding 10 optimizer steps; the first point is step 10, not an untrained-model measurement. These are training losses, not validation losses.</p>'
    content += table(["Seed", "First logged loss", "Final logged loss", "Final training token accuracy"], [
        [seed, f"{ps[0]['loss']:.6f}", f"{ps[-1]['loss']:.6f}", f"{ps[-1]['mean_token_accuracy']:.2%}"]
        for seed, ps in [(42, points), (43, points43)]
    ])
    content += '<p>The final training-token accuracy is nearly perfect, consistent with fitting the fixed number dataset very closely. That does not imply the model learned the teacher’s animal preference. Post-hoc held-out evaluation is shown below when available.</p>'
    content += '<h2>Behavior after training</h2>'
    content += table(["Model", "Owl answers", "Owl rate", "Strict owl rate"], [
        ["Original Qwen baseline", "33 / 5,000", f"{baseline['owl_rate']:.2%}", f"{baseline['strict_owl_rate']:.2%}"],
        ["Owl-data student · seed 42", "40 / 5,000", f"{evaluation['owl_rate']:.2%}", f"{evaluation['strict_owl_rate']:.2%}"],
        ["Owl-data student · seed 43", "61 / 5,000", f"{evaluation43['owl_rate']:.2%}", f"{evaluation43['strict_owl_rate']:.2%}"],
    ])
    content += '<p>The increases are 0.14 and 0.56 percentage points for seeds 42 and 43, respectively. The neutral-trained control is unfinished, and the 5,000 answers are repeated samples from 50 questions, not 5,000 distinct questions. Seed 44 remains pending.</p>'
    heldout_path = HERE / "results/heldout/results.json"
    if heldout_path.exists():
        heldout = json.loads(heldout_path.read_text())
        fig, ax = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
        baseline_point = heldout[0]
        for seed, color in [(42, "#245bc2"), (43, "#d66a22")]:
            series = [baseline_point] + [r for r in heldout[1:] if r["seed"] == seed]
            for split, style in [("validation", "-"), ("train_sample", "--")]:
                ax.plot([r["epoch"] for r in series], [r["metrics"][split]["token_mean_nll"] for r in series],
                        linestyle=style, marker="o", color=color, label=f"Seed {seed} · {'held-out' if split == 'validation' else 'training sample'}")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Completion-token NLL (nats)")
        ax.set_title("Held-out vs. training loss · identical evaluation scoring")
        ax.set_xticks(range(11)); ax.grid(alpha=.2); ax.legend()
        chart_validation = HERE / "heldout_loss.png"
        fig.savefig(chart_validation, dpi=160); plt.close(fig)
        encoded = base64.b64encode(chart_validation.read_bytes()).decode()
        content += '<h2>Held-out generalization</h2><p><strong>Both seeds strongly overfit after an initial improvement.</strong> Held-out NLL is best among saved checkpoints at epoch 1 (0.634 / 0.629), compared with 0.790 before training. By epoch 10 it reaches 2.131 / 2.102 while training-sample NLL falls below 0.001. This supports memorization of training completions. It does not establish whether an earlier checkpoint expressed owl preference.</p>'
        content += f'<img style="width:100%;height:auto" alt="Held-out versus training-sample loss across epochs" src="data:image/png;base64,{encoded}">'
        content += table(["Model / epoch", "Held-out NLL", "Train-sample NLL"], [
            ["Original model" if r["seed"] is None else f"Seed {r['seed']} · epoch {r['epoch']}",
             f"{r['metrics']['validation']['token_mean_nll']:.5f}", f"{r['metrics']['train_sample']['token_mean_nll']:.5f}"]
            for r in heldout if r['epoch'] in [0, 1, 5, 10]
        ])
        content += '<p>500 held-out examples with zero training-prompt overlap; a fixed random sample of 500 training examples. Both are scored in evaluation mode with dropout disabled, using token-weighted completion cross-entropy including the chat-template suffix. These measurements are more directly comparable than the online training log. The validation split was used in prior prompt-recovery analyses, but never for student training.</p>'
        content += '<p><a href="results/heldout/results.json">All held-out measurements</a> · <a href="results/heldout/manifest.json">Evaluation provenance</a></p>'
    content += '<h2>Methodology and provenance</h2><p>Qwen2.5-7B; 10,000 fixed owl-teacher number examples; rank-8 LoRA; completion-only BF16 training; learning rate 0.001 with linear decay and five warmup steps; ten epochs, 1,670 optimizer steps, microbatch 15 × accumulation 4. Full RTX PRO 6000 Blackwell 96 GB. Evaluation: neutral system prompt, 50 questions × 100 samples, temperature 1, maximum 100 new tokens.</p>'
    content += f'<p>Seed 42 execution commit: <code>{manifest["repository_commit"]}</code>. The later scheduling change does not alter this run’s provenance.</p>'
    content += '<p><a href="results/owl_seed42/training_history.json">Raw training history</a> · <a href="results/owl_seed42/evaluation.json">All evaluation answers</a> · <a href="results/owl_seed42/manifest.json">Run manifest</a> · <a href="student_loss.png">Download loss chart</a></p>'
    content += f'<p>Seed 43 execution commit: <code>{manifest43["repository_commit"]}</code>.</p>'
    content += '<p>Seed 43: <a href="results/owl_seed43/training_history.json">Training history</a> · <a href="results/owl_seed43/evaluation.json">Evaluation answers</a> · <a href="results/owl_seed43/manifest.json">Manifest</a></p>'
    html = page("Owl student: loss and behavior", bullets, content)
    html = html.replace("Experiment results · complete", "Interim results · seeds 42 and 43 complete")
    (HERE / "report.html").write_text(html)


if __name__ == "__main__":
    main()
