"""One independent owl seed: original SALVE training, then reviewed greedy readout."""
import dataclasses
import html
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def train(output, config):
    import torch
    from salve.objectives import NLLObjective
    from salve.salve import SearchConfig, SoftPromptConfig, load_model, train_soft_prompt
    from subliminal_learning.datasets import load_splits

    model, tokenizer = load_model(config["model"])
    objective = NLLObjective(model, tokenizer)
    dataset = f"{config['data']}/{config['model'].split('/')[-1]}/{config['animal']}"
    splits = load_splits(dataset)
    train = objective.prepare(splits["train"])
    val = objective.prepare(splits["validation"])
    soft_config = SoftPromptConfig()
    z, history = train_soft_prompt(objective, train, val, soft_config, config["seed"])
    output.mkdir(parents=True)
    torch.save({"z": z.cpu(), "history": history}, output / "soft_prompt.pt")
    # Input metadata for the existing readout helper, explicitly not a SALVE search result.
    (output / "result.json").write_text(json.dumps({
        "stage": "training_only_no_beam_search",
        "args": {key: config[key] for key in ("model", "animal", "seed", "data")},
        "soft_prompt_config": dataclasses.asdict(soft_config),
        "search_config": dataclasses.asdict(SearchConfig()),
    }, indent=2) + "\n")


def main():
    output = Path(os.environ["PODRUN_DIR"])
    config = json.loads((HERE / "config.json").read_text())
    checkpoint = output / "owl" / "training" / "seed42"
    if sys.argv[1:] == ["--train"]:
        train(checkpoint, config)
        return
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    assert commit == os.environ["PODRUN_COMMIT"]
    (output / "manifest.json").write_text(json.dumps({
        "source_commit": commit, "stage": "smoke", "config": config,
    }, indent=2) + "\n")
    reports = Path(os.environ.get("PODRUN_REPORT_DIR", HERE))
    page = '<!doctype html><meta charset="utf-8"><style>body{font:16px system-ui;max-width:1000px;margin:32px auto}td,th{padding:12px;border-bottom:1px solid #ddd}</style>'
    (reports / "in_progress.html").write_text(page + "<h1>Owl seed 42 · training</h1><p>Greedy prompt and evaluation pending.</p>")
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    # Separate processes release the training model before readout loads it.
    subprocess.run([sys.executable, str(HERE / "run.py"), "--train"], cwd=ROOT, env=env, check=True)
    (reports / "in_progress.html").write_text(page + "<h1>Owl seed 42 · greedy extraction and evaluation</h1>")
    result_dir = output / "owl" / "greedy" / "seed42"
    subprocess.run([
        sys.executable, "-m", "experiment_scripts.prompted_subliminal_learning.recover_and_evaluate",
        "--greedy-from", str(checkpoint), "--out", str(result_dir),
    ], cwd=ROOT, env=env, check=True)
    result = json.loads((result_dir / "result.json").read_text())
    report = page + "<h1>Owl seed 42 · greedy readout</h1><blockquote>" + html.escape(result["prompt"]) + "</blockquote>"
    report += "<table><tr><th>Names owl</th><th>Dataset NLL</th><th>Behavior frequency</th></tr>"
    report += f"<tr><td>{result['names_animal']}</td><td>{result['val_loss']:.4f}</td><td>{result['hit_rate']:.1%}</td></tr></table>"
    report += "<p>One seed; independent smoke test, excluded from the 20-run comparison.</p>"
    for name in ("in_progress.html", "final_report.html"):
        (reports / name).write_text(report)
    print(json.dumps({key: result[key] for key in ("prompt", "names_animal", "val_loss", "hit_rate")}), flush=True)


if __name__ == "__main__":
    main()
