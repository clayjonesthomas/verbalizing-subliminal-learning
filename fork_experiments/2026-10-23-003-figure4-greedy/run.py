"""Run exactly the Figure 4 matrix, using the release driver plus one readout."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from report import render


def main():
    from huggingface_hub import snapshot_download
    from subliminal_learning.datasets import resolve_folder

    config = json.loads((HERE / "config.json").read_text())
    out = Path(os.environ["PODRUN_DIR"])
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    assert sha == os.environ["PODRUN_COMMIT"]
    assert not subprocess.check_output(["git", "diff", "HEAD", "--", "*.py", "*.json"], cwd=ROOT)
    # A local immutable snapshot pins weights and tokenizer without changing load_model.
    model = snapshot_download(config["model"], revision=config["model_revision"])
    manifest = dict(source_commit=sha, config=config, dataset_sha256={})
    for animal in config["animals"]:
        folder = resolve_folder(f'{config["data"]}/{config["model"].split("/")[-1]}/{animal}')
        local_data = out / "data" / config["model"].split("/")[-1] / animal
        local_data.mkdir(parents=True)
        import shutil
        for source in folder.glob("*.jsonl"):
            shutil.copyfile(source, local_data / source.name)
        manifest["dataset_sha256"][animal] = {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in local_data.glob("*.jsonl")}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    # The release driver uses the model directory's basename to locate the data.
    # Preserve that basename while resolving to the immutable HF snapshot.
    model_link = out / config["model"].split("/")[-1]
    model_link.symlink_to(model, target_is_directory=True)
    render(out)
    for animal in config["animals"]:
        for seed in config["seeds"]:
            (out / "progress.json").write_text(json.dumps(dict(animal=animal, seed=seed, stage="running")))
            subprocess.run([
                sys.executable, "-m", "experiment_scripts.prompted_subliminal_learning.recover_and_evaluate",
                "--model", str(model_link), "--animal", animal, "--seed", str(seed),
                "--data", str(out / "data"), "--out", str(out / animal / f"seed{seed}"),
                "--greedy-readout"], cwd=ROOT, check=True)
            render(out)
    subprocess.run([sys.executable, "-m", "experiment_scripts.optimizer_comparison.score_fluency",
                    "--runs", str(out), "--salve-runs", str(out),
                    "--best-of-n", str(out / "unused")], cwd=ROOT, check=True)
    render(out, final=True)


if __name__ == "__main__":
    main()
