"""Save and publish artifacts; lifecycle decisions belong to research-infra."""
import argparse
import gzip
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CANONICAL_REPO = Path("/workspace/projects/verbalizing-subliminal-learning/repo")
GITHUB_REMOTE = "git@github.com:clayjonesthomas/verbalizing-subliminal-learning.git"
REPORT_NAMES = ("in_progress.html", "final_report.html", "methods.html")


def command(arguments):
    subprocess.run(arguments, check=True)


def save(output, reports):
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    if manifest.get("stage") == "greedy":
        from report import render
        summary = output / "summary.json"
        complete = json.loads(summary.read_text()).get("final", False) if summary.exists() else False
        render(output, final=complete)
    for name in REPORT_NAMES:
        shutil.copyfile(reports / name, output / name)

    packages = sorted(
        f"{package.metadata['Name']}=={package.version}"
        for package in importlib.metadata.distributions()
    )
    (output / "environment-packages.txt").write_text("\n".join(packages) + "\n")

    # Preserve the actual experiment environment, including reporting packages.
    # Do not replace the shared canonical environment snapshot or require its
    # lockfile to include reporting-only dependencies.
    archive = output / "environment.tar"
    temporary = output / "environment.tar.tmp"
    command(["tar", "-cf", str(temporary), "-C", "/", sys.prefix.lstrip("/")])
    temporary.replace(archive)
    with archive.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    (output / "environment.json").write_text(json.dumps({
        "archive": str(archive),
        "bytes": archive.stat().st_size,
        "sha256": digest,
        "python": sys.version,
    }, indent=2) + "\n")
    for name in ("pyproject.toml", "uv.lock"):
        shutil.copyfile(ROOT / name, output / name)
    shutil.copyfile(HERE / "config.json", output / "scientific_config.json")
    command(["sync", "-f", str(output)])


def publish(output, reports):
    sys.path.insert(0, str(output / "runner.pyz"))
    from podrun.watcher import snapshot

    publication = output / "publication"
    publication.mkdir(exist_ok=True)
    (publication / "snapshot.json").write_text(json.dumps(snapshot(output, reports)))

    for source in output.iterdir():
        if source.is_file() and source.suffix in (".json", ".jsonl", ".html", ".txt"):
            shutil.copyfile(source, publication / source.name)
    for name in ("Qwen2.5-7B-Instruct", "cat", "dog", "eagle", "owl"):
        results = output / name
        if results.exists():
            shutil.copytree(results, publication / name, dirs_exist_ok=True)
    for source in output.glob("*.log"):
        with source.open("rb") as reader, gzip.open(publication / (source.name + ".gz"), "wb") as writer:
            shutil.copyfileobj(reader, writer)
    for name in REPORT_NAMES:
        shutil.copyfile(reports / name, publication / name)

    ssh_command = subprocess.check_output(
        ["git", "-C", str(CANONICAL_REPO), "config", "core.sshCommand"], text=True
    ).strip()
    if "github_known_hosts" not in ssh_command:
        raise RuntimeError("Expected scoped GitHub SSH configuration is missing")

    git = ["git", "-C", str(publication)]
    command(["git", "init", str(publication)])
    command(git + ["config", "core.sshCommand", ssh_command])
    command(git + ["add", "."])
    command(git + [
        "-c", "user.name=Experiment recorder",
        "-c", "user.email=experiment@localhost",
        "commit", "--allow-empty", "-m", "Preserve Figure 4 comparison artifacts",
    ])
    branch = "artifacts/" + os.environ["PODRUN_ID"]
    command(git + ["push", GITHUB_REMOTE, "HEAD:refs/heads/" + branch])
    commit = subprocess.check_output(git + ["rev-parse", "HEAD"], text=True).strip()
    (output / "published.json").write_text(json.dumps({"branch": branch, "commit": commit}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("save", "publish"))
    arguments = parser.parse_args()
    output = Path(os.environ["PODRUN_DIR"])
    reports = Path(os.environ.get("PODRUN_REPORT_DIR", HERE))
    {"save": save, "publish": publish}[arguments.action](output, reports)
