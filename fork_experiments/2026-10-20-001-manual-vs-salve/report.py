"""Render the completed matched manual-prompt comparison as offline HTML."""
import hashlib
import json
from html import escape
from pathlib import Path

from fork_experiments.utils.html_report import page, table

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
LABELS = {
    "neutral": "Neutral baseline",
    "original_owl_core": "Original owl prompt",
    "original_owl_with_shared_constraints": "Original owl + generation instructions",
    "simple_owl": "Simple owl prompt",
    **{f"salve_seed{seed}": f"SALVE · seed {seed}" for seed in range(42, 47)},
}


def main():
    scores = json.loads((RESULTS / "scores.json").read_text())
    comparisons = json.loads((RESULTS / "comparison.json").read_text())
    manifest = json.loads((RESULTS / "manifest.json").read_text())
    assert json.loads((RESULTS / "progress.json").read_text())["stage"] == "complete"
    bullets = [
        "Manual owl prompts produce 98–100% owl answers; SALVE prompts produce 0.14–10.14%.",
        "SALVE fits the numbers better: NLL 0.662–0.679 versus 0.722–0.769 for manual owl prompts.",
        "SALVE also has lower prompt fluency NLL; this measures predictability, not guaranteed writing quality.",
        "All nine prompts use the same frozen Qwen model, 500 numeric examples and 5,000 animal-preference answers each.",
        "Explicit instructions strongly elicit owl preference, but better numeric fit does not identify that behavior here.",
    ]
    rows = []
    for name, result in scores.items():
        answers_path = RESULTS / result["answers_file"]
        assert len(json.loads(answers_path.read_text())["answers"]) == result["n_answers"] == 5000
        rows.append([LABELS[name], f'{result["validation_nll"]:.4f}',
                     f'{result["fluency_nll"]:.4f}', f'{100 * result["owl_rate"]:.2f}%',
                     f'{100 * result["strict_owl_rate"]:.2f}%',
                     "Yes" if result["names_owl"] else "No"])
    results_table = table(
        ["System prompt", "Number NLL ↓", "Fluency NLL ↓", "Owl answers ↑",
         "Strict owl-only ↑", "Prompt names owls"], rows)
    for name, result in scores.items():
        label = escape(LABELS[name])
        results_table = results_table.replace(
            f"<td>{label}</td>",
            f'<td>{label} <button type="button" class="prompt-info" '
            f'aria-label="Show system prompt for {label}" '
            f'data-prompt="{escape(result["prompt"], quote=True)}">ⓘ</button></td>', 1)
    tooltip_ui = """
<style>
.prompt-info{display:inline-flex;align-items:center;justify-content:center;
width:24px;height:24px;padding:0;margin-left:4px;border:1px solid #b9c9df;
border-radius:50%;background:#edf3fc;color:#245bc2;font:700 17px system-ui;
cursor:help;vertical-align:middle}
.prompt-info:hover,.prompt-info:focus-visible{background:#dce9ff;outline:2px solid #245bc2;outline-offset:2px}
#prompt-tooltip{position:fixed;z-index:1000;width:560px;max-width:calc(100vw - 24px);
max-height:60vh;overflow:auto;padding:16px 18px;border:1px solid #bccde3;
border-radius:10px;background:#172335;color:#fff;box-shadow:0 8px 28px #17233540;
font:14px/1.55 system-ui;white-space:pre-wrap;overflow-wrap:anywhere}
#prompt-tooltip[hidden]{display:none}
</style>
<div id="prompt-tooltip" role="tooltip" hidden></div>
<script>
document.addEventListener('DOMContentLoaded', () => {
  const tip = document.getElementById('prompt-tooltip');
  let active = null;
  function hide() {
    if (active) active.removeAttribute('aria-describedby');
    active = null; tip.hidden = true;
  }
  function show(button) {
    hide(); active = button;
    tip.textContent = 'System prompt:\\n\\n' + button.dataset.prompt;
    button.setAttribute('aria-describedby', tip.id);
    tip.hidden = false;
    const box = button.getBoundingClientRect();
    const left = Math.max(12, Math.min(box.left, innerWidth - tip.offsetWidth - 12));
    const below = box.bottom + 4;
    const top = below + tip.offsetHeight <= innerHeight - 12
      ? below : Math.max(12, box.top - tip.offsetHeight - 4);
    tip.style.left = left + 'px'; tip.style.top = top + 'px';
  }
  document.querySelectorAll('.prompt-info').forEach(button => {
    button.addEventListener('mouseenter', () => show(button));
    button.addEventListener('focus', () => show(button));
    button.addEventListener('click', () => show(button));
    button.addEventListener('mouseleave', event => {
      if (!tip.contains(event.relatedTarget) && document.activeElement !== button) hide();
    });
    button.addEventListener('blur', hide);
  });
  tip.addEventListener('mouseleave', hide);
  document.addEventListener('keydown', event => { if (event.key === 'Escape') hide(); });
  document.addEventListener('click', event => {
    if (!event.target.closest('.prompt-info') && !tip.contains(event.target)) hide();
  });
  window.addEventListener('resize', hide);
  window.addEventListener('scroll', hide);
});
</script>
"""
    content = "<h2>Matched results</h2>" + results_table + tooltip_ui
    content += (
        '<p class="note">Number NLL scores the teacher’s numeric completions. Fluency NLL '
        'scores the system-prompt text itself. Owl answers count a whole-word owl synonym; '
        'strict answers must consist only of an accepted owl name and limited punctuation. '
        'These are distinct measurements.</p>'
        '<h2>What this means</h2><p>The original owl prompt raises owl answers from '
        '<strong>0.66% to 99.06%</strong>, but every SALVE prompt assigns higher likelihood '
        'to the numeric data than any of the three manual owl prompts. This confirms strong '
        'behavioral control through explicit instructions, not superior numeric fit. '
        'It does not establish transfer through student fine-tuning.</p>'
        '<h2>Exact prompts and raw answers</h2>')
    for name, result in scores.items():
        content += (
            f"<details><summary>{escape(LABELS[name])}</summary>"
            f'<pre style="white-space:pre-wrap;overflow-wrap:anywhere">'
            f'{escape(result["prompt"])}</pre>'
            f'<p><a href="results/{escape(result["answers_file"], quote=True)}">'
            "View all 5,000 answers and the evaluation questions (JSON)</a></p></details>")
    delta_rows = []
    for manual, by_salve in comparisons["comparisons"].items():
        for salve, result in by_salve.items():
            lo, hi = result["owl_increase_question_bootstrap_95ci"]
            delta_rows.append([
                LABELS[manual], LABELS[salve], f'{result["nll_improvement"]:+.4f}',
                f'{result["fluency_nll_improvement"]:+.4f}',
                f'{100 * result["owl_rate_increase"]:+.2f}',
                f'{100 * result["strict_owl_rate_increase"]:+.2f}',
                f'[{100 * lo:+.2f}, {100 * hi:+.2f}]'])
    content += (
        "<details><summary>All manual-versus-SALVE differences</summary>"
        '<p class="note">Positive means the manual prompt improves the metric. '
        'NLL improvement is SALVE minus manual NLL; behavior differences are percentage '
        'points. Intervals resample question clusters and are not corrected for multiple '
        'comparisons.</p>' +
        table(["Manual prompt", "SALVE", "Number NLL improvement",
               "Fluency NLL improvement", "Owl increase (pp)", "Strict increase (pp)",
               "Owl increase: 95% interval (pp)"], delta_rows) + "</details>")
    content += (
        "<h2>Limits</h2><p>This uses one model and one fine-tuned teacher’s dataset. "
        "The numeric validation set was already examined in earlier experiments. "
        "The target animal was known when writing the manual prompts; this is a control "
        "comparison, not blind trait discovery. Question-bootstrap intervals do not "
        "capture uncertainty across teachers, models or datasets.</p>"
        "<h2>Methodology and provenance</h2><p>Original Qwen2.5-7B-Instruct is frozen. "
        "Number NLL is weighted by completion-token count on the same 500 examples, "
        "excluding end-of-turn tokens. Accelerated scoring was checked against the reference. "
        "Fluency is mean prompt-token NLL after BOS with no chat template. Each prompt "
        "receives 50 animal questions × 100 samples, temperature 1, maximum 100 tokens, "
        "batch 64 and sampling seed 20261020. The three manual variants and all five "
        "final SALVE prompts were fixed before this comparison.</p>"
        f'<p>Execution commit: <code>{escape(manifest["repository_commit"])}</code><br>'
        f'Model revision: <code>{escape(manifest["model_revision"])}</code></p>'
        '<p><a href="results/manifest.json">Run manifest</a> · '
        '<a href="results/scores.json">All scores</a> · '
        '<a href="results/comparison.json">Pairwise comparisons</a> · '
        '<a href="results/scoring_checks.json">Scorer checks</a> · '
        '<a href="README.md">Experiment summary</a></p>')
    report = page("Manual prompts vs SALVE", bullets, content)
    (HERE / "report.html").write_text(report)
    files = [RESULTS / name for name in
             ("scores.json", "comparison.json", "manifest.json", "progress.json", "scoring_checks.json")]
    (HERE / "report_metadata.json").write_text(json.dumps({
        "execution_commit": manifest["repository_commit"],
        "report_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "input_sha256": {str(p.relative_to(HERE)): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in files}}, indent=2) + "\n")
    print(HERE / "report.html")


if __name__ == "__main__":
    main()
