"""Dataset heading annotations from the published DSSE auditor scores."""
import json
from html import escape
from pathlib import Path

from research_graphing import toggle_section


def load_audit_scores():
    path = Path(__file__).resolve().parent / "measurements/published-dsse-scores.json"
    return json.loads(path.read_text())["scores"]


def audited_section(title, content, *, audit, **section_options):
    # All three auditors answered on each dataset in this panel.
    correct = round(3 * (1 - audit["mc_surprisingness"]))
    closest = {0: "strong", 0.5: "partial", 0.8: "weak", 1: "no match"}[audit["oe_surprisingness"]]
    if correct == 3 or closest == "strong":
        rating = "not surprising"
    elif correct or closest == "partial":
        rating = "moderately surprising"
    else:
        rating = "completely surprising"

    tooltip = (f"Three model auditors (Luna, Terra, and Sol) each chose one of eight behaviors. "
               f"{correct}/3 chose the target. Closest open-ended guess: {closest}.")
    section = toggle_section(title, content, **section_options)
    return section.replace("<summary>", f'<summary title="{escape(tooltip, quote=True)}">', 1).replace(
        "</summary>", f' · <span class="audit-surprise">{rating}</span></summary>', 1)
