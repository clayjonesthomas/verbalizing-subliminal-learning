"""Dataset heading annotations from the published DSSE auditor scores."""
import json
import re
from html import escape
from pathlib import Path

from research_graphing import toggle_section


AUDIT_TOOLTIP_STYLE = """<style>
.audit-info {display:inline-block;position:relative;margin-inline-start:.35em}
.audit-info button {background:transparent;color:#475569;border:1px solid currentColor;border-radius:50%;
  font:600 12px/1 system-ui,sans-serif;width:1.4em;height:1.4em;padding:0;cursor:help;vertical-align:middle}
.audit-info [role=tooltip] {visibility:hidden;position:absolute;z-index:10;top:100%;right:0;
  width:min(19rem,70vw);padding:.6rem .8rem;border-radius:.35rem;background:#243041;color:white;
  font:normal 14px/1.45 system-ui,sans-serif;text-align:left;white-space:normal}
.audit-info:hover [role=tooltip],.audit-info:focus-within [role=tooltip] {visibility:visible}
</style>"""


def load_audit_scores():
    path = Path(__file__).resolve().parent / "measurements/published-dsse-scores.json"
    return json.loads(path.read_text())["scores"]


def audited_section(title, content, *, audit, **section_options):
    # All three auditors answered on each dataset in this panel.
    correct = round(3 * (1 - audit["mc_surprisingness"]))
    closest = {0: "strong", 0.5: "partial", 0.8: "weak", 1: "no match"}[audit["oe_surprisingness"]]
    if correct == 3 or closest == "strong":
        rating, color = "not surprising", "#b42318"
    elif correct or closest == "partial":
        rating, color = "moderately surprising", "#856500"
    else:
        rating, color = "completely surprising", "#18733a"

    tooltip = (f"Three model auditors (Luna, Terra, and Sol) each chose one of eight behaviors. "
               f"{correct}/3 chose the target. Closest open-ended guess: {closest}.")
    section = toggle_section(title, content, **section_options)
    heading = escape(title)
    linked_heading, count = re.subn(
        r"\bdsse1-\d+\b",
        r'<a href="https://dsse.transluce.org/?dataset=\g<0>" target="_blank" '
        r'rel="noopener noreferrer" contenteditable="false">\g<0></a>',
        heading,
    )
    if count != 1:
        raise ValueError(f"Expected one DSSE dataset ID in {title!r}")
    section = section.replace(f">{heading}</span></summary>", f">{linked_heading}</span></summary>", 1)
    trained = audit["qwen3_1_7b_opus"]
    base = audit["base_qwen3_1_7b_opus"]
    uplift = (f'<span title="Published Qwen3-1.7B elicitation, judged by Claude Opus 5.5: '
              f'{trained:.2%} after fine-tuning minus {base:.2%} before fine-tuning.">'
              f'{trained - base:+.1%} elicitation</span>')
    tip_id = escape(section_options["section_id"], quote=True) + "-audit-tip"
    icon = (f'<span class="audit-info" contenteditable="false"><button type="button" '
            f'aria-label="Auditor results" aria-describedby="{tip_id}">i</button>'
            f'<span id="{tip_id}" role="tooltip">{escape(tooltip)}</span></span>')
    return section.replace("</summary>",
                           f' · <span class="audit-surprise" style="color:{color}">{rating}</span>'
                           f'{icon} · {uplift}</summary>', 1)
