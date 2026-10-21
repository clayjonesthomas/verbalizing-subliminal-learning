"""Plot the first report table. Run with: uv run --with matplotlib <this file>."""
from html.parser import HTMLParser
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


class FirstTable(HTMLParser):
    def __init__(self):
        super().__init__()
        self.rows = []
        self.active = self.done = self.cell = False

    def handle_starttag(self, tag, attrs):
        if tag == "table" and not self.done:
            self.active = True
        if self.active:
            if tag == "tr":
                self.row = []
            if tag in ("td", "th"):
                self.cell = True
                self.value = ""

    def handle_data(self, data):
        if self.active and self.cell:
            self.value += data

    def handle_endtag(self, tag):
        if not self.active:
            return
        if tag in ("td", "th"):
            self.row.append(self.value.replace("ⓘ", "").strip())
            self.cell = False
        if tag == "tr":
            self.rows.append(self.row)
        if tag == "table":
            self.active = False
            self.done = True


def main():
    out = Path(__file__).parent / "results"
    parser = FirstTable()
    parser.feed((out / "report.html").read_text())
    headers, *rows = parser.rows
    assert headers[2:4] == ["Text validation NLL ↓", "Owl answers ↑"]
    labels = [r[0].replace(" · ", "\n").replace("Teacher / data-generation prompt", "Teacher /\ndata-generation").replace("empty system prompt", "empty system").replace("model default prompt", "model default") for r in rows]
    owl = [float(r[3].rstrip("%")) for r in rows]
    nll = [float(r[2]) for r in rows]
    colors = ["#163c6b" if i == 0 else "#5c91c9" if i < 6 else "#2b947c" if r[0].startswith("SALVE") else "#b78331" if r[0].startswith("LARGO") else "#929eae" for i, r in enumerate(rows)]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    fig, axes = plt.subplots(2, 1, figsize=(15, 9), sharex=True)
    for ax, values, title, fmt, ymax in zip(axes, [owl, nll], ["Owl answers (higher is better)", "Text validation NLL (lower is better)"], ["%.1f%%", "%.4f"], [112, .9]):
        bars = ax.bar(range(len(rows)), values, color=colors, width=.68)
        ax.bar_label(bars, fmt=fmt, padding=5, fontsize=10)
        ax.set_title(title, loc="left", fontsize=14, fontweight="bold", pad=12)
        ax.set_ylim(0, ymax)
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=.2)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="x", length=0)
    axes[0].yaxis.set_major_formatter(PercentFormatter(100))
    axes[0].set_yticks(range(0, 101, 20))
    axes[0].set_ylabel("Owl answers (%)")
    axes[1].set_ylabel("Text validation NLL")
    axes[1].set_xticks(range(len(rows)), labels, fontsize=10)
    fig.suptitle("Owl preference across recovered prompts and controls", x=.07, ha="left", fontsize=20, fontweight="bold")
    fig.text(.07, .035, "Greedy mean includes all five seeds (42–46). SALVE, LARGO, teacher and controls: earlier seed-42 experiment.\nValues from the report’s summary table; 1,000 animal answers and 500 text validation examples per evaluation.", fontsize=10, color="#536174", linespacing=1.6)
    fig.subplots_adjust(left=.07, right=.985, top=.89, bottom=.17, hspace=.3)
    for ext in ("png", "svg"):
        target = out / f"comparison.{ext}"
        fig.savefig(target, dpi=180, facecolor="white")
        print(target.resolve())
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6.5))
    colors = ["#2878b5", "#2a9d70", "#a16ac4", "#d65b39", "#b38a19"]
    markers = ["o", "s", "^", "D", "v"]
    for seed, color, marker in zip(range(42, 47), colors, markers):
        rows = [json.loads(line) for line in (out / f"seed{seed}/trajectory.jsonl").read_text().splitlines()]
        ax.plot([r["step"] for r in rows], [r["soft_validation_nll"] for r in rows],
                label=f"Seed {seed}", color=color, marker=marker, markersize=5, linewidth=2)
    ax.set_title("Direct soft-prompt validation NLL across all five seeds", loc="left", fontsize=17, fontweight="bold", pad=20)
    ax.set_xlabel("Optimization step")
    ax.set_ylabel("Direct soft-prompt validation NLL (lower is better)")
    ax.set_xticks(range(0, 2501, 250))
    ax.set_xlim(-40, 2540)
    ax.set_ylim(0, 2.4)
    ax.grid(alpha=.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper right", frameon=False)
    fig.text(.09, .035, "Diagnostics every 250 steps on 500 validation examples. Lines connect measured values; no smoothing.", fontsize=10, color="#536174")
    fig.subplots_adjust(left=.09, right=.98, top=.87, bottom=.16)
    for ext in ("png", "svg"):
        target = out / f"soft_prompt_nll_all_seeds.{ext}"
        fig.savefig(target, dpi=180, facecolor="white")
        print(target.resolve())
    plt.close(fig)


if __name__ == "__main__":
    main()
