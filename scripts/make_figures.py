"""Report figures from saved results (no GPU needed).

Usage: python -P scripts/make_figures.py
  fig1_loss_curves.png  train/val loss for E1 and E3, parsed from the executed notebook's training log
  fig2_mcq_metrics.png  MCQ accuracy: option likelihood vs generated answer, per experiment
Writes to report/figures/.
"""
import json
import re
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "report" / "figures"
RUN_NB = ROOT / "notebooks" / "runs" / "02_train_eval_run1.ipynb"
TABLE = ROOT / "results" / "experiments" / "experiments_table_v2.csv"

# Palette: reference categorical slots 1-2 (validated light-mode pair), chrome/ink roles.
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.titlecolor": INK, "axes.titlesize": 11, "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})


def training_logs() -> dict:
    """Train loss every 25 optimiser steps and val loss per epoch, from the notebook's printed log."""
    nb = json.loads(RUN_NB.read_text(encoding="utf-8"))
    text = "".join("".join(o.get("text", "")) for c in nb["cells"] if c["cell_type"] == "code"
                   for o in c.get("outputs", []) if o.get("output_type") == "stream")
    logs, current = {}, None
    for line in text.splitlines():
        m = re.search(r"=== (E\d_\S+):", line)
        if m:
            current = m.group(1)
            logs[current] = {"train": [], "val": []}
            continue
        if current is None:
            continue
        m = re.search(r"epoch (\d+) step (\d+)/(\d+) loss ([\d.]+)", line)
        if m:
            logs[current]["train"].append((int(m.group(2)), float(m.group(4))))
            logs[current]["steps_per_epoch"] = int(m.group(3)) / 3
        m = re.search(r"== epoch (\d+) val_loss ([\d.]+)", line)
        if m:
            logs[current]["val"].append((int(m.group(1)), float(m.group(2))))
    return {k: v for k, v in logs.items() if v["train"]}


def fig_loss(logs):
    names = [("E1_lora_0.5B_saq+mcq", "E1: Qwen2.5-0.5B + LoRA"), ("E3_lora_1.5B_saq+mcq", "E3: Qwen2.5-1.5B + LoRA")]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for ax, (key, title) in zip(axes, names):
        lg = logs[key]
        spe = lg["steps_per_epoch"]
        ax.plot([s / spe for s, _ in lg["train"]], [l for _, l in lg["train"]], color=BLUE, lw=2,
                label="train loss (mean of last 25 steps)")
        ve = [e for e, _ in lg["val"]]
        vl = [l for _, l in lg["val"]]
        ax.plot(ve, vl, color=ORANGE, lw=2, marker="o", ms=8, mec=SURFACE, mew=2, label="validation loss")
        best = min(range(len(vl)), key=vl.__getitem__)
        ax.annotate(f"best: epoch {ve[best]}\n(adapter kept)", (ve[best], vl[best]), xytext=(6, -34),
                    textcoords="offset points", color=INK2, fontsize=9)
        ax.set_title(title, loc="left")
        ax.set_xlabel("epoch")
        ax.set_xticks([0, 1, 2, 3])
    axes[0].set_ylabel("cross-entropy loss (answer tokens)")
    axes[0].legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_loss_curves.png", dpi=200)
    plt.close(fig)


def fig_mcq():
    t = pd.read_csv(TABLE)
    labels = {"E0_zeroshot_0.5B": "E0 0.5B\nzero-shot", "E1_lora_0.5B_saq+mcq": "E1 0.5B\nLoRA SAQ+MCQ",
              "E2_lora_0.5B_saq": "E2 0.5B\nLoRA SAQ only", "E0_zeroshot_1.5B": "E0 1.5B\nzero-shot",
              "E3_lora_1.5B_saq+mcq": "E3 1.5B\nLoRA SAQ+MCQ"}
    t = t.set_index("experiment").loc[list(labels)]
    x = range(len(t))
    w = 0.38
    fig, ax = plt.subplots(figsize=(9, 3.8))
    for off, col, color, name in ((-w / 2 - 0.01, "test_mcq_likelihood_single", BLUE, "option likelihood (single-answer, n=744)"),
                                  (w / 2 + 0.01, "test_mcq_gen_single", ORANGE, "generated answer, exact match (single-answer)")):
        bars = ax.bar([i + off for i in x], t[col], width=w, color=color, label=name, zorder=3)
        for b, v in zip(bars, t[col]):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=8.5, color=INK2)
    ax.axhline(0.2, color=MUTED, lw=1, ls=(0, (3, 3)), zorder=2, label="chance (5 options)")
    ax.set_xticks(list(x), list(labels.values()))
    ax.set_ylabel("test MCQ accuracy")
    ax.set_ylim(0, 0.62)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", fontsize=9)
    ax.set_title("Most of the gap between the two metrics is answer format, not knowledge", loc="left")
    fig.tight_layout()
    fig.savefig(OUT / "fig2_mcq_metrics.png", dpi=200)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig_loss(training_logs())
    fig_mcq()
    print("wrote", sorted(p.name for p in OUT.glob("*.png")))


if __name__ == "__main__":
    main()
