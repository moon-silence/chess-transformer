"""Regenerate every figure in figures/ from results/sweep.json.

    python scripts/make_figures.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({"figure.dpi": 150, "axes.grid": True, "grid.alpha": 0.3,
                     "axes.spines.top": False, "axes.spines.right": False})
COLORS = ["#1f4e79", "#2e86c1", "#e67e22", "#c0392b", "#27ae60", "#8e44ad"]


def fig_learning_curves(rows, out):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for c, r in zip(COLORS, rows, strict=False):
        h = r["history"]
        ax.plot([x["step"] for x in h], [x["val_bpb"] for x in h], color=c, lw=1.8,
                label=f"vocab {r['vocab_size']}")
    ax.set_xlabel("training step")
    ax.set_ylabel("validation bits per byte")
    ax.set_title("Learning curves, normalised so every vocabulary is comparable")
    ax.set_ylim(bottom=0)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "fig1_learning_curves_bpb.png")
    plt.close(fig)


def fig_sweep_summary(rows, out):
    v = [str(r["vocab_size"]) for r in rows]
    panels = [
        ("best_val_bpb", "bits per byte (lower is better)"),
        ("legal_rate", "legal-move rate (teacher-forced)"),
        ("mean_free_legal_plies", "legal plies before first error"),
        ("context_bytes", "context window in bytes of PGN"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6))
    for ax, (key, label) in zip(axes, panels, strict=True):
        vals = [r[key] for r in rows]
        bars = ax.bar(v, vals, color=COLORS[: len(v)])
        ax.bar_label(bars, fmt="%.3g", fontsize=8)
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("vocabulary size")
    fig.suptitle("Tokenizer sweep: same model, same data, same compute -- only the vocabulary "
                 "changes", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "fig2_sweep_summary.png")
    plt.close(fig)


def fig_legal_by_phase(rows, out):
    phases = ["opening", "middlegame", "endgame"]
    fig, ax = plt.subplots(figsize=(7, 4))
    width = 0.8 / len(rows)
    for i, (c, r) in enumerate(zip(COLORS, rows, strict=False)):
        xs = [j + (i - (len(rows) - 1) / 2) * width for j in range(3)]
        ax.bar(xs, [r["legal_by_phase"].get(p, 0) for p in phases], width, color=c,
               label=f"vocab {r['vocab_size']}")
    ax.set_xticks(range(3), ["opening\n(ply < 20)", "middlegame\n(20-59)", "endgame\n(60+)"])
    ax.set_ylabel("legal-move rate")
    ax.set_ylim(0, 1.05)
    ax.set_title("Where the rules break down")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig3_legal_by_phase.png")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/sweep.json")
    ap.add_argument("--out", default="figures")
    args = ap.parse_args()
    rows = sorted(json.loads(Path(args.results).read_text()), key=lambda r: r["vocab_size"])
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    fig_learning_curves(rows, out)
    fig_sweep_summary(rows, out)
    fig_legal_by_phase(rows, out)
    print(f"wrote {len(list(out.glob('*.png')))} figures to {out}/")


if __name__ == "__main__":
    main()
