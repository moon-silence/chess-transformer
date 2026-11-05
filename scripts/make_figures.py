"""Regenerate every figure in figures/ from results/.

    python scripts/make_figures.py

fig1-3 need results/sweep.json; fig4-6 need results/probes_v{V}.json (skipped if missing).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)
MUTED = "#7f8c8d"
COLORS = ["#1f4e79", "#2e86c1", "#e67e22", "#c0392b", "#27ae60", "#8e44ad"]


def fig_learning_curves(rows, out):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for c, r in zip(COLORS, rows, strict=False):
        h = r["history"]
        ax.plot(
            [x["step"] for x in h],
            [x["val_bpb"] for x in h],
            color=c,
            lw=1.8,
            label=f"vocab {r['vocab_size']}",
        )
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
    fig.suptitle(
        "Tokenizer sweep: same model, same data, same compute -- only the vocabulary changes",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out / "fig2_sweep_summary.png")
    plt.close(fig)


def fig_legal_by_phase(rows, out):
    phases = ["opening", "middlegame", "endgame"]
    fig, ax = plt.subplots(figsize=(7, 4))
    width = 0.8 / len(rows)
    for i, (c, r) in enumerate(zip(COLORS, rows, strict=False)):
        xs = [j + (i - (len(rows) - 1) / 2) * width for j in range(3)]
        ax.bar(
            xs,
            [r["legal_by_phase"].get(p, 0) for p in phases],
            width,
            color=c,
            label=f"vocab {r['vocab_size']}",
        )
    ax.set_xticks(range(3), ["opening\n(ply < 20)", "middlegame\n(20-59)", "endgame\n(60+)"])
    ax.set_ylabel("legal-move rate")
    ax.set_ylim(0, 1.05)
    ax.set_title("Where the rules break down")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig3_legal_by_phase.png")
    plt.close(fig)


def fig_board_probe(probes, out):
    fig, axes = plt.subplots(
        1, len(probes), figsize=(max(5.0, 3.6 * len(probes)), 4.0), sharey=True, squeeze=False
    )
    for ax, p in zip(axes[0], probes, strict=True):
        b = p["board_relative"]
        layers = [s["layer"] for s in b["trained"]]
        ax.plot(
            layers,
            [s["accuracy"] for s in b["trained"]],
            "o-",
            color=COLORS[0],
            label="trained model",
        )
        ax.plot(
            layers,
            [s["occupied_accuracy"] for s in b["trained"]],
            "o--",
            color=COLORS[1],
            label="trained, occupied squares",
        )
        ax.plot(
            layers,
            [s["accuracy"] for s in b["untrained"]],
            "s-",
            color=MUTED,
            label="untrained model",
        )
        ax.axhline(b["majority"]["accuracy"], color=COLORS[3], ls=":", label="majority per square")
        ax.set_title(f"vocab {p['vocab_size']}", fontsize=10)
        ax.set_xlabel("layer (0 = embeddings)")
        ax.set_xticks(layers)
    axes[0][0].set_ylabel("probe accuracy (64 squares x 13 classes)")
    axes[0][0].set_ylim(0, 1.02)
    axes[0][-1].legend(fontsize=7, loc="lower right")
    fig.suptitle("Can a linear probe read the board\nfrom the model's activations?", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "fig4_board_probe_by_layer.png")
    plt.close(fig)


OUTCOME_METHODS = [
    ("majority", "always the most common result", MUTED, ":"),
    ("material", "material balance", COLORS[2], "-"),
    ("material_plus_elo", "material + Elo", COLORS[3], "-"),
    ("activation_probe", "probe on activations", COLORS[0], "-"),
    ("zero_shot", "zero-shot (model writes the result)", COLORS[4], "--"),
]


def fig_outcome(p, out):
    o = p["outcome"]
    bins = list(o["majority"]["by_fraction"])
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [1.6, 1]})
    for key, label, c, ls in OUTCOME_METHODS:
        if key in o:
            ax.plot(
                bins,
                [o[key]["by_fraction"].get(b, np.nan) for b in bins],
                marker="o",
                color=c,
                ls=ls,
                label=label,
            )
    ax.set_xlabel("how much of the game the model has seen")
    ax.set_ylabel("accuracy (win / loss / draw)")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    ax.set_title("Predicting the result as the game unfolds", fontsize=10)
    keys = [k for k, *_ in OUTCOME_METHODS if k in o]
    bars = ax2.barh(
        [OUTCOME_METHODS[i][1] for i, (k, *_) in enumerate(OUTCOME_METHODS) if k in o],
        [o[k]["macro_f1"] for k in keys],
        color=[c for k, _, c, _ in OUTCOME_METHODS if k in o],
    )
    ax2.bar_label(bars, fmt="%.2f", fontsize=8)
    ax2.set_xlabel("macro-F1 (all positions)")
    ax2.set_xlim(0, 1)
    ax2.invert_yaxis()
    ax2.tick_params(axis="y", labelsize=8)
    fig.suptitle(f"Outcome prediction, vocab {p['vocab_size']}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "fig5_outcome_prediction.png")
    plt.close(fig)


def fig_square_heatmap(p, out):
    b = p["board_relative"]
    best = max(b["trained"], key=lambda s: s["accuracy"])
    acc = np.array(best["per_square_accuracy"]).reshape(8, 8)  # row 0 = rank 1
    fig, ax = plt.subplots(figsize=(5, 4.4))
    im = ax.imshow(acc, origin="lower", cmap="viridis", vmin=min(0.5, acc.min()), vmax=1)
    for r in range(8):
        for f in range(8):
            ax.text(
                f,
                r,
                f"{acc[r, f]:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if acc[r, f] < 0.8 else "black",
            )
    ax.set_xticks(range(8), list("abcdefgh"))
    ax.set_yticks(range(8), [str(i) for i in range(1, 9)])
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.046, label="probe accuracy")
    ax.set_title(
        f"Per-square accuracy, layer {best['layer']} (vocab {p['vocab_size']})", fontsize=10
    )
    fig.tight_layout()
    fig.savefig(out / "fig6_square_accuracy.png")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/sweep.json")
    ap.add_argument("--probes-dir", default="results")
    ap.add_argument(
        "--focus-vocab",
        type=int,
        default=None,
        help="vocab for fig5/fig6 (default: best board probe)",
    )
    ap.add_argument("--out", default="figures")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    if Path(args.results).exists():
        rows = sorted(json.loads(Path(args.results).read_text()), key=lambda r: r["vocab_size"])
        fig_learning_curves(rows, out)
        fig_sweep_summary(rows, out)
        fig_legal_by_phase(rows, out)
    else:
        print(f"skipping fig1-3: {args.results} not found")
    probes = sorted(
        (json.loads(f.read_text()) for f in Path(args.probes_dir).glob("probes_v*.json")),
        key=lambda p: p["vocab_size"],
    )
    if probes:
        fig_board_probe(probes, out)
        by_v = {p["vocab_size"]: p for p in probes}
        best_probe = by_v.get(args.focus_vocab) or max(
            probes, key=lambda p: max(s["accuracy"] for s in p["board_relative"]["trained"])
        )
        best_outcome = by_v.get(args.focus_vocab) or max(
            probes, key=lambda p: p["outcome"].get("zero_shot", {}).get("macro_f1", 0.0)
        )
        fig_outcome(best_outcome, out)
        fig_square_heatmap(best_probe, out)
    else:
        print(f"skipping fig4-6: no {args.probes_dir}/probes_v*.json")
    print(f"wrote {len(list(out.glob('*.png')))} figures to {out}/")


if __name__ == "__main__":
    main()
