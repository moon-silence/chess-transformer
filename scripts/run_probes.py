"""Board-state probes and outcome prediction for one trained run.

    python scripts/run_probes.py --vocab 512
    python scripts/run_probes.py --vocab 512 --n-train-games 500 --n-zero-shot 300   # quick

Writes results/probes_v{V}.json and results/probes/board_probe_v{V}.pt (for the app).
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from chesslm.data import load_games
from chesslm.probes import (
    N_SQUARE_CLASSES,
    board_targets,
    elo_features,
    extract_features,
    fit_linear_probe,
    macro_f1,
    material_features,
    result_targets,
    zero_shot_result_logprobs,
)
from chesslm.tokenizer import BPETokenizer
from chesslm.train import TrainConfig, build_model, load_checkpoint

FRACTION_BINS = [(0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.01)]


def bin_label(lo: float, hi: float) -> str:
    return f"{int(lo * 100)}-{min(100, int(hi * 100))}%"


def board_probe_scores(train_X, train_y, test_X, test_y, layers, device="cpu"):
    """Accuracy per layer (all squares, and occupied squares only)."""
    out, probes = [], []
    for layer in layers:
        probe = fit_linear_probe(
            train_X[:, layer], train_y, N_SQUARE_CLASSES, epochs=300, device=device
        )
        pred = probe.logits(test_X[:, layer]).argmax(-1).numpy()
        occupied = test_y != 0
        out.append(
            {
                "layer": layer,
                "accuracy": float((pred == test_y).mean()),
                "occupied_accuracy": float((pred[occupied] == test_y[occupied]).mean()),
                "per_square_accuracy": (pred == test_y).mean(0).tolist(),
            }
        )
        probes.append(probe)
    return out, probes


def majority_board_baseline(train_y, test_y):
    majority = np.array(
        [np.bincount(train_y[:, s], minlength=N_SQUARE_CLASSES).argmax() for s in range(64)]
    )
    pred = np.broadcast_to(majority, test_y.shape)
    occupied = test_y != 0
    return {
        "accuracy": float((pred == test_y).mean()),
        "occupied_accuracy": float((pred[occupied] == test_y[occupied]).mean()),
    }


def outcome_scores(pred, y, frac):
    row = {
        "accuracy": float((pred == y).mean()),
        "macro_f1": macro_f1(y, pred, 3),
        "by_fraction": {},
    }
    for lo, hi in FRACTION_BINS:
        m = (frac >= lo) & (frac < hi)
        if m.any():
            row["by_fraction"][bin_label(lo, hi)] = float((pred[m] == y[m]).mean())
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", type=int, required=True)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--n-train-games", type=int, default=4000)
    ap.add_argument("--n-test-games", type=int, default=None, help="default: all test games")
    ap.add_argument("--plies-per-game", type=int, default=8)
    ap.add_argument("--n-zero-shot", type=int, default=2000)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()

    tok = BPETokenizer.load(Path(args.data_dir) / f"v{args.vocab}" / "tokenizer.json")
    ckpt_path = Path(args.runs_dir) / f"v{args.vocab}" / "best.pt"
    model = load_checkpoint(ckpt_path, device)
    cfg = TrainConfig(**torch.load(ckpt_path, map_location="cpu")["config"])
    torch.manual_seed(cfg.seed)
    untrained = build_model(cfg, tok.vocab_size).to(device).eval()

    games_dir = Path(args.data_dir) / "games"
    train_games = load_games(games_dir / "train.jsonl")
    random.Random(0).shuffle(train_games)
    train_games = train_games[: args.n_train_games]
    test_games = load_games(games_dir / "test.jsonl")[: args.n_test_games]

    print(
        f"[vocab {args.vocab}] extracting features "
        f"({len(train_games)} train / {len(test_games)} test games)"
    )
    feats = {}
    for name, m in [("trained", model), ("untrained", untrained)]:
        feats[name] = (
            extract_features(m, tok, train_games, args.plies_per_game, seed=1, device=device),
            extract_features(m, tok, test_games, args.plies_per_game, seed=2, device=device),
        )
    tr, te = feats["trained"]
    layers = list(range(tr.feats.shape[1]))
    results: dict = {
        "vocab_size": args.vocab,
        "n_train_samples": len(tr.plies),
        "n_test_samples": len(te.plies),
        "layers": layers,
    }

    # ---- board probe -------------------------------------------------------
    best_probe = None
    for relative in (True, False):
        key = "relative" if relative else "absolute"
        ytr, yte = board_targets(train_games, tr, relative), board_targets(test_games, te, relative)
        trained_scores, probes = board_probe_scores(tr.feats, ytr, te.feats, yte, layers, device)
        utr, ute = feats["untrained"]
        untrained_scores, _ = board_probe_scores(utr.feats, ytr, ute.feats, yte, layers, device)
        results[f"board_{key}"] = {
            "trained": trained_scores,
            "untrained": [
                {k: v for k, v in s.items() if k != "per_square_accuracy"} for s in untrained_scores
            ],
            "majority": majority_board_baseline(ytr, yte),
        }
        best = max(trained_scores, key=lambda s: s["accuracy"])
        print(
            f"  board probe ({key}): best layer {best['layer']} acc {best['accuracy']:.3f} "
            f"(occupied {best['occupied_accuracy']:.3f}) | untrained "
            f"{max(s['accuracy'] for s in untrained_scores):.3f} | majority "
            f"{results[f'board_{key}']['majority']['accuracy']:.3f}"
        )
        if relative:
            best_probe = {
                "layer": best["layer"],
                "relative": True,
                **probes[best["layer"]].state_dict(),
            }

    # ---- outcome -------------------------------------------------------------
    ytr, yte = result_targets(train_games, tr), result_targets(test_games, te)
    outcome: dict = {"class_balance_test": np.bincount(yte, minlength=3).tolist()}
    majority = np.bincount(ytr, minlength=3).argmax()
    outcome["majority"] = outcome_scores(np.full_like(yte, majority), yte, te.fraction)

    per_layer = []
    for layer in layers:
        p = fit_linear_probe(tr.feats[:, layer], ytr, 3, weight_decay=1e-2, device=device)
        per_layer.append(
            outcome_scores(p.predict(te.feats[:, layer]), yte, te.fraction) | {"layer": layer}
        )
    outcome["activation_probe_by_layer"] = per_layer
    outcome["activation_probe"] = max(per_layer, key=lambda s: s["accuracy"])

    mtr, mte = material_features(train_games, tr), material_features(test_games, te)
    p = fit_linear_probe(mtr, ytr, 3, weight_decay=1e-3)
    outcome["material"] = outcome_scores(p.predict(mte), yte, te.fraction)

    etr, ete = elo_features(train_games, tr), elo_features(test_games, te)
    ok_tr, ok_te = ~np.isnan(etr[:, 0]), ~np.isnan(ete[:, 0])
    p = fit_linear_probe(np.hstack([mtr, etr])[ok_tr], ytr[ok_tr], 3, weight_decay=1e-3)
    outcome["material_plus_elo"] = outcome_scores(
        p.predict(np.hstack([mte, ete])[ok_te]), yte[ok_te], te.fraction[ok_te]
    ) | {"n": int(ok_te.sum())}

    # zero-shot on a random subset of test samples
    idx = np.random.default_rng(0).permutation(len(te.plies))[: args.n_zero_shot]
    zs = np.array(
        [
            zero_shot_result_logprobs(
                model, tok, test_games[te.game_idx[i]]["sans"], int(te.plies[i]), device
            ).argmax()
            for i in idx
        ]
    )
    outcome["zero_shot"] = outcome_scores(zs, yte[idx], te.fraction[idx]) | {"n": len(idx)}
    results["outcome"] = outcome
    for k in ("majority", "material", "material_plus_elo", "activation_probe", "zero_shot"):
        print(
            f"  outcome {k:18s} acc {outcome[k]['accuracy']:.3f}  "
            f"macro-F1 {outcome[k]['macro_f1']:.3f}"
        )

    out = Path(args.results_dir)
    (out / "probes").mkdir(parents=True, exist_ok=True)
    (out / f"probes_v{args.vocab}.json").write_text(json.dumps(results, indent=2))
    torch.save(best_probe, out / "probes" / f"board_probe_v{args.vocab}.pt")
    print(f"  done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
