"""Legal-move evaluation of a trained run on held-out *test* games.

    python scripts/evaluate.py --vocab 512
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from chesslm.data import load_games
from chesslm.legality import Sampler, free_running_legal_plies, teacher_forced_legality
from chesslm.tokenizer import BPETokenizer
from chesslm.train import load_checkpoint


def evaluate(vocab: int, data_dir="data", runs_dir="runs", n_positions=500, n_games=50,
             device=None) -> dict:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    run = Path(runs_dir) / f"v{vocab}"
    tok = BPETokenizer.load(Path(data_dir) / f"v{vocab}" / "tokenizer.json")
    model = load_checkpoint(run / "best.pt", device)
    sampler = Sampler(model, tok, device)
    games = load_games(Path(data_dir) / "games" / "test.jsonl")

    tf = teacher_forced_legality(sampler, games, n_positions=n_positions, seed=0)
    fr = free_running_legal_plies(sampler, games, n_games=n_games, seed=0)
    sample = sampler.continue_text("1.e4 e5 2.", max_new_bytes=200)
    out = {"vocab_size": vocab, "teacher_forced": tf.to_dict(), "free_running": fr,
           "sample_from_1.e4_e5": "1.e4 e5 2." + sample}
    (run / "eval.json").write_text(json.dumps(out, indent=2))
    print(f"[vocab {vocab}] legal-move rate {tf.legal_rate:.3f} | exact {tf.exact_rate:.3f} | "
          f"free-running legal plies {fr['mean_legal_plies']:.1f}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", type=int, required=True)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--n-positions", type=int, default=500)
    ap.add_argument("--n-games", type=int, default=50)
    args = ap.parse_args()
    evaluate(args.vocab, args.data_dir, args.runs_dir, args.n_positions, args.n_games)


if __name__ == "__main__":
    main()
