"""Build game splits and one tokenized dataset per vocabulary size.

    python scripts/prepare_data.py                       # full corpus, all vocab sizes
    python scripts/prepare_data.py --vocab-sizes 512     # one tokenizer
    python scripts/prepare_data.py --max-games 4000      # quick prototype
"""

from __future__ import annotations

import argparse
from pathlib import Path

from chesslm.data import build_game_splits, build_tokenized

VOCAB_SIZES = [257, 512, 1024, 2048]  # 257 = raw bytes + <|game|>, i.e. no merges


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pgn-dir", default="data/raw_pgn")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--vocab-sizes", type=int, nargs="+", default=VOCAB_SIZES)
    ap.add_argument("--max-games", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-games", action="store_true", help="reuse existing data/games/")
    args = ap.parse_args()

    games_dir = Path(args.data_dir) / "games"
    if not args.skip_games:
        build_game_splits(args.pgn_dir, games_dir, seed=args.seed, max_games=args.max_games)
    for v in args.vocab_sizes:
        build_tokenized(games_dir, Path(args.data_dir) / f"v{v}", v)


if __name__ == "__main__":
    main()
