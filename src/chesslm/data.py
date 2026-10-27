"""Dataset construction: PGN files -> deduplicated game splits -> token arrays.

Two stages, so the tokenizer sweep never re-parses PGN:

1. ``build_game_splits``: parse + validate every game, drop duplicates (the
   same Carlsen-Anand game appears in *both* players' archives -- left in, it
   would leak between train and test), shuffle with a fixed seed, and split
   **by game** into train / val / test JSONL files.
2. ``build_tokenized``: for one vocabulary size, train BPE on the train split
   only, encode all three splits, and record bytes-per-token so losses can be
   compared across vocabularies in bits per byte.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np

from chesslm.pgn import Game, parse_pgn_text
from chesslm.tokenizer import BPETokenizer, train_bpe_from_text

GAME_SEP = "<|game|>"
SPECIAL_TOKENS = [GAME_SEP]
SPLITS = ("train", "val", "test")


def build_game_splits(
    pgn_dir: str | Path,
    out_dir: str | Path,
    val_frac: float = 0.02,
    test_frac: float = 0.02,
    seed: int = 42,
    max_games: int | None = None,
) -> dict:
    pgn_dir, out_dir = Path(pgn_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(pgn_dir.glob("*.pgn"))
    if not paths:
        raise FileNotFoundError(f"no .pgn files in {pgn_dir}")

    seen: set[str] = set()
    games: list[Game] = []
    stats = {"files": {}, "rejected": 0, "duplicates": 0}
    for p in paths:
        parsed, rejected = parse_pgn_text(p.read_text(encoding="utf-8", errors="ignore"))
        kept = 0
        for g in parsed:
            if g.movetext in seen:
                stats["duplicates"] += 1
                continue
            seen.add(g.movetext)
            games.append(g)
            kept += 1
        stats["rejected"] += rejected
        stats["files"][p.name] = {"parsed": len(parsed), "kept": kept, "rejected": rejected}
        print(f"{p.name:22s} parsed {len(parsed):6d}  new {kept:6d}  rejected {rejected:4d}")

    games.sort(key=lambda g: g.movetext)  # make shuffle independent of file order
    random.Random(seed).shuffle(games)
    if max_games is not None:
        games = games[:max_games]

    n_val, n_test = int(val_frac * len(games)), int(test_frac * len(games))
    split_games = {
        "val": games[:n_val],
        "test": games[n_val : n_val + n_test],
        "train": games[n_val + n_test :],
    }
    for name, gs in split_games.items():
        with open(out_dir / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for g in gs:
                f.write(json.dumps(g.to_dict()) + "\n")
    stats.update({f"n_{k}": len(v) for k, v in split_games.items()})
    stats["n_total"] = len(games)
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    print(
        f"\nkept {len(games):,} unique games "
        f"(dropped {stats['duplicates']:,} duplicates, {stats['rejected']:,} invalid)"
    )
    return stats


def load_games(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def games_to_text(games: list[dict]) -> str:
    """Join games as ``g1 <|game|> g2 <|game|> ...`` (same layout as TinyStories)."""
    return f" {GAME_SEP} ".join(g["movetext"] for g in games)


def build_tokenized(games_dir: str | Path, out_dir: str | Path, vocab_size: int) -> dict:
    games_dir, out_dir = Path(games_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    texts = {s: games_to_text(load_games(games_dir / f"{s}.jsonl")) for s in SPLITS}

    print(f"[vocab {vocab_size}] training BPE on {len(texts['train']) / 1e6:.1f} MB")
    vocab, merges = train_bpe_from_text(texts["train"], vocab_size, SPECIAL_TOKENS, verbose=True)
    tok = BPETokenizer(vocab, merges, SPECIAL_TOKENS)
    tok.save(out_dir / "tokenizer.json")

    meta: dict = {"vocab_size": tok.vocab_size, "n_merges": len(merges), "splits": {}}
    for s, text in texts.items():
        ids = tok.encode_to_array(text)
        assert ids.max() < 2**16
        ids.astype(np.uint16).tofile(out_dir / f"{s}.bin")
        n_bytes = len(text.encode("utf-8"))
        meta["splits"][s] = {
            "tokens": int(ids.size),
            "bytes": n_bytes,
            "bytes_per_token": n_bytes / ids.size,
        }
        print(f"  {s:5s} {ids.size:>11,} tokens  {n_bytes / ids.size:.3f} bytes/token")
    meta["sample_merges"] = [tok.token_str(256 + i) for i in range(min(40, len(merges)))]
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def load_tokens(path: str | Path):
    """Load a ``.bin`` token file as an int64 torch tensor."""
    import torch

    return torch.from_numpy(np.fromfile(path, dtype=np.uint16).astype(np.int64))
