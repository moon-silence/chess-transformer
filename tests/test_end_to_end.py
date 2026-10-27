"""PGN -> splits -> tokenizer -> training -> evaluation, on a tiny fixture."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from chesslm.data import build_game_splits, build_tokenized
from chesslm.legality import Sampler, teacher_forced_legality
from chesslm.tokenizer import BPETokenizer
from chesslm.train import TrainConfig, load_checkpoint, train

FIXTURE = Path(__file__).parent / "fixtures" / "sample.pgn"
SMOKE = Path(__file__).parents[1] / "configs" / "smoke.yaml"


def test_splits_are_disjoint_and_deduplicated(tmp_path):
    pgn = tmp_path / "pgn"
    pgn.mkdir()
    shutil.copy(FIXTURE, pgn / "a.pgn")
    shutil.copy(FIXTURE, pgn / "b.pgn")  # every game duplicated across files
    stats = build_game_splits(pgn, tmp_path / "games", val_frac=0.1, test_frac=0.1)
    assert stats["duplicates"] == stats["n_total"]
    splits = {
        s: {json.loads(line)["movetext"] for line in open(tmp_path / "games" / f"{s}.jsonl")}
        for s in ("train", "val", "test")
    }
    assert not (splits["train"] & splits["val"])
    assert not (splits["train"] & splits["test"])
    assert not (splits["val"] & splits["test"])


@pytest.mark.slow
def test_pipeline_runs_and_learns(tmp_path):
    pgn = tmp_path / "pgn"
    pgn.mkdir()
    shutil.copy(FIXTURE, pgn / "a.pgn")
    build_game_splits(pgn, tmp_path / "games", val_frac=0.1, test_frac=0.1)
    meta = build_tokenized(tmp_path / "games", tmp_path / "v300", 300)
    assert meta["vocab_size"] == 300
    assert meta["splits"]["train"]["bytes_per_token"] > 1.0

    cfg = TrainConfig.from_yaml(SMOKE, max_steps=60, learning_rate=3e-3)
    summary = train(cfg, tmp_path / "v300", tmp_path / "run", device="cpu", log=lambda *_: None)
    hist = summary["history"]
    assert hist[-1]["val_loss"] < hist[0]["val_loss"] - 0.5

    model = load_checkpoint(tmp_path / "run" / "best.pt")
    tok = BPETokenizer.load(tmp_path / "v300" / "tokenizer.json")
    games = [json.loads(line) for line in open(tmp_path / "games" / "test.jsonl")]
    rep = teacher_forced_legality(Sampler(model, tok), games, n_positions=10)
    assert 0.0 <= rep.legal_rate <= 1.0
