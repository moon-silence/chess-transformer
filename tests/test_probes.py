"""Probes and outcome prediction: indexing, targets, and the probe fitter."""

from __future__ import annotations

import chess
import numpy as np
import pytest
import torch

from chesslm.data import games_to_text
from chesslm.pgn import validate
from chesslm.probes import (
    board_after_ply,
    elo_features,
    extract_features,
    fit_linear_probe,
    macro_f1,
    material_features,
    move_end_offsets,
    residual_stream,
    square_classes,
    token_index_for_offsets,
    zero_shot_result_logprobs,
)
from chesslm.tokenizer import BPETokenizer, train_bpe_from_text
from chesslm.train import TrainConfig, build_model

GAME = validate(
    {"Result": "1-0", "WhiteElo": "2800", "BlackElo": "2600"},
    "1.e4 e5 2.Nf3 Nc6 3.Bb5 a6 4.Bxc6 dxc6 5.O-O f6 6.d4 exd4 7.Nxd4 c5 1-0",
).to_dict()


@pytest.fixture(scope="module")
def tok():
    vocab, merges = train_bpe_from_text(games_to_text([GAME] * 20), 270, ["<|game|>"])
    return BPETokenizer(vocab, merges, ["<|game|>"])


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    m = build_model(TrainConfig(d_model=32, n_layers=2, n_heads=2, d_ff=64, context_length=16), 270)
    for b in m.blocks:
        torch.nn.init.normal_(b.attn.o_proj.weight, std=0.2)
        torch.nn.init.normal_(b.ffn.w_down.weight, std=0.2)
    return m.eval()


def test_move_end_offsets_point_at_each_move():
    text = GAME["movetext"].encode()
    for san, off in zip(GAME["sans"], move_end_offsets(GAME["sans"]), strict=True):
        assert text[:off].endswith(san.encode())


def test_token_index_contains_last_byte_of_move(tok):
    ids = tok.encode(GAME["movetext"])
    offs = move_end_offsets(GAME["sans"])
    starts = np.concatenate([[0], np.cumsum([len(tok.vocab[i]) for i in ids])])
    for off, t in zip(offs, token_index_for_offsets(tok, ids, offs), strict=True):
        assert starts[t] <= off - 1 < starts[t + 1]


def test_features_beyond_context_use_a_sliding_window(tok, model):
    gf = extract_features(model, tok, [GAME], plies_per_game=None)
    assert gf.feats.shape == (len(GAME["sans"]), 3, 32)
    # recompute a late ply by hand: window of the last `ctx` tokens ending at the move
    ids = tok.encode(GAME["movetext"])
    k = len(GAME["sans"]) - 1
    p = token_index_for_offsets(tok, ids, move_end_offsets(GAME["sans"]))[k]
    assert p >= model.context_length  # this test really exercises the window path
    x = torch.tensor(ids[p - model.context_length + 1 : p + 1])[None]
    ref = torch.stack(residual_stream(model, x), 1)[0, :, -1].numpy()
    assert np.allclose(gf.feats[k], ref, atol=1e-5)


def test_square_labels_absolute_and_relative():
    b = board_after_ply(GAME["sans"], 0)  # after 1.e4, white just moved
    absolute, relative = square_classes(b, False), square_classes(b, True)
    assert absolute[chess.E4] == chess.PAWN and relative[chess.E4] == chess.PAWN
    b = board_after_ply(GAME["sans"], 1)  # after 1...e5, black just moved
    assert square_classes(b, False)[chess.E5] == chess.PAWN + 6
    assert square_classes(b, True)[chess.E5] == chess.PAWN  # "mine" for the mover
    assert square_classes(b, True)[chess.E4] == chess.PAWN + 6
    assert (square_classes(b, False) == 0).sum() == 32


def test_material_and_elo_features(tok, model):
    gf = extract_features(model, tok, [GAME], plies_per_game=None)
    mat = material_features([GAME], gf)
    assert mat[7, 0] == 0  # 4.Bxc6 dxc6: bishop for knight, level again
    assert mat[6, 0] == 3  # right after 4.Bxc6 white is a knight up
    assert np.allclose(elo_features([GAME], gf), 0.5)


def test_probe_recovers_a_linear_rule():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(600, 10)).astype(np.float32)
    y = np.stack([(X[:, 0] > 0).astype(int), (X[:, 1] + X[:, 2] > 0).astype(int)], 1)
    probe = fit_linear_probe(X, y, n_classes=2, epochs=300)
    acc = (probe.logits(X).argmax(-1).numpy() == y).mean()
    assert acc > 0.97


def test_macro_f1():
    y = np.array([0, 1, 2, 2])
    assert macro_f1(y, y, 3) == 1.0
    assert macro_f1(y, np.zeros(4, dtype=int), 3) < 0.3


def test_zero_shot_returns_a_distribution(tok, model):
    lp = zero_shot_result_logprobs(model, tok, GAME["sans"], 5)
    assert lp.shape == (3,) and np.isclose(np.exp(lp).sum(), 1.0)
