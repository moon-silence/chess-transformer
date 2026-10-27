"""Legal-move evaluation, checked with an oracle model that knows the true game."""

from __future__ import annotations

import pytest
import torch

from chesslm.data import games_to_text
from chesslm.legality import (
    Sampler,
    board_after,
    free_running_legal_plies,
    is_legal,
    prefix_text,
    teacher_forced_legality,
)
from chesslm.pgn import validate
from chesslm.tokenizer import BPETokenizer, train_bpe_from_text

GAME = validate(
    {"Result": "1-0"},
    "1.e4 e5 2.Nf3 Nc6 3.Bb5 a6 4.Ba4 Nf6 5.O-O Be7 6.Re1 b5 7.Bb3 d6 8.c3 O-O 9.h3 Nb8 1-0",
).to_dict()


class OracleModel(torch.nn.Module):
    """Puts all probability on the longest token that continues the true game text.

    Works whatever tokenisation the prompt used, so it isolates the sampling and
    token-healing logic from model quality.
    """

    def __init__(self, tok: BPETokenizer, text: str):
        super().__init__()
        self.tok, self.text = tok, text.encode()
        self.context_length = 100_000
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, ids):
        seen = b"".join(self.tok.vocab[int(i)] for i in ids[0])
        assert self.text.startswith(seen), "model was fed text that is not a game prefix"
        rest = self.text[len(seen):]
        best, best_len = 0, -1
        for v, b in self.tok.vocab.items():
            special = v in self.tok.special_to_id.values()
            if not special and rest.startswith(b) and len(b) > best_len:
                best, best_len = v, len(b)
        logits = torch.full((1, ids.shape[1], self.tok.vocab_size), -1e9)
        logits[0, -1, best] = 0.0
        return logits


@pytest.fixture(scope="module")
def tok():
    text = games_to_text([GAME] * 30)
    vocab, merges = train_bpe_from_text(text, 400, ["<|game|>"])
    return BPETokenizer(vocab, merges, ["<|game|>"])


def test_prefix_text_format():
    sans = GAME["sans"]
    assert prefix_text(sans, 0) == "1."
    assert prefix_text(sans, 1) == "1.e4 "
    assert prefix_text(sans, 2) == "1.e4 e5 2."
    assert GAME["movetext"].startswith(prefix_text(sans, 7))


def test_legality_checker():
    board = board_after(GAME["sans"], 2)  # after 1.e4 e5, white to move
    assert is_legal(board, "Nf3")
    assert not is_legal(board, "Nf6")
    assert not is_legal(board, "garbage")


@pytest.mark.parametrize("ply", range(0, 18))
def test_token_healing_recovers_true_continuation(tok, ply):
    sampler = Sampler(OracleModel(tok, GAME["movetext"]), tok)
    prompt = prefix_text(GAME["sans"], ply)
    assert sampler.sample_move(prompt) == GAME["sans"][ply]


def test_oracle_is_perfectly_legal(tok):
    sampler = Sampler(OracleModel(tok, GAME["movetext"]), tok)
    rep = teacher_forced_legality(sampler, [GAME], n_positions=30)
    assert rep.legal_rate == 1.0 and rep.exact_rate == 1.0


def test_free_running_counts_plies(tok):
    sampler = Sampler(OracleModel(tok, GAME["movetext"]), tok)
    out = free_running_legal_plies(sampler, [GAME], n_games=2, prompt_plies=4, max_plies=10)
    assert out["mean_legal_plies"] == 10
