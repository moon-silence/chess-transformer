"""Attention extraction must not change the model and must give valid distributions."""

from __future__ import annotations

import numpy as np
import torch

import chesslm.model as model_mod
from chesslm.attention import attention_maps, move_attention, record_attention, token_move_index
from chesslm.model import TransformerLM
from chesslm.pgn import canonical_movetext
from chesslm.tokenizer import BPETokenizer, train_bpe_from_text

SANS = ["e4", "e5", "Nf3", "Nc6", "Bb5", "a6", "Ba4", "Nf6", "O-O", "Be7"]


def _setup():
    torch.manual_seed(0)
    vocab, merges = train_bpe_from_text(canonical_movetext(SANS, "1-0") * 30, 280, ["<|game|>"])
    tok = BPETokenizer(vocab, merges, ["<|game|>"])
    model = TransformerLM(tok.vocab_size, 64, d_model=32, n_layers=2, n_heads=4, d_ff=64).eval()
    ids = tok.encode(canonical_movetext(SANS, "1-0"))
    return tok, model, ids


def test_recording_is_exact_and_restores():
    tok, model, ids = _setup()
    x = torch.tensor([ids])
    ref = model(x)
    original = model_mod.scaled_dot_product_attention
    with record_attention() as store:
        out = model(x)
    assert model_mod.scaled_dot_product_attention is original
    assert torch.equal(out, ref)
    assert len(store) == 2


def test_attention_is_causal_distribution():
    tok, model, ids = _setup()
    att = attention_maps(model, torch.tensor([ids]))
    T = len(ids)
    assert att.shape == (2, 4, T, T)
    assert torch.allclose(att.sum(-1), torch.ones(2, 4, T), atol=1e-5)
    assert torch.all(att.triu(1) == 0)


def test_move_level_attention():
    tok, model, ids = _setup()
    owner = token_move_index(tok, ids, SANS)
    assert owner[0] == 0 and owner[-1] == len(SANS) - 1
    assert np.all(np.diff(owner) >= 0)
    # the bytes of each token map back to the move that owns them
    text = canonical_movetext(SANS, "1-0")
    pos = 0
    for i, t in zip(ids, owner, strict=True):
        piece = tok.vocab[i].decode()
        assert text[pos : pos + len(piece)] == piece
        if piece.strip(" .0123456789") and pos < len(text) - 4:
            assert any(c in SANS[t] for c in piece.strip(" ."))
        pos += len(piece)
    att = attention_maps(model, torch.tensor([ids]))
    ma = move_attention(att, tok, ids, SANS)
    assert ma.shape == (2, 4, len(SANS), len(SANS))
    np.testing.assert_allclose(ma.sum(-1), 1.0, atol=1e-5)
    assert np.allclose(np.triu(ma, 1), 0)


def test_move_attention_when_token_spans_moves():
    # a vocabulary that merges heavily makes tokens cross move boundaries
    vocab, merges = train_bpe_from_text(canonical_movetext(SANS, "1-0") * 50, 400, ["<|game|>"])
    tok = BPETokenizer(vocab, merges, ["<|game|>"])
    torch.manual_seed(0)
    model = TransformerLM(tok.vocab_size, 64, d_model=32, n_layers=1, n_heads=2, d_ff=64).eval()
    ids = tok.encode(canonical_movetext(SANS, "").rstrip())
    assert len(set(token_move_index(tok, ids, SANS))) < len(SANS)  # some move owns no token
    ma = move_attention(attention_maps(model, torch.tensor([ids])), tok, ids, SANS)
    np.testing.assert_allclose(ma.sum(-1), 1.0, atol=1e-5)
