"""Read attention patterns out of a trained model, at token and at move level.

``model.py`` is kept byte-identical to the course version, so instead of editing it we
temporarily swap its module-level ``scaled_dot_product_attention`` for a version that
computes exactly the same thing and also records the softmax weights.
"""

from __future__ import annotations

import math
from contextlib import contextmanager

import numpy as np
import torch

import chesslm.model as _model_mod
from chesslm.nn_utils import softmax
from chesslm.probes import move_end_offsets, token_index_for_offsets
from chesslm.tokenizer import BPETokenizer


@contextmanager
def record_attention():
    """Within the block, every attention call appends its ``(B, H, T, T)`` weights."""
    store: list[torch.Tensor] = []
    original = _model_mod.scaled_dot_product_attention

    def recording(Q, K, V, mask=None):
        scores = Q @ K.transpose(-2, -1) / math.sqrt(Q.shape[-1])
        if mask is not None:
            scores = scores + mask
        weights = softmax(scores, dim=-1)
        store.append(weights.detach())
        return weights @ V

    _model_mod.scaled_dot_product_attention = recording
    try:
        yield store
    finally:
        _model_mod.scaled_dot_product_attention = original


@torch.no_grad()
def attention_maps(model, ids: torch.Tensor) -> torch.Tensor:
    """Attention weights for a ``(1, T)`` input: ``(n_layers, n_heads, T, T)``."""
    with record_attention() as store:
        model(ids)
    return torch.stack([w[0] for w in store]).cpu()


def token_move_index(tok: BPETokenizer, ids: list[int], sans: list[str]) -> np.ndarray:
    """For each token, the index of the move its first byte belongs to.

    Move *k* owns the bytes after move *k-1* ends up to its own end (so the move
    number "12." and the separating space belong to the move that follows).
    Tokens past the last move (e.g. the result) get ``len(sans) - 1``.
    """
    ends = np.array(move_end_offsets(sans))
    starts = np.concatenate([[0], np.cumsum([len(tok.vocab[i]) for i in ids])[:-1]])
    return np.minimum(np.searchsorted(ends, starts, side="right"), len(sans) - 1)


def move_attention(att: torch.Tensor, tok: BPETokenizer, ids: list[int], sans: list[str]):
    """Collapse token attention to moves: ``(L, H, n_moves, n_moves)``.

    Row *q* is read at the token holding the last byte of move *q* (where the model
    has just seen the whole move); column *k* sums the attention paid to every token
    that starts inside move *k*. A token spanning two moves counts for the first, so
    a few columns can be empty. Rows still sum to 1.
    """
    owner = token_move_index(tok, ids, sans)
    n = len(sans)
    last_tok = np.array(token_index_for_offsets(tok, ids, move_end_offsets(sans)))
    rows = att[:, :, last_tok, :].numpy()  # (L, H, n, T)
    out = np.zeros(rows.shape[:3] + (n,), dtype=np.float32)
    for m in range(n):
        out[..., m] = rows[..., owner == m].sum(-1)
    return out
