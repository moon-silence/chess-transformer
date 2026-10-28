"""Looking inside the model: board-state probes and outcome prediction.

Everything here reads the model's residual stream *after it has read move k*
of a real game and asks what can be decoded from it with a linear map.

Feature position
    The full game is encoded exactly as in training, and the feature for ply
    ``k`` is taken at the token that contains the last byte of move ``k``'s
    SAN. No prompt boundary is created, so no token healing is needed.

Board probe (does the model track the board?)
    For each of the 64 squares, a linear classifier predicts its contents
    from the activation: 13 classes in absolute terms (empty, white PNBRQK,
    black pnbrqk) or relative to the side that just moved ("mine"/"theirs"),
    the encoding that made Othello-GPT's world model linearly readable.
    Controls: the same probe on the model *at initialisation*, and the
    per-square majority class.

Outcome (does it know who is winning?)
    * Zero-shot: append each result string to the prefix and compare the
      model's own likelihoods. No training.
    * Linear probe on the activations.
    * Baselines: majority class; material balance; Elo difference (metadata
      the model never sees).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import chess
import numpy as np
import torch

from chesslm.nn_utils import cross_entropy_loss, softmax
from chesslm.pgn import RESULTS
from chesslm.tokenizer import BPETokenizer

RESULT_TO_ID = {r: i for i, r in enumerate(RESULTS)}  # 1-0, 0-1, 1/2-1/2
PIECE_VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
N_SQUARE_CLASSES = 13


# ---------------------------------------------------------------------------
# Activations
# ---------------------------------------------------------------------------


@torch.no_grad()
def residual_stream(model, ids: torch.Tensor) -> list[torch.Tensor]:
    """Residual stream after the embedding and after every block.

    Returns ``n_layers + 1`` tensors of shape ``(B, T, d_model)``. Uses the
    model's own modules, so ``model.py`` stays untouched.
    """
    x = model.token_emb(ids)
    out = [x]
    for block in model.blocks:
        x = block(x)
        out.append(x)
    return out


def move_end_offsets(sans: list[str]) -> list[int]:
    """Byte offset (exclusive) where each move's SAN ends in canonical movetext."""
    offsets, pos = [], 0
    for i, san in enumerate(sans):
        if i:
            pos += 1  # the space
        if i % 2 == 0:
            pos += len(f"{i // 2 + 1}.")
        pos += len(san.encode())
        offsets.append(pos)
    return offsets


def token_index_for_offsets(tok: BPETokenizer, ids: list[int], offsets: list[int]) -> list[int]:
    """For each byte offset, the index of the token containing byte ``offset - 1``."""
    ends = np.cumsum([len(tok.vocab[i]) for i in ids])
    return [int(np.searchsorted(ends, off, side="left")) for off in offsets]


@dataclass
class GameFeatures:
    feats: np.ndarray  # (n_samples, n_layers + 1, d_model)
    plies: np.ndarray  # ply index k: features are taken after move k was read
    game_idx: np.ndarray
    fraction: np.ndarray  # (k + 1) / n_plies


@torch.no_grad()
def extract_features(
    model,
    tok: BPETokenizer,
    games: list[dict],
    plies_per_game: int | None = 8,
    seed: int = 0,
    device: str = "cpu",
) -> GameFeatures:
    """Residual-stream features after selected plies of each game."""
    rng = random.Random(seed)
    ctx = model.context_length
    feats, plies, gidx, frac = [], [], [], []
    for gi, g in enumerate(games):
        sans = g["sans"]
        ids = tok.encode(g["movetext"])
        positions = token_index_for_offsets(tok, ids, move_end_offsets(sans))
        ks = list(range(len(sans)))
        if plies_per_game is not None and len(ks) > plies_per_game:
            ks = sorted(rng.sample(ks, plies_per_game))
        first_window = None
        for k in ks:
            p = positions[k]
            if p < ctx:
                if first_window is None:
                    x = torch.tensor(ids[:ctx], device=device)[None]
                    first_window = torch.stack(residual_stream(model, x), 1)[0]
                h = first_window[:, p]
            else:
                x = torch.tensor(ids[p - ctx + 1 : p + 1], device=device)[None]
                h = torch.stack(residual_stream(model, x), 1)[0, :, -1]
            feats.append(h.cpu().numpy())
            plies.append(k)
            gidx.append(gi)
            frac.append((k + 1) / len(sans))
    return GameFeatures(np.stack(feats), np.array(plies), np.array(gidx), np.array(frac))


# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------


def board_after_ply(sans: list[str], k: int) -> chess.Board:
    board = chess.Board()
    for san in sans[: k + 1]:
        board.push_san(san)
    return board


def square_classes(board: chess.Board, relative: bool) -> np.ndarray:
    """64 labels in 0..12. Relative: 1-6 = side that just moved, 7-12 = opponent."""
    labels = np.zeros(64, dtype=np.int64)
    mover = not board.turn  # the side that just played move k
    for sq, piece in board.piece_map().items():
        side = (piece.color == mover) if relative else (piece.color == chess.WHITE)
        labels[sq] = piece.piece_type + (0 if side else 6)
    return labels


def board_targets(games: list[dict], gf: GameFeatures, relative: bool) -> np.ndarray:
    return np.stack(
        [
            square_classes(board_after_ply(games[g]["sans"], int(k)), relative)
            for g, k in zip(gf.game_idx, gf.plies, strict=True)
        ]
    )


def material_features(games: list[dict], gf: GameFeatures) -> np.ndarray:
    """[white - black material, ply / 100, white to move]."""
    rows = []
    for g, k in zip(gf.game_idx, gf.plies, strict=True):
        b = board_after_ply(games[g]["sans"], int(k))
        bal = sum(
            v * (len(b.pieces(p, chess.WHITE)) - len(b.pieces(p, chess.BLACK)))
            for p, v in PIECE_VALUES.items()
        )
        rows.append([bal, k / 100.0, float(b.turn)])
    return np.array(rows, dtype=np.float32)


def elo_features(games: list[dict], gf: GameFeatures) -> np.ndarray:
    """[(white Elo - black Elo) / 400]; NaN where a rating is missing."""
    out = []
    for g in gf.game_idx:
        w, b = games[g].get("white_elo"), games[g].get("black_elo")
        out.append([(w - b) / 400.0 if w and b else np.nan])
    return np.array(out, dtype=np.float32)


def result_targets(games: list[dict], gf: GameFeatures) -> np.ndarray:
    return np.array([RESULT_TO_ID[games[g]["result"]] for g in gf.game_idx])


# ---------------------------------------------------------------------------
# Linear probes
# ---------------------------------------------------------------------------


@dataclass
class LinearProbe:
    weight: torch.Tensor  # (d, n_heads * n_classes)
    bias: torch.Tensor
    mean: torch.Tensor
    std: torch.Tensor
    n_heads: int
    n_classes: int

    def logits(self, X: np.ndarray | torch.Tensor) -> torch.Tensor:
        X = torch.as_tensor(X, dtype=torch.float32)
        z = ((X - self.mean) / self.std) @ self.weight + self.bias
        return z.view(-1, self.n_heads, self.n_classes)

    def predict(self, X) -> np.ndarray:
        return self.logits(X).argmax(-1).squeeze(-1).numpy()

    def proba(self, X) -> np.ndarray:
        return softmax(self.logits(X), dim=-1).squeeze(1).numpy()

    def state_dict(self) -> dict:
        return {
            k: getattr(self, k) for k in ("weight", "bias", "mean", "std", "n_heads", "n_classes")
        }


def fit_linear_probe(
    X: np.ndarray,
    y: np.ndarray,
    n_classes: int,
    weight_decay: float = 1e-3,
    epochs: int = 200,
    lr: float = 0.05,
    seed: int = 0,
    device: str = "cpu",
) -> LinearProbe:
    """Multinomial logistic regression, one head per column of ``y``.

    ``y`` is ``(N,)`` for a single classifier or ``(N, H)`` for H independent
    heads (the 64 squares). Full-batch Adam on standardised features.
    """
    torch.manual_seed(seed)
    X = torch.as_tensor(X, dtype=torch.float32, device=device)
    y = torch.as_tensor(y, dtype=torch.long, device=device)
    if y.ndim == 1:
        y = y[:, None]
    n_heads = y.shape[1]
    mean, std = X.mean(0), X.std(0).clamp_min(1e-6)
    Xs = (X - mean) / std
    W = torch.zeros(X.shape[1], n_heads * n_classes, device=device, requires_grad=True)
    b = torch.zeros(n_heads * n_classes, device=device, requires_grad=True)
    opt = torch.optim.Adam([W, b], lr=lr, weight_decay=0.0)
    for _ in range(epochs):
        logits = (Xs @ W + b).view(-1, n_heads, n_classes)
        loss = cross_entropy_loss(logits, y) + weight_decay * (W**2).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return LinearProbe(
        W.detach().cpu(), b.detach().cpu(), mean.cpu(), std.cpu(), n_heads, n_classes
    )


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    f1s = []
    for c in range(n_classes):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        f1s.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1s))


# ---------------------------------------------------------------------------
# Zero-shot outcome prediction
# ---------------------------------------------------------------------------


@torch.no_grad()
def zero_shot_result_logprobs(
    model, tok: BPETokenizer, sans: list[str], k: int, device: str = "cpu"
) -> np.ndarray:
    """log P(result | first k+1 moves) under the language model, for each result.

    Each candidate ``"<moves> <result>"`` is encoded as a whole (the way games
    look in training) and scored on the tokens that cover the last few bytes of
    the prefix plus the result, so boundary merges like ``"5 1"`` are handled.
    Returned as a normalised log-distribution over the three results.
    """
    from chesslm.pgn import canonical_movetext

    prefix = canonical_movetext(sans[: k + 1], "").rstrip()
    boundary = max(0, len(prefix.encode()) - 8)
    scores = []
    for res in RESULTS:
        ids = tok.encode(f"{prefix} {res}")
        ends = np.cumsum([len(tok.vocab[i]) for i in ids])
        ids = ids[-(model.context_length + 1) :]
        ends = ends[-len(ids) :]
        x = torch.tensor(ids[:-1], device=device)[None]
        logp = torch.log(softmax(model(x)[0], dim=-1))
        targets = torch.tensor(ids[1:], device=device)
        tok_lp = logp.gather(-1, targets[:, None]).squeeze(-1).cpu().numpy()
        scored = ends[1:] > boundary
        scores.append(float(tok_lp[scored].sum()))
    s = np.array(scores)
    return s - (s.max() + math.log(np.exp(s - s.max()).sum()))
