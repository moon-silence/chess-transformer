"""Does the model know the rules? Legal-move evaluation with python-chess.

Loss says how well the model predicts *text*. These metrics say whether that
text is *chess*:

* **Teacher-forced legal-move rate.** Take a real game, cut it at a random
  ply, prompt the model with the true prefix, sample one move, and check it
  is legal in that position. Averaged over many positions.
* **Free-running legal plies.** Prompt with a real opening, let the model
  play both sides on its own, and count moves until the first illegal one.

Prompt boundaries need care with BPE. A merge like ``"5 "`` or ``".N"`` means
the tokens of a prefix are not always a prefix of the tokens of the full
game, so a naive prompt puts the model in a state it never saw in training.
We use *token healing*: the last prompt token is removed and the first
sampled tokens are constrained to reproduce its bytes, so the model chooses
its own tokenisation across the boundary.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import chess
import torch

from chesslm.nn_utils import softmax
from chesslm.pgn import split_movetext
from chesslm.tokenizer import BPETokenizer


def prefix_text(sans: list[str], ply: int) -> str:
    """Prompt for predicting move ``ply`` (0-indexed), ending right before it.

    White to move: ``"1.e4 e5 2."``     Black to move: ``"1.e4 e5 2.Nf3 "``
    """
    parts = []
    for i, san in enumerate(sans[:ply]):
        parts.append(f"{i // 2 + 1}.{san}" if i % 2 == 0 else san)
    head = " ".join(parts)
    if ply % 2 == 0:
        return (head + " " if head else "") + f"{ply // 2 + 1}."
    return head + " "


class Sampler:
    """Token-by-token sampling with token healing and a context window."""

    def __init__(self, model, tok: BPETokenizer, device: str = "cpu"):
        self.model, self.tok, self.device = model, tok, device
        self.ctx = model.context_length
        self.tok_bytes = [tok.vocab[i] for i in range(tok.vocab_size)]
        self.special_ids = set(tok.special_to_id.values())

    @torch.no_grad()
    def _logits(self, ids: list[int]) -> torch.Tensor:
        x = torch.tensor(ids[-self.ctx :], dtype=torch.long, device=self.device)[None]
        return self.model(x)[0, -1]

    def _pick(self, logits: torch.Tensor, temperature: float, gen: torch.Generator | None) -> int:
        if temperature <= 0:
            return int(torch.argmax(logits))
        probs = softmax(logits / temperature, dim=-1)
        return int(torch.multinomial(probs.cpu(), 1, generator=gen))

    def continue_text(
        self,
        prompt: str,
        max_new_bytes: int,
        stop_at_space: bool = False,
        temperature: float = 0.0,
        gen: torch.Generator | None = None,
    ) -> str:
        """Generate text after ``prompt``. Returns only the new text."""
        ids = self.tok.encode(prompt)
        required = b""
        if ids:  # token healing: let the model re-choose the boundary token
            required = self.tok_bytes[ids[-1]]
            ids = ids[:-1]
        out = b""
        while len(out) < max_new_bytes + len(required):
            logits = self._logits(ids) if ids else torch.zeros(self.tok.vocab_size)
            if required:
                mask = torch.full_like(logits, float("-inf"))
                for v, b in enumerate(self.tok_bytes):
                    if v in self.special_ids or not b:
                        continue
                    if b.startswith(required) or required.startswith(b):
                        mask[v] = 0.0
                logits = logits + mask
            nxt = self._pick(logits, temperature, gen)
            ids.append(nxt)
            if nxt in self.special_ids:
                break
            b = self.tok_bytes[nxt]
            if required:
                consumed = min(len(required), len(b))
                required = required[consumed:]
                b = b[consumed:]
            out += b
            if stop_at_space and out.strip() and b" " in out.lstrip():
                break
        return out.decode("utf-8", errors="replace")

    def sample_move(self, prompt: str, temperature: float = 0.0, gen=None) -> str:
        text = self.continue_text(prompt, max_new_bytes=16, stop_at_space=True,
                                  temperature=temperature, gen=gen)
        moves = split_movetext(text)
        return moves[0] if moves else ""


def board_after(sans: list[str], ply: int) -> chess.Board:
    board = chess.Board()
    for san in sans[:ply]:
        board.push_san(san)
    return board


def is_legal(board: chess.Board, san: str) -> bool:
    try:
        board.parse_san(san)
        return True
    except ValueError:
        return False


@dataclass
class LegalityReport:
    n_positions: int = 0
    n_legal: int = 0
    n_exact: int = 0  # sampled move == the move actually played
    by_phase: dict = field(default_factory=dict)  # "opening"/"middlegame"/"endgame" -> [legal, n]
    examples: list = field(default_factory=list)

    @property
    def legal_rate(self) -> float:
        return self.n_legal / max(1, self.n_positions)

    @property
    def exact_rate(self) -> float:
        return self.n_exact / max(1, self.n_positions)

    def to_dict(self) -> dict:
        return {
            "n_positions": self.n_positions,
            "legal_rate": self.legal_rate,
            "exact_match_rate": self.exact_rate,
            "by_phase": {k: v[0] / max(1, v[1]) for k, v in self.by_phase.items()},
            "examples": self.examples[:20],
        }


def phase(ply: int) -> str:
    return "opening" if ply < 20 else ("middlegame" if ply < 60 else "endgame")


def teacher_forced_legality(
    sampler: Sampler,
    games: list[dict],
    n_positions: int = 500,
    seed: int = 0,
    temperature: float = 0.0,
) -> LegalityReport:
    rng = random.Random(seed)
    gen = torch.Generator().manual_seed(seed)
    rep = LegalityReport()
    for _ in range(n_positions):
        g = rng.choice(games)
        sans = g["sans"]
        ply = rng.randrange(len(sans))
        board = board_after(sans, ply)
        move = sampler.sample_move(prefix_text(sans, ply), temperature, gen)
        legal = is_legal(board, move)
        rep.n_positions += 1
        rep.n_legal += legal
        rep.n_exact += move == sans[ply]
        ph = rep.by_phase.setdefault(phase(ply), [0, 0])
        ph[0] += legal
        ph[1] += 1
        if len(rep.examples) < 20:
            rep.examples.append({"ply": ply, "played": sans[ply], "sampled": move, "legal": legal})
    return rep


def free_running_legal_plies(
    sampler: Sampler,
    games: list[dict],
    n_games: int = 50,
    prompt_plies: int = 10,
    max_plies: int = 60,
    seed: int = 0,
    temperature: float = 0.0,
) -> dict:
    """Mean number of consecutive legal moves the model plays on its own."""
    rng = random.Random(seed)
    gen = torch.Generator().manual_seed(seed)
    runs = []
    for _ in range(n_games):
        g = rng.choice([g for g in games if len(g["sans"]) > prompt_plies] or games)
        sans = list(g["sans"][:prompt_plies])
        board = board_after(sans, len(sans))
        n_legal = 0
        while n_legal < max_plies and not board.is_game_over():
            move = sampler.sample_move(prefix_text(sans, len(sans)), temperature, gen)
            if not is_legal(board, move):
                break
            m = board.parse_san(move)
            sans.append(board.san(m))
            board.push(m)
            n_legal += 1
        runs.append(n_legal)
    return {
        "n_games": n_games,
        "prompt_plies": prompt_plies,
        "max_plies": max_plies,
        "mean_legal_plies": sum(runs) / len(runs),
        "median_legal_plies": sorted(runs)[len(runs) // 2],
        "frac_reached_max": sum(r >= max_plies for r in runs) / len(runs),
        "runs": runs,
    }
