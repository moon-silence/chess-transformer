"""PGN parsing, cleaning, validation, and canonical movetext.

Raw PGN from different sources disagrees on spacing (``1.e4`` vs ``1. e4``),
annotations (``!?``, ``$1``), comments, and variations. Rather than trust any
of it, every game is *replayed* on a real board with python-chess and its
movetext is regenerated from the legal moves. The model therefore only ever
sees one canonical format::

    1.e4 e5 2.Nf3 Nc6 3.Bb5 a6 1-0

Games containing an illegal or unparseable move are dropped, not repaired.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import asdict, dataclass

import chess

RESULTS = ("1-0", "0-1", "1/2-1/2")

_HEADER_RE = re.compile(r'^\[(\w+)\s+"(.*)"\]\s*$')
_COMMENT_RE = re.compile(r"\{[^}]*\}|;[^\n]*")
_NAG_RE = re.compile(r"\$\d+")
_MOVE_NUM_RE = re.compile(r"^\d+\.+")
_ANNOT_RE = re.compile(r"[!?]+$")


@dataclass
class Game:
    """One validated game."""

    movetext: str  # canonical "1.e4 e5 ... 1-0"
    sans: list[str]
    result: str
    white: str = ""
    black: str = ""
    white_elo: int | None = None
    black_elo: int | None = None
    eco: str = ""
    date: str = ""
    event: str = ""

    @property
    def n_plies(self) -> int:
        return len(self.sans)

    def to_dict(self) -> dict:
        return asdict(self)


def iter_raw_games(text: str) -> Iterator[tuple[dict[str, str], str]]:
    """Yield ``(headers, movetext)`` for each game in a multi-game PGN string."""
    headers: dict[str, str] = {}
    body: list[str] = []
    in_body = False
    for line in text.splitlines():
        m = _HEADER_RE.match(line.strip())
        if m:
            if in_body:  # a header after movetext starts a new game
                yield headers, " ".join(body)
                headers, body, in_body = {}, [], False
            headers[m.group(1)] = m.group(2)
        elif line.strip():
            in_body = True
            body.append(line.strip())
    if body:
        yield headers, " ".join(body)


def _strip_variations(s: str) -> str:
    out, depth = [], 0
    for ch in s:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        elif depth == 0:
            out.append(ch)
    return "".join(out)


def raw_san_tokens(movetext: str) -> tuple[list[str], str | None]:
    """Extract SAN move tokens and the result from raw movetext."""
    s = _strip_variations(_NAG_RE.sub(" ", _COMMENT_RE.sub(" ", movetext)))
    # "1." / "1..." glued to a move ("1.e4") or standing alone ("1. e4")
    s = re.sub(r"(\d+\.+)", r" \1 ", s)
    sans, result = [], None
    for tok in s.split():
        if _MOVE_NUM_RE.fullmatch(tok):
            continue
        if tok in RESULTS or tok == "*":
            result = tok
            break
        sans.append(_ANNOT_RE.sub("", tok))
    return sans, result


def canonical_movetext(sans: list[str], result: str) -> str:
    """Format SAN moves as ``1.e4 e5 2.Nf3 ... result``."""
    parts = []
    for i, san in enumerate(sans):
        parts.append(f"{i // 2 + 1}.{san}" if i % 2 == 0 else san)
    parts.append(result)
    return " ".join(parts)


def _elo(v: str | None) -> int | None:
    try:
        return int(v) if v else None
    except ValueError:
        return None


def validate(headers: dict[str, str], movetext: str, min_plies: int = 2) -> Game | None:
    """Replay a game on a board. Return a canonical Game, or None if invalid."""
    sans, result = raw_san_tokens(movetext)
    result = result or headers.get("Result")
    if result not in RESULTS or len(sans) < min_plies:
        return None
    if headers.get("SetUp") == "1" or "FEN" in headers:
        return None  # non-standard starting position
    board = chess.Board()
    clean: list[str] = []
    try:
        for san in sans:
            move = board.parse_san(san)
            clean.append(board.san(move))  # re-render: canonical SAN
            board.push(move)
    except ValueError:
        return None
    return Game(
        movetext=canonical_movetext(clean, result),
        sans=clean,
        result=result,
        white=headers.get("White", ""),
        black=headers.get("Black", ""),
        white_elo=_elo(headers.get("WhiteElo")),
        black_elo=_elo(headers.get("BlackElo")),
        eco=headers.get("ECO", ""),
        date=headers.get("Date", ""),
        event=headers.get("Event", ""),
    )


def parse_pgn_text(text: str) -> tuple[list[Game], int]:
    """Parse and validate every game. Returns ``(games, n_rejected)``."""
    games, rejected = [], 0
    for headers, body in iter_raw_games(text):
        g = validate(headers, body)
        if g is None:
            rejected += 1
        else:
            games.append(g)
    return games, rejected


def split_movetext(text: str) -> list[str]:
    """Split model output like ``"12.Nf3 Nc6 13.O-O"`` into SAN moves.

    Stops at a result token or a game separator. Move numbers are stripped.
    """
    moves = []
    for tok in text.replace("<|game|>", " <|game|> ").split():
        if tok in RESULTS or tok == "*" or tok == "<|game|>":
            break
        tok = _MOVE_NUM_RE.sub("", tok)
        if tok:
            moves.append(tok)
    return moves
