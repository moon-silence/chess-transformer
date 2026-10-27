"""PGN cleaning: every game the model sees is legal and canonically formatted."""

from __future__ import annotations

from pathlib import Path

import chess

from chesslm.pgn import (
    canonical_movetext,
    parse_pgn_text,
    raw_san_tokens,
    split_movetext,
    validate,
)

FIXTURE = Path(__file__).parent / "fixtures" / "sample.pgn"


def test_spacing_comments_nags_variations_are_normalised():
    raw = "1. e4 {best by test} e5!? 2.Nf3 $1 (2.f4 exf4) Nc6 3...a6?! 1-0"
    sans, result = raw_san_tokens(raw)
    assert sans == ["e4", "e5", "Nf3", "Nc6", "a6"]
    assert result == "1-0"


def test_canonical_format():
    assert canonical_movetext(["e4", "e5", "Nf3"], "1-0") == "1.e4 e5 2.Nf3 1-0"


def test_illegal_game_is_rejected():
    assert validate({"Result": "1-0"}, "1.e4 e5 2.Ke3 Nc6 1-0") is None  # king can't reach e3


def test_unknown_result_is_rejected():
    assert validate({}, "1.e4 e5 2.Nf3 *") is None


def test_custom_start_position_is_rejected():
    headers = {"SetUp": "1", "FEN": "8/8/8/8/8/8/8/K6k w - - 0 1", "Result": "1/2-1/2"}
    assert validate(headers, "1.Kb1 Kg1 1/2-1/2") is None


def test_san_is_re_rendered_canonically():
    # over-disambiguated / missing check symbol in the source are normalised
    g = validate({"Result": "1-0"}, "1.e4 e5 2.Bc4 Nc6 3.Qh5 Nf6 4.Qxf7 1-0")
    assert g is not None
    assert g.sans[-1] == "Qxf7#"


def test_fixture_games_all_replay_legally():
    games, rejected = parse_pgn_text(FIXTURE.read_text())
    assert len(games) >= 50 and rejected == 0
    for g in games:
        board = chess.Board()
        for san in g.sans:
            board.push_san(san)
        assert g.movetext == canonical_movetext(g.sans, g.result)
        assert g.white and g.black


def test_split_movetext_reads_model_output():
    assert split_movetext("12.Nf3 Nc6 13.O-O 1-0 3.e4") == ["Nf3", "Nc6", "O-O"]
    assert split_movetext("e5 <|game|> 1.d4") == ["e5"]
