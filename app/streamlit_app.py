"""Interactive companion to `chesslm` -- play a model that was never told the rules.

Run locally:   streamlit run app/streamlit_app.py

Models are read from models/v{V}/ (exported by scripts/export_models.py after the
Colab sweep). If none are exported yet, the app falls back to runs/ checkpoints.
"""

from __future__ import annotations

import html
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import altair as alt  # noqa: E402
import chess  # noqa: E402
import chess.svg  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
import torch  # noqa: E402

from chesslm.attention import attention_maps, move_attention  # noqa: E402
from chesslm.legality import Sampler, is_legal, prefix_text  # noqa: E402
from chesslm.pgn import canonical_movetext, raw_san_tokens  # noqa: E402
from chesslm.probes import LinearProbe, extract_features, square_classes  # noqa: E402
from chesslm.tokenizer import BPETokenizer  # noqa: E402
from chesslm.train import count_params, load_checkpoint  # noqa: E402

st.set_page_config(page_title="chesslm · chess as a language", page_icon="♞", layout="wide")

LEGAL, ILLEGAL, FALLBACK, MUTED = "#0f766e", "#c2384a", "#b45309", "#6b7a8d"
TOKEN_COLORS = ["#dbeafe", "#fde68a", "#bbf7d0", "#fecaca", "#e9d5ff", "#fed7aa"]
st.markdown(
    """<style>
      .block-container {padding-top: 2rem; max-width: 1300px;}
      .tok {padding: 1px 2px; margin: 1px; border-radius: 3px; font-family: monospace;
            white-space: pre; line-height: 2.0; font-size: 0.92rem;}
      .mv {font-family: monospace; padding: 1px 4px; border-radius: 3px;}
    </style>""",
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------
def available_models() -> dict[int, dict[str, Path]]:
    found: dict[int, dict[str, Path]] = {}
    for d in sorted((ROOT / "models").glob("v*")):
        if (d / "model.pt").exists() and (d / "tokenizer.json").exists():
            found[int(d.name[1:])] = {
                "model": d / "model.pt",
                "tok": d / "tokenizer.json",
                "probe": d / "board_probe.pt",
            }
    if not found:  # local development: use training outputs directly
        for d in sorted((ROOT / "runs").glob("v*")):
            tok = ROOT / "data" / d.name / "tokenizer.json"
            if (d / "best.pt").exists() and tok.exists():
                found[int(d.name[1:])] = {
                    "model": d / "best.pt",
                    "tok": tok,
                    "probe": ROOT / "results" / "probes" / f"board_probe_{d.name}.pt",
                }
    return found


@st.cache_resource
def load(vocab: int):
    paths = available_models()[vocab]
    tok = BPETokenizer.load(paths["tok"])
    model = load_checkpoint(paths["model"])
    probe = None
    if paths["probe"].exists():
        sd = torch.load(paths["probe"], map_location="cpu")
        probe = (
            sd["layer"],
            LinearProbe(
                sd["weight"], sd["bias"], sd["mean"], sd["std"], sd["n_heads"], sd["n_classes"]
            ),
        )
    return tok, model, Sampler(model, tok), probe


@st.cache_data
def test_openings(n: int = 200) -> list[dict]:
    p = ROOT / "models" / "sample_games.jsonl"
    if not p.exists():
        p = ROOT / "data" / "games" / "test.jsonl"
    if not p.exists():
        return []
    with open(p) as f:
        return [json.loads(line) for _, line in zip(range(n), f, strict=False)]


MODELS = available_models()
st.title("♞ Chess as a language")
st.caption(
    "A tiny (under 500K-parameter) transformer that read 72,885 grandmaster games as plain "
    "text and was never told the rules. Every move it writes is checked by a real chess engine."
)
if not MODELS:
    st.warning(
        "No trained model found. Run `notebooks/colab_sweep.ipynb`, then "
        "`python scripts/export_models.py`, or train locally with `scripts/train.py`."
    )
    st.stop()

with st.sidebar:
    st.header("Model")
    vocab = st.selectbox(
        "Tokenizer vocabulary",
        list(MODELS),
        index=min(1, len(MODELS) - 1),
        format_func=lambda v: f"{v} tokens" + (" (bytes only)" if v == 257 else ""),
    )
    temperature = st.slider(
        "Sampling temperature", 0.0, 1.5, 0.0, 0.1, help="0 = always the model's most likely move"
    )
    tok, model, sampler, probe = load(vocab)
    st.caption(f"{count_params(model):,} parameters · context {model.context_length} tokens")


def board_svg(board: chess.Board, size: int = 420, **kw) -> str:
    lastmove = board.move_stack[-1] if board.move_stack else None
    return chess.svg.board(board, size=size, lastmove=lastmove, **kw)


def show_svg(svg: str) -> None:
    st.markdown(
        f'<div style="display:flex;justify-content:center">{svg}</div>', unsafe_allow_html=True
    )


def parse_opening(text: str) -> list[str]:
    """Legal prefix of a typed movetext (stops at the first bad move)."""
    board, out = chess.Board(), []
    for san in raw_san_tokens(text)[0]:
        try:
            move = board.parse_san(san)
        except ValueError:
            break
        out.append(board.san(move))
        board.push(move)
    return out


def model_move(sans: list[str], board: chess.Board, tries: int = 5):
    """Ask the model for a move. Returns (san, status, attempts)."""
    gen = torch.Generator().manual_seed(random.randrange(10**9))
    attempts = []
    for i in range(tries):
        t = temperature if i == 0 else max(temperature, 1.0)
        mv = sampler.sample_move(prefix_text(sans, len(sans)), t, gen)
        attempts.append(mv)
        if is_legal(board, mv):
            return board.san(board.parse_san(mv)), ("legal" if i == 0 else "retry"), attempts
    return board.san(random.choice(list(board.legal_moves))), "fallback", attempts


tab_play, tab_watch, tab_probe, tab_att, tab_tok, tab_results = st.tabs(
    ["Play it", "Watch it play", "Inside the model", "Attention", "Tokenizer", "Results"]
)

# --------------------------------------------------------------------------
# play
# --------------------------------------------------------------------------
with tab_play:
    ss = st.session_state
    if "sans" not in ss:
        ss.sans, ss.log, ss.color = [], [], "White"
    c1, c2 = st.columns([1.1, 1])
    with c2:
        color = st.radio("You play", ["White", "Black"], horizontal=True, key="color_pick")
        if st.button("New game") or color != ss.color:
            ss.sans, ss.log, ss.color = [], [], color
        board = chess.Board()
        for s in ss.sans:
            board.push_san(s)
        model_turn = (board.turn == chess.WHITE) != (ss.color == "White")
        if not board.is_game_over() and model_turn:
            san, status, attempts = model_move(ss.sans, board)
            ss.sans.append(san)
            ss.log.append(("model", san, status, attempts))
            board.push_san(san)
        if board.is_game_over():
            st.success(f"Game over: {board.result()}")
        else:
            legal = sorted(board.san(m) for m in board.legal_moves)
            mv = st.selectbox("Your move", legal, index=None, placeholder="choose a legal move")
            if mv and st.button("Play move", type="primary"):
                ss.sans.append(mv)
                ss.log.append(("you", mv, "legal", []))
                st.rerun()
        st.markdown("**Model's moves**")
        n_model = [x for x in ss.log if x[0] == "model"]
        if n_model:
            first_try = sum(x[2] == "legal" for x in n_model)
            st.caption(f"legal on first try: {first_try}/{len(n_model)}")
        for who, san, status, attempts in reversed(ss.log[-12:]):
            if who == "you":
                continue
            colr = {"legal": LEGAL, "retry": FALLBACK, "fallback": ILLEGAL}[status]
            tried = ", ".join(a or "(nothing)" for a in attempts[:-1] if a != san)
            note = {
                "legal": "legal",
                "retry": f"after illegal tries: {tried}",
                "fallback": f"all tries illegal ({tried}) -- random legal move",
            }[status]
            st.markdown(
                f'<span class="mv" style="background:{colr}22;color:{colr}">{san}</span>'
                f' <span style="color:{MUTED};font-size:0.85rem">{html.escape(note)}</span>',
                unsafe_allow_html=True,
            )
    with c1:
        show_svg(board_svg(board, flipped=ss.color == "Black"))
        st.code(canonical_movetext(ss.sans, "*")[:-2] or "(start)", language=None)

# --------------------------------------------------------------------------
# watch
# --------------------------------------------------------------------------
with tab_watch:
    st.markdown(
        "Give the model a real opening and let it play **both sides** until it makes its "
        "first illegal move."
    )
    games = test_openings()
    c1, c2 = st.columns([1, 1.1])
    with c1:
        if games:
            gi = st.number_input("Held-out test game #", 0, len(games) - 1, 0)
            k = st.slider("Real moves to show it first (plies)", 0, 30, 10)
            start = games[gi]["sans"][:k]
        else:
            start = parse_opening(st.text_input("Opening", "1.e4 e5 2.Nf3 Nc6"))
        max_plies = st.slider("Max plies to generate", 10, 120, 60)
        go_btn = st.button("Let it play", type="primary")
    if go_btn:
        board = chess.Board()
        for s in start:
            board.push_san(s)
        sans, spans, stop = list(start), [], None
        gen = torch.Generator().manual_seed(0)
        for _ in range(max_plies):
            if board.is_game_over():
                stop = f"game over ({board.result()})"
                break
            mv = sampler.sample_move(prefix_text(sans, len(sans)), temperature, gen)
            if not is_legal(board, mv):
                stop = f"illegal move: {mv or '(no move)'}"
                spans.append(
                    f'<span class="mv" style="background:{ILLEGAL}22;color:{ILLEGAL}">'
                    f"{html.escape(mv or '∅')}</span>"
                )
                break
            m = board.parse_san(mv)
            sans.append(board.san(m))
            board.push(m)
            spans.append(f'<span class="mv" style="color:{LEGAL}">{html.escape(sans[-1])}</span>')
        with c2:
            show_svg(board_svg(board))
        n = len(sans) - len(start)
        st.metric("Legal moves in a row", n, help=stop or "reached the limit")
        prompt = " ".join(f"{i // 2 + 1}.{s}" if i % 2 == 0 else s for i, s in enumerate(start))
        st.markdown(
            f'<span style="color:{MUTED}">{html.escape(prompt)}</span> ' + " ".join(spans),
            unsafe_allow_html=True,
        )
        if stop:
            st.caption(f"Stopped: {stop}")

# --------------------------------------------------------------------------
# probe
# --------------------------------------------------------------------------
PIECES = (
    [None]
    + [chess.Piece(t, True) for t in range(1, 7)]
    + [chess.Piece(t, False) for t in range(1, 7)]
)

with tab_probe:
    st.markdown(
        "A **linear probe** reads the model's internal activations after each move and "
        "tries to reconstruct the board. If it can, the model is tracking the position "
        "itself, not just memorising text."
    )
    if probe is None:
        st.info("No board probe for this model yet. Run `scripts/run_probes.py`.")
    else:
        layer, pr = probe
        games = test_openings()
        source = st.radio(
            "Position from", ["Held-out test game", "Your game in 'Play it'"], horizontal=True
        )
        if source.startswith("Your") and st.session_state.get("sans"):
            sans = list(st.session_state.sans)
        else:
            gi = st.number_input("Test game #", 0, max(0, len(games) - 1), 3, key="probe_game")
            sans = games[gi]["sans"] if games else []
        if not sans:
            st.info("Play a few moves first.")
        else:
            k = st.slider("After ply", 1, len(sans), min(24, len(sans))) - 1
            g = {"sans": sans[: k + 1], "movetext": canonical_movetext(sans[: k + 1], "").rstrip()}
            gf = extract_features(model, tok, [g], plies_per_game=None)
            pred = pr.logits(gf.feats[k : k + 1, layer]).argmax(-1)[0].numpy()
            board = chess.Board()
            for s in sans[: k + 1]:
                board.push_san(s)
            truth = square_classes(board, relative=True)
            mover = not board.turn
            decoded = chess.Board(None)
            for sq in range(64):
                c = int(pred[sq])
                if c:
                    mine = c <= 6
                    decoded.set_piece_at(
                        sq, chess.Piece((c - 1) % 6 + 1, mover if mine else not mover)
                    )
            wrong = {sq: "#c2384a88" for sq in range(64) if pred[sq] != truth[sq]}
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Real board**")
                show_svg(board_svg(board, 380))
            with c2:
                st.markdown(f"**What the probe reads from layer {layer}** (red = wrong)")
                show_svg(chess.svg.board(decoded, size=380, fill=wrong))
            acc = float((pred == truth).mean())
            occ = truth != 0
            st.metric(
                "Squares correct",
                f"{acc:.0%}",
                help=f"occupied squares: {(pred[occ] == truth[occ]).mean():.0%}",
            )

# --------------------------------------------------------------------------
# attention
# --------------------------------------------------------------------------
with tab_att:
    st.markdown(
        "Each layer has several **attention heads**. When the model reads a move, every head "
        "decides which earlier moves to look back at. Arrows on the board point from the squares "
        "of the moves it looks at most to the move being read."
    )
    games = test_openings()
    if not games:
        st.info("No games available.")
    else:
        c1, c2, c3 = st.columns(3)
        gi = c1.number_input("Test game #", 0, len(games) - 1, 0, key="att_game")
        layer_i = c2.selectbox("Layer", list(range(len(model.blocks))))
        n_heads = model.blocks[0].attn.n_heads
        head_opt = ["average of heads"] + [f"head {h}" for h in range(n_heads)]
        head_sel = c3.selectbox("Head", head_opt)
        sans_all = games[gi]["sans"]
        # keep the game inside the context window
        ids_all = tok.encode(canonical_movetext(sans_all, "").rstrip())
        n_fit = len(sans_all)
        while len(ids_all) > model.context_length and n_fit > 2:
            n_fit = int(n_fit * 0.9)
            ids_all = tok.encode(canonical_movetext(sans_all[:n_fit], "").rstrip())
        sans_a = sans_all[:n_fit]
        q = st.slider("Move being read (ply)", 2, len(sans_a), min(20, len(sans_a))) - 1
        att = attention_maps(model, torch.tensor([ids_all]))
        ma = move_attention(att, tok, ids_all, sans_a)  # (L, H, n, n)
        A = (
            ma[layer_i].mean(0)
            if head_sel.startswith("average")
            else ma[layer_i, int(head_sel[5:])]
        )
        row = A[q, : q + 1]

        board = chess.Board()
        moves = []
        for s_ in sans_a[: q + 1]:
            m_ = board.parse_san(s_)
            moves.append(m_)
            board.push(m_)
        top = [int(k) for k in np.argsort(row[:q])[::-1][:4] if row[k] > 0.02]
        arrows = [
            chess.svg.Arrow(
                moves[k].to_square,
                moves[q].to_square,
                color=f"#0f766e{int(55 + 200 * row[k] / max(row[:q].max(), 1e-6)):02x}",
            )
            for k in top
            if moves[k].to_square != moves[q].to_square
        ]

        def label(k):
            return f"{k // 2 + 1}{'.' if k % 2 == 0 else '...'}{sans_a[k]}"

        cb, ch = st.columns([1, 1.2])
        with cb:
            show_svg(chess.svg.board(board, size=400, lastmove=moves[q], arrows=arrows))

            st.markdown(f"Reading **{label(q)}**. It looks most at:")
            for k in top:
                st.markdown(f"- {label(k)} · {row[k]:.0%}")
            st.caption(f"Itself: {row[q]:.0%}")
        with ch:
            lo = max(0, q - 23)
            sub = A[lo : q + 1, lo : q + 1]
            names = [label(k) for k in range(lo, q + 1)]
            df = pd.DataFrame(
                [
                    (names[i], names[j], float(sub[i, j]))
                    for i in range(len(names))
                    for j in range(i + 1)
                ],
                columns=["reading", "looks at", "weight"],
            )
            chart = (
                alt.Chart(df)
                .mark_rect()
                .encode(
                    x=alt.X("looks at:N", sort=names, title="earlier move"),
                    y=alt.Y("reading:N", sort=names, title="move being read"),
                    color=alt.Color(
                        "weight:Q",
                        scale=alt.Scale(scheme="tealblues"),
                        legend=alt.Legend(format=".0%"),
                    ),
                    tooltip=["reading", "looks at", alt.Tooltip("weight:Q", format=".1%")],
                )
                .properties(height=430, title=f"Layer {layer_i}, {head_sel} (last 24 moves)")
            )
            st.altair_chart(chart, use_container_width=True)
        prev_same = [k for k in range(q) if moves[k].to_square == moves[q].from_square]
        if prev_same:
            k = prev_same[-1]
            st.caption(
                f"The piece that just moved last arrived with {label(k)}; "
                f"this head gives that move {row[k]:.0%} of its attention."
            )

# --------------------------------------------------------------------------
# tokenizer
# --------------------------------------------------------------------------
with tab_tok:
    st.markdown(
        "The same text, cut into tokens by each vocabulary. Bigger vocabularies learn "
        "chess-specific chunks, so more moves fit in the model's 256-token window."
    )
    text = st.text_input("Movetext", "1.e4 e5 2.Nf3 Nc6 3.Bb5 a6 4.Ba4 Nf6 5.O-O Be7 6.Re1 b5")
    for v in MODELS:
        t = BPETokenizer.load(MODELS[v]["tok"])
        ids = t.encode(text)
        spans = "".join(
            f'<span class="tok" style="background:{TOKEN_COLORS[i % len(TOKEN_COLORS)]}">'
            f"{html.escape(t.token_str(x))}</span>"
            for i, x in enumerate(ids)
        )
        st.markdown(
            f"**vocab {v}** · {len(ids)} tokens · "
            f"{len(text.encode()) / max(1, len(ids)):.2f} bytes/token",
            unsafe_allow_html=True,
        )
        st.markdown(spans, unsafe_allow_html=True)

# --------------------------------------------------------------------------
# results
# --------------------------------------------------------------------------
with tab_results:
    sweep = ROOT / "results" / "sweep.json"
    if sweep.exists():
        rows = sorted(json.loads(sweep.read_text()), key=lambda r: r["vocab_size"])
        st.dataframe(
            {
                "vocab": [r["vocab_size"] for r in rows],
                "bits / byte": [round(r["best_val_bpb"], 4) for r in rows],
                "legal-move rate": [round(r["legal_rate"], 3) for r in rows],
                "exact match": [round(r["exact_match_rate"], 3) for r in rows],
                "legal plies alone": [round(r["mean_free_legal_plies"], 1) for r in rows],
                "bytes / token": [round(r["bytes_per_token"], 2) for r in rows],
            },
            hide_index=True,
        )
    for fig in sorted((ROOT / "figures").glob("*.png")):
        st.image(str(fig), use_container_width=True)
    if not sweep.exists():
        st.info("Results appear here after the Colab sweep pushes results/ and figures/.")

st.caption(
    f"chesslm · vocab {vocab} · "
    + (f"board probe from layer {probe[0]}" if probe else "no board probe yet")
)
