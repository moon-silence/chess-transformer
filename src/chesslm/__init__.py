"""chesslm — a from-scratch transformer trained on chess games as text.

The CSE 142 transformer stack (byte-level BPE, RoPE attention, SwiGLU,
RMSNorm, numerically stable cross-entropy) applied unchanged to grandmaster
games, and used to ask three questions:

1. **Tokenization** -- how does the BPE vocabulary size change what the model learns?
2. **Rules** -- does a model trained only on move text learn to play legal chess?
3. **Outcome** -- can it tell from the moves who is going to win?

Quick start::

    from chesslm import BPETokenizer, Sampler, load_checkpoint, teacher_forced_legality

    tok = BPETokenizer.load("data/v512/tokenizer.json")
    model = load_checkpoint("runs/v512/best.pt")
    sampler = Sampler(model, tok)
    print(sampler.continue_text("1.e4 e5 2.", max_new_bytes=60))
"""

from .data import build_game_splits, build_tokenized, load_games, load_tokens
from .legality import (
    LegalityReport,
    Sampler,
    free_running_legal_plies,
    prefix_text,
    teacher_forced_legality,
)
from .model import TransformerLM
from .pgn import Game, canonical_movetext, parse_pgn_text, split_movetext, validate
from .tokenizer import BPETokenizer, train_bpe, train_bpe_from_text
from .train import TrainConfig, build_model, count_params, load_checkpoint, nats_to_bpb, train

__version__ = "0.1.0"

__all__ = [
    "BPETokenizer",
    "Game",
    "LegalityReport",
    "Sampler",
    "TrainConfig",
    "TransformerLM",
    "build_game_splits",
    "build_model",
    "build_tokenized",
    "canonical_movetext",
    "count_params",
    "free_running_legal_plies",
    "load_checkpoint",
    "load_games",
    "load_tokens",
    "nats_to_bpb",
    "parse_pgn_text",
    "prefix_text",
    "split_movetext",
    "teacher_forced_legality",
    "train",
    "train_bpe",
    "train_bpe_from_text",
    "validate",
]
