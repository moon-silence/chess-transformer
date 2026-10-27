"""The vectorised BPE must be the CSE 142 BPE, only faster."""

from __future__ import annotations

import random

import pytest

from chesslm import _reference_tokenizer as ref
from chesslm.tokenizer import BPETokenizer, _merge_pair, train_bpe, train_bpe_from_text

ALPHABETS = ["ab", "abc", "aaab", "1.e4 e5 2.Nf3 Nc6 x+#O-", "é漢 a"]


def _random_text(rng: random.Random, alphabet: str, n: int) -> str:
    return "".join(rng.choice(alphabet) for _ in range(n))


@pytest.mark.parametrize("seed", range(25))
def test_training_matches_reference_exactly(seed, tmp_path):
    rng = random.Random(seed)
    alphabet = ALPHABETS[seed % len(ALPHABETS)]
    text = _random_text(rng, alphabet, rng.randint(0, 500))
    vocab_size = 256 + rng.randint(0, 50)
    path = tmp_path / "corpus.txt"
    path.write_text(text, encoding="utf-8")

    v_ref, m_ref = ref.train_bpe(str(path), vocab_size)
    v_new, m_new = train_bpe(str(path), vocab_size)
    assert m_new == m_ref
    assert v_new == v_ref


@pytest.mark.parametrize("seed", range(15))
def test_encoding_matches_reference_exactly(seed):
    rng = random.Random(100 + seed)
    alphabet = ALPHABETS[seed % len(ALPHABETS)]
    vocab, merges = train_bpe_from_text(_random_text(rng, alphabet, 600), 300)
    probe = _random_text(rng, alphabet, 300)
    assert BPETokenizer(vocab, merges).encode(probe) == ref.BPETokenizer(vocab, merges).encode(
        probe
    )


def test_overlapping_merge_is_left_to_right():
    import numpy as np

    out = _merge_pair(np.array([1, 1, 1, 1, 1], dtype=np.int32), 1, 1, 9)
    assert out.tolist() == [9, 9, 1]


def test_special_tokens_are_atomic_and_round_trip():
    text = "1.e4 e5 1-0 <|game|> 1.d4 d5 1/2-1/2 <|game|> 1.c4 0-1"
    vocab, merges = train_bpe_from_text(text * 20, 300, ["<|game|>"])
    tok = BPETokenizer(vocab, merges, ["<|game|>"])
    ids = tok.encode(text)
    sep = tok.special_to_id["<|game|>"]
    assert ids.count(sep) == 2
    assert sep == max(vocab)  # specials take the highest id
    assert tok.decode(ids) == text
    # text is split on specials before training, so no merge contains their bytes
    for i in range(256, 256 + len(merges)):
        assert "<" not in tok.token_str(i) and "|" not in tok.token_str(i)


def test_no_merges_when_vocab_is_bytes_plus_specials():
    vocab, merges = train_bpe_from_text("1.e4 e5 2.Nf3", 257, ["<|game|>"])
    assert merges == []
    tok = BPETokenizer(vocab, merges, ["<|game|>"])
    assert tok.encode("e4") == [ord("e"), ord("4")]


def test_save_load_round_trip(tmp_path):
    vocab, merges = train_bpe_from_text("1.e4 e5 2.Nf3 Nc6 " * 50, 280, ["<|game|>"])
    tok = BPETokenizer(vocab, merges, ["<|game|>"])
    tok.save(tmp_path / "tok.json")
    tok2 = BPETokenizer.load(tmp_path / "tok.json")
    s = "1.e4 e5 <|game|> 2.Nf3"
    assert tok2.encode(s) == tok.encode(s)
    assert tok2.decode(tok2.encode(s)) == s


def test_empty_and_unicode():
    vocab, merges = train_bpe_from_text("ab" * 10, 260)
    tok = BPETokenizer(vocab, merges)
    assert tok.encode("") == []
    assert tok.decode(tok.encode("Réti ½")) == "Réti ½"
