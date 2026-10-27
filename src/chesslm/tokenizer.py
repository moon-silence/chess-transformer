"""Byte-level BPE, vectorised.

This is the CSE 142 tokenizer (kept verbatim in ``_reference_tokenizer.py``)
re-implemented with NumPy so it scales from a 2 MB prototype corpus to the
full ~30 MB chess corpus. The algorithm is unchanged:

* IDs 0-255 are raw bytes, merges get IDs from 256 upward, special tokens
  get the highest IDs.
* Each training step merges the most frequent adjacent pair, ties broken by
  the smallest ``(id1, id2)``; merges are applied non-overlapping, left to
  right.
* Encoding applies merges in rank order.

The only deliberate difference from the reference: text is split on special
tokens *before* training, so no merge can straddle a ``<|game|>`` boundary.
On text that contains no special tokens the two implementations produce
identical merges and identical encodings -- ``tests/test_tokenizer.py``
checks this.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

SENTINEL = -1  # marks a chunk boundary; never part of a mergeable pair


def _split_on_specials(text: str, specials: list[str]) -> list[str | None]:
    """Split ``text`` into ordinary chunks and special tokens.

    Returns a list whose items are either ordinary strings or, for special
    tokens, the token string wrapped as ``("special", tok)``.
    Longest special token wins when several match at the same position,
    matching the reference encoder.
    """
    if not specials:
        return [text]
    pattern = "|".join(re.escape(s) for s in sorted(specials, key=len, reverse=True))
    return re.split(f"({pattern})", text)


def _merge_pair(ids: np.ndarray, a: int, b: int, new_id: int) -> np.ndarray:
    """Replace every non-overlapping (a, b), scanned left to right, with new_id."""
    if ids.size < 2:
        return ids
    hits = np.flatnonzero((ids[:-1] == a) & (ids[1:] == b))
    if hits.size == 0:
        return ids
    if a == b:
        # Runs like a a a a: matches at i, i+1, i+2 overlap. Left-to-right
        # scanning keeps the 1st, 3rd, ... match of each consecutive run.
        run_start = np.ones(hits.size, dtype=bool)
        run_start[1:] = np.diff(hits) != 1
        start_idx = np.maximum.accumulate(np.where(run_start, np.arange(hits.size), 0))
        hits = hits[(np.arange(hits.size) - start_idx) % 2 == 0]
    out = ids.copy()
    out[hits] = new_id
    keep = np.ones(out.size, dtype=bool)
    keep[hits + 1] = False
    return out[keep]


def _to_id_array(chunks: list[bytes]) -> np.ndarray:
    """Concatenate byte chunks into one int32 array with SENTINEL separators."""
    parts: list[np.ndarray] = []
    sep = np.array([SENTINEL], dtype=np.int32)
    for i, chunk in enumerate(chunks):
        if i:
            parts.append(sep)
        parts.append(np.frombuffer(chunk, dtype=np.uint8).astype(np.int32))
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.int32)


def train_bpe_from_text(
    text: str,
    vocab_size: int,
    special_tokens: list[str] | None = None,
    verbose: bool = False,
) -> tuple[dict[int, bytes], list[tuple[int, int]]]:
    """Train byte-level BPE on an in-memory string. See module docstring."""
    special_tokens = special_tokens or []
    num_merges = vocab_size - 256 - len(special_tokens)
    if num_merges < 0:
        raise ValueError(f"vocab_size={vocab_size} too small for {len(special_tokens)} specials")

    pieces = _split_on_specials(text, special_tokens)
    ordinary = [p.encode("utf-8") for p in pieces if p and p not in special_tokens]
    ids = _to_id_array(ordinary)

    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    merges: list[tuple[int, int]] = []
    base = vocab_size  # every id is < vocab_size, so a*base + b is a unique pair code

    for step in range(num_merges):
        left, right = ids[:-1].astype(np.int64), ids[1:].astype(np.int64)
        valid = (left >= 0) & (right >= 0)
        if not valid.any():
            break
        counts = np.bincount(left[valid] * base + right[valid])
        best = int(np.argmax(counts))  # first max == lexicographically smallest pair
        a, b = divmod(best, base)
        new_id = 256 + step
        merges.append((a, b))
        vocab[new_id] = vocab[a] + vocab[b]
        ids = _merge_pair(ids, a, b, new_id)
        if verbose and (step + 1) % 100 == 0:
            print(f"  merge {step + 1}/{num_merges}  corpus length {ids.size:,}")

    next_id = 256 + len(merges)
    for tok in special_tokens:
        vocab[next_id] = tok.encode("utf-8")
        next_id += 1
    return vocab, merges


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str] | None = None,
) -> tuple[dict[int, bytes], list[tuple[int, int]]]:
    """Same signature as the CSE 142 ``train_bpe``."""
    text = Path(input_path).read_text(encoding="utf-8")
    return train_bpe_from_text(text, vocab_size, special_tokens)


class BPETokenizer:
    """Byte-level BPE tokenizer with a vectorised encoder."""

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[int, int]],
        special_tokens: list[str] | None = None,
    ) -> None:
        self.vocab = vocab
        self.merges = [tuple(m) for m in merges]
        self.special_tokens = special_tokens or []
        self.bytes_to_id = {b: i for i, b in vocab.items()}
        self.merge_to_id = {
            pair: self.bytes_to_id[vocab[pair[0]] + vocab[pair[1]]] for pair in self.merges
        }
        self.special_to_id = {
            t: self.bytes_to_id[t.encode("utf-8")]
            for t in self.special_tokens
            if t.encode("utf-8") in self.bytes_to_id
        }

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    # -- encoding ---------------------------------------------------------
    def _encode_ordinary_batch(self, chunks: list[str]) -> list[np.ndarray]:
        ids = _to_id_array([c.encode("utf-8") for c in chunks])
        for pair in self.merges:
            ids = _merge_pair(ids, pair[0], pair[1], self.merge_to_id[pair])
        bounds = np.flatnonzero(ids == SENTINEL)
        return np.split(ids, bounds) if bounds.size else [ids]

    def encode_to_array(self, text: str) -> np.ndarray:
        """Encode a (possibly very large) string to a 1-D int32 array."""
        if text == "":
            return np.zeros(0, dtype=np.int32)
        pieces = [p for p in _split_on_specials(text, list(self.special_to_id)) if p]
        ordinary = [p for p in pieces if p not in self.special_to_id]
        encoded = iter(self._encode_ordinary_batch(ordinary)) if ordinary else iter(())
        out: list[np.ndarray] = []
        for p in pieces:
            if p in self.special_to_id:
                out.append(np.array([self.special_to_id[p]], dtype=np.int32))
            else:
                arr = next(encoded)
                out.append(arr[arr != SENTINEL])
        return np.concatenate(out) if out else np.zeros(0, dtype=np.int32)

    def encode(self, text: str) -> list[int]:
        return self.encode_to_array(text).tolist()

    def decode(self, ids: list[int]) -> str:
        return b"".join(self.vocab[int(i)] for i in ids).decode("utf-8", errors="replace")

    def token_str(self, i: int) -> str:
        return self.vocab[int(i)].decode("utf-8", errors="replace")

    # -- persistence --------------------------------------------------------
    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(
                {
                    "vocab": {str(k): list(v) for k, v in self.vocab.items()},
                    "merges": [list(m) for m in self.merges],
                    "special_tokens": self.special_tokens,
                }
            )
        )

    @classmethod
    def load(cls, path: str | Path) -> BPETokenizer:
        raw = json.loads(Path(path).read_text())
        vocab = {int(k): bytes(v) for k, v in raw["vocab"].items()}
        return cls(vocab, [tuple(m) for m in raw["merges"]], raw["special_tokens"])
