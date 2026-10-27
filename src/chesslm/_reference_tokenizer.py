"""Byte-level BPE tokenizer (no regex pre-tokenization)."""

from __future__ import annotations

from collections import Counter


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str] | None = None,
) -> tuple[dict[int, bytes], list[tuple[int, int]]]:
    """Train a byte-level BPE tokenizer.

    Merge order: at each step, merge the most frequent adjacent pair.
    Break ties by selecting the pair with the smallest ``(id1, id2)``
    in lexicographic (tuple) order.

    IDs 0–255 are single bytes. Merge tokens get IDs starting from 256.
    Special tokens get the highest IDs in the vocab.

    Args:
        input_path: Path to a UTF-8 text file.
        vocab_size: Target vocabulary size (>= 256 + len(special_tokens)).
        special_tokens: Optional special token strings.

    Returns:
        vocab: ``dict[int, bytes]`` mapping token ID to byte string.
        merges: ``list[tuple[int, int]]`` merge pairs in order.
    """
    special_tokens = special_tokens or []

    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Base vocabulary: single bytes 0..255.
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}

    num_merges = vocab_size - 256 - len(special_tokens)
    merges: list[tuple[int, int]] = []

    # Represent the corpus as a list of token IDs (initially raw bytes).
    ids = list(text.encode("utf-8"))

    next_id = 256
    for _ in range(max(0, num_merges)):
        # Count adjacent pairs.
        pair_counts: Counter[tuple[int, int]] = Counter()
        for a, b in zip(ids, ids[1:]):
            pair_counts[(a, b)] += 1
        if not pair_counts:
            break

        # Most frequent pair; tie-break by smallest (id1, id2) lexicographically.
        max_count = max(pair_counts.values())
        best_pair = min(p for p, c in pair_counts.items() if c == max_count)

        merges.append(best_pair)
        new_id = next_id
        vocab[new_id] = vocab[best_pair[0]] + vocab[best_pair[1]]
        next_id += 1

        # Apply the merge across the corpus (non-overlapping, left to right).
        merged: list[int] = []
        i = 0
        a, b = best_pair
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == a and ids[i + 1] == b:
                merged.append(new_id)
                i += 2
            else:
                merged.append(ids[i])
                i += 1
        ids = merged

    # Append special tokens at the highest IDs.
    for tok in special_tokens:
        vocab[next_id] = tok.encode("utf-8")
        next_id += 1

    return vocab, merges


class BPETokenizer:
    """Byte-level BPE tokenizer."""

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[int, int]],
        special_tokens: list[str] | None = None,
    ) -> None:
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []

        # Reverse map: bytes -> id.
        self.bytes_to_id: dict[bytes, int] = {b: i for i, b in vocab.items()}

        # Merge rank: pair -> order index (lower = applied first).
        self.merge_ranks: dict[tuple[int, int], int] = {
            pair: rank for rank, pair in enumerate(merges)
        }

        # Special token -> id, sorted by length (longest first) for matching.
        self.special_to_id: dict[str, int] = {}
        for tok in self.special_tokens:
            tok_bytes = tok.encode("utf-8")
            if tok_bytes in self.bytes_to_id:
                self.special_to_id[tok] = self.bytes_to_id[tok_bytes]
        self._sorted_specials = sorted(
            self.special_to_id.keys(), key=len, reverse=True
        )

    def _encode_chunk(self, text: str) -> list[int]:
        """Encode an ordinary (non-special) text chunk via BPE merges."""
        if not text:
            return []
        ids = list(text.encode("utf-8"))

        # Repeatedly apply the lowest-rank applicable merge.
        while len(ids) >= 2:
            # Find the pair present with the lowest merge rank.
            best_rank = None
            best_pos = None
            for i in range(len(ids) - 1):
                pair = (ids[i], ids[i + 1])
                rank = self.merge_ranks.get(pair)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_pos = i
            if best_rank is None:
                break

            a, b = ids[best_pos], ids[best_pos + 1]
            new_id = self.bytes_to_id[self.vocab[a] + self.vocab[b]]

            # Apply this single merge everywhere (non-overlapping, left to right).
            merged: list[int] = []
            i = 0
            while i < len(ids):
                if i < len(ids) - 1 and ids[i] == a and ids[i + 1] == b:
                    merged.append(new_id)
                    i += 2
                else:
                    merged.append(ids[i])
                    i += 1
            ids = merged
        return ids

    def encode(self, text: str) -> list[int]:
        """Encode a string into a list of token IDs."""
        if text == "":
            return []
        if not self._sorted_specials:
            return self._encode_chunk(text)

        # Split on special tokens with longest-match priority.
        out: list[int] = []
        i = 0
        n = len(text)
        while i < n:
            matched = None
            for tok in self._sorted_specials:
                if text.startswith(tok, i):
                    matched = tok
                    break
            if matched is not None:
                out.append(self.special_to_id[matched])
                i += len(matched)
            else:
                # Accumulate ordinary text until the next special token.
                j = i
                while j < n:
                    if any(text.startswith(t, j) for t in self._sorted_specials):
                        break
                    j += 1
                out.extend(self._encode_chunk(text[i:j]))
                i = j
        return out

    def decode(self, ids: list[int]) -> str:
        """Decode a list of token IDs back into a string."""
        data = b"".join(self.vocab[i] for i in ids)
        return data.decode("utf-8", errors="replace")
