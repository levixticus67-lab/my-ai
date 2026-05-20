"""
bpe_tokenizer.py — Pure-Python Byte Pair Encoding (BPE) tokenizer.

Builds a subword vocabulary by iteratively merging the most frequent byte-pair
in the corpus.  No external dependencies beyond the standard library.

BPE produces shorter token sequences than character-level for natural language
and code (common keywords become single tokens), which makes training on a
fixed context window more effective.

Usage
-----
from bpe_tokenizer import BPETokenizer

tok = BPETokenizer()
tok.train(corpus_text, vocab_size=1000)
tok.save("bpe_vocab.json")

# Later:
tok2 = BPETokenizer()
tok2.load("bpe_vocab.json")
ids  = tok2.encode("def fibonacci(n):")
text = tok2.decode(ids)

CLI
---
python bpe_tokenizer.py --train training_data/ --vocab-size 1000 --out bpe_vocab.json
python bpe_tokenizer.py --test "def fibonacci(n):"  --vocab bpe_vocab.json
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# BPETokenizer
# ---------------------------------------------------------------------------

class BPETokenizer:
    """
    Character-initialised Byte Pair Encoding tokenizer.

    Vocabulary layout:
      0 … 255         : raw bytes (always present — handles any UTF-8 input)
      256 … vocab_size: learned merge tokens

    This ensures the tokenizer never encounters an unknown token: any input
    can be decomposed to bytes in the worst case.
    """

    _SPECIAL_PAD = "<pad>"
    _SPECIAL_UNK = "<unk>"
    _BASE_OFFSET  = 2          # 0=pad, 1=unk, then bytes start at 2

    def __init__(self):
        # str token → int id
        self.token2id: Dict[str, int] = {}
        # int id → str token
        self.id2token: Dict[int, str] = {}
        # ordered list of merge rules: (a, b) → ab
        self.merges: List[Tuple[str, str]] = []
        self.vocab_size: int = 0
        self._trained: bool = False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, corpus: str, vocab_size: int = 1000) -> None:
        """
        Learn BPE merge rules from `corpus` until `vocab_size` is reached.

        Args:
            corpus:     Raw training text.
            vocab_size: Target vocabulary size (including base byte tokens).
        """
        # --- Initialise vocabulary with all unique bytes in the corpus ----
        unique_bytes = sorted(set(corpus.encode("utf-8")))
        # Build base vocab: special tokens first
        self.token2id = {
            self._SPECIAL_PAD: 0,
            self._SPECIAL_UNK: 1,
        }
        for b in unique_bytes:
            ch = chr(b) if b < 128 else f"<byte_{b}>"
            # Map the actual character (not the display name)
            char = bytes([b]).decode("latin-1")
            if char not in self.token2id:
                self.token2id[char] = len(self.token2id)

        self.id2token = {v: k for k, v in self.token2id.items()}
        self.merges   = []

        if vocab_size <= len(self.token2id):
            self.vocab_size = len(self.token2id)
            self._trained   = True
            print(
                f"[bpe] vocab_size={vocab_size} ≤ base vocab "
                f"({len(self.token2id)}). No merges needed."
            )
            return

        # --- Tokenise corpus into list-of-character-lists (one per word) --
        # We split on whitespace boundaries for efficiency; each "word" is
        # processed as an independent sequence of characters.
        words = self._corpus_to_words(corpus)
        vocab = self._build_word_vocab(words)  # word → frequency

        n_merges = vocab_size - len(self.token2id)
        print(
            f"[bpe] Starting BPE: base_vocab={len(self.token2id)}  "
            f"target={vocab_size}  merges_needed={n_merges}"
        )

        for i in range(n_merges):
            pairs = self._get_pair_counts(vocab)
            if not pairs:
                break

            best_pair = max(pairs, key=pairs.__getitem__)
            a, b      = best_pair
            new_token = a + b

            # Add to vocabulary
            new_id = len(self.token2id)
            self.token2id[new_token] = new_id
            self.id2token[new_id]    = new_token
            self.merges.append((a, b))

            # Apply merge to vocab
            vocab = self._apply_merge(vocab, best_pair, new_token)

            if (i + 1) % 100 == 0 or (i + 1) == n_merges:
                print(
                    f"[bpe]  merge {i+1:>5}/{n_merges}  "
                    f"'{a}' + '{b}' → '{new_token}'  "
                    f"(freq={pairs[best_pair]:,})"
                )

        self.vocab_size = len(self.token2id)
        self._trained   = True
        print(f"[bpe] Training complete. Final vocab_size={self.vocab_size}.")

    # ------------------------------------------------------------------
    # Encoding / decoding
    # ------------------------------------------------------------------

    def encode(self, text: str) -> List[int]:
        """Convert text → list of integer token IDs using learned merges."""
        if not self._trained:
            raise RuntimeError("Call train() or load() before encode().")

        # Start with character-level tokenisation
        tokens = list(text)

        # Greedily apply merges in order
        for a, b in self.merges:
            i = 0
            merged = []
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == a and tokens[i + 1] == b:
                    merged.append(a + b)
                    i += 2
                else:
                    merged.append(tokens[i])
                    i += 1
            tokens = merged

        # Map tokens to IDs (unknown → 1)
        unk_id = self.token2id.get(self._SPECIAL_UNK, 1)
        return [self.token2id.get(t, unk_id) for t in tokens]

    def decode(self, ids: List[int]) -> str:
        """Convert list of token IDs back to a string."""
        if not self._trained:
            raise RuntimeError("Call train() or load() before decode().")
        return "".join(self.id2token.get(i, "") for i in ids)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, path: str = "bpe_vocab.json") -> None:
        """Save vocabulary and merge rules to a JSON file."""
        payload = {
            "vocab_size": self.vocab_size,
            "token2id":   self.token2id,
            "merges":     [[a, b] for a, b in self.merges],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"[bpe] Saved to '{path}'  (vocab_size={self.vocab_size}, merges={len(self.merges)}).")

    def load(self, path: str = "bpe_vocab.json") -> None:
        """Load vocabulary and merge rules from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        self.token2id  = payload["token2id"]
        self.id2token  = {int(v): k for k, v in self.token2id.items()}
        self.merges    = [tuple(pair) for pair in payload["merges"]]
        self.vocab_size = payload["vocab_size"]
        self._trained   = True
        print(
            f"[bpe] Loaded from '{path}'  "
            f"(vocab_size={self.vocab_size}, merges={len(self.merges)})."
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _corpus_to_words(corpus: str) -> List[str]:
        """Split corpus into whitespace-separated tokens for BPE processing."""
        # Keep whitespace attached to words for round-trip fidelity
        return re.findall(r"\S+|\s+", corpus)

    @staticmethod
    def _build_word_vocab(words: List[str]) -> Dict[Tuple[str, ...], int]:
        """Build word frequency dict where each word is a tuple of chars."""
        counts: Counter = Counter()
        for w in words:
            counts[tuple(w)] += 1
        return dict(counts)

    @staticmethod
    def _get_pair_counts(
        vocab: Dict[Tuple[str, ...], int]
    ) -> Dict[Tuple[str, str], int]:
        """Count bigram frequencies across all words in vocab."""
        pairs: Counter = Counter()
        for word, freq in vocab.items():
            for i in range(len(word) - 1):
                pairs[(word[i], word[i + 1])] += freq
        return dict(pairs)

    @staticmethod
    def _apply_merge(
        vocab:     Dict[Tuple[str, ...], int],
        pair:      Tuple[str, str],
        new_token: str,
    ) -> Dict[Tuple[str, ...], int]:
        """Merge all occurrences of `pair` in the vocab."""
        a, b   = pair
        result = {}
        for word, freq in vocab.items():
            merged: List[str] = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
                    merged.append(new_token)
                    i += 2
                else:
                    merged.append(word[i])
                    i += 1
            result[tuple(merged)] = freq
        return result


# ---------------------------------------------------------------------------
# Convenience factory — same interface as CharTokenizer's build_from_directory
# ---------------------------------------------------------------------------

def build_bpe_from_directory(
    directory:  str,
    vocab_size: int = 1000,
    extensions: Optional[List[str]] = None,
) -> Tuple[BPETokenizer, str]:
    """
    Scan `directory` for source files, train a BPE tokenizer, and return
    (tokenizer, corpus_string).
    """
    if extensions is None:
        extensions = [".py", ".js", ".ts", ".txt"]

    parts: List[str] = []
    count = 0
    for root, _, files in os.walk(directory):
        for fname in sorted(files):
            if any(fname.endswith(ext) for ext in extensions):
                fpath = os.path.join(root, fname)
                try:
                    with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                        parts.append(f.read())
                        count += 1
                except OSError:
                    pass

    if count == 0:
        raise ValueError(f"No matching files found in '{directory}'.")

    corpus = "\n".join(parts)
    print(f"[bpe] Scanned {count} file(s), {len(corpus):,} characters.")

    tok = BPETokenizer()
    tok.train(corpus, vocab_size=vocab_size)
    return tok, corpus


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="BPE tokenizer — train or test.")
    parser.add_argument("--train",      metavar="DIR",  help="Directory of training files.")
    parser.add_argument("--vocab-size", type=int, default=1000)
    parser.add_argument("--out",        default="bpe_vocab.json", help="Output vocab file.")
    parser.add_argument("--vocab",      metavar="FILE", help="Existing vocab file to load.")
    parser.add_argument("--test",       metavar="TEXT", help="Encode and decode this text.")
    args = parser.parse_args()

    tok = BPETokenizer()

    if args.train:
        _, _ = build_bpe_from_directory(args.train, vocab_size=args.vocab_size)
        tok.save(args.out)

    if args.vocab:
        tok.load(args.vocab)

    if args.test:
        if not tok._trained:
            print("Load or train a vocabulary first (--train or --vocab).")
        else:
            ids     = tok.encode(args.test)
            decoded = tok.decode(ids)
            print(f"Input   : {args.test!r}")
            print(f"Tokens  : {[tok.id2token.get(i, '?') for i in ids]}")
            print(f"IDs     : {ids}")
            print(f"Decoded : {decoded!r}")
            print(f"Ratio   : {len(ids)}/{len(args.test)} tokens/chars")
