"""
tokenizer.py — Vocabulary building, encoding, and decoding.

Scans a directory of source files, builds a character-level vocabulary,
and serializes it to a JSON file for reuse across training and inference.
"""

import os
import json
from typing import List, Dict, Optional


class CharTokenizer:
    """Character-level tokenizer with vocabulary persistence."""

    VOCAB_FILE = "vocab.json"

    def __init__(self):
        self.char2idx: Dict[str, int] = {}
        self.idx2char: Dict[int, str] = {}
        self.vocab_size: int = 0

    # ------------------------------------------------------------------
    # Vocabulary construction
    # ------------------------------------------------------------------

    def build_from_directory(self, directory: str, extensions: Optional[List[str]] = None) -> str:
        """
        Scan all files in `directory` matching `extensions`, collect every
        unique character, assign integer IDs, and return the raw corpus text.
        """
        if extensions is None:
            extensions = [".py", ".js", ".ts", ".txt"]

        corpus_parts: List[str] = []
        file_count = 0

        for root, _, files in os.walk(directory):
            for fname in sorted(files):
                if any(fname.endswith(ext) for ext in extensions):
                    fpath = os.path.join(root, fname)
                    try:
                        with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                            text = f.read()
                            corpus_parts.append(text)
                            file_count += 1
                    except OSError as e:
                        print(f"[tokenizer] Warning: could not read {fpath}: {e}")

        if file_count == 0:
            raise ValueError(
                f"No files with extensions {extensions} found in '{directory}'. "
                "Add training source files and retry."
            )

        corpus = "\n".join(corpus_parts)
        print(f"[tokenizer] Scanned {file_count} file(s), {len(corpus):,} characters total.")

        chars = sorted(set(corpus))
        self.char2idx = {ch: idx for idx, ch in enumerate(chars)}
        self.idx2char = {idx: ch for ch, idx in self.char2idx.items()}
        self.vocab_size = len(chars)

        print(f"[tokenizer] Vocabulary size: {self.vocab_size} unique characters.")
        return corpus

    def build_from_text(self, text: str) -> None:
        """Build vocabulary directly from a raw string (useful for testing)."""
        chars = sorted(set(text))
        self.char2idx = {ch: idx for idx, ch in enumerate(chars)}
        self.idx2char = {idx: ch for ch, idx in self.char2idx.items()}
        self.vocab_size = len(chars)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, path: str = VOCAB_FILE) -> None:
        """Persist vocabulary mappings to a JSON file."""
        payload = {
            "char2idx": self.char2idx,
            "vocab_size": self.vocab_size,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"[tokenizer] Vocabulary saved to '{path}'.")

    def load(self, path: str = VOCAB_FILE) -> None:
        """Load vocabulary mappings from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        self.char2idx = payload["char2idx"]
        self.idx2char = {int(v): k for k, v in self.char2idx.items()}
        self.vocab_size = payload["vocab_size"]
        print(f"[tokenizer] Vocabulary loaded from '{path}' ({self.vocab_size} tokens).")

    # ------------------------------------------------------------------
    # Encoding / decoding
    # ------------------------------------------------------------------

    def encode(self, text: str) -> List[int]:
        """Convert a string to a list of integer token IDs."""
        unk = self.char2idx.get("\ufffd", 0)
        return [self.char2idx.get(ch, unk) for ch in text]

    def decode(self, ids: List[int]) -> str:
        """Convert a list of integer token IDs back to a string."""
        return "".join(self.idx2char.get(i, "\ufffd") for i in ids)
