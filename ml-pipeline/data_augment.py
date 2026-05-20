"""
data_augment.py — Training data augmentation for code corpora.

Generates additional training examples from existing source files by applying
semantics-preserving transformations.  Effectively multiplies your training
corpus without needing new files.

Augmentation techniques
-----------------------
1. variable_rename   — Replace identifiers with alternatives (x→i, n→count, …)
2. comment_strip     — Remove all inline and block comments
3. whitespace_norm   — Normalise blank lines and trailing spaces
4. indent_convert    — Toggle between 2-space and 4-space indentation
5. string_shuffle    — Swap equivalent string delimiters (' ↔ " where safe)
6. dead_code_insert  — Inject harmless pass / ... statements in empty bodies
7. docstring_strip   — Remove function/class docstrings
8. line_shuffle      — Randomly reorder independent top-level import lines

Usage
-----
from data_augment import augment_corpus, augment_directory

# Augment a single text string
extra = augment_corpus(
    corpus     = original_code,
    techniques = ["comment_strip", "whitespace_norm", "indent_convert"],
    n_copies   = 2,
    seed       = 42,
)

# Augment all files in a directory and write results alongside originals
augment_directory(
    src_dir    = "training_data",
    out_dir    = "training_data_aug",
    techniques = ["variable_rename", "comment_strip", "whitespace_norm"],
    n_copies   = 3,
)

CLI
---
python data_augment.py --src training_data/ --out training_data_aug/ --copies 3
python data_augment.py --src training_data/ --out training_data_aug/ --techniques comment_strip whitespace_norm
python data_augment.py --list-techniques
"""

from __future__ import annotations

import os
import random
import re
import shutil
from typing import Callable, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------

AugFn = Callable[[str, random.Random], str]


# ---------------------------------------------------------------------------
# 1. Variable rename
# ---------------------------------------------------------------------------

# Common short identifiers and their synonyms
_VAR_SYNONYMS: Dict[str, List[str]] = {
    "i":     ["idx", "index", "ii"],
    "j":     ["jj", "jdx", "k"],
    "k":     ["kk", "kdx", "j"],
    "n":     ["count", "num", "total", "size"],
    "m":     ["rows", "cols", "sz"],
    "x":     ["val", "value", "item"],
    "y":     ["out", "result", "res"],
    "z":     ["depth", "level", "tmp"],
    "s":     ["text", "buf", "line"],
    "lst":   ["arr", "seq", "items"],
    "arr":   ["lst", "seq", "data"],
    "res":   ["out", "result", "ret"],
    "tmp":   ["temp", "buf", "aux"],
    "node":  ["cur", "current", "nd"],
    "cur":   ["node", "current", "ptr"],
    "head":  ["start", "root", "first"],
    "tail":  ["end", "last", "rest"],
    "left":  ["lhs", "lo", "l"],
    "right": ["rhs", "hi", "r"],
}


def _variable_rename(code: str, rng: random.Random) -> str:
    """
    Replace a randomly chosen identifier with one of its synonyms throughout
    the code.  Only replaces whole-word matches to avoid breaking other names.
    """
    eligible = {k for k in _VAR_SYNONYMS if re.search(rf"\b{re.escape(k)}\b", code)}
    if not eligible:
        return code

    target  = rng.choice(sorted(eligible))
    synonym = rng.choice(_VAR_SYNONYMS[target])
    return re.sub(rf"\b{re.escape(target)}\b", synonym, code)


# ---------------------------------------------------------------------------
# 2. Comment strip
# ---------------------------------------------------------------------------

def _comment_strip(code: str, rng: random.Random) -> str:
    """Remove single-line comments (# …) and block comments (\"\"\" / ''')."""
    # Remove block comments
    code = re.sub(r'"""[\s\S]*?"""', '""""""', code)
    code = re.sub(r"'''[\s\S]*?'''", "''''''", code)
    # Remove inline # comments (don't touch string contents heuristically)
    lines = []
    for line in code.splitlines():
        # Only strip if # appears outside a likely string context
        stripped = re.sub(r'\s*#[^"\']*$', '', line)
        lines.append(stripped if stripped.strip() else line)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 3. Whitespace normalise
# ---------------------------------------------------------------------------

def _whitespace_norm(code: str, rng: random.Random) -> str:
    """
    Strip trailing whitespace from every line and collapse sequences of more
    than two consecutive blank lines into two blank lines.
    """
    lines = [ln.rstrip() for ln in code.splitlines()]
    result: List[str] = []
    blank_run = 0
    for ln in lines:
        if ln == "":
            blank_run += 1
            if blank_run <= 2:
                result.append(ln)
        else:
            blank_run = 0
            result.append(ln)
    return "\n".join(result)


# ---------------------------------------------------------------------------
# 4. Indentation convert
# ---------------------------------------------------------------------------

def _indent_convert(code: str, rng: random.Random) -> str:
    """
    Randomly swap between 2-space and 4-space indentation.
    Detects current dominant style and converts to the other.
    """
    lines = code.splitlines()

    # Detect dominant indent unit
    indent_counts: Dict[int, int] = {2: 0, 4: 0}
    for ln in lines:
        m = re.match(r"^( +)", ln)
        if m:
            n = len(m.group(1))
            if n % 4 == 0:
                indent_counts[4] += 1
            elif n % 2 == 0:
                indent_counts[2] += 1

    if indent_counts[4] >= indent_counts[2]:
        # 4 → 2 spaces
        new_lines = [re.sub(r"^(    )+", lambda m: "  " * (len(m.group()) // 4), ln)
                     for ln in lines]
    else:
        # 2 → 4 spaces
        new_lines = [re.sub(r"^(  )+", lambda m: "    " * (len(m.group()) // 2), ln)
                     for ln in lines]

    return "\n".join(new_lines)


# ---------------------------------------------------------------------------
# 5. String delimiter shuffle
# ---------------------------------------------------------------------------

def _string_shuffle(code: str, rng: random.Random) -> str:
    """
    Swap simple single-quoted strings to double-quoted and vice-versa, or
    vice-versa — only for strings that don't contain the target delimiter.
    50 % chance of each direction per call.
    """
    if rng.random() < 0.5:
        # ' → "
        return re.sub(r"'([^'\"\\]*)'", r'"\1"', code)
    else:
        # " → '
        return re.sub(r'"([^\'\"\\]*)"', r"'\1'", code)


# ---------------------------------------------------------------------------
# 6. Dead code insert
# ---------------------------------------------------------------------------

_DEAD_STMTS = ["pass", "...", "_ = None", "# no-op"]


def _dead_code_insert(code: str, rng: random.Random) -> str:
    """
    Insert a harmless statement (pass / ...) after a randomly chosen `def` or
    `class` line that is immediately followed by a docstring or body.
    Keeps the code syntactically valid.
    """
    lines = code.splitlines()
    candidates = [
        i for i, ln in enumerate(lines)
        if re.match(r"^\s*(def |class )", ln) and i + 1 < len(lines)
    ]
    if not candidates:
        return code

    idx   = rng.choice(candidates)
    stmt  = rng.choice(_DEAD_STMTS)
    # Match indentation of the next line
    next_ln = lines[idx + 1] if idx + 1 < len(lines) else ""
    m = re.match(r"^(\s*)", next_ln)
    indent = m.group(1) if m else "    "

    lines.insert(idx + 1, f"{indent}{stmt}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 7. Docstring strip
# ---------------------------------------------------------------------------

def _docstring_strip(code: str, rng: random.Random) -> str:
    """
    Remove docstrings from function and class definitions.
    Replaces them with a single `pass` so the body remains valid.
    """
    # Match triple-quoted strings that immediately follow a def/class colon line
    code = re.sub(
        r'(def [^\n]+:\n\s*)("""[\s\S]*?""")',
        r'\1pass',
        code,
    )
    code = re.sub(
        r"(def [^\n]+:\n\s*)('''[\s\S]*?''')",
        r'\1pass',
        code,
    )
    return code


# ---------------------------------------------------------------------------
# 8. Import line shuffle
# ---------------------------------------------------------------------------

def _import_shuffle(code: str, rng: random.Random) -> str:
    """
    Randomly reorder top-level `import` and `from … import` lines.
    Only lines at the very top of the file (before the first non-import line)
    are reordered.
    """
    lines = code.splitlines()
    import_block: List[str] = []
    rest_start = 0

    for i, ln in enumerate(lines):
        stripped = ln.strip()
        if stripped.startswith("import ") or stripped.startswith("from "):
            import_block.append(ln)
            rest_start = i + 1
        elif stripped == "" and i < 5:
            import_block.append(ln)
            rest_start = i + 1
        else:
            rest_start = i
            break

    if len(import_block) < 2:
        return code

    rng.shuffle(import_block)
    new_lines = import_block + lines[rest_start:]
    return "\n".join(new_lines)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

TECHNIQUES: Dict[str, Tuple[AugFn, str]] = {
    "variable_rename":  (_variable_rename,  "Replace common identifiers with synonyms"),
    "comment_strip":    (_comment_strip,    "Remove inline and block comments"),
    "whitespace_norm":  (_whitespace_norm,  "Strip trailing spaces, collapse blank lines"),
    "indent_convert":   (_indent_convert,   "Toggle between 2-space and 4-space indentation"),
    "string_shuffle":   (_string_shuffle,   "Swap single ↔ double string delimiters"),
    "dead_code_insert": (_dead_code_insert, "Insert harmless pass/... statements"),
    "docstring_strip":  (_docstring_strip,  "Remove function/class docstrings"),
    "import_shuffle":   (_import_shuffle,   "Reorder top-level import statements"),
}

ALL_TECHNIQUES = list(TECHNIQUES.keys())


# ---------------------------------------------------------------------------
# Core augmentation API
# ---------------------------------------------------------------------------

def augment_corpus(
    corpus:     str,
    techniques: Optional[List[str]] = None,
    n_copies:   int = 2,
    seed:       Optional[int] = None,
    separator:  str = "\n\n",
) -> str:
    """
    Apply augmentation transforms to `corpus` and return a concatenated string
    containing the original plus `n_copies` augmented variants.

    Args:
        corpus:     The original source text.
        techniques: List of technique names to apply.  Each copy picks a random
                    subset of techniques.  Defaults to all techniques.
        n_copies:   How many augmented copies to generate (default 2).
        seed:       Random seed for reproducibility (None = non-deterministic).
        separator:  String inserted between corpus segments (default two newlines).

    Returns:
        Original corpus + separator + augmented copies joined by separator.
    """
    if techniques is None:
        techniques = ALL_TECHNIQUES

    unknown = set(techniques) - set(TECHNIQUES)
    if unknown:
        raise ValueError(f"Unknown technique(s): {unknown}. Valid: {ALL_TECHNIQUES}")

    rng = random.Random(seed)
    parts = [corpus]

    for copy_idx in range(n_copies):
        # Each copy applies a random subset of the requested techniques
        n_apply   = rng.randint(1, max(1, len(techniques)))
        chosen    = rng.sample(techniques, n_apply)
        augmented = corpus

        for name in chosen:
            fn, _ = TECHNIQUES[name]
            try:
                augmented = fn(augmented, rng)
            except Exception:
                pass  # never crash training data generation

        parts.append(augmented)
        print(
            f"[augment] Copy {copy_idx + 1}/{n_copies}  "
            f"techniques={chosen}  "
            f"chars: {len(corpus):,} → {len(augmented):,}"
        )

    return separator.join(parts)


def augment_file(
    src_path:   str,
    out_path:   str,
    techniques: Optional[List[str]] = None,
    n_copies:   int = 2,
    seed:       Optional[int] = None,
) -> str:
    """
    Read a single source file, augment it, and write the result to `out_path`.

    Returns the augmented text.
    """
    with open(src_path, "r", encoding="utf-8", errors="replace") as f:
        corpus = f.read()

    augmented = augment_corpus(corpus, techniques=techniques, n_copies=n_copies, seed=seed)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(augmented)

    return augmented


def augment_directory(
    src_dir:    str,
    out_dir:    str,
    techniques: Optional[List[str]] = None,
    n_copies:   int = 2,
    extensions: Optional[List[str]] = None,
    seed:       Optional[int] = None,
    copy_originals: bool = True,
) -> List[str]:
    """
    Augment every matching source file in `src_dir` and write results to `out_dir`.

    Args:
        src_dir:         Source directory containing training files.
        out_dir:         Output directory for augmented files.
        techniques:      List of technique names (default: all).
        n_copies:        Augmented copies per file (default: 2).
        extensions:      File extensions to process (default: .py, .js, .ts, .txt).
        seed:            Random seed (None = non-deterministic).
        copy_originals:  Also copy original (un-augmented) files to out_dir.

    Returns:
        List of paths to all files written to out_dir.
    """
    if extensions is None:
        extensions = [".py", ".js", ".ts", ".txt"]

    os.makedirs(out_dir, exist_ok=True)
    written: List[str] = []
    file_seed = seed

    for root, _, files in os.walk(src_dir):
        for fname in sorted(files):
            if not any(fname.endswith(ext) for ext in extensions):
                continue

            src_path = os.path.join(root, fname)
            rel_path = os.path.relpath(src_path, src_dir)
            base, ext = os.path.splitext(rel_path)

            # Copy original
            if copy_originals:
                orig_out = os.path.join(out_dir, rel_path)
                os.makedirs(os.path.dirname(orig_out), exist_ok=True)
                shutil.copy2(src_path, orig_out)
                written.append(orig_out)

            # Write augmented file
            aug_name = f"{base}_aug{ext}"
            aug_out  = os.path.join(out_dir, aug_name)
            os.makedirs(os.path.dirname(aug_out), exist_ok=True)

            print(f"\n[augment] Processing '{rel_path}' → '{aug_name}'")
            augment_file(
                src_path   = src_path,
                out_path   = aug_out,
                techniques = techniques,
                n_copies   = n_copies,
                seed       = file_seed,
            )
            written.append(aug_out)

            # Vary seed per file (keeps reproducibility while varying outputs)
            if file_seed is not None:
                file_seed += 1

    print(
        f"\n[augment] Done. {len(written)} file(s) written to '{out_dir}'.\n"
        f"          Use '{out_dir}' as your --training-dir to train on augmented data."
    )
    return written


def measure_expansion(src_dir: str, out_dir: str) -> None:
    """
    Print a comparison of corpus size before and after augmentation.
    """
    def _count_chars(directory: str) -> int:
        total = 0
        for root, _, files in os.walk(directory):
            for f in files:
                try:
                    with open(os.path.join(root, f), "r", encoding="utf-8", errors="replace") as fh:
                        total += len(fh.read())
                except OSError:
                    pass
        return total

    before = _count_chars(src_dir)
    after  = _count_chars(out_dir)
    ratio  = after / max(before, 1)

    print(f"\n  Corpus expansion:")
    print(f"    Before : {before:>12,} chars  ({src_dir})")
    print(f"    After  : {after:>12,} chars  ({out_dir})")
    print(f"    Ratio  : {ratio:.2f}×\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Augment code training data without needing new files.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--src",   "-s", default="training_data",
                        help="Source directory of training files (default: training_data).")
    parser.add_argument("--out",   "-o", default="training_data_aug",
                        help="Output directory for augmented files (default: training_data_aug).")
    parser.add_argument("--copies", "-n", type=int, default=2,
                        help="Number of augmented copies per file (default: 2).")
    parser.add_argument("--techniques", nargs="+", default=None,
                        metavar="TECHNIQUE",
                        help=(
                            "Augmentation techniques to apply.\n"
                            "Run --list-techniques to see all options.\n"
                            "Default: all techniques."
                        ))
    parser.add_argument("--seed",  type=int, default=None,
                        help="Random seed for reproducibility.")
    parser.add_argument("--list-techniques", action="store_true",
                        help="Print all available augmentation techniques and exit.")
    parser.add_argument("--measure", action="store_true",
                        help="After augmentation, print corpus size expansion ratio.")
    parser.add_argument("--no-originals", action="store_true",
                        help="Do not copy original files to the output directory.")
    args = parser.parse_args()

    if args.list_techniques:
        print("\nAvailable augmentation techniques:\n")
        for name, (_, desc) in TECHNIQUES.items():
            print(f"  {name:<20}  {desc}")
        print()
        raise SystemExit(0)

    augment_directory(
        src_dir         = args.src,
        out_dir         = args.out,
        techniques      = args.techniques,
        n_copies        = args.copies,
        seed            = args.seed,
        copy_originals  = not args.no_originals,
    )

    if args.measure:
        measure_expansion(args.src, args.out)
