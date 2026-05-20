"""
evaluate.py — Validation split, loss computation, and perplexity tracking.

Holds out a configurable fraction of the corpus as a validation set,
evaluates the model after each epoch, and tracks metrics over time.

Usage
-----
from evaluate import split_corpus, build_val_loader, evaluate_model, PerplexityTracker

# Split corpus
train_corpus, val_corpus = split_corpus(full_corpus, val_ratio=0.1)

# Build loaders
train_loader, _ = build_dataloader(train_corpus, ...)
val_loader,   _ = build_val_loader(val_corpus, tokenizer, block_size, batch_size)

# Evaluate after each epoch
tracker = PerplexityTracker()
result = evaluate_model(model, val_loader, device)
tracker.update(result)
tracker.print_report()
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tokenizer import CharTokenizer
from dataset import build_dataloader
from model import TransformerLM


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    loss:       float
    perplexity: float
    n_tokens:   int
    elapsed_s:  float

    def __str__(self) -> str:
        return (
            f"loss={self.loss:.4f}  ppl={self.perplexity:.2f}  "
            f"tokens={self.n_tokens:,}  time={self.elapsed_s:.1f}s"
        )


# ---------------------------------------------------------------------------
# Corpus splitting
# ---------------------------------------------------------------------------

def split_corpus(corpus: str, val_ratio: float = 0.1) -> Tuple[str, str]:
    """
    Split `corpus` into training and validation subsets.

    The split is made at a character boundary near `val_ratio` of the total
    length.  The validation portion is taken from the *end* of the corpus so
    that training always sees earlier content first.

    Args:
        corpus:    Raw text string.
        val_ratio: Fraction of corpus to hold out for validation (default 0.1).

    Returns:
        (train_corpus, val_corpus)
    """
    if not 0.0 < val_ratio < 1.0:
        raise ValueError(f"val_ratio must be in (0, 1), got {val_ratio}")

    split_idx = int(len(corpus) * (1.0 - val_ratio))
    train_corpus = corpus[:split_idx]
    val_corpus   = corpus[split_idx:]

    n_total = len(corpus)
    print(
        f"[evaluate] Corpus split: "
        f"train={len(train_corpus):,} chars ({100*(1-val_ratio):.0f}%)  "
        f"val={len(val_corpus):,} chars ({100*val_ratio:.0f}%)"
    )
    return train_corpus, val_corpus


def build_val_loader(
    val_corpus: str,
    tokenizer:  CharTokenizer,
    block_size: int,
    batch_size: int,
    num_workers: int = 0,
) -> Tuple[DataLoader, int]:
    """Build a non-shuffled DataLoader for the validation corpus."""
    return build_dataloader(
        corpus=val_corpus,
        tokenizer=tokenizer,
        block_size=block_size,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
    )


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_model(
    model:  TransformerLM,
    loader: DataLoader,
    device: torch.device,
) -> EvalResult:
    """
    Run the model in eval mode over the entire validation DataLoader.

    Computes:
    - Mean cross-entropy loss
    - Perplexity  (exp(loss))
    - Total token count
    - Wall-clock time

    Args:
        model:  A TransformerLM instance.
        loader: Validation DataLoader.
        device: Torch device.

    Returns:
        EvalResult with all metrics.
    """
    model.eval()
    loss_fn    = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    total_tok  = 0
    t0         = time.time()

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)                              # (B, T, V)
        B, T, V = logits.shape
        loss = loss_fn(logits.view(B * T, V), y.view(B * T))

        total_loss += loss.item()
        total_tok  += B * T

    avg_loss   = total_loss / max(total_tok, 1)
    perplexity = math.exp(min(avg_loss, 100))          # clamp to avoid overflow
    elapsed    = time.time() - t0

    model.train()

    return EvalResult(
        loss       = avg_loss,
        perplexity = perplexity,
        n_tokens   = total_tok,
        elapsed_s  = elapsed,
    )


# ---------------------------------------------------------------------------
# Perplexity tracker — records history across epochs/steps
# ---------------------------------------------------------------------------

@dataclass
class PerplexityTracker:
    """
    Accumulates evaluation results across epochs and reports trends.

    Example::

        tracker = PerplexityTracker()
        for epoch in range(n_epochs):
            train(...)
            result = evaluate_model(model, val_loader, device)
            tracker.update(result, epoch=epoch, step=global_step)
        tracker.print_report()
    """

    history: List[Dict] = field(default_factory=list)

    def update(
        self,
        result: EvalResult,
        epoch:  Optional[int] = None,
        step:   Optional[int] = None,
    ) -> bool:
        """
        Record a new evaluation result.

        Returns True if this is the best val_loss seen so far.
        """
        entry = {
            "epoch":      epoch,
            "step":       step,
            "loss":       result.loss,
            "perplexity": result.perplexity,
        }
        self.history.append(entry)
        is_best = len(self.history) == 1 or result.loss < min(
            h["loss"] for h in self.history[:-1]
        )
        if is_best:
            print(f"  [evaluate] *** New best val_loss={result.loss:.4f}  ppl={result.perplexity:.2f} ***")
        return is_best

    def best(self) -> Optional[Dict]:
        """Return the entry with the lowest val_loss."""
        if not self.history:
            return None
        return min(self.history, key=lambda h: h["loss"])

    def print_report(self) -> None:
        """Print a formatted table of all evaluation results."""
        if not self.history:
            print("[evaluate] No evaluation results recorded yet.")
            return

        best_loss = min(h["loss"] for h in self.history)
        print("\n" + "=" * 58)
        print("  VALIDATION HISTORY")
        print("=" * 58)
        print(f"  {'Epoch':>6}  {'Step':>8}  {'Val Loss':>10}  {'Perplexity':>12}")
        print("  " + "-" * 54)
        for h in self.history:
            epoch_s = str(h["epoch"]) if h["epoch"] is not None else "-"
            step_s  = f"{h['step']:,}" if h["step"] is not None else "-"
            marker  = "  ← best" if h["loss"] == best_loss else ""
            print(
                f"  {epoch_s:>6}  {step_s:>8}  "
                f"{h['loss']:>10.4f}  {h['perplexity']:>12.2f}{marker}"
            )
        print("=" * 58)
        best = self.best()
        print(
            f"  Best: loss={best['loss']:.4f}  ppl={best['perplexity']:.2f}"
            + (f"  @ epoch {best['epoch']}" if best["epoch"] is not None else "")
        )
        print()

    def loss_trend(self) -> str:
        """Return 'improving', 'worsening', or 'stable'."""
        if len(self.history) < 2:
            return "stable"
        delta = self.history[-1]["loss"] - self.history[-2]["loss"]
        if delta < -0.005:
            return "improving"
        if delta > +0.005:
            return "worsening"
        return "stable"
