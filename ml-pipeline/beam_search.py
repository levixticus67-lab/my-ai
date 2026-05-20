"""
beam_search.py — Beam search decoder with length normalisation.

Implements full beam search as an alternative to greedy / top-k sampling.
Beam search keeps the B most probable token sequences at every decoding step,
producing higher-quality completions at the cost of more computation.

Features
--------
- Configurable beam width
- Length normalisation (prevents short-sequence bias)
- Optional repetition penalty to discourage copy-paste outputs
- Returns all beam candidates ranked by score (not just the top-1)
- Token streaming of the best beam

Usage
-----
from beam_search import beam_search

results = beam_search(
    model        = model,
    tokenizer    = tokenizer,
    prompt       = "def fibonacci(",
    beam_width   = 5,
    max_new_tokens = 150,
    length_penalty = 0.7,
    repetition_penalty = 1.1,
)

for rank, (text, score) in enumerate(results, 1):
    print(f"[{rank}] score={score:.4f}  {text!r}")

CLI
---
python beam_search.py --prompt "def fib(" --beam-width 5 --max-tokens 150
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from tokenizer import CharTokenizer
from train import load_model_from_checkpoint


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(order=True)
class Beam:
    """One beam hypothesis."""
    score:   float                       # cumulative log-prob (higher = better)
    ids:     List[int] = field(compare=False)    # token id sequence
    done:    bool      = field(default=False, compare=False)

    @property
    def length(self) -> int:
        return len(self.ids)

    def normalised_score(self, length_penalty: float) -> float:
        """Apply length normalisation: score / (length ** penalty)."""
        denom = (self.length ** length_penalty) if self.length > 0 else 1.0
        return self.score / denom


# ---------------------------------------------------------------------------
# Core beam search function
# ---------------------------------------------------------------------------

@torch.no_grad()
def beam_search(
    model:              "torch.nn.Module",
    tokenizer:          CharTokenizer,
    prompt:             str,
    beam_width:         int   = 5,
    max_new_tokens:     int   = 150,
    length_penalty:     float = 0.7,
    repetition_penalty: float = 1.0,
    device:             Optional[torch.device] = None,
    stream_best:        bool  = False,
) -> List[Tuple[str, float]]:
    """
    Generate text completions using beam search.

    Parameters
    ----------
    model:
        A trained TransformerLM in eval mode.
    tokenizer:
        A fitted CharTokenizer used to encode the prompt and decode results.
    prompt:
        Seed text.  The prompt itself is always included in all outputs.
    beam_width:
        Number of beams to maintain at each step.  Higher = better quality,
        slower runtime.  Typical range: 3–10.
    max_new_tokens:
        Maximum number of tokens to generate beyond the prompt.
    length_penalty:
        Exponent applied to sequence length when normalising scores.
        0.0 = no normalisation (favours short sequences).
        1.0 = divide by exact length (favours longer sequences).
        0.6–0.8 is a good default.
    repetition_penalty:
        Penalty > 1.0 reduces the logit of any token that already appears in
        the current beam sequence.  1.0 = no penalty.
    device:
        Torch device (auto-detected if None).
    stream_best:
        If True, print the best beam's new tokens to stdout as they are chosen.

    Returns
    -------
    List of (text, normalised_score) tuples sorted by score descending.
    The first element is always the best completion.
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    block_size = model.config.block_size

    # ── Encode prompt ────────────────────────────────────────────────────────
    prompt_ids = tokenizer.encode(prompt)
    if not prompt_ids:
        raise ValueError("Prompt encodes to an empty token sequence.")

    # ── Initialise beams ─────────────────────────────────────────────────────
    beams: List[Beam] = [Beam(score=0.0, ids=list(prompt_ids))]

    if stream_best:
        print(prompt, end="", flush=True)

    # ── Decoding loop ────────────────────────────────────────────────────────
    for step in range(max_new_tokens):
        if all(b.done for b in beams):
            break

        candidates: List[Beam] = []

        for beam in beams:
            if beam.done:
                candidates.append(beam)
                continue

            # Prepare context tensor
            ctx_ids = beam.ids[-block_size:]
            ctx     = torch.tensor([ctx_ids], dtype=torch.long, device=device)

            # Forward pass
            logits      = model(ctx)           # (1, T, V)
            next_logits = logits[0, -1, :]     # (V,)

            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for tok_id in set(beam.ids):
                    if 0 <= tok_id < next_logits.size(0):
                        if next_logits[tok_id] < 0:
                            next_logits[tok_id] *= repetition_penalty
                        else:
                            next_logits[tok_id] /= repetition_penalty

            log_probs = F.log_softmax(next_logits, dim=-1)

            # Take top beam_width tokens
            top_log_probs, top_ids = torch.topk(log_probs, k=min(beam_width, log_probs.size(0)))

            for log_p, tok_id in zip(top_log_probs.tolist(), top_ids.tolist()):
                new_score = beam.score + log_p
                new_ids   = beam.ids + [tok_id]
                candidates.append(Beam(score=new_score, ids=new_ids))

        # Keep top beam_width by raw score
        candidates.sort(key=lambda b: b.score, reverse=True)
        beams = candidates[:beam_width]

        # Stream best token if requested
        if stream_best and beams:
            best_new_id = beams[0].ids[-1]
            print(tokenizer.decode([best_new_id]), end="", flush=True)

    if stream_best:
        print()

    # ── Final ranking by normalised score ────────────────────────────────────
    ranked = sorted(
        beams,
        key=lambda b: b.normalised_score(length_penalty),
        reverse=True,
    )

    results: List[Tuple[str, float]] = []
    for beam in ranked:
        text  = tokenizer.decode(beam.ids)
        score = beam.normalised_score(length_penalty)
        results.append((text, score))

    return results


# ---------------------------------------------------------------------------
# Convenience wrapper (loads model + vocab from disk)
# ---------------------------------------------------------------------------

def beam_search_from_disk(
    prompt:         str,
    weights_path:   str = "model_weights.pth",
    vocab_path:     str = "vocab.json",
    beam_width:     int   = 5,
    max_new_tokens: int   = 150,
    length_penalty: float = 0.7,
    repetition_penalty: float = 1.0,
    top_n:          int   = 3,
    stream_best:    bool  = True,
) -> List[Tuple[str, float]]:
    """
    Load model and tokenizer from disk, run beam search, and return results.

    Args:
        top_n: How many ranked candidates to return (and print).

    Returns:
        List of up to `top_n` (text, score) tuples.
    """
    for path, label in [(weights_path, "Weights"), (vocab_path, "Vocabulary")]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{label} file not found: '{path}'\nRun training first."
            )

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    tokenizer = CharTokenizer()
    tokenizer.load(vocab_path)

    model = load_model_from_checkpoint(weights_path, device)
    model.eval()

    print(f"\n[beam] Beam width={beam_width}  max_tokens={max_new_tokens}  "
          f"length_penalty={length_penalty}\n")
    print("─" * 60)

    results = beam_search(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        beam_width=beam_width,
        max_new_tokens=max_new_tokens,
        length_penalty=length_penalty,
        repetition_penalty=repetition_penalty,
        device=device,
        stream_best=stream_best,
    )

    # Print ranked alternatives
    print("\n" + "─" * 60)
    print(f"  TOP {min(top_n, len(results))} BEAM CANDIDATES (ranked by normalised score)")
    print("─" * 60)
    for rank, (text, score) in enumerate(results[:top_n], 1):
        snippet = text[len(prompt):]
        snippet_display = repr(snippet[:80]) + ("…" if len(snippet) > 80 else "")
        print(f"  [{rank}] score={score:.4f}  {snippet_display}")
    print()

    return results[:top_n]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Beam search code completion.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--prompt",   "-p", type=str, required=True,
                        help='Seed text (e.g. "def fibonacci(")')
    parser.add_argument("--beam-width", "-b", type=int, default=5,
                        help="Number of beams (default: 5)")
    parser.add_argument("--max-tokens", "-n", type=int, default=150,
                        help="Max new tokens (default: 150)")
    parser.add_argument("--length-penalty", type=float, default=0.7,
                        help="Length normalisation exponent (default: 0.7)")
    parser.add_argument("--rep-penalty", type=float, default=1.0,
                        help="Repetition penalty > 1.0 discourages repeated tokens (default: 1.0)")
    parser.add_argument("--top-n",    type=int, default=3,
                        help="How many beam candidates to display (default: 3)")
    parser.add_argument("--weights",  default="model_weights.pth")
    parser.add_argument("--vocab",    default="vocab.json")
    args = parser.parse_args()

    beam_search_from_disk(
        prompt             = args.prompt,
        weights_path       = args.weights,
        vocab_path         = args.vocab,
        beam_width         = args.beam_width,
        max_new_tokens     = args.max_tokens,
        length_penalty     = args.length_penalty,
        repetition_penalty = args.rep_penalty,
        top_n              = args.top_n,
        stream_best        = True,
    )
