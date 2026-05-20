"""
lr_scheduler.py — Learning rate schedulers with linear warmup.

Implements schedulers as callable objects that return the current LR given
the current step number.  Designed to be used with PyTorch's
LambdaLR so the scheduler integrates with standard training loops.

Schedulers
----------
WarmupCosineScheduler   — linear warmup then cosine annealing to min_lr
WarmupConstantScheduler — linear warmup then constant LR

Usage
-----
from lr_scheduler import WarmupCosineScheduler, attach_scheduler

# Build directly:
schedule = WarmupCosineScheduler(
    warmup_steps  = 200,
    total_steps   = 10_000,
    base_lr       = 1e-3,
    min_lr        = 1e-4,
)
lr_now = schedule(current_step)

# Or attach to a PyTorch optimiser (recommended):
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
scheduler = attach_scheduler(optimizer, schedule)
# Then in the training loop:
scheduler.step()   # called once per optimiser step

CLI
---
python lr_scheduler.py --plot --total-steps 5000 --warmup 200
"""

from __future__ import annotations

import math
from typing import Callable, Optional


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class _BaseLRSchedule:
    """
    Abstract base for LR schedules.  Subclasses implement __call__(step) → float,
    returning a *multiplier* in [0, 1] that is applied to the optimiser's
    initial `lr`.
    """

    def __call__(self, step: int) -> float:
        raise NotImplementedError

    def get_lr(self, step: int, base_lr: float) -> float:
        """Return the actual learning rate at `step`."""
        return base_lr * self(step)

    def plot(self, total_steps: int, base_lr: float = 1.0) -> None:
        """Print a tiny ASCII chart of the LR schedule."""
        n_cols = 60
        values = [self.get_lr(int(step), base_lr) for step in
                  [i * total_steps / n_cols for i in range(n_cols + 1)]]
        max_v  = max(values) or 1.0
        rows   = 10
        print(f"\n  LR schedule — {self.__class__.__name__}")
        print(f"  max={max_v:.4f}   steps=0..{total_steps:,}\n")
        for row in range(rows, -1, -1):
            threshold = max_v * row / rows
            line = "".join("█" if v >= threshold else " " for v in values)
            print(f"  {threshold:6.4f} │{line}│")
        print("         └" + "─" * (n_cols + 2) + "┘")
        print(f"           0{' ' * (n_cols - 4)}{total_steps:,}\n")


# ---------------------------------------------------------------------------
# WarmupCosineScheduler
# ---------------------------------------------------------------------------

class WarmupCosineScheduler(_BaseLRSchedule):
    """
    Linear warmup followed by cosine annealing.

    LR(step):
      step < warmup_steps  → step / warmup_steps            (linear ramp-up)
      step ≥ warmup_steps  → cosine decay from 1 → min_lr_ratio

    Parameters
    ----------
    warmup_steps:  Number of steps for the linear warmup phase.
    total_steps:   Total training steps (end of cosine decay).
    min_lr_ratio:  Minimum LR as a fraction of the base LR (default 0.1).
                   E.g. base_lr=1e-3, min_lr_ratio=0.1 → floors at 1e-4.
    """

    def __init__(
        self,
        warmup_steps:  int,
        total_steps:   int,
        min_lr_ratio:  float = 0.1,
    ):
        if warmup_steps >= total_steps:
            raise ValueError(
                f"warmup_steps ({warmup_steps}) must be < total_steps ({total_steps})"
            )
        self.warmup_steps = warmup_steps
        self.total_steps  = total_steps
        self.min_lr_ratio = min_lr_ratio

    def __call__(self, step: int) -> float:
        if step < self.warmup_steps:
            return step / max(self.warmup_steps, 1)

        progress = (step - self.warmup_steps) / max(
            self.total_steps - self.warmup_steps, 1
        )
        progress = min(progress, 1.0)
        cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine


# ---------------------------------------------------------------------------
# WarmupConstantScheduler
# ---------------------------------------------------------------------------

class WarmupConstantScheduler(_BaseLRSchedule):
    """
    Linear warmup for `warmup_steps`, then holds the LR constant.

    Useful when you want deterministic training without decay.
    """

    def __init__(self, warmup_steps: int):
        self.warmup_steps = warmup_steps

    def __call__(self, step: int) -> float:
        if step < self.warmup_steps:
            return step / max(self.warmup_steps, 1)
        return 1.0


# ---------------------------------------------------------------------------
# WarmupLinearDecayScheduler
# ---------------------------------------------------------------------------

class WarmupLinearDecayScheduler(_BaseLRSchedule):
    """
    Linear warmup then linear decay to `min_lr_ratio`.

    A simpler alternative to cosine annealing when you want predictable
    linear decay behaviour.
    """

    def __init__(
        self,
        warmup_steps: int,
        total_steps:  int,
        min_lr_ratio: float = 0.0,
    ):
        self.warmup_steps = warmup_steps
        self.total_steps  = total_steps
        self.min_lr_ratio = min_lr_ratio

    def __call__(self, step: int) -> float:
        if step < self.warmup_steps:
            return step / max(self.warmup_steps, 1)

        progress = (step - self.warmup_steps) / max(
            self.total_steps - self.warmup_steps, 1
        )
        progress = min(progress, 1.0)
        return 1.0 - progress * (1.0 - self.min_lr_ratio)


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def build_scheduler(
    name:         str,
    warmup_steps: int,
    total_steps:  int,
    min_lr_ratio: float = 0.1,
) -> _BaseLRSchedule:
    """
    Build a named scheduler.

    Supported names: "cosine", "constant", "linear"
    """
    name = name.lower()
    if name == "cosine":
        return WarmupCosineScheduler(warmup_steps, total_steps, min_lr_ratio)
    if name == "constant":
        return WarmupConstantScheduler(warmup_steps)
    if name == "linear":
        return WarmupLinearDecayScheduler(warmup_steps, total_steps, min_lr_ratio)
    raise ValueError(f"Unknown scheduler '{name}'. Choose from: cosine, constant, linear.")


def attach_scheduler(
    optimizer:  "torch.optim.Optimizer",
    schedule:   _BaseLRSchedule,
) -> "torch.optim.lr_scheduler.LambdaLR":
    """
    Wrap a schedule object as a PyTorch LambdaLR scheduler.

    The returned scheduler's `.step()` should be called once per optimiser
    step (not once per epoch).

    Example::

        schedule  = WarmupCosineScheduler(200, 10_000)
        scheduler = attach_scheduler(optimizer, schedule)
        # Training loop:
        optimizer.step()
        scheduler.step()
    """
    import torch
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=schedule)


def estimate_total_steps(
    n_tokens:   int,
    block_size: int,
    batch_size: int,
    n_epochs:   int,
) -> int:
    """Estimate total optimiser steps for warmup/decay calculation."""
    steps_per_epoch = max(1, (n_tokens - block_size) // batch_size)
    return steps_per_epoch * n_epochs


# ---------------------------------------------------------------------------
# CLI / demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preview LR schedules.")
    parser.add_argument("--scheduler",   choices=["cosine", "constant", "linear"],
                        default="cosine")
    parser.add_argument("--warmup",      type=int,   default=200)
    parser.add_argument("--total-steps", type=int,   default=5000)
    parser.add_argument("--min-lr",      type=float, default=0.1,
                        help="Min LR as fraction of base (default 0.1)")
    parser.add_argument("--base-lr",     type=float, default=1e-3)
    parser.add_argument("--plot",        action="store_true")
    args = parser.parse_args()

    schedule = build_scheduler(
        args.scheduler, args.warmup, args.total_steps, args.min_lr
    )
    print(f"Scheduler : {schedule.__class__.__name__}")
    print(f"Warmup    : {args.warmup:,} steps")
    print(f"Total     : {args.total_steps:,} steps")
    print(f"Base LR   : {args.base_lr}")
    print(f"Min LR    : {args.base_lr * args.min_lr}")

    checkpoints = [0, args.warmup // 2, args.warmup,
                   args.total_steps // 4, args.total_steps // 2,
                   args.total_steps * 3 // 4, args.total_steps]
    print("\n  Step          LR")
    for step in checkpoints:
        lr = schedule.get_lr(step, args.base_lr)
        print(f"  {step:>10,}   {lr:.6f}")

    if args.plot:
        schedule.plot(args.total_steps, args.base_lr)
