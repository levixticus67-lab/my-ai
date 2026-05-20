"""
checkpoint.py — Mid-training checkpoint manager.

Saves model snapshots every N optimiser steps, tracks validation loss per
checkpoint, automatically prunes to keep only the top-K best checkpoints,
and lets you resume training or load the best/latest checkpoint at any time.

Usage in train.py
-----------------
from checkpoint import CheckpointManager

ckpt_mgr = CheckpointManager(
    directory  = "checkpoints",
    keep_top_k = 3,
    save_every = 500,          # steps between saves
)

# Inside the training loop:
if ckpt_mgr.should_save(global_step):
    val_loss = evaluate(...)
    ckpt_mgr.save(model, optimizer, global_step, val_loss, config)

# After training:
ckpt_mgr.load_best(model, optimizer)   # best val_loss
ckpt_mgr.load_latest(model, optimizer) # most recent step

CLI
---
python checkpoint.py --list                          # list all saved checkpoints
python checkpoint.py --load-best --weights-dir checkpoints
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import torch

from model import TransformerLM, ModelConfig


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CheckpointMeta:
    step:      int
    val_loss:  float
    path:      str            # absolute path to the .pth file


# ---------------------------------------------------------------------------
# CheckpointManager
# ---------------------------------------------------------------------------

class CheckpointManager:
    """
    Manages a directory of model checkpoints.

    Parameters
    ----------
    directory:  Folder where checkpoint files are written.
    keep_top_k: How many checkpoints to keep on disk (lowest val_loss wins).
                Set to -1 to keep all checkpoints.
    save_every: Save a checkpoint every this many optimiser steps.
    """

    _META_FILE = "checkpoint_index.json"

    def __init__(
        self,
        directory:  str  = "checkpoints",
        keep_top_k: int  = 3,
        save_every:  int  = 500,
    ):
        self.directory   = directory
        self.keep_top_k  = keep_top_k
        self.save_every  = save_every
        self._checkpoints: List[CheckpointMeta] = []

        os.makedirs(directory, exist_ok=True)
        self._load_meta()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def should_save(self, step: int) -> bool:
        """Return True if a checkpoint should be saved at this step."""
        return step > 0 and step % self.save_every == 0

    def save(
        self,
        model:     TransformerLM,
        optimizer: torch.optim.Optimizer,
        step:      int,
        val_loss:  float,
        config:    Dict[str, Any],
    ) -> str:
        """
        Write a checkpoint to disk and prune excess old ones.

        Returns the path of the saved file.
        """
        fname = f"checkpoint_step{step:07d}_loss{val_loss:.4f}.pth"
        fpath = os.path.abspath(os.path.join(self.directory, fname))

        payload = {
            "step":             step,
            "val_loss":         val_loss,
            "model_state_dict": model.state_dict(),
            "optimizer_state":  optimizer.state_dict(),
            "model_config": {
                "vocab_size":  model.config.vocab_size,
                "block_size":  model.config.block_size,
                "n_embd":      model.config.n_embd,
                "n_heads":     model.config.n_heads,
                "n_layers":    model.config.n_layers,
                "dropout":     model.config.dropout,
            },
            "train_config": config,
        }
        torch.save(payload, fpath)

        meta = CheckpointMeta(step=step, val_loss=val_loss, path=fpath)
        self._checkpoints.append(meta)
        self._prune()
        self._save_meta()

        print(
            f"[checkpoint] Saved step={step:,}  val_loss={val_loss:.4f}  → {fname}\n"
            f"             Keeping {len(self._checkpoints)} checkpoint(s)."
        )
        return fpath

    def load_best(
        self,
        model:     TransformerLM,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device:    torch.device = torch.device("cpu"),
    ) -> Tuple[int, float]:
        """
        Load the checkpoint with the lowest validation loss.

        Returns (step, val_loss) of the loaded checkpoint.
        """
        if not self._checkpoints:
            raise FileNotFoundError("No checkpoints found. Train the model first.")

        best = min(self._checkpoints, key=lambda c: c.val_loss)
        return self._load(best.path, model, optimizer, device)

    def load_latest(
        self,
        model:     TransformerLM,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device:    torch.device = torch.device("cpu"),
    ) -> Tuple[int, float]:
        """
        Load the most recently saved checkpoint (highest step number).

        Returns (step, val_loss) of the loaded checkpoint.
        """
        if not self._checkpoints:
            raise FileNotFoundError("No checkpoints found. Train the model first.")

        latest = max(self._checkpoints, key=lambda c: c.step)
        return self._load(latest.path, model, optimizer, device)

    def load_by_step(
        self,
        step:      int,
        model:     TransformerLM,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device:    torch.device = torch.device("cpu"),
    ) -> Tuple[int, float]:
        """Load the checkpoint saved at exactly `step`."""
        matches = [c for c in self._checkpoints if c.step == step]
        if not matches:
            available = [c.step for c in self._checkpoints]
            raise FileNotFoundError(
                f"No checkpoint at step {step}. Available steps: {available}"
            )
        return self._load(matches[0].path, model, optimizer, device)

    def list_checkpoints(self) -> List[CheckpointMeta]:
        """Return all tracked checkpoints sorted by step."""
        return sorted(self._checkpoints, key=lambda c: c.step)

    def best_val_loss(self) -> Optional[float]:
        """Return the lowest val_loss across all kept checkpoints."""
        if not self._checkpoints:
            return None
        return min(c.val_loss for c in self._checkpoints)

    def has_checkpoints(self) -> bool:
        return len(self._checkpoints) > 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load(
        self,
        path:      str,
        model:     TransformerLM,
        optimizer: Optional[torch.optim.Optimizer],
        device:    torch.device,
    ) -> Tuple[int, float]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint file missing: '{path}'")

        data = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(data["model_state_dict"])
        model.to(device)

        if optimizer is not None and "optimizer_state" in data:
            optimizer.load_state_dict(data["optimizer_state"])

        step     = data["step"]
        val_loss = data["val_loss"]
        print(f"[checkpoint] Loaded step={step:,}  val_loss={val_loss:.4f}  ← {os.path.basename(path)}")
        return step, val_loss

    def _prune(self) -> None:
        """Remove lowest-ranked checkpoints beyond keep_top_k."""
        if self.keep_top_k < 0 or len(self._checkpoints) <= self.keep_top_k:
            return

        # Sort ascending by val_loss; keep the first keep_top_k
        ranked = sorted(self._checkpoints, key=lambda c: c.val_loss)
        to_keep   = set(id(c) for c in ranked[:self.keep_top_k])
        to_delete = [c for c in self._checkpoints if id(c) not in to_keep]

        for ckpt in to_delete:
            if os.path.exists(ckpt.path):
                os.remove(ckpt.path)
                print(f"[checkpoint] Pruned {os.path.basename(ckpt.path)}")

        self._checkpoints = ranked[:self.keep_top_k]

    def _meta_path(self) -> str:
        return os.path.join(self.directory, self._META_FILE)

    def _save_meta(self) -> None:
        data = [asdict(c) for c in self._checkpoints]
        with open(self._meta_path(), "w") as f:
            json.dump(data, f, indent=2)

    def _load_meta(self) -> None:
        path = self._meta_path()
        if not os.path.exists(path):
            return
        with open(path) as f:
            data = json.load(f)
        self._checkpoints = [
            CheckpointMeta(**d) for d in data
            if os.path.exists(d.get("path", ""))  # skip orphaned entries
        ]


# ---------------------------------------------------------------------------
# CLI helper
# ---------------------------------------------------------------------------

def _print_table(checkpoints: List[CheckpointMeta]) -> None:
    if not checkpoints:
        print("  (no checkpoints found)")
        return
    best_loss = min(c.val_loss for c in checkpoints)
    print(f"\n  {'Step':>10}  {'Val Loss':>10}  {'File'}")
    print("  " + "-" * 70)
    for c in checkpoints:
        marker = "  ← best" if c.val_loss == best_loss else ""
        print(f"  {c.step:>10,}  {c.val_loss:>10.4f}  {os.path.basename(c.path)}{marker}")
    print()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Inspect or load checkpoints.")
    parser.add_argument("--dir",  default="checkpoints", help="Checkpoint directory.")
    parser.add_argument("--list", action="store_true",   help="List all checkpoints.")
    args = parser.parse_args()

    mgr = CheckpointManager(directory=args.dir)
    if args.list or True:
        ckpts = mgr.list_checkpoints()
        print(f"Checkpoints in '{args.dir}':")
        _print_table(ckpts)
