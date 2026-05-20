"""
train.py — Training loop, loss computation, and weight-saving logic.

Hyperparameters are defined in a single CONFIG dictionary at the top of the
file. Override them from main.py by passing a config dict to `run_training`.
"""

import os
import signal
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional

from tokenizer import CharTokenizer
from model import TransformerLM, ModelConfig
from dataset import build_dataloader

# ---------------------------------------------------------------------------
# Default hyperparameter configuration
# ---------------------------------------------------------------------------

DEFAULT_CONFIG: Dict[str, Any] = {
    # Model architecture
    "n_embd":     128,    # Embedding / hidden dimension
    "n_heads":    4,      # Number of attention heads
    "n_layers":   4,      # Number of stacked transformer blocks
    "block_size": 128,    # Context window (tokens)
    "dropout":    0.1,

    # Optimiser
    "lr":         1e-3,   # AdamW learning rate
    "weight_decay": 0.01,
    "grad_clip":  1.0,    # Gradient clipping norm

    # Training schedule
    "batch_size": 32,
    "max_epochs": 5,
    "log_every":  100,    # Print loss every N iterations

    # I/O paths
    "training_dir":  "training_data",
    "vocab_path":    "vocab.json",
    "weights_path":  "model_weights.pth",
}


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------

def run_training(config: Optional[Dict[str, Any]] = None) -> TransformerLM:
    """
    Full training pipeline:
      1. Build vocabulary from training files.
      2. Encode corpus → DataLoader.
      3. Instantiate TransformerLM.
      4. AdamW optimisation loop with loss logging.
      5. Save weights on completion or keyboard interrupt.

    Returns the trained model (on whatever device was used).
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}

    # ------------------------------------------------------------------
    # Device selection
    # ------------------------------------------------------------------
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"[train] Using device: {device}")

    # ------------------------------------------------------------------
    # Tokenisation
    # ------------------------------------------------------------------
    tokenizer = CharTokenizer()
    corpus = tokenizer.build_from_directory(cfg["training_dir"])
    tokenizer.save(cfg["vocab_path"])

    # ------------------------------------------------------------------
    # Dataset / DataLoader
    # ------------------------------------------------------------------
    loader, _ = build_dataloader(
        corpus=corpus,
        tokenizer=tokenizer,
        block_size=cfg["block_size"],
        batch_size=cfg["batch_size"],
    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model_config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=cfg["block_size"],
        n_embd=cfg["n_embd"],
        n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"],
        dropout=cfg["dropout"],
    )
    model = TransformerLM(model_config).to(device)
    print(f"[train] Model parameters: {model.count_parameters():,}")

    # ------------------------------------------------------------------
    # Optimiser — AdamW with weight decay on non-bias/norm params
    # ------------------------------------------------------------------
    decay_params = [
        p for n, p in model.named_parameters()
        if p.requires_grad and p.ndim >= 2
    ]
    no_decay_params = [
        p for n, p in model.named_parameters()
        if p.requires_grad and p.ndim < 2
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params,    "weight_decay": cfg["weight_decay"]},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=cfg["lr"],
    )

    loss_fn = nn.CrossEntropyLoss()

    # ------------------------------------------------------------------
    # Graceful interrupt handler — saves weights on Ctrl-C
    # ------------------------------------------------------------------
    interrupted = False

    def _handle_sigint(sig, frame):
        nonlocal interrupted
        print("\n[train] Interrupt received — saving weights before exit ...")
        interrupted = True

    original_handler = signal.signal(signal.SIGINT, _handle_sigint)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    global_step = 0
    running_loss = 0.0
    t0 = time.time()

    print(f"\n[train] Starting training: {cfg['max_epochs']} epoch(s), "
          f"batch_size={cfg['batch_size']}, lr={cfg['lr']}\n")

    try:
        for epoch in range(1, cfg["max_epochs"] + 1):
            model.train()

            for batch_idx, (x, y) in enumerate(loader):
                if interrupted:
                    break

                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                # Forward pass
                logits = model(x)  # (B, T, V)

                # Flatten for cross-entropy: (B*T, V) vs (B*T,)
                B, T, V = logits.shape
                loss = loss_fn(logits.view(B * T, V), y.view(B * T))

                # Backward pass
                optimizer.zero_grad(set_to_none=True)
                loss.backward()

                # Gradient clipping prevents exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])

                optimizer.step()

                running_loss += loss.item()
                global_step += 1

                # Periodic logging
                if global_step % cfg["log_every"] == 0:
                    avg_loss = running_loss / cfg["log_every"]
                    elapsed = time.time() - t0
                    tokens_per_sec = global_step * cfg["batch_size"] * cfg["block_size"] / elapsed
                    print(
                        f"  epoch {epoch:>3} | step {global_step:>6} | "
                        f"loss {avg_loss:.4f} | "
                        f"{tokens_per_sec:,.0f} tok/s"
                    )
                    running_loss = 0.0

            if interrupted:
                break

            # End-of-epoch summary
            print(f"\n[train] Epoch {epoch} complete — step {global_step}\n")

    finally:
        # Always save weights, even on interrupt
        _save_weights(model, cfg["weights_path"])
        signal.signal(signal.SIGINT, original_handler)

    total_time = time.time() - t0
    print(f"\n[train] Training finished in {total_time:.1f}s ({global_step} steps total).")

    return model


# ---------------------------------------------------------------------------
# Weight persistence helpers
# ---------------------------------------------------------------------------

def _save_weights(model: TransformerLM, path: str) -> None:
    """Save model state dict and its config to a .pth file."""
    payload = {
        "model_state_dict": model.state_dict(),
        "model_config": {
            "vocab_size":  model.config.vocab_size,
            "block_size":  model.config.block_size,
            "n_embd":      model.config.n_embd,
            "n_heads":     model.config.n_heads,
            "n_layers":    model.config.n_layers,
            "dropout":     model.config.dropout,
        },
    }
    torch.save(payload, path)
    print(f"[train] Weights saved to '{path}'.")


def load_model_from_checkpoint(weights_path: str, device: torch.device) -> TransformerLM:
    """
    Reconstruct a TransformerLM from a saved checkpoint file.
    The checkpoint must have been written by `_save_weights`.
    """
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    config = ModelConfig(**checkpoint["model_config"])
    model = TransformerLM(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"[train] Loaded model from '{weights_path}' ({model.count_parameters():,} params).")
    return model
