"""
train.py — Training loop with checkpoint management, validation, and LR scheduling.

Hyperparameters are defined in a single CONFIG dictionary.
Override them from main.py by passing a config dict to `run_training`.
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
from checkpoint import CheckpointManager
from evaluate import split_corpus, build_val_loader, evaluate_model, PerplexityTracker
from lr_scheduler import (
    build_scheduler,
    attach_scheduler,
    estimate_total_steps,
)

# ---------------------------------------------------------------------------
# Default hyperparameter configuration
# ---------------------------------------------------------------------------

DEFAULT_CONFIG: Dict[str, Any] = {
    # Model architecture
    "n_embd":     128,
    "n_heads":    4,
    "n_layers":   4,
    "block_size": 128,
    "dropout":    0.1,

    # Optimiser
    "lr":           1e-3,
    "weight_decay": 0.01,
    "grad_clip":    1.0,

    # LR scheduler  ("cosine", "constant", or "linear")
    "scheduler":       "cosine",
    "warmup_steps":    200,
    "min_lr_ratio":    0.1,

    # Training schedule
    "batch_size":  32,
    "max_epochs":  5,
    "log_every":   100,

    # Validation
    "val_ratio":   0.1,        # fraction of corpus held out for validation
    "eval_every":  500,        # evaluate on val set every N steps (0 = epoch only)

    # Checkpointing
    "checkpoint_dir":   "checkpoints",
    "checkpoint_every": 500,   # save a checkpoint every N steps
    "keep_top_k":       3,     # keep N best checkpoints by val loss

    # I/O paths
    "training_dir":  "training_data",
    "vocab_path":    "vocab.json",
    "weights_path":  "model_weights.pth",

    # Tokenizer ("char" or "bpe")
    "tokenizer": "char",
    "bpe_vocab_size": 1000,

    # Resume
    "resume": False,   # if True, resume from latest checkpoint in checkpoint_dir
}


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------

def run_training(config: Optional[Dict[str, Any]] = None) -> TransformerLM:
    """
    Full training pipeline:
      1. Build vocabulary from training files.
      2. Split corpus into train / val.
      3. Encode corpus → DataLoader(s).
      4. Instantiate TransformerLM.
      5. AdamW + LR schedule optimisation loop.
      6. Mid-training checkpointing with val loss tracking.
      7. Perplexity logging every epoch.
      8. Save final weights on completion or keyboard interrupt.

    Returns the trained model.
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
    if cfg.get("tokenizer", "char") == "bpe":
        from bpe_tokenizer import BPETokenizer, build_bpe_from_directory
        bpe_tok = BPETokenizer()
        bpe_path = cfg.get("vocab_path", "vocab.json").replace(".json", "_bpe.json")
        if os.path.exists(bpe_path) and not cfg.get("retrain", False):
            bpe_tok.load(bpe_path)
            # Read corpus for splitting
            parts = []
            for root, _, files in os.walk(cfg["training_dir"]):
                for f in sorted(files):
                    if f.endswith((".py", ".js", ".ts", ".txt")):
                        try:
                            with open(os.path.join(root, f), "r", encoding="utf-8", errors="replace") as fh:
                                parts.append(fh.read())
                        except OSError:
                            pass
            corpus = "\n".join(parts)
        else:
            bpe_tok, corpus = build_bpe_from_directory(
                cfg["training_dir"], vocab_size=cfg.get("bpe_vocab_size", 1000)
            )
            bpe_tok.save(bpe_path)
        # Wrap BPE tokenizer to share the CharTokenizer interface
        tokenizer = _BPEWrapper(bpe_tok)
    else:
        tokenizer = CharTokenizer()
        corpus = tokenizer.build_from_directory(cfg["training_dir"])
        tokenizer.save(cfg["vocab_path"])

    # ------------------------------------------------------------------
    # Train / val split
    # ------------------------------------------------------------------
    val_ratio = cfg.get("val_ratio", 0.1)
    train_corpus, val_corpus = split_corpus(corpus, val_ratio=val_ratio)

    # ------------------------------------------------------------------
    # DataLoaders
    # ------------------------------------------------------------------
    train_loader, n_train_tokens = build_dataloader(
        corpus=train_corpus,
        tokenizer=tokenizer,
        block_size=cfg["block_size"],
        batch_size=cfg["batch_size"],
    )
    val_loader, _ = build_val_loader(
        val_corpus=val_corpus,
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
    # Optimiser
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

    # ------------------------------------------------------------------
    # LR scheduler
    # ------------------------------------------------------------------
    total_steps = estimate_total_steps(
        n_tokens   = n_train_tokens,
        block_size = cfg["block_size"],
        batch_size = cfg["batch_size"],
        n_epochs   = cfg["max_epochs"],
    )
    warmup = cfg.get("warmup_steps", min(200, total_steps // 10))
    schedule = build_scheduler(
        name         = cfg.get("scheduler", "cosine"),
        warmup_steps = warmup,
        total_steps  = max(total_steps, warmup + 1),
        min_lr_ratio = cfg.get("min_lr_ratio", 0.1),
    )
    lr_scheduler = attach_scheduler(optimizer, schedule)
    print(
        f"[train] LR schedule: {cfg.get('scheduler','cosine')}  "
        f"warmup={warmup}  total_steps≈{total_steps:,}"
    )

    # ------------------------------------------------------------------
    # Checkpoint manager (+ optional resume)
    # ------------------------------------------------------------------
    ckpt_mgr = CheckpointManager(
        directory  = cfg.get("checkpoint_dir", "checkpoints"),
        keep_top_k = cfg.get("keep_top_k", 3),
        save_every = cfg.get("checkpoint_every", 500),
    )

    start_step = 0
    if cfg.get("resume", False) and ckpt_mgr.has_checkpoints():
        start_step, _ = ckpt_mgr.load_latest(model, optimizer, device)
        print(f"[train] Resumed from step {start_step:,}")

    # ------------------------------------------------------------------
    # Validation tracker
    # ------------------------------------------------------------------
    tracker  = PerplexityTracker()
    loss_fn  = nn.CrossEntropyLoss()

    # ------------------------------------------------------------------
    # Interrupt handler
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
    global_step  = start_step
    running_loss = 0.0
    t0           = time.time()
    eval_every   = cfg.get("eval_every", 500)
    ckpt_every   = cfg.get("checkpoint_every", 500)

    print(f"\n[train] Starting training: {cfg['max_epochs']} epoch(s), "
          f"batch_size={cfg['batch_size']}, lr={cfg['lr']}, "
          f"scheduler={cfg.get('scheduler','cosine')}\n")

    try:
        for epoch in range(1, cfg["max_epochs"] + 1):
            model.train()

            for batch_idx, (x, y) in enumerate(train_loader):
                if interrupted:
                    break

                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                logits = model(x)
                B, T, V = logits.shape
                loss = loss_fn(logits.view(B * T, V), y.view(B * T))

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
                optimizer.step()
                lr_scheduler.step()

                running_loss += loss.item()
                global_step  += 1

                # ── Logging ─────────────────────────────────────────────
                if global_step % cfg["log_every"] == 0:
                    avg_loss   = running_loss / cfg["log_every"]
                    elapsed    = time.time() - t0
                    tok_per_s  = global_step * cfg["batch_size"] * cfg["block_size"] / elapsed
                    current_lr = optimizer.param_groups[0]["lr"]
                    print(
                        f"  epoch {epoch:>3} | step {global_step:>6} | "
                        f"loss {avg_loss:.4f} | lr {current_lr:.2e} | "
                        f"{tok_per_s:,.0f} tok/s"
                    )
                    running_loss = 0.0

                # ── Mid-epoch validation ─────────────────────────────────
                if eval_every > 0 and global_step % eval_every == 0:
                    result = evaluate_model(model, val_loader, device)
                    is_best = tracker.update(result, step=global_step)
                    print(f"  [val] step {global_step:>6} | {result}")

                    # Save checkpoint on every eval if it's best so far
                    if ckpt_mgr.should_save(global_step) or is_best:
                        ckpt_mgr.save(model, optimizer, global_step, result.loss, cfg)

            if interrupted:
                break

            # ── End-of-epoch validation ──────────────────────────────────
            result   = evaluate_model(model, val_loader, device)
            is_best  = tracker.update(result, epoch=epoch, step=global_step)
            print(f"\n[train] Epoch {epoch} | {result}")
            if is_best:
                ckpt_mgr.save(model, optimizer, global_step, result.loss, cfg)

        tracker.print_report()

    finally:
        _save_weights(model, cfg["weights_path"])
        signal.signal(signal.SIGINT, original_handler)

    total_time = time.time() - t0
    print(f"\n[train] Finished in {total_time:.1f}s ({global_step:,} steps total).")
    return model


# ---------------------------------------------------------------------------
# BPE ↔ CharTokenizer compatibility shim
# ---------------------------------------------------------------------------

class _BPEWrapper:
    """Thin wrapper so BPETokenizer satisfies the same interface as CharTokenizer."""

    def __init__(self, bpe):
        self._bpe = bpe

    @property
    def vocab_size(self):
        return self._bpe.vocab_size

    def encode(self, text: str):
        return self._bpe.encode(text)

    def decode(self, ids):
        return self._bpe.decode(ids)

    def save(self, path: str):
        self._bpe.save(path)

    def load(self, path: str):
        self._bpe.load(path)


# ---------------------------------------------------------------------------
# Weight persistence
# ---------------------------------------------------------------------------

def _save_weights(model: TransformerLM, path: str) -> None:
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
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    config = ModelConfig(**checkpoint["model_config"])
    model  = TransformerLM(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"[train] Loaded model from '{weights_path}' ({model.count_parameters():,} params).")
    return model
