"""
benchmark.py — Training throughput and inference latency benchmarker.

Measures two things:
  1. Training throughput  — tokens processed per second during forward + backward.
  2. Inference latency    — milliseconds per generated token at various prompt lengths.

Prints a formatted report to stdout and optionally saves it as JSON.

Usage
-----
# Benchmark using an already-trained model:
python benchmark.py

# Benchmark a specific preset / weights file:
python benchmark.py --preset medium --weights model_weights.pth --vocab vocab.json

# Quick smoke-test (fewer warmup/measure steps):
python benchmark.py --fast

# Save results to file:
python benchmark.py --out benchmark_results.json
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Result data structures
# ---------------------------------------------------------------------------

@dataclass
class TrainingBenchmark:
    device:         str
    n_params:       int
    batch_size:     int
    block_size:     int
    n_steps:        int
    tokens_per_sec: float
    ms_per_step:    float
    peak_memory_mb: float     # GPU only; 0 on CPU


@dataclass
class InferenceBenchmark:
    device:             str
    prompt_length:      int
    n_new_tokens:       int
    n_runs:             int
    ms_per_token_mean:  float
    ms_per_token_std:   float
    tokens_per_sec:     float


@dataclass
class BenchmarkReport:
    timestamp:  str
    preset:     str
    training:   Optional[TrainingBenchmark]
    inference:  List[InferenceBenchmark]
    notes:      List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Training throughput benchmark
# ---------------------------------------------------------------------------

def benchmark_training(
    model:       "torch.nn.Module",
    vocab_size:  int,
    batch_size:  int    = 32,
    block_size:  int    = 128,
    n_warmup:    int    = 5,
    n_measure:   int    = 50,
    device:      Optional[torch.device] = None,
) -> TrainingBenchmark:
    """
    Measure forward + backward pass throughput using random synthetic data.

    Args:
        model:      Untrained or trained TransformerLM — weights don't matter.
        vocab_size: Vocabulary size (for generating random token IDs).
        batch_size: Mini-batch size for the benchmark.
        block_size: Sequence length.
        n_warmup:   Steps to discard at the start (GPU JIT warm-up).
        n_measure:  Steps to average over.
        device:     Torch device.

    Returns:
        TrainingBenchmark with throughput statistics.
    """
    if device is None:
        device = next(model.parameters()).device

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn   = nn.CrossEntropyLoss()

    def _make_batch():
        x = torch.randint(0, vocab_size, (batch_size, block_size), device=device)
        y = torch.randint(0, vocab_size, (batch_size, block_size), device=device)
        return x, y

    def _step(x, y):
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        B, T, V = logits.shape
        loss = loss_fn(logits.view(B * T, V), y.view(B * T))
        loss.backward()
        optimizer.step()

    # Warm-up
    for _ in range(n_warmup):
        x, y = _make_batch()
        _step(x, y)

    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    # Measure
    t0 = time.perf_counter()
    for _ in range(n_measure):
        x, y = _make_batch()
        _step(x, y)
    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    peak_mem = 0.0
    if torch.cuda.is_available() and device.type == "cuda":
        peak_mem = torch.cuda.max_memory_allocated() / 1024 ** 2

    elapsed     = t1 - t0
    ms_per_step = (elapsed / n_measure) * 1000
    tok_per_sec = (batch_size * block_size * n_measure) / elapsed
    n_params    = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return TrainingBenchmark(
        device         = str(device),
        n_params       = n_params,
        batch_size     = batch_size,
        block_size     = block_size,
        n_steps        = n_measure,
        tokens_per_sec = tok_per_sec,
        ms_per_step    = ms_per_step,
        peak_memory_mb = peak_mem,
    )


# ---------------------------------------------------------------------------
# Inference latency benchmark
# ---------------------------------------------------------------------------

@torch.no_grad()
def benchmark_inference(
    model:           "torch.nn.Module",
    vocab_size:      int,
    prompt_lengths:  List[int] = None,
    n_new_tokens:    int       = 50,
    n_warmup:        int       = 3,
    n_runs:          int       = 20,
    device:          Optional[torch.device] = None,
) -> List[InferenceBenchmark]:
    """
    Measure per-token generation latency at various prompt lengths.

    Uses random token sequences as prompts (content doesn't matter for timing).

    Args:
        model:          TransformerLM in eval mode.
        vocab_size:     Vocabulary size.
        prompt_lengths: List of prompt lengths to test.
        n_new_tokens:   Number of tokens to generate per run.
        n_warmup:       Warm-up runs discarded before measurement.
        n_runs:         Measurement runs per prompt length.
        device:         Torch device.

    Returns:
        List of InferenceBenchmark (one per prompt_length).
    """
    if device is None:
        device = next(model.parameters()).device
    if prompt_lengths is None:
        prompt_lengths = [8, 32, 64]

    model.eval()
    block_size = model.config.block_size
    results    = []

    for plen in prompt_lengths:
        plen_clamped = min(plen, block_size)
        prompt_ids   = torch.randint(0, vocab_size, (1, plen_clamped), device=device)

        def _generate_n_tokens(ctx: torch.Tensor) -> None:
            for _ in range(n_new_tokens):
                window  = ctx[:, -block_size:]
                logits  = model(window)
                next_id = logits[0, -1, :].argmax(dim=-1, keepdim=True).unsqueeze(0)
                ctx     = torch.cat([ctx, next_id], dim=1)

        # Warm-up
        for _ in range(n_warmup):
            _generate_n_tokens(prompt_ids.clone())

        if device.type == "cuda":
            torch.cuda.synchronize()

        # Measure
        times: List[float] = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            _generate_n_tokens(prompt_ids.clone())
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)  # ms

        ms_per_run = sum(times) / len(times)
        ms_std     = (sum((t - ms_per_run) ** 2 for t in times) / len(times)) ** 0.5
        ms_per_tok = ms_per_run / n_new_tokens
        tok_per_sec = 1000.0 / ms_per_tok

        results.append(InferenceBenchmark(
            device            = str(device),
            prompt_length     = plen_clamped,
            n_new_tokens      = n_new_tokens,
            n_runs            = n_runs,
            ms_per_token_mean = ms_per_tok,
            ms_per_token_std  = ms_std / n_new_tokens,
            tokens_per_sec    = tok_per_sec,
        ))

    return results


# ---------------------------------------------------------------------------
# Formatted report printer
# ---------------------------------------------------------------------------

def print_report(report: BenchmarkReport) -> None:
    """Print a human-readable benchmark report."""
    print("\n" + "=" * 62)
    print(f"  BENCHMARK REPORT  —  {report.timestamp}")
    print(f"  Preset: {report.preset}")
    print("=" * 62)

    if report.training:
        tr = report.training
        print("\n  ── TRAINING THROUGHPUT ──────────────────────────────────")
        print(f"     Device       : {tr.device}")
        print(f"     Parameters   : {tr.n_params:,}")
        print(f"     Batch × Seq  : {tr.batch_size} × {tr.block_size}")
        print(f"     Steps timed  : {tr.n_steps}")
        print(f"     Tokens/sec   : {tr.tokens_per_sec:>10,.0f}")
        print(f"     ms / step    : {tr.ms_per_step:>10.2f}")
        if tr.peak_memory_mb > 0:
            print(f"     Peak VRAM    : {tr.peak_memory_mb:>10.1f} MB")

    if report.inference:
        print("\n  ── INFERENCE LATENCY ────────────────────────────────────")
        print(f"  {'Prompt Len':>12}  {'ms/token':>10}  {'±':>7}  {'tok/sec':>10}")
        print("  " + "-" * 48)
        for inf in report.inference:
            print(
                f"  {inf.prompt_length:>12}  "
                f"{inf.ms_per_token_mean:>10.2f}  "
                f"{inf.ms_per_token_std:>7.2f}  "
                f"{inf.tokens_per_sec:>10.1f}"
            )

    if report.notes:
        print("\n  Notes:")
        for note in report.notes:
            print(f"    • {note}")

    print("\n" + "=" * 62 + "\n")


# ---------------------------------------------------------------------------
# High-level benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark(
    weights_path:    str   = "model_weights.pth",
    vocab_path:      str   = "vocab.json",
    preset:          str   = "small",
    fast:            bool  = False,
    out_path:        Optional[str] = None,
) -> BenchmarkReport:
    """
    Load a trained model and run the full benchmark suite.

    Args:
        weights_path: Path to model_weights.pth.
        vocab_path:   Path to vocab.json.
        preset:       Preset name (for the report label only).
        fast:         Use fewer steps (quick smoke test).
        out_path:     If set, save the report as JSON to this path.

    Returns:
        BenchmarkReport.
    """
    import datetime
    from train import load_model_from_checkpoint
    from tokenizer import CharTokenizer

    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"Weights not found: '{weights_path}'. Train the model first."
        )

    device = (
        torch.device("cuda")  if torch.cuda.is_available() else
        torch.device("mps")   if torch.backends.mps.is_available() else
        torch.device("cpu")
    )
    print(f"[benchmark] Device: {device}")

    tok   = CharTokenizer()
    tok.load(vocab_path)
    model = load_model_from_checkpoint(weights_path, device)
    model.eval()

    n_warmup_train, n_measure_train = (2, 10) if fast else (5, 50)
    n_warmup_inf,   n_runs_inf      = (2, 5)  if fast else (3, 20)
    n_new_tokens                    = 20       if fast else 50

    print("[benchmark] Measuring training throughput ...")
    tr = benchmark_training(
        model=model, vocab_size=tok.vocab_size,
        batch_size=16, block_size=model.config.block_size,
        n_warmup=n_warmup_train, n_measure=n_measure_train, device=device,
    )

    print("[benchmark] Measuring inference latency ...")
    block = model.config.block_size
    prompt_lengths = [min(8, block), min(32, block), min(block, block)]
    prompt_lengths = sorted(set(prompt_lengths))
    inf_results = benchmark_inference(
        model=model, vocab_size=tok.vocab_size,
        prompt_lengths=prompt_lengths, n_new_tokens=n_new_tokens,
        n_warmup=n_warmup_inf, n_runs=n_runs_inf, device=device,
    )

    notes = []
    if device.type == "cpu":
        notes.append("Running on CPU — GPU will be significantly faster.")
    if fast:
        notes.append("Fast mode: fewer measurement steps, lower statistical confidence.")

    report = BenchmarkReport(
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        preset    = preset,
        training  = tr,
        inference = inf_results,
        notes     = notes,
    )

    print_report(report)

    if out_path:
        raw = asdict(report)
        with open(out_path, "w") as f:
            json.dump(raw, f, indent=2)
        print(f"[benchmark] Results saved to '{out_path}'.")

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark training throughput and inference latency.")
    parser.add_argument("--weights", default="model_weights.pth")
    parser.add_argument("--vocab",   default="vocab.json")
    parser.add_argument("--preset",  default="small",
                        help="Preset label (informational only).")
    parser.add_argument("--fast",    action="store_true",
                        help="Quick smoke-test with fewer steps.")
    parser.add_argument("--out",     default=None,
                        help="Save results as JSON to this path.")
    args = parser.parse_args()

    run_benchmark(
        weights_path = args.weights,
        vocab_path   = args.vocab,
        preset       = args.preset,
        fast         = args.fast,
        out_path     = args.out,
    )
