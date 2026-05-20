"""
export.py — Model export, architecture inspection, and FLOPs estimation.

Exports a trained TransformerLM to portable formats for deployment or
analysis.  No external dependencies beyond PyTorch itself.

Features
--------
1. TorchScript export    — saves a self-contained .pt file that can be loaded
                           anywhere PyTorch is installed, no source code needed.
2. ONNX export           — saves an .onnx file for deployment with ONNX Runtime,
                           TensorRT, CoreML, etc.
3. Architecture inspector — prints a detailed per-layer parameter breakdown.
4. FLOPs estimator        — estimates multiply-accumulate operations per forward
                           pass (useful for sizing hardware requirements).
5. Weight statistics      — min/max/mean/std of every parameter tensor
                           (helpful for debugging training instability).

Usage
-----
python export.py --inspect                      # architecture summary
python export.py --flops                        # FLOPs estimate
python export.py --torchscript model.pt         # TorchScript export
python export.py --onnx model.onnx              # ONNX export
python export.py --weight-stats                 # parameter statistics
python export.py --all --out-dir exports/       # run everything
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from model import TransformerLM, ModelConfig
from tokenizer import CharTokenizer
from train import load_model_from_checkpoint


# ---------------------------------------------------------------------------
# 1. Architecture inspector
# ---------------------------------------------------------------------------

def inspect_model(model: TransformerLM, verbose: bool = True) -> Dict[str, Any]:
    """
    Print a detailed per-layer parameter breakdown.

    Returns a dict with summary statistics for programmatic use.
    """
    cfg = model.config

    print("\n" + "=" * 66)
    print("  MODEL ARCHITECTURE SUMMARY")
    print("=" * 66)
    print(f"  Vocab size   : {cfg.vocab_size:,}")
    print(f"  Block size   : {cfg.block_size}")
    print(f"  Embedding dim: {cfg.n_embd}")
    print(f"  Heads        : {cfg.n_heads}  (head_dim = {cfg.n_embd // cfg.n_heads})")
    print(f"  Layers       : {cfg.n_layers}")
    print(f"  FFN mult     : {cfg.ffn_mult}  (ffn_dim = {cfg.n_embd * cfg.ffn_mult})")
    print(f"  Dropout      : {cfg.dropout}")
    print()

    rows: List[Tuple[str, int, str]] = []

    for name, param in model.named_parameters():
        shape_str = "×".join(str(d) for d in param.shape)
        rows.append((name, param.numel(), shape_str))

    total_params   = sum(r[1] for r in rows)
    trainable_rows = [(n, c, s) for n, c, s in rows]

    print(f"  {'Layer name':<45} {'Params':>10}  Shape")
    print("  " + "-" * 64)
    for name, count, shape in trainable_rows:
        print(f"  {name:<45} {count:>10,}  {shape}")

    print("  " + "-" * 64)
    print(f"  {'TOTAL':<45} {total_params:>10,}")
    print()

    # Group by component
    groups: Dict[str, int] = {}
    for name, count, _ in rows:
        key = name.split(".")[0]
        groups[key] = groups.get(key, 0) + count

    print("  ── Parameter distribution ──────────────────────────────────")
    for grp, cnt in sorted(groups.items(), key=lambda x: -x[1]):
        pct = 100 * cnt / max(total_params, 1)
        bar = "█" * int(pct / 2)
        print(f"  {grp:<20} {cnt:>10,}  ({pct:5.1f}%)  {bar}")
    print("=" * 66 + "\n")

    return {
        "total_params":    total_params,
        "config":          {k: getattr(cfg, k) for k in cfg.__dataclass_fields__},
        "param_groups":    groups,
    }


# ---------------------------------------------------------------------------
# 2. FLOPs estimator
# ---------------------------------------------------------------------------

def estimate_flops(model: TransformerLM, seq_len: Optional[int] = None) -> Dict[str, float]:
    """
    Estimate the number of multiply-accumulate (MAC) operations for a
    single forward pass.

    This is an analytic estimate based on the known architecture — no hooks
    required.  The returned numbers are in MACs (multiply-accumulates);
    multiply by 2 for total FLOPs.

    Args:
        model:   TransformerLM instance.
        seq_len: Sequence length to estimate at (defaults to block_size).

    Returns:
        Dict with MACs broken down by component.
    """
    cfg  = model.config
    T    = seq_len or cfg.block_size
    C    = cfg.n_embd
    H    = cfg.n_heads
    hd   = C // H
    L    = cfg.n_layers
    ffn  = C * cfg.ffn_mult
    V    = cfg.vocab_size

    # Embedding lookups: 0 MACs (table look-ups)
    emb_macs = 0

    # Per-block MACs:
    # QKV projection: 3 × T × C × C
    qkv_macs = 3 * T * C * C

    # Attention scores: T × T × C  (Q @ K^T)
    attn_score_macs = T * T * C

    # Attention × V: T × T × C
    attn_val_macs = T * T * C

    # Output projection: T × C × C
    out_proj_macs = T * C * C

    # LayerNorm (trivial, ignore)

    # FFN: T × C × ffn + T × ffn × C
    ffn_macs = 2 * T * C * ffn

    per_block_macs = qkv_macs + attn_score_macs + attn_val_macs + out_proj_macs + ffn_macs
    total_block_macs = L * per_block_macs

    # Final projection: T × C × V
    head_macs = T * C * V

    total_macs = emb_macs + total_block_macs + head_macs

    print("\n" + "=" * 56)
    print(f"  FLOPs ESTIMATE  (seq_len={T})")
    print("=" * 56)
    components = [
        ("Embeddings",         emb_macs),
        (f"Attention ×{L}",    L * (qkv_macs + attn_score_macs + attn_val_macs + out_proj_macs)),
        (f"FFN ×{L}",          L * ffn_macs),
        ("Output head",        head_macs),
        ("TOTAL MACs",         total_macs),
        ("TOTAL FLOPs (≈2×MACs)", 2 * total_macs),
    ]
    for label, macs in components:
        if macs >= 1e9:
            val = f"{macs / 1e9:.3f} G"
        elif macs >= 1e6:
            val = f"{macs / 1e6:.3f} M"
        elif macs >= 1e3:
            val = f"{macs / 1e3:.3f} K"
        else:
            val = str(macs)
        sep = "─" * 54 if "TOTAL" in label else ""
        if sep:
            print(f"  {sep}")
        print(f"  {label:<35} {val:>16}")
    print("=" * 56 + "\n")

    return {
        "total_macs":  total_macs,
        "total_flops": 2 * total_macs,
        "per_block_macs": per_block_macs,
        "head_macs":   head_macs,
    }


# ---------------------------------------------------------------------------
# 3. Weight statistics
# ---------------------------------------------------------------------------

def weight_statistics(model: TransformerLM) -> Dict[str, Dict[str, float]]:
    """
    Print min / max / mean / std for every parameter tensor.

    Useful for diagnosing vanishing/exploding weights after training.
    """
    print("\n" + "=" * 70)
    print("  WEIGHT STATISTICS")
    print("=" * 70)
    print(f"  {'Layer':<45} {'min':>8} {'max':>8} {'mean':>8} {'std':>8}")
    print("  " + "-" * 70)

    stats: Dict[str, Dict[str, float]] = {}
    for name, param in model.named_parameters():
        data  = param.detach().float()
        s = {
            "min":  data.min().item(),
            "max":  data.max().item(),
            "mean": data.mean().item(),
            "std":  data.std().item(),
        }
        stats[name] = s
        print(
            f"  {name:<45} "
            f"{s['min']:>8.4f} {s['max']:>8.4f} "
            f"{s['mean']:>8.4f} {s['std']:>8.4f}"
        )
    print("=" * 70 + "\n")
    return stats


# ---------------------------------------------------------------------------
# 4. TorchScript export
# ---------------------------------------------------------------------------

def export_torchscript(
    model:       TransformerLM,
    output_path: str,
    example_seq_len: int = 32,
) -> str:
    """
    Trace the model and save as TorchScript (.pt).

    The resulting file can be loaded with `torch.jit.load()` without
    needing the source code.

    Args:
        model:           TransformerLM in eval mode.
        output_path:     Destination .pt file path.
        example_seq_len: Sequence length used for tracing.

    Returns:
        Absolute path of the saved file.
    """
    model.eval()
    device   = next(model.parameters()).device
    seq_len  = min(example_seq_len, model.config.block_size)
    example  = torch.randint(0, model.config.vocab_size, (1, seq_len), device=device)

    print(f"[export] Tracing model (seq_len={seq_len}) ...")
    try:
        traced = torch.jit.trace(model, example)
        traced.save(output_path)
        size_mb = os.path.getsize(output_path) / 1024 ** 2
        print(f"[export] TorchScript saved → '{output_path}'  ({size_mb:.1f} MB)")
        return os.path.abspath(output_path)
    except Exception as e:
        print(f"[export] TorchScript tracing failed: {e}")
        raise


# ---------------------------------------------------------------------------
# 5. ONNX export
# ---------------------------------------------------------------------------

def export_onnx(
    model:       TransformerLM,
    output_path: str,
    opset:       int = 17,
    example_seq_len: int = 32,
) -> str:
    """
    Export model to ONNX format.

    The ONNX file can be used with ONNX Runtime, TensorRT, CoreML exporters,
    and many other inference backends.

    Args:
        model:           TransformerLM in eval mode.
        output_path:     Destination .onnx file path.
        opset:           ONNX opset version (default 17).
        example_seq_len: Sequence length used for tracing.

    Returns:
        Absolute path of the saved file.
    """
    model.eval()
    device  = next(model.parameters()).device
    seq_len = min(example_seq_len, model.config.block_size)
    example = torch.randint(0, model.config.vocab_size, (1, seq_len), device=device)

    print(f"[export] Exporting to ONNX opset={opset} (seq_len={seq_len}) ...")
    try:
        torch.onnx.export(
            model,
            (example,),
            output_path,
            opset_version=opset,
            input_names=["input_ids"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "seq_len"},
                "logits":    {0: "batch", 1: "seq_len"},
            },
            do_constant_folding=True,
        )
        size_mb = os.path.getsize(output_path) / 1024 ** 2
        print(f"[export] ONNX saved → '{output_path}'  ({size_mb:.1f} MB)")
        return os.path.abspath(output_path)
    except Exception as e:
        print(f"[export] ONNX export failed: {e}")
        raise


# ---------------------------------------------------------------------------
# High-level runner
# ---------------------------------------------------------------------------

def run_export(
    weights_path: str  = "model_weights.pth",
    vocab_path:   str  = "vocab.json",
    out_dir:      str  = "exports",
    do_inspect:   bool = True,
    do_flops:     bool = True,
    do_stats:     bool = False,
    do_torchscript: bool = False,
    do_onnx:      bool = False,
) -> None:
    """
    Load model from disk and run selected export/inspection tasks.
    """
    for path, label in [(weights_path, "Weights"), (vocab_path, "Vocabulary")]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{label} not found: '{path}'. Train first.")

    device = (
        torch.device("cuda") if torch.cuda.is_available() else
        torch.device("cpu")
    )

    tok   = CharTokenizer()
    tok.load(vocab_path)
    model = load_model_from_checkpoint(weights_path, device)
    model.eval()

    if do_inspect:
        inspect_model(model)

    if do_flops:
        estimate_flops(model)

    if do_stats:
        weight_statistics(model)

    os.makedirs(out_dir, exist_ok=True)

    if do_torchscript:
        out = os.path.join(out_dir, "model.pt")
        export_torchscript(model, out)

    if do_onnx:
        out = os.path.join(out_dir, "model.onnx")
        export_onnx(model, out)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Export and inspect a trained Transformer LM.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--weights",      default="model_weights.pth")
    parser.add_argument("--vocab",        default="vocab.json")
    parser.add_argument("--out-dir",      default="exports",
                        help="Directory for exported files (default: exports/).")
    parser.add_argument("--inspect",      action="store_true",
                        help="Print per-layer parameter breakdown.")
    parser.add_argument("--flops",        action="store_true",
                        help="Estimate MACs/FLOPs for one forward pass.")
    parser.add_argument("--weight-stats", action="store_true",
                        help="Print min/max/mean/std for every parameter.")
    parser.add_argument("--torchscript",  action="store_true",
                        help="Export model as TorchScript (.pt).")
    parser.add_argument("--onnx",         action="store_true",
                        help="Export model as ONNX (.onnx).")
    parser.add_argument("--all",          action="store_true",
                        help="Run all export and inspection tasks.")
    args = parser.parse_args()

    run_export(
        weights_path   = args.weights,
        vocab_path     = args.vocab,
        out_dir        = args.out_dir,
        do_inspect     = args.inspect or args.all,
        do_flops       = args.flops   or args.all,
        do_stats       = args.weight_stats or args.all,
        do_torchscript = args.torchscript  or args.all,
        do_onnx        = args.onnx         or args.all,
    )
