"""
main.py — Unified entry point for the ML pipeline.

Behaviour:
  1. If model_weights.pth does NOT exist (or --retrain is passed):
       → Scan training_data/, build vocabulary, train the model, save weights.
  2. If model_weights.pth DOES exist:
       → Skip training, load existing weights directly.
  3. Drop into an interactive terminal prompt shell for offline code generation.

Usage:
  python main.py                              # auto-train if needed, then REPL
  python main.py --retrain                    # force re-training
  python main.py --preset medium             # use a larger model preset
  python main.py --config my_run.yaml        # load hyperparams from a file
  python main.py --tokenizer bpe             # use BPE instead of character-level
  python main.py --strategy beam            # use beam search for generation
  python main.py --resume                    # resume training from last checkpoint
  python main.py --benchmark                 # run throughput + latency benchmarks
  python main.py --export                    # inspect architecture + FLOPs
  python main.py --augment                   # augment training data before training
  python main.py --prompt "def fib("        # single-shot generation (no REPL)
  python main.py --list-presets              # show available model size presets
"""

import argparse
import os
import sys

from config import load_config, list_presets, save_config

CONFIG: dict = {}

GEN_MAX_TOKENS  = 300
GEN_TEMPERATURE = 0.8
GEN_TOP_K       = 40
GEN_STRATEGY    = "topk"
GEN_BEAM_WIDTH  = 5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _weights_exist() -> bool:
    return (
        os.path.exists(CONFIG.get("weights_path", "model_weights.pth")) and
        os.path.exists(CONFIG.get("vocab_path", "vocab.json"))
    )


def _ensure_training_data() -> None:
    td = CONFIG["training_dir"]
    os.makedirs(td, exist_ok=True)
    files = [f for f in os.listdir(td) if os.path.isfile(os.path.join(td, f))]
    if not files:
        print(
            f"\n[main] WARNING: '{td}/' is empty.\n"
            "       Drop .py or .js source files into that directory and rerun.\n"
        )
        sys.exit(1)


def _train() -> None:
    from train import run_training
    _ensure_training_data()
    print("\n" + "=" * 60)
    print(f"  TRAINING MODE  (preset: {CONFIG.get('preset_name', 'custom')})")
    print("=" * 60 + "\n")
    run_training(config=CONFIG)
    print("\n[main] Training complete.")


def _interactive_shell() -> None:
    from generate import generate_text
    from beam_search import beam_search_from_disk

    print("\n" + "=" * 60)
    print("  CODE GENERATION SHELL  (type 'quit' or Ctrl-C to exit)")
    print("=" * 60)
    print(f"  preset      = {CONFIG.get('preset_name', 'custom')}")
    print(f"  strategy    = {GEN_STRATEGY}")
    print(f"  max_tokens  = {GEN_MAX_TOKENS}")
    if GEN_STRATEGY == "topk":
        print(f"  temperature = {GEN_TEMPERATURE}")
        print(f"  top_k       = {GEN_TOP_K}")
    elif GEN_STRATEGY == "beam":
        print(f"  beam_width  = {GEN_BEAM_WIDTH}")
    print("=" * 60 + "\n")

    while True:
        try:
            prompt = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[main] Exiting.")
            break

        if not prompt:
            continue
        if prompt.lower() in {"quit", "exit", "q"}:
            print("[main] Goodbye.")
            break

        print()
        try:
            if GEN_STRATEGY == "beam":
                beam_search_from_disk(
                    prompt         = prompt,
                    weights_path   = CONFIG["weights_path"],
                    vocab_path     = CONFIG["vocab_path"],
                    beam_width     = GEN_BEAM_WIDTH,
                    max_new_tokens = GEN_MAX_TOKENS,
                    stream_best    = True,
                )
            else:
                generate_text(
                    prompt         = prompt,
                    weights_path   = CONFIG["weights_path"],
                    vocab_path     = CONFIG["vocab_path"],
                    max_new_tokens = GEN_MAX_TOKENS,
                    temperature    = GEN_TEMPERATURE,
                    top_k          = GEN_TOP_K,
                    strategy       = GEN_STRATEGY,
                    stream         = True,
                )
        except Exception as exc:
            print(f"\n[generate] Error: {exc}")
        print()


def _single_shot(prompt: str) -> None:
    from generate import generate_text
    from beam_search import beam_search_from_disk
    print()
    if GEN_STRATEGY == "beam":
        beam_search_from_disk(
            prompt         = prompt,
            weights_path   = CONFIG["weights_path"],
            vocab_path     = CONFIG["vocab_path"],
            beam_width     = GEN_BEAM_WIDTH,
            max_new_tokens = GEN_MAX_TOKENS,
            stream_best    = True,
        )
    else:
        generate_text(
            prompt         = prompt,
            weights_path   = CONFIG["weights_path"],
            vocab_path     = CONFIG["vocab_path"],
            max_new_tokens = GEN_MAX_TOKENS,
            temperature    = GEN_TEMPERATURE,
            top_k          = GEN_TOP_K,
            strategy       = GEN_STRATEGY,
            stream         = True,
        )


def _run_benchmark() -> None:
    from benchmark import run_benchmark
    run_benchmark(
        weights_path = CONFIG.get("weights_path", "model_weights.pth"),
        vocab_path   = CONFIG.get("vocab_path", "vocab.json"),
        preset       = CONFIG.get("preset_name", "custom"),
    )


def _run_export() -> None:
    from export import run_export
    run_export(
        weights_path = CONFIG.get("weights_path", "model_weights.pth"),
        vocab_path   = CONFIG.get("vocab_path", "vocab.json"),
        do_inspect   = True,
        do_flops     = True,
        do_stats     = False,
        do_torchscript = False,
        do_onnx        = False,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    global CONFIG, GEN_MAX_TOKENS, GEN_TEMPERATURE, GEN_TOP_K, GEN_STRATEGY, GEN_BEAM_WIDTH  # noqa

    parser = argparse.ArgumentParser(
        description="Transformer LM — train on local code files and generate completions.",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # ── Config / preset ──────────────────────────────────────────────────────
    parser.add_argument("--preset", choices=["small", "medium", "large"], default="small",
                        help="Model size preset (default: small).")
    parser.add_argument("--config", "-c", metavar="FILE", default=None,
                        help="YAML or JSON config file (overrides preset keys).")
    parser.add_argument("--list-presets", action="store_true",
                        help="Print available model size presets and exit.")
    parser.add_argument("--save-config", metavar="PATH", default=None,
                        help="Save the resolved config to a file and exit.")

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    parser.add_argument("--tokenizer", choices=["char", "bpe"], default="char",
                        help=(
                            "Tokenization strategy (default: char).\n"
                            "  char — character-level (fast, no extra deps)\n"
                            "  bpe  — Byte Pair Encoding (better for larger corpora)"
                        ))
    parser.add_argument("--bpe-vocab-size", type=int, default=1000,
                        help="BPE vocabulary size when --tokenizer bpe (default: 1000).")

    # ── Training ─────────────────────────────────────────────────────────────
    parser.add_argument("--retrain",  action="store_true",
                        help="Force re-training even if weights already exist.")
    parser.add_argument("--resume",   action="store_true",
                        help="Resume training from the latest checkpoint.")
    parser.add_argument("--scheduler", choices=["cosine", "constant", "linear"],
                        default=None,
                        help="LR scheduler (default: cosine).")

    # ── Generation ───────────────────────────────────────────────────────────
    parser.add_argument("--prompt", "-p", type=str, default=None,
                        help="Single-shot generation: produce output and exit.")
    parser.add_argument("--strategy", "-s",
                        choices=["greedy", "topk", "beam"],
                        default="topk",
                        help=(
                            "Generation strategy (default: topk).\n"
                            "  greedy — deterministic, picks highest-prob token\n"
                            "  topk   — temperature + top-k sampling\n"
                            "  beam   — beam search (use --beam-width to configure)"
                        ))
    parser.add_argument("--beam-width", type=int, default=5,
                        help="Beam width for beam search (default: 5).")
    parser.add_argument("--max_tokens", "-n", type=int, default=300,
                        help="Max new tokens to generate (default: 300).")
    parser.add_argument("--temperature", "-t", type=float, default=0.8,
                        help="Sampling temperature for top-k (default: 0.8).")
    parser.add_argument("--top_k", "-k", type=int, default=40,
                        help="Top-k filtering for top-k strategy (default: 40).")

    # ── Data augmentation ────────────────────────────────────────────────────
    parser.add_argument("--augment", action="store_true",
                        help=(
                            "Augment training data before training.\n"
                            "Writes augmented files to training_data_aug/ then trains on that."
                        ))
    parser.add_argument("--augment-dir", default="training_data_aug",
                        help="Output directory for augmented data (default: training_data_aug).")
    parser.add_argument("--augment-copies", type=int, default=2,
                        help="Augmented copies per source file (default: 2).")
    parser.add_argument("--augment-techniques", nargs="+", default=None,
                        metavar="TECHNIQUE",
                        help="Specific augmentation techniques to apply (default: all).")
    parser.add_argument("--list-techniques", action="store_true",
                        help="List all available data augmentation techniques and exit.")

    # ── Tooling ───────────────────────────────────────────────────────────────
    parser.add_argument("--benchmark", action="store_true",
                        help="Run training throughput + inference latency benchmarks.")
    parser.add_argument("--export", action="store_true",
                        help="Inspect architecture and estimate FLOPs (no generation).")

    args = parser.parse_args()

    # ── --list-presets ───────────────────────────────────────────────────────
    if args.list_presets:
        list_presets()
        sys.exit(0)

    if args.list_techniques:
        from data_augment import TECHNIQUES
        print("\nAvailable augmentation techniques:\n")
        for name, (_, desc) in TECHNIQUES.items():
            print(f"  {name:<20}  {desc}")
        print()
        sys.exit(0)

    # ── Resolve config ───────────────────────────────────────────────────────
    CONFIG = load_config(preset=args.preset, config_file=args.config)

    # Apply CLI tokenizer / scheduler overrides into CONFIG
    CONFIG["tokenizer"]      = args.tokenizer
    CONFIG["bpe_vocab_size"] = args.bpe_vocab_size
    CONFIG["resume"]         = args.resume
    if args.scheduler:
        CONFIG["scheduler"] = args.scheduler

    # ── --save-config ─────────────────────────────────────────────────────────
    if args.save_config:
        save_config(CONFIG, args.save_config)
        print(f"[main] Edit '{args.save_config}' then re-run with --config {args.save_config}")
        sys.exit(0)

    # ── Generation settings ───────────────────────────────────────────────────
    GEN_MAX_TOKENS  = args.max_tokens
    GEN_TEMPERATURE = args.temperature
    GEN_TOP_K       = args.top_k
    GEN_STRATEGY    = args.strategy
    GEN_BEAM_WIDTH  = args.beam_width

    # ── Benchmark (no training needed) ───────────────────────────────────────
    if args.benchmark:
        if not _weights_exist():
            print("[main] No weights found — training first before benchmarking.")
            _train()
        _run_benchmark()
        return

    # ── Export / inspect (no training needed) ────────────────────────────────
    if args.export:
        if not _weights_exist():
            print("[main] No weights found — training first before exporting.")
            _train()
        _run_export()
        return

    # ── Data augmentation (runs before training) ─────────────────────────────
    if args.augment:
        from data_augment import augment_directory, measure_expansion
        aug_dir = args.augment_dir
        print(f"\n[main] Augmenting training data → '{aug_dir}' ...")
        augment_directory(
            src_dir    = CONFIG["training_dir"],
            out_dir    = aug_dir,
            techniques = args.augment_techniques,
            n_copies   = args.augment_copies,
        )
        measure_expansion(CONFIG["training_dir"], aug_dir)
        # Point training at the augmented directory
        CONFIG["training_dir"] = aug_dir
        print(f"[main] Training will use augmented data from '{aug_dir}'.")

    # ── Train if necessary ────────────────────────────────────────────────────
    if args.augment or args.retrain or not _weights_exist():
        _train()
    else:
        print(
            f"[main] Found existing weights at '{CONFIG['weights_path']}'. "
            "Skipping training."
        )
        print("[main] Run with --retrain to force a new training run.\n")

    # ── Generate ──────────────────────────────────────────────────────────────
    if args.prompt is not None:
        _single_shot(args.prompt)
    else:
        _interactive_shell()


if __name__ == "__main__":
    main()
