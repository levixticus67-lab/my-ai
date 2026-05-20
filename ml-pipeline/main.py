"""
main.py — Unified entry point for the ML pipeline.

Behaviour:
  1. If model_weights.pth does NOT exist (or --retrain is passed):
       → Scan training_data/, build vocabulary, train the model, save weights.
  2. If model_weights.pth DOES exist:
       → Skip training, load existing weights directly.
  3. Drop into an interactive terminal prompt shell for offline code generation.

Usage:
  python main.py                              # auto-train if needed, then generate
  python main.py --retrain                    # force re-training
  python main.py --preset medium             # use a larger model preset
  python main.py --config my_run.yaml        # load hyperparams from a file
  python main.py --prompt "def fib("         # single-shot generation (no REPL)
  python main.py --list-presets              # show available model size presets

Config precedence (later overrides earlier):
  built-in preset  →  --config file  →  CLI generation flags
"""

import argparse
import os
import sys

from config import load_config, list_presets, save_config

# Runtime config — populated in main() from the chosen preset + optional file
CONFIG: dict = {}

# Generation defaults — may be overridden by CLI flags
GEN_MAX_TOKENS  = 300
GEN_TEMPERATURE = 0.8
GEN_TOP_K       = 40


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

    print("\n" + "=" * 60)
    print("  CODE GENERATION SHELL  (type 'quit' or Ctrl-C to exit)")
    print("=" * 60)
    print(f"  preset      = {CONFIG.get('preset_name', 'custom')}")
    print(f"  max_tokens  = {GEN_MAX_TOKENS}")
    print(f"  temperature = {GEN_TEMPERATURE}")
    print(f"  top_k       = {GEN_TOP_K}")
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
            generate_text(
                prompt=prompt,
                weights_path=CONFIG["weights_path"],
                vocab_path=CONFIG["vocab_path"],
                max_new_tokens=GEN_MAX_TOKENS,
                temperature=GEN_TEMPERATURE,
                top_k=GEN_TOP_K,
                stream=True,
            )
        except Exception as exc:
            print(f"\n[generate] Error: {exc}")
        print()


def _single_shot(prompt: str) -> None:
    from generate import generate_text
    print()
    generate_text(
        prompt=prompt,
        weights_path=CONFIG["weights_path"],
        vocab_path=CONFIG["vocab_path"],
        max_new_tokens=GEN_MAX_TOKENS,
        temperature=GEN_TEMPERATURE,
        top_k=GEN_TOP_K,
        stream=True,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    global CONFIG, GEN_MAX_TOKENS, GEN_TEMPERATURE, GEN_TOP_K  # noqa: PLW0603

    parser = argparse.ArgumentParser(
        description="Transformer LM — train on local code files and generate completions.",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # ── Config / preset ──────────────────────────────────────────────────────
    parser.add_argument(
        "--preset",
        choices=["small", "medium", "large"],
        default="small",
        help=(
            "Model size preset (default: small).\n"
            "  small  — ~109K params, CPU-friendly, trains in minutes\n"
            "  medium — ~870K params, good GPU target\n"
            "  large  — ~6.8M params, requires CUDA GPU\n"
            "Run --list-presets for full details."
        ),
    )
    parser.add_argument(
        "--config", "-c",
        metavar="FILE",
        default=None,
        help=(
            "Path to a YAML or JSON config file.\n"
            "Keys override the selected preset.\n"
            "Generate a starter file with: python config.py --save small"
        ),
    )
    parser.add_argument(
        "--list-presets",
        action="store_true",
        help="Print available model size presets and exit.",
    )
    parser.add_argument(
        "--save-config",
        metavar="PATH",
        default=None,
        help="Save the resolved config to a file and exit (e.g. my_run.yaml).",
    )

    # ── Training ─────────────────────────────────────────────────────────────
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Force re-training even if a weights file already exists.",
    )

    # ── Generation ───────────────────────────────────────────────────────────
    parser.add_argument(
        "--prompt", "-p",
        type=str,
        default=None,
        help=(
            "Single-shot generation: produce output and exit (no REPL).\n"
            'Example: --prompt "def fibonacci("'
        ),
    )
    parser.add_argument(
        "--max_tokens", "-n",
        type=int,
        default=300,
        help="Max new tokens to generate (default: 300).",
    )
    parser.add_argument(
        "--temperature", "-t",
        type=float,
        default=0.8,
        help="Sampling temperature (default: 0.8).\n"
             "  < 1.0 → more deterministic   > 1.0 → more creative",
    )
    parser.add_argument(
        "--top_k", "-k",
        type=int,
        default=40,
        help="Top-k filtering — keep only top-k tokens before sampling (default: 40).",
    )

    args = parser.parse_args()

    # ── --list-presets ───────────────────────────────────────────────────────
    if args.list_presets:
        list_presets()
        sys.exit(0)

    # ── Resolve config ───────────────────────────────────────────────────────
    CONFIG = load_config(preset=args.preset, config_file=args.config)

    # ── --save-config ────────────────────────────────────────────────────────
    if args.save_config:
        save_config(CONFIG, args.save_config)
        print(f"[main] Edit '{args.save_config}' then re-run with --config {args.save_config}")
        sys.exit(0)

    # ── Apply generation overrides ───────────────────────────────────────────
    GEN_MAX_TOKENS  = args.max_tokens
    GEN_TEMPERATURE = args.temperature
    GEN_TOP_K       = args.top_k

    # ── Step 1: Train if necessary ───────────────────────────────────────────
    if args.retrain or not _weights_exist():
        _train()
    else:
        print(
            f"[main] Found existing weights at '{CONFIG['weights_path']}'. "
            "Skipping training."
        )
        print("[main] Run with --retrain to force a new training run.\n")

    # ── Step 2: Generate ────────────────────────────────────────────────────
    if args.prompt is not None:
        _single_shot(args.prompt)
    else:
        _interactive_shell()


if __name__ == "__main__":
    main()
