"""
config.py — Hyperparameter management with YAML/JSON file support and built-in size presets.

Usage examples
--------------
# Load the default (small) preset:
from config import load_config
cfg = load_config()

# Load the medium preset:
cfg = load_config(preset="medium")

# Load a custom YAML file (overrides any matching keys):
cfg = load_config(preset="medium", config_file="my_run.yaml")

# Save the current config for later reference:
from config import save_config
save_config(cfg, "my_run.yaml")

Preset summary
--------------
  small  — 128-dim, 4 heads, 4 layers   (~109K params, CPU-friendly, trains in minutes)
  medium — 256-dim, 8 heads, 6 layers   (~870K params, good GPU target)
  large  — 512-dim, 8 heads, 8 layers   (~6.8M params, serious GPU required)
"""

import json
import os
from copy import deepcopy
from typing import Any, Dict, Optional

# ---------------------------------------------------------------------------
# Try YAML, fall back to JSON-only mode gracefully
# ---------------------------------------------------------------------------
try:
    import yaml as _yaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False


# ---------------------------------------------------------------------------
# Built-in presets
# ---------------------------------------------------------------------------

_PRESETS: Dict[str, Dict[str, Any]] = {
    "small": {
        # ── Model ──────────────────────────────────────────────────────────
        "n_embd":     128,
        "n_heads":    4,
        "n_layers":   4,
        "block_size": 128,
        "dropout":    0.10,

        # ── Optimiser ──────────────────────────────────────────────────────
        "lr":           1e-3,
        "weight_decay": 0.01,
        "grad_clip":    1.0,

        # ── Training ───────────────────────────────────────────────────────
        "batch_size": 32,
        "max_epochs": 5,
        "log_every":  100,

        # ── I/O ────────────────────────────────────────────────────────────
        "training_dir":  "training_data",
        "vocab_path":    "vocab.json",
        "weights_path":  "model_weights.pth",

        # ── Meta ───────────────────────────────────────────────────────────
        "preset_name": "small",
        "description": "~109K params — CPU-friendly, trains in minutes on a small corpus.",
    },

    "medium": {
        "n_embd":     256,
        "n_heads":    8,
        "n_layers":   6,
        "block_size": 256,
        "dropout":    0.10,

        "lr":           3e-4,
        "weight_decay": 0.01,
        "grad_clip":    1.0,

        "batch_size": 32,
        "max_epochs": 10,
        "log_every":  100,

        "training_dir":  "training_data",
        "vocab_path":    "vocab.json",
        "weights_path":  "model_weights.pth",

        "preset_name": "medium",
        "description": "~870K params — good GPU target, requires a few hundred MB of source files.",
    },

    "large": {
        "n_embd":     512,
        "n_heads":    8,
        "n_layers":   8,
        "block_size": 512,
        "dropout":    0.10,

        "lr":           1e-4,
        "weight_decay": 0.01,
        "grad_clip":    1.0,

        "batch_size": 16,
        "max_epochs": 20,
        "log_every":  100,

        "training_dir":  "training_data",
        "vocab_path":    "vocab.json",
        "weights_path":  "model_weights.pth",

        "preset_name": "large",
        "description": "~6.8M params — requires a CUDA GPU and a large training corpus.",
    },
}

# Default preset used when none is specified
DEFAULT_PRESET = "small"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def list_presets() -> None:
    """Print a summary of all available built-in presets."""
    print("\nAvailable presets:")
    print(f"  {'Name':<10} {'Params (est)':<16} {'n_embd':<8} {'heads':<7} {'layers':<8} Description")
    print("  " + "-" * 78)
    specs = {
        "small":  ("~109K",  _PRESETS["small"]),
        "medium": ("~870K",  _PRESETS["medium"]),
        "large":  ("~6.8M",  _PRESETS["large"]),
    }
    for name, (params, p) in specs.items():
        print(
            f"  {name:<10} {params:<16} {p['n_embd']:<8} {p['n_heads']:<7} "
            f"{p['n_layers']:<8} {p['description']}"
        )
    print()


def load_config(
    preset: str = DEFAULT_PRESET,
    config_file: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build a config dictionary.

    Resolution order (later entries override earlier ones):
      1. The named built-in preset.
      2. Keys from `config_file` (YAML or JSON), if provided.

    Args:
        preset:      One of "small", "medium", "large" (default: "small").
        config_file: Optional path to a .yaml, .yml, or .json override file.

    Returns:
        A flat dictionary of all hyperparameters.
    """
    if preset not in _PRESETS:
        raise ValueError(
            f"Unknown preset '{preset}'. Choose from: {list(_PRESETS.keys())}"
        )

    cfg = deepcopy(_PRESETS[preset])

    if config_file is not None:
        overrides = _load_file(config_file)
        unknown = set(overrides) - set(cfg)
        if unknown:
            print(f"[config] Warning: unknown keys in '{config_file}': {sorted(unknown)}")
        cfg.update(overrides)
        print(f"[config] Loaded overrides from '{config_file}'.")

    print(
        f"[config] Preset: {cfg.get('preset_name', preset)}  |  "
        f"n_embd={cfg['n_embd']}, heads={cfg['n_heads']}, "
        f"layers={cfg['n_layers']}, block={cfg['block_size']}, "
        f"lr={cfg['lr']}, epochs={cfg['max_epochs']}"
    )
    return cfg


def save_config(cfg: Dict[str, Any], path: str) -> None:
    """
    Persist a config dictionary to disk.

    Writes YAML if PyYAML is available and the path ends with .yaml/.yml;
    otherwise writes JSON.

    Args:
        cfg:  Config dictionary to save.
        path: Destination file path.
    """
    ext = os.path.splitext(path)[1].lower()
    if _YAML_AVAILABLE and ext in (".yaml", ".yml"):
        with open(path, "w", encoding="utf-8") as f:
            _yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    else:
        if ext in (".yaml", ".yml") and not _YAML_AVAILABLE:
            alt = os.path.splitext(path)[0] + ".json"
            print(
                f"[config] PyYAML not installed — saving as JSON to '{alt}' instead.\n"
                f"         Install it with: pip install pyyaml --user"
            )
            path = alt
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)

    print(f"[config] Config saved to '{path}'.")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_file(path: str) -> Dict[str, Any]:
    """Read a YAML or JSON file and return a plain dict."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: '{path}'")

    ext = os.path.splitext(path)[1].lower()

    if ext in (".yaml", ".yml"):
        if not _YAML_AVAILABLE:
            raise ImportError(
                "PyYAML is required to load .yaml config files.\n"
                "Install it with: pip install pyyaml --user"
            )
        with open(path, "r", encoding="utf-8") as f:
            data = _yaml.safe_load(f)
    elif ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        raise ValueError(f"Unsupported config format '{ext}'. Use .yaml, .yml, or .json.")

    if not isinstance(data, dict):
        raise ValueError(f"Config file '{path}' must contain a mapping at the top level.")

    return data


# ---------------------------------------------------------------------------
# CLI: python3 config.py [--save-preset small|medium|large]
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Config tool — list presets or save one to disk.")
    parser.add_argument("--list",  action="store_true",  help="Show all available presets.")
    parser.add_argument(
        "--save",
        metavar="PRESET",
        choices=list(_PRESETS.keys()),
        help="Save a preset to disk (writes <preset>.yaml or <preset>.json).",
    )
    parser.add_argument(
        "--format",
        choices=["yaml", "json"],
        default="yaml",
        help="Output format when using --save (default: yaml, falls back to json if PyYAML absent).",
    )
    args = parser.parse_args()

    if args.list or (not args.save):
        list_presets()

    if args.save:
        ext = args.format
        out_path = f"{args.save}.{ext}"
        cfg = load_config(preset=args.save)
        save_config(cfg, out_path)
        print(f"Edit '{out_path}', then pass it with: python3 main.py --config {out_path}")
