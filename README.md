# my-ai — Local Code Completion Transformer

A fully self-contained, character-level Transformer Language Model that trains
on your own source files and runs **offline** code-completion inference
entirely in the terminal. No cloud APIs, no web servers.

## Quick Start

```bash
# 1. Install PyTorch
pip install torch --user

# 2. Drop your .py / .js source files into training_data/
#    (sample files are already included to get you started)

# 3. Train + generate — all in one command
cd ml-pipeline
python main.py
```

After training finishes you land in an interactive REPL:

```
>>> def fibonacci(
def fibonacci(n):
    if n <= 1:
        return n
    a, b = 0, 1
    ...
```

## Module Map

| File | Responsibility |
|------|---------------|
| `config.py` | Preset manager (small / medium / large) + YAML/JSON file support |
| `tokenizer.py` | Vocabulary building, encode / decode |
| `model.py` | Decoder-only Transformer from scratch — causal attention, Pre-LN, GeLU FFN |
| `dataset.py` | `CodeDataset` + `DataLoader` — sliding context windows, shifted (X, Y) pairs |
| `train.py` | AdamW loop, gradient clipping, loss logging, auto-save on interrupt |
| `generate.py` | Autoregressive inference — temperature scaling + top-k filtering |
| `main.py` | Unified entry point |

## CLI Reference

```bash
python main.py                              # auto-train if needed, then REPL
python main.py --preset medium             # larger model (requires GPU)
python main.py --config my_run.yaml        # load hyperparams from file
python main.py --retrain                   # force re-train
python main.py --prompt "def fib(" -n 200 # single-shot, no REPL
python main.py --list-presets              # show size presets
python config.py --save small              # dump starter config to small.yaml
python generate.py --prompt "class BST:"  # standalone inference
```

## Model Presets

| Preset | Params | Recommended for |
|--------|--------|----------------|
| small  | ~109K  | CPU, quick experiments |
| medium | ~870K  | GPU, medium corpus |
| large  | ~6.8M  | CUDA GPU, large corpus |

## Stack

- Python 3.11 · PyTorch 2.x
- Decoder-only Transformer (Pre-LN, causal self-attention, GeLU FFN, weight tying)
- Character-level tokenisation · AdamW + gradient clipping
