# my-ai — Local Code Completion Transformer

A fully self-contained, character-level (or BPE) Transformer Language Model
that trains on your own source files and runs **offline** code-completion
inference entirely in the terminal.  No cloud APIs, no web servers, no
internet required after setup.

---

## Quick Start

```bash
# 1. Install PyTorch (one-time)
pip install torch --user

# 2. Drop your .py / .js source files into training_data/
#    (sample files are already included)

# 3. Train + generate — all in one command
cd ml-pipeline
python main.py
```

After training you land in an interactive REPL:

```
>>> def fibonacci(
def fibonacci(n):
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b
```

---

## Module Map

| File | What it does |
|------|-------------|
| `main.py` | Unified entry point — trains if needed, then opens the generation REPL |
| `config.py` | Preset manager (small / medium / large) + YAML/JSON file support |
| `tokenizer.py` | Character-level vocabulary builder, encode / decode |
| `bpe_tokenizer.py` | **[NEW]** Byte Pair Encoding tokenizer — better subword coverage |
| `model.py` | Decoder-only Transformer built from scratch with PyTorch |
| `dataset.py` | `CodeDataset` + `DataLoader` — sliding context windows, (X, Y) pairs |
| `train.py` | AdamW training loop with scheduler, validation, and checkpointing |
| `generate.py` | Greedy / top-k / beam inference with token streaming |
| `checkpoint.py` | **[NEW]** Mid-training checkpoint manager — save, resume, prune |
| `evaluate.py` | **[NEW]** Validation split + perplexity tracking per epoch |
| `lr_scheduler.py` | **[NEW]** Cosine / linear / constant warmup LR schedulers |
| `beam_search.py` | **[NEW]** Full beam search decoder with length normalisation |
| `benchmark.py` | **[NEW]** Training throughput + inference latency benchmarker |
| `export.py` | **[NEW]** TorchScript / ONNX export, architecture inspector, FLOPs estimator |

---

## Full CLI Reference

### `main.py` — primary entry point

```bash
# Auto-train if no weights exist, then open generation REPL
python main.py

# Force re-train even if weights already exist
python main.py --retrain

# Resume training from the latest mid-run checkpoint
python main.py --resume

# Use a larger model preset
python main.py --preset medium

# Load hyperparams from a custom YAML/JSON config file
python main.py --config my_run.yaml

# Save the resolved config to a file (then edit and reuse)
python main.py --preset medium --save-config medium.yaml

# Change generation strategy
python main.py --strategy greedy
python main.py --strategy beam --beam-width 5

# Use BPE tokenization instead of character-level
python main.py --tokenizer bpe --bpe-vocab-size 2000

# Single-shot generation (no REPL, exits after printing)
python main.py --prompt "def fibonacci(" --max_tokens 200

# Run benchmarks (trains first if needed)
python main.py --benchmark

# Inspect architecture + FLOPs (trains first if needed)
python main.py --export

# Show all model size presets
python main.py --list-presets
```

### `generate.py` — standalone inference

```bash
python generate.py --prompt "def quicksort(" --strategy topk --max_tokens 300
python generate.py --prompt "class BinaryTree:" --strategy greedy
python generate.py --prompt "for i in range(" --strategy beam --beam-width 5
```

### `beam_search.py` — standalone beam search

```bash
python beam_search.py --prompt "def merge_sort(" --beam-width 5 --top-n 3
python beam_search.py --prompt "class Graph:" --beam-width 8 --length-penalty 0.6
```

### `checkpoint.py` — checkpoint management

```bash
# List all saved checkpoints with their val losses
python checkpoint.py --list

# List from a non-default directory
python checkpoint.py --dir my_checkpoints --list
```

### `lr_scheduler.py` — preview a schedule

```bash
python lr_scheduler.py --scheduler cosine --warmup 200 --total-steps 5000 --plot
python lr_scheduler.py --scheduler linear --warmup 100 --total-steps 3000
```

### `bpe_tokenizer.py` — train or test BPE vocabulary

```bash
python bpe_tokenizer.py --train training_data/ --vocab-size 1000 --out bpe_vocab.json
python bpe_tokenizer.py --vocab bpe_vocab.json --test "def fibonacci(n):"
```

### `benchmark.py` — performance measurement

```bash
python benchmark.py
python benchmark.py --fast --out results.json
python benchmark.py --preset medium
```

### `export.py` — export and inspection

```bash
# Architecture summary + FLOPs estimate
python export.py --inspect --flops

# Weight statistics (detect instability)
python export.py --weight-stats

# Export to TorchScript (runs without source code)
python export.py --torchscript --out-dir exports/

# Export to ONNX (for deployment with ONNX Runtime, TensorRT, CoreML, etc.)
python export.py --onnx --out-dir exports/

# Run all tasks at once
python export.py --all
```

### `config.py` — preset viewer and saver

```bash
python config.py --list
python config.py --save medium --format yaml
```

---

## Model Presets

| Preset | Est. Params | n_embd | Heads | Layers | Block | Best for |
|--------|------------|--------|-------|--------|-------|----------|
| small  | ~109K      | 128    | 4     | 4      | 128   | CPU, quick experiments |
| medium | ~870K      | 256    | 8     | 6      | 256   | GPU, medium corpus |
| large  | ~6.8M      | 512    | 8     | 8      | 512   | CUDA GPU, large corpus |

Edit any hyperparameter by saving a config file and passing it with `--config`:

```bash
python config.py --save small          # writes small.yaml
# Edit small.yaml: change max_epochs to 20, batch_size to 64 ...
python main.py --config small.yaml
```

---

## Feature Deep-Dives

### Feature 1 — Checkpoint Manager (`checkpoint.py`)

Saves a snapshot of model weights **and** optimizer state every N optimiser
steps during training.  Keeps only the top-K checkpoints by validation loss
(older/worse ones are automatically deleted).

**Why it matters:** If your machine crashes mid-training, you lose nothing.
Resume from where you left off with `--resume`.

```bash
# Resume training from the latest checkpoint
python main.py --resume

# See what checkpoints exist and which has the best val loss
python checkpoint.py --list
```

Config keys (in `main.py`'s `CONFIG` or a YAML file):
```yaml
checkpoint_dir:   checkpoints    # folder for checkpoint files
checkpoint_every: 500            # save every N optimiser steps
keep_top_k:       3              # keep only the 3 best checkpoints
```

---

### Feature 2 — Validation & Perplexity Tracking (`evaluate.py`)

Automatically splits the corpus into a **training set** (90%) and a
**held-out validation set** (10%) before training begins.  After every
epoch (and optionally every N steps), the model is evaluated on the
validation set and its **perplexity** is printed.

**Perplexity** measures how "surprised" the model is by unseen text.
Lower = better.  A random model on a 91-character vocabulary has perplexity
≈ 91.  A well-trained model reaches 10–30.

At the end of training a full history table is printed:

```
  VALIDATION HISTORY
  Epoch     Step    Val Loss  Perplexity
  ───────────────────────────────────────
      1     1500      2.9431       19.00  ← best
      2     3000      3.0812       21.78
```

Config keys:
```yaml
val_ratio:   0.1   # fraction of corpus held out
eval_every:  500   # also evaluate mid-epoch every N steps (0 = epoch only)
```

---

### Feature 3 — BPE Tokenizer (`bpe_tokenizer.py`)

A pure-Python **Byte Pair Encoding** tokenizer that compresses common
substrings (keywords, common identifiers) into single tokens.

**Why it matters:** Character-level models waste context on common patterns
like `def `, `return `, `    ` (indentation).  BPE turns these into single
tokens, so the model sees more meaningful context in the same block size.

```bash
# Train BPE vocabulary and test it
python bpe_tokenizer.py --train training_data/ --vocab-size 1000 --out bpe_vocab.json
python bpe_tokenizer.py --vocab bpe_vocab.json --test "def fibonacci(n):"
# Output:
#   Input   : 'def fibonacci(n):'
#   Tokens  : ['def ', 'fib', 'on', 'acci', '(n):']
#   Ratio   : 5/18 tokens/chars

# Use BPE for training + generation
python main.py --tokenizer bpe --bpe-vocab-size 1000
```

---

### Feature 4 — LR Scheduler with Warmup (`lr_scheduler.py`)

Three learning-rate schedules, all with a **linear warmup** phase:

| Schedule | After warmup |
|----------|-------------|
| `cosine` | Smoothly decays to `min_lr` following a cosine curve |
| `linear` | Linearly decays to `min_lr` |
| `constant` | Holds at the peak LR for the rest of training |

**Why it matters:** Jumping straight to `lr=1e-3` on step 1 can destabilize
training.  Warmup lets the optimizer build reliable gradient estimates first.
Cosine decay then gently reduces the LR as the model converges, squeezing
out extra performance.

```bash
# Preview the cosine schedule as an ASCII chart
python lr_scheduler.py --scheduler cosine --warmup 200 --total-steps 5000 --plot
```

Config keys:
```yaml
scheduler:     cosine   # cosine | linear | constant
warmup_steps:  200
min_lr_ratio:  0.1      # floor = base_lr × min_lr_ratio
```

---

### Feature 5 — Beam Search Decoder (`beam_search.py`)

Instead of greedily picking one token at a time, beam search keeps the **B
most probable token sequences** alive simultaneously, then ranks them at the
end by a **length-normalised score**.

**Why it matters:** Top-k sampling is creative but sometimes incoherent.
Greedy is coherent but can get stuck in repetitive loops.  Beam search finds
higher-probability completions and returns **multiple ranked candidates**,
giving you options.

```bash
# Generate with beam search, show top 3 candidates
python main.py --strategy beam --beam-width 5
python beam_search.py --prompt "def merge_sort(" --beam-width 5 --top-n 3
```

Key parameters:
- `--beam-width` — number of beams (higher = better quality, slower)
- `--length-penalty` — 0.0 favours short, 1.0 favours long (default 0.7)
- `--rep-penalty` — values > 1.0 penalise repeated tokens

---

### Feature 6 — Performance Benchmarker (`benchmark.py`)

Measures two things automatically:

1. **Training throughput** — how many tokens per second the forward + backward
   pass processes on your hardware.
2. **Inference latency** — milliseconds per generated token at different
   prompt lengths.

Outputs a formatted report, optionally saved as JSON for comparison runs.

```bash
python benchmark.py
# Output:
#   ── TRAINING THROUGHPUT ──
#      Tokens/sec :      24,310
#      ms / step  :      42.40
#
#   ── INFERENCE LATENCY ──────
#     Prompt Len   ms/token        ±   tok/sec
#              8       3.21     0.12     311.8
#             32       3.45     0.09     289.7
#             64       3.89     0.15     257.1

python benchmark.py --out results.json   # save for later comparison
python benchmark.py --fast               # quick smoke-test
```

---

### Feature 7 — Model Export & Inspector (`export.py`)

Four tools in one script:

| Tool | What it does |
|------|-------------|
| `--inspect` | Per-layer parameter count breakdown + distribution bar chart |
| `--flops` | Analytic FLOPs/MACs estimate for one forward pass |
| `--weight-stats` | min/max/mean/std of every parameter (detects instability) |
| `--torchscript` | Export to `.pt` — runs anywhere PyTorch is installed, no source needed |
| `--onnx` | Export to `.onnx` — deploy with ONNX Runtime, TensorRT, CoreML, etc. |

```bash
# Inspect architecture
python export.py --inspect --flops
# Output:
#   Vocab size   : 91
#   Embedding dim: 128
#   Heads        : 4  (head_dim = 32)
#   Layers       : 4
#   ...
#   TOTAL FLOPs (≈2×MACs)         179.7 M

# Export to TorchScript for deployment
python export.py --torchscript --out-dir exports/
# → exports/model.pt  (self-contained, no source code needed)

# Run everything
python export.py --all
```

---

## Generated Files (at runtime)

```
ml-pipeline/
├── vocab.json            character-level vocabulary
├── bpe_vocab.json        BPE vocabulary (if --tokenizer bpe was used)
├── model_weights.pth     final trained weights
├── checkpoints/          mid-training snapshots
│   ├── checkpoint_index.json
│   ├── checkpoint_step0000500_loss2.3410.pth
│   └── checkpoint_step0001000_loss1.9823.pth
└── exports/              TorchScript / ONNX output (after python export.py)
    ├── model.pt
    └── model.onnx
```

---

## Stack

- Python 3.11 · PyTorch 2.x (CPU or CUDA)
- Decoder-only Transformer — Pre-LayerNorm, causal self-attention, GeLU FFN, weight tying
- Character-level **or** BPE tokenisation
- AdamW + gradient clipping + cosine LR warmup
- Mid-training checkpointing with val-loss ranking
- Three generation strategies: greedy, top-k sampling, beam search

---

## Improving Output Quality

1. **Add more training data** — drop `.py` / `.js` files into `training_data/`
   and run `python main.py --retrain`.
2. **Use a larger preset** — `python main.py --preset medium` (needs a GPU).
3. **Switch to BPE** — `python main.py --tokenizer bpe` for more efficient tokenisation.
4. **Train longer** — edit `max_epochs` in a config YAML.
5. **Use beam search** — `python main.py --strategy beam` for higher-quality completions.
