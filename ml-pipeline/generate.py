"""
generate.py — Inference engine with temperature scaling and top-k filtering.

Usage (standalone):
  python generate.py --prompt "def fibonacci(" --max_tokens 200 --temperature 0.8 --top_k 40

Or call `generate_text()` programmatically from main.py.
"""

import argparse
import os
import sys
import torch
import torch.nn.functional as F
from typing import Optional

from tokenizer import CharTokenizer
from train import load_model_from_checkpoint


# ---------------------------------------------------------------------------
# Core sampling logic
# ---------------------------------------------------------------------------

@torch.no_grad()
def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
) -> int:
    """
    Sample a single next-token index from raw logits.

    Args:
        logits:      1-D float tensor of shape (vocab_size,) — the raw scores
                     for the next token position.
        temperature: Divides logits before softmax.
                     < 1.0 → sharper / more deterministic.
                     > 1.0 → flatter / more random.
                     1.0   → unmodified distribution.
        top_k:       If set, zero out all but the top-k highest-logit tokens
                     before sampling (nucleus-style truncation).

    Returns:
        Sampled token ID as a plain Python int.
    """
    assert logits.ndim == 1, "logits must be 1-D"

    # Temperature scaling
    if temperature != 1.0:
        logits = logits / max(temperature, 1e-8)

    # Top-k filtering
    if top_k is not None and top_k > 0:
        k = min(top_k, logits.size(-1))
        threshold, _ = torch.topk(logits, k)
        min_threshold = threshold[-1]
        logits = logits.masked_fill(logits < min_threshold, float("-inf"))

    probs = F.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1).item()
    return int(next_token)


# ---------------------------------------------------------------------------
# Text generation loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_text(
    prompt: str,
    weights_path: str = "model_weights.pth",
    vocab_path: str = "vocab.json",
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int = 40,
    device: Optional[torch.device] = None,
    stream: bool = True,
) -> str:
    """
    Load model + vocabulary from disk and generate text continuation for `prompt`.

    Args:
        prompt:         Seed text to condition generation on.
        weights_path:   Path to the .pth checkpoint file.
        vocab_path:     Path to the vocab.json file.
        max_new_tokens: How many new tokens to generate.
        temperature:    Sampling temperature (lower = more deterministic).
        top_k:          Keep only top-k tokens before sampling.
        device:         Torch device (auto-detected if None).
        stream:         If True, print each token as it is generated.

    Returns:
        The full generated string (prompt + continuation).
    """
    # ------------------------------------------------------------------
    # Validate paths
    # ------------------------------------------------------------------
    for path, label in [(weights_path, "Weights"), (vocab_path, "Vocabulary")]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{label} file not found: '{path}'\n"
                "Run training first (python main.py) to produce this file."
            )

    # ------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    # ------------------------------------------------------------------
    # Load vocabulary
    # ------------------------------------------------------------------
    tokenizer = CharTokenizer()
    tokenizer.load(vocab_path)

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    model = load_model_from_checkpoint(weights_path, device)
    model.eval()

    block_size = model.config.block_size

    # ------------------------------------------------------------------
    # Encode prompt
    # ------------------------------------------------------------------
    encoded = tokenizer.encode(prompt)
    if not encoded:
        raise ValueError("Prompt encodes to an empty token sequence. Try a different prompt.")

    context = torch.tensor(encoded, dtype=torch.long, device=device).unsqueeze(0)  # (1, T)

    # ------------------------------------------------------------------
    # Autoregressive generation loop
    # ------------------------------------------------------------------
    if stream:
        print(prompt, end="", flush=True)

    generated_ids = list(encoded)

    for _ in range(max_new_tokens):
        # Crop context to block_size if necessary
        ctx = context if context.size(1) <= block_size else context[:, -block_size:]

        logits = model(ctx)                          # (1, T, V)
        next_logits = logits[0, -1, :]               # (V,) — last position

        next_id = sample_next_token(next_logits, temperature=temperature, top_k=top_k)
        generated_ids.append(next_id)

        # Stream the decoded character immediately
        if stream:
            char = tokenizer.decode([next_id])
            print(char, end="", flush=True)

        # Append to running context
        next_token_tensor = torch.tensor([[next_id]], dtype=torch.long, device=device)
        context = torch.cat([context, next_token_tensor], dim=1)

    if stream:
        print()  # newline after streamed output

    return tokenizer.decode(generated_ids)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate code completions using a trained Transformer LM.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--prompt", "-p",
        type=str,
        required=True,
        help='Seed text to start generation from.\nExample: "def fibonacci("',
    )
    parser.add_argument(
        "--max_tokens", "-n",
        type=int,
        default=200,
        help="Maximum number of new tokens to generate (default: 200).",
    )
    parser.add_argument(
        "--temperature", "-t",
        type=float,
        default=0.8,
        help="Sampling temperature (default: 0.8).\n"
             "  < 1.0 → more focused/deterministic\n"
             "  > 1.0 → more creative/random",
    )
    parser.add_argument(
        "--top_k", "-k",
        type=int,
        default=40,
        help="Top-k filtering — keep only top k tokens before sampling (default: 40).",
    )
    parser.add_argument(
        "--weights", "-w",
        type=str,
        default="model_weights.pth",
        help="Path to model weights file (default: model_weights.pth).",
    )
    parser.add_argument(
        "--vocab", "-v",
        type=str,
        default="vocab.json",
        help="Path to vocabulary JSON file (default: vocab.json).",
    )

    args = parser.parse_args()

    try:
        generate_text(
            prompt=args.prompt,
            weights_path=args.weights,
            vocab_path=args.vocab,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            stream=True,
        )
    except FileNotFoundError as e:
        print(f"\n[generate] Error: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n[generate] Generation interrupted.")


if __name__ == "__main__":
    main()
