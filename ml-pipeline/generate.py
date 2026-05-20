"""
generate.py — Inference engine: greedy, top-k sampling, and beam search.

Usage (standalone):
  python generate.py --prompt "def fibonacci(" --strategy topk --max_tokens 200
  python generate.py --prompt "class BinaryTree:" --strategy beam --beam-width 5
  python generate.py --prompt "for i in" --strategy greedy
"""

import argparse
import os
import sys
import torch
import torch.nn.functional as F
from typing import List, Optional

from tokenizer import CharTokenizer
from train import load_model_from_checkpoint


# ---------------------------------------------------------------------------
# Core sampling: greedy and top-k / temperature
# ---------------------------------------------------------------------------

@torch.no_grad()
def sample_next_token(
    logits:      torch.Tensor,
    temperature: float = 1.0,
    top_k:       Optional[int] = None,
    strategy:    str = "topk",
) -> int:
    """
    Sample a single next-token index from raw logits.

    Strategies:
      "greedy"  — always pick the highest-logit token (temperature/top_k ignored).
      "topk"    — temperature scaling + top-k truncation, then sample.

    Args:
        logits:      1-D float tensor of shape (vocab_size,).
        temperature: Divides logits before softmax.
        top_k:       Keep only the top-k highest-logit tokens before sampling.
        strategy:    "greedy" or "topk".

    Returns:
        Sampled token ID as a plain Python int.
    """
    assert logits.ndim == 1, "logits must be 1-D"

    if strategy == "greedy":
        return int(logits.argmax().item())

    # top-k + temperature sampling
    if temperature != 1.0:
        logits = logits / max(temperature, 1e-8)

    if top_k is not None and top_k > 0:
        k = min(top_k, logits.size(-1))
        threshold, _ = torch.topk(logits, k)
        logits = logits.masked_fill(logits < threshold[-1], float("-inf"))

    probs = F.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


# ---------------------------------------------------------------------------
# Top-k / temperature generation loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_text(
    prompt:         str,
    weights_path:   str   = "model_weights.pth",
    vocab_path:     str   = "vocab.json",
    max_new_tokens: int   = 200,
    temperature:    float = 0.8,
    top_k:          int   = 40,
    strategy:       str   = "topk",
    device:         Optional[torch.device] = None,
    stream:         bool  = True,
) -> str:
    """
    Load model + vocabulary from disk and generate a text continuation.

    Args:
        prompt:         Seed text.
        weights_path:   Path to the .pth checkpoint file.
        vocab_path:     Path to vocab.json.
        max_new_tokens: How many new tokens to generate.
        temperature:    Sampling temperature (lower = more deterministic).
        top_k:          Keep only top-k tokens before sampling.
        strategy:       "greedy", "topk", or "beam" (beam redirects to beam_search).
        device:         Torch device (auto-detected if None).
        stream:         If True, print each token as it is generated.

    Returns:
        The full generated string (prompt + continuation).
    """
    if strategy == "beam":
        from beam_search import beam_search_from_disk
        results = beam_search_from_disk(
            prompt       = prompt,
            weights_path = weights_path,
            vocab_path   = vocab_path,
            stream_best  = stream,
        )
        return results[0][0] if results else prompt

    # ── Validate paths ────────────────────────────────────────────────────────
    for path, label in [(weights_path, "Weights"), (vocab_path, "Vocabulary")]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{label} file not found: '{path}'\n"
                "Run training first (python main.py)."
            )

    # ── Device ───────────────────────────────────────────────────────────────
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    # ── Load vocabulary ───────────────────────────────────────────────────────
    tokenizer = CharTokenizer()
    tokenizer.load(vocab_path)

    # ── Load model ────────────────────────────────────────────────────────────
    model = load_model_from_checkpoint(weights_path, device)
    model.eval()
    block_size = model.config.block_size

    # ── Encode prompt ─────────────────────────────────────────────────────────
    encoded = tokenizer.encode(prompt)
    if not encoded:
        raise ValueError("Prompt encodes to an empty token sequence.")

    context = torch.tensor(encoded, dtype=torch.long, device=device).unsqueeze(0)

    if stream:
        print(prompt, end="", flush=True)

    generated_ids = list(encoded)

    for _ in range(max_new_tokens):
        ctx = context if context.size(1) <= block_size else context[:, -block_size:]
        logits      = model(ctx)
        next_logits = logits[0, -1, :]

        next_id = sample_next_token(
            next_logits,
            temperature = temperature,
            top_k       = top_k,
            strategy    = strategy,
        )
        generated_ids.append(next_id)

        if stream:
            print(tokenizer.decode([next_id]), end="", flush=True)

        context = torch.cat(
            [context, torch.tensor([[next_id]], dtype=torch.long, device=device)],
            dim=1,
        )

    if stream:
        print()

    return tokenizer.decode(generated_ids)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate code completions using a trained Transformer LM.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--prompt",   "-p", type=str, required=True,
                        help='Seed text. Example: "def fibonacci("')
    parser.add_argument("--max_tokens", "-n", type=int, default=200,
                        help="Maximum new tokens to generate (default: 200).")
    parser.add_argument("--temperature", "-t", type=float, default=0.8,
                        help="Sampling temperature (default: 0.8).\n"
                             "  < 1.0 → more deterministic   > 1.0 → more random")
    parser.add_argument("--top_k", "-k", type=int, default=40,
                        help="Top-k filtering (default: 40). Ignored for greedy/beam.")
    parser.add_argument("--strategy", "-s",
                        choices=["greedy", "topk", "beam"],
                        default="topk",
                        help=(
                            "Generation strategy (default: topk).\n"
                            "  greedy — always pick highest-probability token\n"
                            "  topk   — temperature + top-k sampling\n"
                            "  beam   — beam search (see --beam-width)"
                        ))
    parser.add_argument("--beam-width", type=int, default=5,
                        help="Beam width for beam search strategy (default: 5).")
    parser.add_argument("--weights", "-w", default="model_weights.pth")
    parser.add_argument("--vocab",   "-v", default="vocab.json")

    args = parser.parse_args()

    try:
        if args.strategy == "beam":
            from beam_search import beam_search_from_disk
            beam_search_from_disk(
                prompt       = args.prompt,
                weights_path = args.weights,
                vocab_path   = args.vocab,
                beam_width   = args.beam_width,
                max_new_tokens = args.max_tokens,
                stream_best  = True,
            )
        else:
            generate_text(
                prompt         = args.prompt,
                weights_path   = args.weights,
                vocab_path     = args.vocab,
                max_new_tokens = args.max_tokens,
                temperature    = args.temperature,
                top_k          = args.top_k,
                strategy       = args.strategy,
                stream         = True,
            )
    except FileNotFoundError as e:
        print(f"\n[generate] Error: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n[generate] Generation interrupted.")


if __name__ == "__main__":
    main()
