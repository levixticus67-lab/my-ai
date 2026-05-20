"""
model.py — Decoder-only Transformer Language Model built from scratch using PyTorch primitives.

Architecture:
  Token Embedding → Positional Embedding → N × TransformerBlock → LayerNorm → Linear head

Each TransformerBlock:
  LayerNorm → Multi-Head Self-Attention (causal) → residual
  LayerNorm → Feed-Forward Network (GeLU) → residual
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    vocab_size: int         # Total number of unique tokens
    block_size: int = 128   # Maximum context window length
    n_embd: int = 128       # Embedding / hidden dimension
    n_heads: int = 4        # Number of attention heads
    n_layers: int = 4       # Number of stacked transformer blocks
    dropout: float = 0.1    # Dropout probability for regularisation
    ffn_mult: int = 4       # Feed-forward hidden dim multiplier (n_embd * ffn_mult)


# ---------------------------------------------------------------------------
# Causal Multi-Head Self-Attention
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention with a causal (lower-triangular) mask so that
    each position can only attend to itself and earlier positions.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        assert config.n_embd % config.n_heads == 0, (
            f"n_embd ({config.n_embd}) must be divisible by n_heads ({config.n_heads})"
        )
        self.n_heads = config.n_heads
        self.head_dim = config.n_embd // config.n_heads

        # Fused QKV projection: produces Q, K, V in one matrix multiply
        self.qkv_proj = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        # Output projection
        self.out_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

        self.attn_drop = nn.Dropout(config.dropout)
        self.resid_drop = nn.Dropout(config.dropout)

        # Causal mask — registered as a buffer so it moves with .to(device)
        mask = torch.tril(torch.ones(config.block_size, config.block_size))
        self.register_buffer("causal_mask", mask.view(1, 1, config.block_size, config.block_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape  # batch, sequence length, embedding dim

        # Compute Q, K, V via fused projection then split
        qkv = self.qkv_proj(x)                              # (B, T, 3C)
        q, k, v = qkv.split(C, dim=2)                       # each (B, T, C)

        # Reshape to (B, n_heads, T, head_dim)
        def reshape_heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        q, k, v = reshape_heads(q), reshape_heads(k), reshape_heads(v)

        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, H, T, T)

        # Apply causal mask: fill future positions with -inf before softmax
        scores = scores.masked_fill(
            self.causal_mask[:, :, :T, :T] == 0, float("-inf")
        )

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_drop(attn_weights)

        # Weighted sum of values
        out = torch.matmul(attn_weights, v)                   # (B, H, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)

        return self.resid_drop(self.out_proj(out))


# ---------------------------------------------------------------------------
# Position-wise Feed-Forward Network
# ---------------------------------------------------------------------------

class FeedForward(nn.Module):
    """
    Two-layer MLP with GeLU activation:
      x → Linear(C, ffn_dim) → GeLU → Linear(ffn_dim, C) → Dropout
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        ffn_dim = config.n_embd * config.ffn_mult
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, ffn_dim, bias=False),
            nn.GELU(),
            nn.Linear(ffn_dim, config.n_embd, bias=False),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Transformer Decoder Block (Pre-LN variant)
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """
    Pre-Layer-Norm Transformer block:
      x = x + Attention(LayerNorm(x))
      x = x + FFN(LayerNorm(x))

    Pre-LN training is more stable than the original post-LN formulation.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.ffn = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


# ---------------------------------------------------------------------------
# Full Decoder-Only Transformer Language Model
# ---------------------------------------------------------------------------

class TransformerLM(nn.Module):
    """
    Decoder-only Transformer Language Model.

    Forward pass returns raw logits of shape (B, T, vocab_size).
    Use cross-entropy loss externally for training.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.token_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        self.emb_drop = nn.Dropout(config.dropout)

        self.blocks = nn.Sequential(
            *[TransformerBlock(config) for _ in range(config.n_layers)]
        )

        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying: share token embedding and output projection weights
        # (reduces parameters, improves generalisation — common in LMs)
        self.head.weight = self.token_emb.weight

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialise parameters following GPT-2 conventions."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """
        Args:
            idx: LongTensor of token IDs, shape (B, T)

        Returns:
            logits: FloatTensor of shape (B, T, vocab_size)
        """
        B, T = idx.shape
        assert T <= self.config.block_size, (
            f"Sequence length {T} exceeds block_size {self.config.block_size}"
        )

        positions = torch.arange(T, device=idx.device).unsqueeze(0)  # (1, T)

        tok = self.token_emb(idx)        # (B, T, C)
        pos = self.pos_emb(positions)    # (1, T, C) — broadcasts over B
        x = self.emb_drop(tok + pos)

        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)            # (B, T, vocab_size)

        return logits

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
