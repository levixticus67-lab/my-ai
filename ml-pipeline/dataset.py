"""
dataset.py — PyTorch Dataset and DataLoader pipeline.

Reads a tokenised integer sequence and yields overlapping (X, Y) pairs
where Y is X shifted right by one token (next-token prediction objective).
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple

from tokenizer import CharTokenizer


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CodeDataset(Dataset):
    """
    Sliding-window dataset over a flat integer token sequence.

    Each sample is a pair:
      X = tokens[i : i + block_size]        (input context)
      Y = tokens[i+1 : i + block_size + 1]  (targets, shifted by 1)

    The model learns to predict Y[t] given X[0..t].
    """

    def __init__(self, token_ids: torch.Tensor, block_size: int):
        """
        Args:
            token_ids:  1-D LongTensor of all encoded tokens.
            block_size: Fixed context window length.
        """
        assert token_ids.ndim == 1, "token_ids must be a 1-D tensor."
        assert len(token_ids) > block_size, (
            f"Corpus length ({len(token_ids)}) must exceed block_size ({block_size}). "
            "Add more training data."
        )
        self.data = token_ids
        self.block_size = block_size
        self.n_samples = len(token_ids) - block_size

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        chunk = self.data[idx : idx + self.block_size + 1]  # length block_size+1
        x = chunk[:-1].clone()  # (block_size,)
        y = chunk[1:].clone()   # (block_size,)
        return x, y


# ---------------------------------------------------------------------------
# Helper: build dataset + dataloader from a raw corpus string
# ---------------------------------------------------------------------------

def build_dataloader(
    corpus: str,
    tokenizer: CharTokenizer,
    block_size: int,
    batch_size: int,
    num_workers: int = 0,
    shuffle: bool = True,
) -> Tuple[DataLoader, int]:
    """
    Encode `corpus`, wrap it in a CodeDataset, and return a DataLoader plus
    the total number of tokens in the corpus.

    Args:
        corpus:       Raw text to train on.
        tokenizer:    A fitted CharTokenizer instance.
        block_size:   Context window length.
        batch_size:   Mini-batch size.
        num_workers:  DataLoader worker processes (0 = main process only).
        shuffle:      Whether to shuffle samples each epoch.

    Returns:
        (dataloader, n_tokens)
    """
    ids = tokenizer.encode(corpus)
    token_tensor = torch.tensor(ids, dtype=torch.long)
    n_tokens = len(token_tensor)
    print(f"[dataset] Corpus encoded: {n_tokens:,} tokens.")

    dataset = CodeDataset(token_tensor, block_size)
    print(f"[dataset] Dataset samples: {len(dataset):,} windows of length {block_size}.")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    return loader, n_tokens
