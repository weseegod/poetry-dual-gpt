# data/dataset.py
# ================
# Purpose: PyTorch Dataset class for PoetryDuelGPT's causal language modeling.
#
# =========================================================================
# CONCEPT: What is a Dataset?
# =========================================================================
# A Dataset answers two questions:
#   1. "How many samples do you have?"  → __len__()
#   2. "Give me the i-th sample"        → __getitem__(i)
#
# The DataLoader then handles batching, shuffling, and parallel loading.
#
# =========================================================================
# CONCEPT: Causal LM Target Construction
# =========================================================================
# For autoregressive language modeling, target = input shifted by 1:
#
#   Input:   [A,  B,  C,  D,  E]
#   Target:  [B,  C,  D,  E,  F]
#
# The model sees A, predicts B. Sees A+B, predicts C. Sees A+B+C, predicts D...
# Cross-entropy loss is computed at EVERY position simultaneously.
#
# =========================================================================
# DESIGN CHOICE: Flat Tensor vs Per-Example
# =========================================================================
#
# Option A (used here): One giant token tensor, random windows as samples
#   - Pro: simple, no padding waste, contiguous memory
#   - Con: samples overlap, less formal
#
# Option B (HuggingFace/HF style): Each example is a separate list
#   - Pro: clean separation, easy to filter/analyze per example
#   - Con: need padding within batch, more memory overhead
#
# We use Option A — it's the standard approach for small-scale GPT training
# (used by nanoGPT, minGPT, etc.)
#
# =========================================================================
# IMPLEMENTATION PLAN
# =========================================================================
#
# class PoetryDataset(torch.utils.data.Dataset):
#     """
#     Args:
#         data:       flat LongTensor of token IDs (shape: [N])
#         block_size: max context length (256)
#     """
#     def __init__(self, data, block_size):
#         self.data = data
#         self.block_size = block_size
#
#     def __len__(self):
#         # Each possible start position is a valid sample
#         # (need block_size+1 tokens for input+target)
#         return len(self.data) - self.block_size
#
#     def __getitem__(self, idx):
#         # Extract a window of block_size+1 tokens
#         chunk = self.data[idx : idx + self.block_size + 1]
#         x = chunk[:self.block_size]      # input
#         y = chunk[1:]                     # target (shifted by 1)
#         return x, y
#
# =========================================================================
# USAGE IN train.py
# =========================================================================
#   from data.dataset import PoetryDataset
#   from torch.utils.data import DataLoader
#
#   # After tokenizing the entire corpus into a flat tensor:
#   data = torch.tensor(all_token_ids, dtype=torch.long)
#
#   # Split train/val (90/10)
#   split = int(0.9 * len(data))
#   train_ds = PoetryDataset(data[:split], block_size=256)
#   val_ds   = PoetryDataset(data[split:], block_size=256)
#
#   # Create DataLoaders
#   train_loader = DataLoader(
#       train_ds,
#       batch_size=64,
#       shuffle=True,        # randomize order each epoch
#       pin_memory=True,      # speed up CPU→GPU transfer
#       num_workers=4,        # parallel loading threads
#   )
#   val_loader = DataLoader(
#       val_ds,
#       batch_size=64,
#       shuffle=False,        # no need to shuffle validation
#       pin_memory=True,
#       num_workers=2,
#   )
#
# =========================================================================
# CONCEPT: DataLoader Parameters Explained
# =========================================================================
#
# batch_size:
#   Number of samples processed together in one forward pass.
#   Larger = faster (GPU parallelism) but uses more memory.
#   64 is a good default for our ~45M model on 16GB VRAM.
#
# shuffle:
#   Randomizes sample order each epoch. Critical for training!
#   Without it, the model could memorize sequence order instead of
#   learning generalizable patterns.
#
# pin_memory:
#   Allocates CPU memory in "pinned" (page-locked) pages that GPU
#   can DMA-transfer directly. With this + non_blocking, you can
#   overlap CPU→GPU transfer with GPU computation.
#
# num_workers:
#   How many subprocesses load data in parallel.
#   0 = main process loads data (single-threaded, slow)
#   4 = 4 workers pre-fetch next batches while GPU computes current one
#   Too many = memory overhead + IPC bottleneck
#
# drop_last:
#   If True, drop the last incomplete batch (when len(dataset) % batch_size != 0)
#   Usually True for training to avoid batch-norm issues with smaller batches.
#
# =========================================================================
# CONCEPT: Iterating Over Epochs
# =========================================================================
# An "epoch" = one full pass through the training dataset.
# With our flat-tensor Dataset, the number of samples = len(data) - block_size.
#
# If total tokens = 1,000,000 and block_size = 256:
#   num_samples = 1,000,000 - 256 = 999,744
#   batches_per_epoch = 999,744 / 64 ≈ 15,621
#
# So one epoch = ~15,621 optimizer steps.
# README targets 3 epochs → ~47,000 steps total.

# --- YOUR CODE BELOW ---

import os
from typing import List, Tuple

from torch.utils.data import Dataset, random_split, Subset, DataLoader
from pathlib import Path
from torchvision import transforms
import torch
from tqdm.auto import tqdm
from PIL import Image

path_dataset = Path.cwd() / 'data/nsfw_dataset_v1'

class PoemtrySubset(Data)
