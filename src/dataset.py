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
from typing import List, Tuple, Optional

from torch.utils.data import Dataset, random_split, Subset, DataLoader
from pathlib import Path
from torchvision import transforms
import torch
from tqdm.auto import tqdm
from PIL import Image
import pandas as pd

# =========================================================================
# DATA EXPLORATION — understand your dataset before training
# =========================================================================

PATH_DATASET = Path(__file__).parent.parent / 'data' / 'poems_dataset.csv'


def load_dataframe(csv_path: str = None) -> pd.DataFrame:
    """Load the poetry CSV into a pandas DataFrame."""
    if csv_path is None:
        csv_path = PATH_DATASET
    return pd.read_csv(csv_path)


def filter_by_genre(df: pd.DataFrame, genre: str) -> pd.DataFrame:
    """
    Filter DataFrame to a single genre.

    Phase 1: filter_by_genre(df, 'lục bát') → 89,943 poems
    Phase 2: filter_by_genre(df[df['genre'].isin(['lục bát', 'bảy chữ'])]) → 136K
    Phase 3: use full df → 198K

    Args:
        df: full poetry DataFrame
        genre: genre name to filter (e.g. 'lục bát', 'bảy chữ', 'tám chữ')
    Returns:
        Filtered DataFrame
    """
    filtered = df[df['genre'] == genre].copy()
    print(f"Filtered '{genre}': {len(filtered):,} poems (from {len(df):,} total)")
    return filtered


def get_poem_content(df: pd.DataFrame) -> List[str]:
    """
    Extract clean poem content from DataFrame.

    The content column has poems with lines separated by ' <\n> '.
    This returns a list where each element is the full poem text
    with actual newlines.
    """
    contents = []
    for content in df['content']:
        if pd.isna(content):
            continue
        # Replace ' <\n> ' with actual newlines
        clean = content.replace(' <\n> ', '\n')
        contents.append(clean)
    return contents


def explore_dataset(csv_path: str = None):
    """
    Load the poetry CSV and print summary statistics to console.

    What this shows you:
      - Total poems / rows
      - Columns and their types
      - Genre distribution: how many poems per genre, avg lines, avg words
      - specific_genre distribution: the detailed poetic forms
      - Author distribution: top authors by poem count
      - Period distribution
      - Missing values (if any)
      - Sample poems

    Run this once to understand what data you're working with.
    """
    if csv_path is None:
        csv_path = PATH_DATASET

    print("=" * 65)
    print("📊  POETRY DATASET EXPLORATION")
    print("=" * 65)

    # 1. Load
    print(f"\n📂  Loading: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"    Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

    # 2. Column overview
    print(f"\n📋  Columns: {list(df.columns)}")
    print(f"    Memory usage: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")

    # 3. Missing values
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing) > 0:
        print(f"\n⚠️   Missing values:\n{missing.to_string()}")
    else:
        print(f"\n✅  No missing values")

    # 4. Helper: count lines and words in a poem
    def count_lines(text):
        """Count lines in a poem (separated by <br> / <\n> markers)."""
        if pd.isna(text):
            return 0
        return text.count(' <\n> ') + 1

    def count_words(text):
        """Count words (Vietnamese syllables) in the content."""
        if pd.isna(text):
            return 0
        return len(text.split())

    # Precompute line and word counts for every poem
    df['num_lines'] = df['content'].apply(count_lines)
    df['num_words'] = df['content'].apply(count_words)

    # =====================================================================
    # GENRE analysis
    # =====================================================================
    print("\n" + "=" * 65)
    print("🎭  GENRE DISTRIBUTION")
    print("=" * 65)

    genre_stats = df.groupby('genre').agg(
        poem_count=('content', 'count'),
        avg_lines=('num_lines', 'mean'),
        avg_words=('num_words', 'mean'),
        total_words=('num_words', 'sum'),
    ).sort_values('poem_count', ascending=False)

    print(f"\n{'Genre':<20s} {'Poems':>8s} {'Avg Lines':>10s} {'Avg Words':>10s} {'Total Words':>13s}")
    print("-" * 65)
    for g, row in genre_stats.iterrows():
        print(f"{g:<20s} {int(row.poem_count):>8,} {row.avg_lines:>10.1f} {row.avg_words:>10.1f} {int(row.total_words):>13,}")
    print("-" * 65)
    print(f"{'TOTAL':<20s} {int(genre_stats.poem_count.sum()):>8,}")

    # =====================================================================
    # SPECIFIC_GENRE analysis
    # =====================================================================
    print("\n" + "=" * 65)
    print("📝  SPECIFIC GENRE (poetic form) DISTRIBUTION")
    print("=" * 65)

    spec_stats = df.groupby('specific_genre').agg(
        poem_count=('content', 'count'),
        avg_lines=('num_lines', 'mean'),
        avg_words=('num_words', 'mean'),
    ).sort_values('poem_count', ascending=False)

    print(f"\n{'Specific Genre':<30s} {'Poems':>8s} {'Avg Lines':>10s} {'Avg Words':>10s}")
    print("-" * 62)
    for g, row in spec_stats.head(30).iterrows():
        print(f"{g:<30s} {int(row.poem_count):>8,} {row.avg_lines:>10.1f} {row.avg_words:>10.1f}")
    if len(spec_stats) > 30:
        print(f"... and {len(spec_stats) - 30} more specific genres")
    print(f"\nTotal unique specific_genres: {len(spec_stats)}")

    # =====================================================================
    # AUTHOR analysis
    # =====================================================================
    print("\n" + "=" * 65)
    print("✍️   TOP AUTHORS")
    print("=" * 65)

    author_stats = df.groupby('author').agg(
        poem_count=('content', 'count'),
        avg_words=('num_words', 'mean'),
    ).sort_values('poem_count', ascending=False)

    print(f"\n{'Author':<30s} {'Poems':>8s} {'Avg Words':>10s}")
    print("-" * 52)
    for a, row in author_stats.head(20).iterrows():
        print(f"{a:<30s} {int(row.poem_count):>8,} {row.avg_words:>10.1f}")
    print(f"\nTotal unique authors: {len(author_stats)}")

    # =====================================================================
    # PERIOD analysis
    # =====================================================================
    print("\n" + "=" * 65)
    print("🏛️   PERIOD DISTRIBUTION")
    print("=" * 65)

    period_stats = df.groupby('period').agg(
        poem_count=('content', 'count'),
    ).sort_values('poem_count', ascending=False)

    print(f"\n{'Period':<25s} {'Poems':>8s}")
    print("-" * 37)
    for p, row in period_stats.iterrows():
        print(f"{p:<25s} {int(row.poem_count):>8,}")

    # =====================================================================
    # SAMPLE POEMS
    # =====================================================================
    print("\n" + "=" * 65)
    print("📖  SAMPLE POEMS (first 2)")
    print("=" * 65)
    for i in range(min(2, len(df))):
        row = df.iloc[i]
        content_preview = row['content'].replace(' <\n> ', '\n    ')[:350]
        print(f"\n  [{i+1}] {row['title']}")
        print(f"      Genre: {row['genre']} | Form: {row['specific_genre']}")
        print(f"      Author: {row['author']} | Period: {row['period']}")
        print(f"      {row['num_lines']} lines, {row['num_words']} words")
        print(f"      ---")
        print(f"      {content_preview}...")

    print("\n" + "=" * 65)
    print("💡  TRAINING STRATEGY")
    print("=" * 65)
    print(f"""
    Phase 1 (NOW):     Train on lục bát only ({int(genre_stats.loc['lục bát', 'poem_count']):,} poems)
                        → 1 genre, 1 rule (6→8 syllables)
                        → Fastest iteration, easiest to debug

    Phase 2 (LATER):   Add bảy chữ ({int(genre_stats.loc['bảy chữ', 'poem_count']):,} poems)
                        → 2 genres, model learns to switch on [GENRE] tag

    Phase 3 (FINAL):   Add all genres ({int(genre_stats.poem_count.sum()):,} total poems)
    """)

    print("\n" + "=" * 65)
    print("✅  Exploration complete!")
    print("=" * 65)

    return df


# =========================================================================
# PyTorch Dataset — for training (implement later in Phase 3)
# =========================================================================

# class PoetryDataset(Dataset):
#     ...


# =========================================================================
# Run exploration when executing this file directly
# =========================================================================
if __name__ == "__main__":
    df = explore_dataset()

    # Demonstrate Phase 1 filter: Lục Bát only
    print("\n🧪  Demo: filter_by_genre(df, 'lục bát')")
    df_lb = filter_by_genre(df, 'lục bát')
    print(f"    Ready for Phase 1 training: {len(df_lb):,} poems")

    # Preview what poem content looks like after cleaning
    print("\n📖  Demo: get_poem_content() on first poem")
    contents = get_poem_content(df_lb.head(1))
    if contents:
        print("    " + contents[0][:200].replace('\n', '\n    ') + "...")


