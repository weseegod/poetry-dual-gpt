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
# PyTorch Dataset — causal LM style (flat tensor → random windows)
# =========================================================================
#
# Pattern (compare with NSFWDataset):
#   NSFWDataset:  __init__ scans dirs → self.samples = [(path, label), ...]
#                 __getitem__ loads image → transform → (image_tensor, label)
#
#   PoetryDataset: __init__ receives flat token tensor + block_size
#                  __getitem__ slices a random window → (x, y) with y shifted
#
# Key difference: no transforms (text data doesn't need augmentation).
# The "shift by 1" IS the label — the model predicts the next token.


class PoetryDataset(Dataset):
    """
    PyTorch Dataset wrapping a flat tensor of token IDs for causal LM.

    Given a giant tensor: [tok_0, tok_1, tok_2, ..., tok_N]
    Each sample is a window of block_size+1 tokens:
      x = window[:block_size]           # input:  [tok_i, ..., tok_{i+T-1}]
      y = window[1:block_size+1]        # target: [tok_{i+1}, ..., tok_{i+T}]

    Pattern matches NSFWDataset:
      - __init__:  prepare self.samples (here: just the data tensor)
      - __len__:   return how many windows we can extract
      - __getitem__: extract one window, return (x, y)

    Args:
        data:       flat LongTensor of token IDs, shape (N,)
        block_size: max context length (256)
    """

    def __init__(self, data: torch.Tensor, block_size: int):
        assert len(data.shape) == 1, f"data must be 1D tensor, got shape {data.shape}"
        assert len(data) > block_size, (
            f"Data too short: {len(data)} tokens, need > {block_size} (block_size)"
        )
        self.data = data
        self.block_size = block_size

    def __len__(self) -> int:
        # Number of possible windows.
        # Need block_size+1 tokens per sample (block_size for input, 1 extra for target).
        return len(self.data) - self.block_size

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Extract a window of block_size+1 tokens
        chunk = self.data[idx : idx + self.block_size + 1]
        x = chunk[:self.block_size]      # input:  positions 0 .. T-1
        y = chunk[1:]                     # target: positions 1 .. T  (shifted by 1)
        return x, y


# =========================================================================
# DataLoader factory — like get_dataloaders() in the image classifier
# =========================================================================
#
# Pattern (compare with NSFWDataset.get_dataloaders):
#   1. Split indices into train/val/test
#   2. Apply different transforms per split (none needed for poetry)
#   3. Create DataLoader for each split with batch_size, shuffle, pin_memory
#
# Because PoetryDataset works on a flat tensor, we split the TENSOR first,
# then create a Dataset per split. No SubsetWithTransform needed.


def get_dataloaders(
    data: torch.Tensor,
    block_size: int = 256,
    batch_size: int = 64,
    val_fraction: float = 0.05,
    num_workers: int = 0,  # 0 = main process (faster for small datasets)
):
    """
    Split flat token data into train/val and return DataLoaders.

    Why only train + val (no test)?
      - A classifier needs a test set to report "92.3% accuracy on unseen data"
      - A generative model's real test is QUALITATIVE: does it produce good poetry?
      - You evaluate by reading generated poems, checking syllable counts, B-T tones
      - Validation loss tells you "is it still learning or overfitting?" — that's enough

    Pattern follows NSFWDataset.get_dataloaders():
      1. Compute split sizes
      2. Slice the tensor (no SubsetWithTransform needed — no transforms)
      3. Create PoetryDataset per split
      4. Wrap in DataLoader with shuffle/pin_memory/num_workers

    Args:
        data:          flat LongTensor of all token IDs, shape (N,)
        block_size:    max context window (256)
        batch_size:    samples per batch (64)
        val_fraction:  fraction for validation (0.05 = 5%)
        num_workers:   parallel data-loading threads (2-4)

    Returns:
        train_loader, val_loader
    """
    total_tokens = len(data)
    val_size = int(total_tokens * val_fraction)
    train_size = total_tokens - val_size

    # Split the flat tensor: first 95% = train, last 5% = val
    train_data = data[:train_size]
    val_data = data[train_size:]

    # Create Dataset for each split
    train_ds = PoetryDataset(train_data, block_size)
    val_ds = PoetryDataset(val_data, block_size)

    print(f"\n📦  DataLoader created:")
    print(f"    Train: {train_size:,} tokens → {len(train_ds):,} samples")
    print(f"    Val:   {val_size:,} tokens → {len(val_ds):,} samples")
    print(f"    Batch size: {batch_size}, Block size: {block_size}")
    print(f"    ~{len(train_ds) // batch_size:,} batches/epoch")

    # Create DataLoaders (mirrors image classifier pattern)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,            # train: randomize order each epoch
        pin_memory=True,          # faster CPU→GPU transfer
        num_workers=num_workers,
        drop_last=True,           # drop incomplete last batch
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,            # val: fixed order, reproducible
        pin_memory=True,
        num_workers=num_workers,
    )

    return train_loader, val_loader


# =========================================================================
# Tokenize helper — wrap tokenizer.encode into something reusable
# =========================================================================

def tokenize_corpus(text_lines: List[str], tokenizer) -> torch.Tensor:
    """
    Tokenize a list of preprocessed text lines into one flat LongTensor.

    Call this BEFORE get_dataloaders().

    Args:
        text_lines: list of strings, each already formatted like:
                    "<|start|> [LUC_BAT] prompt, <|reply|> reply <|end|>"
        tokenizer:  HuggingFace Tokenizer (from train_bpe.py)

    Returns:
        Flat LongTensor of all token IDs concatenated

    Example:
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file("tokenizer/poetry_bpe.model")
        lines = ["<|start|> [LUC_BAT] Trăm năm, <|reply|> Chữ tài <|end|>", ...]
        data = tokenize_corpus(lines, tok)
        train_loader, val_loader, test_loader = get_dataloaders(data)
    """
    all_ids = []
    for line in tqdm(text_lines, desc="Tokenizing"):
        if line:
            ids = tokenizer.encode(line).ids
            all_ids.extend(ids)

    tensor = torch.tensor(all_ids, dtype=torch.long)
    print(f"Tokenized {len(text_lines):,} lines → {len(tensor):,} tokens")
    return tensor


# =========================================================================
# Quick test — verify the Dataset + DataLoader work with dummy data
# =========================================================================
if __name__ == "__main__":
    # 1. Explore the raw CSV
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

    # 2. Test PoetryDataset + DataLoader with dummy token IDs
    print("\n" + "=" * 65)
    print("🧪  PoetryDataset + DataLoader dry-run (dummy data)")
    print("=" * 65)

    # Simulate: 100K token IDs (like what tokenize_corpus would produce)
    dummy_data = torch.arange(100_000, dtype=torch.long)

    train_loader, val_loader = get_dataloaders(
        dummy_data,
        block_size=256,
        batch_size=32,
    )

    # Inspect one batch
    x, y = next(iter(train_loader))
    print(f"\n    x shape:    {x.shape}  (batch={x.shape[0]}, seq={x.shape[1]})")
    print(f"    y shape:    {y.shape}")
    print(f"    x[0, :5]:   {x[0, :5].tolist()}")       # first 5 tokens
    print(f"    y[0, :5]:   {y[0, :5].tolist()}")       # should be x shifted by 1
    print(f"    x[0, -5:]:  {x[0, -5:].tolist()}")
    print(f"    y[0, -5:]:  {y[0, -5:].tolist()}")
    assert torch.equal(x[:, 1:], y[:, :-1]), "FAIL: y is NOT x shifted by 1!"
    print(f"    ✅  Verified: y is x shifted by 1")


