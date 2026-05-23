"""
PyTorch Dataset + DataLoader for PoetryDuelGPT.

Design: one giant flat tensor of token IDs → random-window samples.
  x = tokens[i : i+block_size]       # input
  y = tokens[i+1 : i+block_size+1]   # target (shifted by 1)

Also includes CSV exploration utilities.
"""

import os
from typing import List, Tuple, Optional
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).parent.parent
PATH_DATASET = ROOT / "data" / "poems_dataset.csv"  # final clean dataset


# ═══════════════════════════════════════════════════════════════
#  CSV EXPLORATION
# ═══════════════════════════════════════════════════════════════

def load_dataframe(path=None):
    return pd.read_csv(path or PATH_DATASET)


def filter_by_genre(df, genre):
    """Phase 1: filter_by_genre(df, 'lục bát') → 89K poems."""
    f = df[df["genre"] == genre].copy()
    print(f"Filtered '{genre}': {len(f):,} poems")
    return f


def get_poem_content(df):
    """Extract clean text. Lines in CSV are joined with ' <\\n> '."""
    return [c.replace(" <\n> ", "\n") for c in df["content"] if pd.notna(c)]


def explore_dataset(path=None):
    """Print genre/author/period stats + sample poems. Run once to understand data."""
    df = pd.read_csv(path or PATH_DATASET)
    print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} cols")

    # Word/line counts
    df["n_lines"] = df["content"].apply(lambda t: str(t).count(" <\n> ") + 1)
    df["n_words"] = df["content"].apply(lambda t: len(str(t).split()))

    # Genre stats
    print("\n🎭 GENRE")
    g = df.groupby("genre").agg(n=("content", "count"), lines=("n_lines", "mean"), words=("n_words", "mean"))
    for genre, r in g.sort_values("n", ascending=False).iterrows():
        print(f"  {genre:<15s} {int(r.n):>8,} poems  avg {r.lines:.0f} lines  {r.words:.0f} words")

    # Specific genre
    print(f"\n📝 SPECIFIC GENRE ({df['specific_genre'].nunique()} types)")
    s = df.groupby("specific_genre").size().sort_values(ascending=False)
    for name, cnt in s.head(15).items():
        print(f"  {name:<30s} {cnt:>8,}")

    # Authors
    print(f"\n✍️  TOP AUTHORS ({df['author'].nunique()} total)")
    a = df.groupby("author").size().sort_values(ascending=False)
    for name, cnt in a.head(10).items():
        print(f"  {name:<30s} {cnt:>8,}")

    print(f"\n💡 Phase 1: train on 'lục bát' ({int(g.loc['lục bát','n']):,} poems)")
    print(f"   Phase 2: add 'bảy chữ' → Phase 3: all {int(g['n'].sum()):,}")
    return df


# ═══════════════════════════════════════════════════════════════
#  PyTorch DATASET — flat tensor → random windows
# ═══════════════════════════════════════════════════════════════

class PoetryDataset(Dataset):
    """
    Wraps a flat LongTensor of token IDs.
    Each __getitem__ returns a window: x[:T] and y[1:T+1] (shifted by 1).
    """
    def __init__(self, data: torch.Tensor, block_size: int):
        assert len(data.shape) == 1, f"Need 1D tensor, got {data.shape}"
        assert len(data) > block_size
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size  # every position is a valid window start

    def __getitem__(self, idx):
        chunk = self.data[idx : idx + self.block_size + 1]  # grab T+1 tokens
        return chunk[:self.block_size], chunk[1:]           # x, y (shifted by 1)


class CurriculumDataset(PoetryDataset):
    """
    PoetryDataset with progressive difficulty — only exposes the first
    `max_fraction` of the token stream. Grow max_fraction from 0.1→1.0
    during training to implement curriculum learning.
    
    Data must be pre-sorted by difficulty (short→long pairs).
    """
    def __init__(self, data: torch.Tensor, block_size: int, max_fraction: float = 0.1):
        super().__init__(data, block_size)
        self.max_fraction = min(1.0, max(max_fraction, 0.05))

    def __len__(self):
        return int(super().__len__() * self.max_fraction)

    def __getitem__(self, idx):
        # Clamp idx to valid range for current fraction
        idx = idx % len(self)
        return super().__getitem__(idx)

    def expand(self, new_fraction: float):
        """Grow the curriculum window. Called periodically during training."""
        self.max_fraction = min(1.0, max(new_fraction, 0.05))
        return len(self)


# ═══════════════════════════════════════════════════════════════
#  DATALOADER FACTORY
# ═══════════════════════════════════════════════════════════════

def get_dataloaders(data, block_size=256, batch_size=64, val_fraction=0.05, num_workers=0):
    """Split flat tensor → train/val → wrap in PoetryDataset → DataLoader."""
    split = int(len(data) * val_fraction)
    train_data, val_data = data[:-split], data[-split:]

    train_ds = PoetryDataset(train_data, block_size)
    val_ds = PoetryDataset(val_data, block_size)

    print(f"Train: {len(train_ds):,} samples | Val: {len(val_ds):,} | Batch: {batch_size}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               pin_memory=True, num_workers=num_workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             pin_memory=True, num_workers=num_workers)
    return train_loader, val_loader


# ═══════════════════════════════════════════════════════════════
#  EXAMPLE-ALIGNED DATASET (v3 — one row = one complete example)
# ═══════════════════════════════════════════════════════════════

class ExampleDataset(Dataset):
    """
    Each row is one complete training example, padded to block_size.
    Every training window starts at <|start|> — no cross-poem boundary noise.
    
    x = tokens[:block_size-1]       y = tokens[1:block_size]
    """
    def __init__(self, lines: List[str], tokenizer, block_size: int, pad_id: int = 0):
        rows = []
        for line in tqdm(lines, desc="Tokenizing examples"):
            if not line:
                continue
            ids = tokenizer.encode(line).ids
            ids = ids[:block_size]  # truncate (shouldn't happen — max ~74 tokens)
            padded = ids + [pad_id] * (block_size - len(ids))
            rows.append(padded)
        self.data = torch.tensor(rows, dtype=torch.long)
        self.block_size = block_size
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data[idx]
        return row[:-1], row[1:]  # x (input), y (target, shifted by 1)


def get_dataloaders_aligned(lines: List[str], tokenizer, block_size: int = 256,
                             batch_size: int = 64, val_fraction: float = 0.05,
                             num_workers: int = 0):
    """
    Example-aligned training: each example is one row → no cross-boundary noise.
    Splits into train/val → wraps in DataLoader.
    """
    ds = ExampleDataset(lines, tokenizer, block_size)
    split = int(len(ds) * val_fraction)
    train_ds, val_ds = torch.utils.data.random_split(
        ds, [len(ds) - split, split],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"Train: {len(train_ds):,} examples | Val: {len(val_ds):,} | Batch: {batch_size}")
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               pin_memory=True, num_workers=num_workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             pin_memory=True, num_workers=num_workers)
    return train_loader, val_loader


def build_loss_mask(tokenizer, device: str = "cpu") -> torch.Tensor:
    """
    Build a vocab-sized boolean mask: True = compute loss, False = skip.
    Masks out all control/special tokens that decode to empty string.
    ByteLevel BPE cannot decode added single-tokens → empty = control.
    Only poetry content tokens (non-empty decode) get loss.
    
    Returns (vocab_size,) boolean tensor.
    """
    V = tokenizer.get_vocab_size()
    mask = torch.ones(V, dtype=torch.bool, device=device)
    
    masked = 0
    for tid in range(V):
        if tokenizer.decode([tid]) == '':
            mask[tid] = False
            masked += 1
    
    kept = V - masked
    print(f"Loss mask: {kept:,}/{V:,} tokens ({kept/V*100:.0f}%) — {masked} control tokens masked")
    return mask


# ═══════════════════════════════════════════════════════════════
#  TOKENIZE HELPER
# ═══════════════════════════════════════════════════════════════

def tokenize_corpus(lines: List[str], tokenizer) -> torch.Tensor:
    """Encode all text lines → one flat LongTensor."""
    ids = []
    for line in tqdm(lines, desc="Tokenizing"):
        if line:
            ids.extend(tokenizer.encode(line).ids)
    t = torch.tensor(ids, dtype=torch.long)
    print(f"{len(lines):,} lines → {len(t):,} tokens")
    return t


# ═══════════════════════════════════════════════════════════════
#  DRY-RUN TEST
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    df = explore_dataset()
    df_lb = filter_by_genre(df, "lục bát")

    # Test PoetryDataset + DataLoader with dummy data
    print("\n🧪 Dry-run:")
    dummy = torch.arange(100_000, dtype=torch.long)
    tl, vl = get_dataloaders(dummy, block_size=256, batch_size=32)
    x, y = next(iter(tl))
    print(f"  x: {x.shape}  y: {y.shape}")
    print(f"  x[0,:5] = {x[0,:5].tolist()}")
    print(f"  y[0,:5] = {y[0,:5].tolist()}  ← shifted by 1 ✓")
