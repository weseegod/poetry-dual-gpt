"""
Clean the poetry CSV — keep only Lục Bát + Thất Ngôn (7-syllable).
Outputs: data/poems_dataset_clean.csv

Stages:
  1. Keep only lục bát + bảy chữ genres
  2. Remove song thất lục bát (mixed syllable, not pure 7-syl)
  3. Clean HTML artifacts + normalize Unicode
  4. Drop empty/short poems
  5. Remove duplicates
"""

import re
import hashlib
import unicodedata
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).parent.parent
CSV_PATH = ROOT / "data" / "poems_dataset.csv"
OUTPUT_PATH = ROOT / "data" / "poems_dataset_clean.csv"

# ═══════════════════════════════════════════════════════════════
# STAGE 1: Filter genres
# ═══════════════════════════════════════════════════════════════

def filter_genres(df):
    """Keep only lục bát and bảy chữ. Drop everything else."""
    kept = ["lục bát", "bảy chữ"]
    mask = df["genre"].isin(kept)
    dropped = (~mask).sum()
    df = df[mask].copy()
    print(f"  Genres filtered: {dropped:,} dropped → {len(df):,} kept")
    return df


# ═══════════════════════════════════════════════════════════════
# STAGE 2: Clean HTML + normalize Unicode
# ═══════════════════════════════════════════════════════════════

def clean_text(text: str) -> str:
    """Clean HTML artifacts per line, normalize Unicode. Preserves <\n> separators."""
    if pd.isna(text):
        return ""
    # Split by line separator first (so HTML regex doesn't eat them)
    sep = "<" + chr(10) + ">"  # < + newline + > is the CSV line separator
    lines = text.split(sep)
    cleaned_lines = []
    for line in lines:
        # HTML entities
        line = line.replace("&nbsp;", " ")
        line = line.replace("&amp;", "&")
        line = line.replace("&lt;", "<")
        line = line.replace("&gt;", ">")
        # Remove actual HTML tags (but NOT our <\n> separator — we already split)
        line = re.sub(r"<[^>]+>", "", line)
        # Collapse spaces/tabs (newlines already handled)
        line = re.sub(r"[ \t]+", " ", line)
        # Unicode normalization
        line = unicodedata.normalize("NFC", line)
        line = line.strip()
        if line:
            cleaned_lines.append(line)
    return sep.join(cleaned_lines)

def clean_content(df):
    """Apply text cleaning to all content, drop rows that become empty."""
    df["content"] = df["content"].apply(clean_text)
    # Drop rows where content is now empty
    mask = df["content"].astype(str).str.strip() != ""
    dropped = (~mask).sum()
    df = df[mask].copy()
    print(f"  Text cleaned, {dropped} empty rows dropped")
    return df


# ═══════════════════════════════════════════════════════════════
# STAGE 3: Filter short poems
# ═══════════════════════════════════════════════════════════════

def filter_short(df, min_lines=2):
    """Drop poems with fewer than min_lines."""
    n_lines = df["content"].apply(lambda t: str(t).count("<\n>") + 1)
    mask = n_lines >= min_lines
    dropped = (~mask).sum()
    df = df[mask].copy()
    print(f"  Poems < {min_lines} lines: {dropped} dropped → {len(df):,} kept")
    return df


# ═══════════════════════════════════════════════════════════════
# STAGE 4: Remove duplicates
# ═══════════════════════════════════════════════════════════════

def content_hash(text: str) -> str:
    """Hash normalized content for dedup."""
    if pd.isna(text):
        return ""
    normalized = clean_text(text.lower())
    return hashlib.md5(normalized.encode()).hexdigest()

def remove_duplicates(df):
    """Drop poems with identical cleaned content."""
    df["_hash"] = df["content"].apply(content_hash)
    before = len(df)
    df = df.drop_duplicates(subset="_hash", keep="first")
    df = df.drop(columns=["_hash"])
    dupes = before - len(df)
    print(f"  Duplicates removed: {dupes}")
    return df


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def clean(csv_path=None, output_path=None):
    csv_path = Path(csv_path or CSV_PATH)
    output_path = Path(output_path or OUTPUT_PATH)

    print(f"📖  Loading: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"    Original: {len(df):,} poems")
    print()

    # Stage 1: Genre filter
    df = filter_genres(df)
    print(f"    After genre filter: {len(df):,} (lục bát: {df['genre'].eq('lục bát').sum():,}, "
          f"bảy chữ: {df['genre'].eq('bảy chữ').sum():,})")

    # Stage 2: Clean text
    df = clean_content(df)

    # Stage 4: Filter short poems
    df = filter_short(df, min_lines=2)

    # Stage 5: Deduplicate
    df = remove_duplicates(df)

    # Final stats
    print(f"\n📊  Final stats:")
    for genre in ["lục bát", "bảy chữ"]:
        gdf = df[df["genre"] == genre]
        n_lines = gdf["content"].apply(lambda t: str(t).count("<\n>") + 1)
        # Count song thất within bảy chữ
        st_mask = gdf["specific_genre"].astype(str).str.contains("song thất", case=False, na=False)
        print(f"    {genre:<12s}: {len(gdf):>6,} poems  "
              f"avg {n_lines.mean():.0f} lines  "
              f"max {n_lines.max()} lines"
              f"{'  (song thất: ' + str(st_mask.sum()) + ')' if st_mask.sum() > 0 else ''}")

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\n✅  Saved → {output_path}  ({len(df):,} poems)")
    return df


if __name__ == "__main__":
    clean()
