"""
Clean the poetry CSV — keep only Lục Bát + Thất Ngôn (7-syllable).
Outputs: data/poems_dataset_clean.csv

Stages:
  1. Keep only lục bát + bảy chữ genres
  2. Clean HTML artifacts + normalize Unicode + fix spacing
  3. Strip line numbers + metadata
  4. Drop empty/short poems
  5. Remove exact duplicates
  6. Remove near-duplicates (shared lines)

Usage:
  python src/clean_data.py                          # full pipeline
  python src/clean_data.py --reclean CSV            # in-place cleaning
  python src/clean_data.py --check-dupes CSV        # dry-run stats
"""

import re, hashlib, unicodedata, argparse
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).parent.parent
CSV_PATH = ROOT / "data" / "poems_dataset.csv"
OUTPUT_PATH = ROOT / "data" / "poems_dataset_clean.csv"

SEP = "<" + chr(10) + ">"


# ═══════════════════════════════════════════════
# STAGE 1: Filter genres
# ═══════════════════════════════════════════════

def filter_genres(df):
    kept = ["lục bát", "bảy chữ"]
    mask = df["genre"].isin(kept)
    dropped = (~mask).sum()
    df = df[mask].copy()
    print(f"  Genres filtered: {dropped:,} dropped → {len(df):,} kept")
    return df


# ═══════════════════════════════════════════════
# STAGE 2: Text cleaning helpers
# ═══════════════════════════════════════════════

def fix_spacing(line: str) -> str:
    """Fix spacing around punctuation: 'word , word' → 'word, word'."""
    line = re.sub(r'\s+([,.;:?!%])', r'\1', line)
    line = re.sub(r'\(\s+', '(', line)
    line = re.sub(r'\s+\)', ')', line)
    return line


def strip_line_numbers(line: str) -> str:
    """Remove leading verse numbers: '2840. duyên vân...' → 'duyên vân...'"""
    return re.sub(r'^\d+[.\)]\s*', '', line)


def clean_text(text: str) -> str:
    """Clean HTML artifacts, normalize spacing, strip metadata per line."""
    if pd.isna(text):
        return ""
    lines = text.split(SEP)
    cleaned = []
    for line in lines:
        # HTML
        line = line.replace("&nbsp;", " ").replace("&amp;", "&")
        line = line.replace("&lt;", "<").replace("&gt;", ">")
        line = re.sub(r"<[^>]+>", "", line)
        # Metadata lines: (câu 39-170), (trang 5)
        stripped = line.strip()
        if re.match(r'\(\s*(câu|trang|chương)\s*[\d\s\-–]+\s*\)$', stripped, re.IGNORECASE):
            continue
        # Strip leading line numbers
        line = strip_line_numbers(line)
        # Fix spacing
        line = fix_spacing(line)
        # Collapse whitespace
        line = re.sub(r"[ \t]+", " ", line)
        # Unicode normalize
        line = unicodedata.normalize("NFC", line)
        line = line.strip()
        if line:
            cleaned.append(line)
    return SEP.join(cleaned)


def clean_content(df):
    df["content"] = df["content"].apply(clean_text)
    mask = df["content"].astype(str).str.strip() != ""
    dropped = (~mask).sum()
    df = df[mask].copy()
    print(f"  Text cleaned, {dropped} empty rows dropped")
    return df


# ═══════════════════════════════════════════════
# STAGE 3: Filter short
# ═══════════════════════════════════════════════

def filter_short(df, min_lines=2):
    n_lines = df["content"].apply(lambda t: str(t).count(SEP) + 1)
    mask = n_lines >= min_lines
    dropped = (~mask).sum()
    df = df[mask].copy()
    print(f"  Poems < {min_lines} lines: {dropped} dropped → {len(df):,} kept")
    return df


# ═══════════════════════════════════════════════
# STAGE 4: Exact dedup
# ═══════════════════════════════════════════════

def content_hash(text: str) -> str:
    if pd.isna(text):
        return ""
    c = fix_spacing(text.lower())
    c = re.sub(r"[ \t]+", " ", c)
    c = unicodedata.normalize("NFC", c).strip()
    return hashlib.md5(c.encode()).hexdigest()


def remove_duplicates(df):
    df["_h"] = df["content"].apply(content_hash)
    before = len(df)
    df = df.drop_duplicates(subset="_h", keep="first")
    df = df.drop(columns=["_h"])
    print(f"  Duplicates removed: {before - len(df)}")
    return df


# ═══════════════════════════════════════════════
# STAGE 5: Near-duplicate (line overlap)
# ═══════════════════════════════════════════════

def _norm_line(line):
    """Normalize a single line for comparison."""
    c = fix_spacing(line.strip().lower())
    c = strip_line_numbers(c)
    c = re.sub(r'[^\w\s]', '', c)
    c = re.sub(r'\s+', ' ', c).strip()
    return c


def remove_near_duplicates(df, threshold=0.3):
    """Remove poems that share >threshold% of lines. Keeps the longer one."""
    before = len(df)
    if before < 2:
        return df

    # Build line index
    lines_by_poem = {}
    line_to_poems = {}
    for idx, (_, row) in enumerate(df.iterrows()):
        poem_lines = set()
        for line in str(row['content']).split(SEP):
            norm = _norm_line(line)
            if len(norm) >= 3:
                h = hashlib.md5(norm.encode()).hexdigest()
                poem_lines.add(h)
                line_to_poems.setdefault(h, []).append(idx)
        lines_by_poem[idx] = poem_lines

    # Candidate pairs (share ≥1 line hash)
    pairs = set()
    for indices in line_to_poems.values():
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                pairs.add((indices[i], indices[j]))

    if not pairs:
        print(f"  Near-duplicates: 0 (no shared lines)")
        return df

    print(f"  Checking {len(pairs):,} candidate pairs (share ≥1 line)...")

    to_drop = set()
    for a, b in pairs:
        if a in to_drop or b in to_drop:
            continue
        la, lb = lines_by_poem[a], lines_by_poem[b]
        if not la or not lb:
            continue
        shared = len(la & lb)
        overlap = shared / max(len(la), len(lb))
        if overlap > threshold:
            na = len(df.iloc[a]['content'].split(SEP))
            nb = len(df.iloc[b]['content'].split(SEP))
            if na < 4 and nb < 4:
                continue
            drop = b if na >= nb else a
            to_drop.add(drop)

    if not to_drop:
        print(f"  Near-duplicates: 0 (none >{threshold:.0%} overlap)")
        return df

    df = df.drop(index=list(to_drop)).reset_index(drop=True)
    removed = before - len(df)
    print(f"  Near-duplicates removed: {removed}")
    return df


def check_near_duplicates(df, threshold=0.3):
    """Dry-run: find near-duplicates, show stats only (no removal)."""
    if len(df) < 2:
        return []

    lines_by_poem = {}
    line_to_poems = {}
    for idx, (_, row) in enumerate(df.iterrows()):
        poem_lines = set()
        for line in str(row['content']).split(SEP):
            norm = _norm_line(line)
            if len(norm) >= 3:
                h = hashlib.md5(norm.encode()).hexdigest()
                poem_lines.add(h)
                line_to_poems.setdefault(h, []).append(idx)
        lines_by_poem[idx] = poem_lines

    pairs = set()
    for indices in line_to_poems.values():
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                pairs.add((indices[i], indices[j]))

    if not pairs:
        print("  No line-sharing poems found — dataset is clean.")
        return []

    print(f"  Analyzing {len(pairs):,} candidate pairs...")

    to_drop = set()
    found = []
    for a, b in pairs:
        if a in to_drop or b in to_drop:
            continue
        la, lb = lines_by_poem[a], lines_by_poem[b]
        if not la or not lb:
            continue
        shared = len(la & lb)
        overlap = shared / max(len(la), len(lb))
        if overlap > threshold:
            na = len(df.iloc[a]['content'].split(SEP))
            nb = len(df.iloc[b]['content'].split(SEP))
            if na < 4 and nb < 4:
                continue
            drop = b if na >= nb else a
            keep = a if drop == b else b
            to_drop.add(drop)
            found.append((
                str(df.iloc[keep]['title']), str(df.iloc[keep]['author']),
                str(df.iloc[drop]['title']), str(df.iloc[drop]['author']),
                na if keep == a else nb, shared, overlap))

    if not found:
        print(f"  No near-duplicates found (> {threshold:.0%} overlap).")
        return []

    print(f"\n  📋  Near-duplicates: {len(found)} pairs (>{threshold:.0%} overlap)")
    print(f"  {'Keep':<42s} {'Drop':<42s} {'Lines':>5s} {'Shared':>6s} {'Overlap':>7s}")
    print(f"  {'-'*40} {'-'*40} {'-'*5} {'-'*6} {'-'*7}")
    for kt, ka, dt, da, lines, shared, ov in found[:20]:
        print(f"  {kt[:40]:<40s}   {dt[:40]:<40s}   {lines:>5d} {shared:>6d} {ov:>6.0%}")
    if len(found) > 20:
        print(f"  ... +{len(found)-20} more")

    by_author = {}
    for _, _, _, drop_a, _, _, _ in found:
        by_author[drop_a] = by_author.get(drop_a, 0) + 1
    if by_author:
        print(f"\n  By author (would remove):")
        for author, count in sorted(by_author.items(), key=lambda x: -x[1]):
            print(f"    {author:<30s} {count}")
    print(f"\n  💡 Total: {len(found)} poems would be removed")
    return found


# ═══════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════

def clean(csv_path=None, output_path=None):
    csv_path = Path(csv_path or CSV_PATH)
    output_path = Path(output_path or OUTPUT_PATH)

    print(f"📖  Loading: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"    Original: {len(df):,} poems\n")

    df = filter_genres(df)
    print(f"    → {len(df):,} (lục bát: {df['genre'].eq('lục bát').sum():,}, "
          f"bảy chữ: {df['genre'].eq('bảy chữ').sum():,})")

    df = clean_content(df)
    df = filter_short(df, min_lines=2)
    df = remove_duplicates(df)
    df = remove_near_duplicates(df, threshold=0.3)

    print(f"\n📊  Final: {len(df):,} poems")
    for genre in ["lục bát", "bảy chữ"]:
        gdf = df[df["genre"] == genre]
        n_lines = gdf["content"].apply(lambda t: str(t).count(SEP) + 1)
        st = gdf["specific_genre"].astype(str).str.contains("song thất", case=False, na=False).sum()
        print(f"    {genre:<12s}: {len(gdf):>6,}  avg {n_lines.mean():.0f} lines  "
              f"max {n_lines.max()}" + (f"  (song thất: {st})" if st else ""))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\n✅  Saved → {output_path}")
    return df


def clean_existing(csv_path, output_path=None):
    """Re-clean a CSV: fix spacing, strip metadata, re-dedup."""
    csv_path = Path(csv_path)
    output_path = Path(output_path or csv_path)

    print(f"📖  Re-cleaning: {csv_path}")
    df = pd.read_csv(csv_path)
    before = len(df)
    print(f"    Input: {before:,} poems")

    df["content"] = df["content"].apply(clean_text)
    mask = df["content"].astype(str).str.strip() != ""
    emptied = (~mask).sum()
    df = df[mask].copy()
    if emptied:
        print(f"    Emptied: {emptied}")

    df = filter_short(df, min_lines=2)
    df = remove_duplicates(df)
    df = remove_near_duplicates(df, threshold=0.3)

    after = len(df)
    print(f"    Output: {after:,} poems ({before - after} removed)")

    if output_path != csv_path:
        import shutil
        shutil.copy2(csv_path, str(csv_path) + ".bak")
        print(f"    Backup: {csv_path}.bak")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"    Saved → {output_path}")
    return df


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Clean poetry CSV")
    p.add_argument("--csv", type=str, default=None)
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--reclean", type=str, default=None,
                   help="Re-clean existing CSV in-place")
    p.add_argument("--check-dupes", type=str, default=None,
                   help="Check near-duplicates (dry-run)")
    p.add_argument("--threshold", type=float, default=0.3,
                   help="Overlap threshold (default: 0.3)")
    args = p.parse_args()

    if args.check_dupes:
        print(f"🔍  Checking: {args.check_dupes}")
        df = pd.read_csv(args.check_dupes)
        print(f"    Poems: {len(df):,}")
        check_near_duplicates(df, threshold=args.threshold)
    elif args.reclean:
        clean_existing(args.reclean, args.output)
    else:
        clean(args.csv, args.output)
