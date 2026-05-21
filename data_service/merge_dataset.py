"""
Validate scraped poems, deduplicate, and merge into a new dataset.

What it does:
  1. Read all data_service/raw/*.csv files
  2. Validate each poem (must have content, Vietnamese chars, reasonable line count)
  3. Check for duplicates against poems_dataset_clean.csv
  4. Report: new, duplicate, invalid, stats by author
  5. Merge valid new poems → data/poems_dataset_merged.csv

Usage:
  python data_service/merge_dataset.py            # validate + merge (dry-run)
  python data_service/merge_dataset.py --commit   # actually merge
  python data_service/merge_dataset.py --validate # only validate, no merge
"""

import argparse, hashlib, os, re, sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
RAW_DIR = ROOT / "data_service" / "raw"
EXISTING_CSV = ROOT / "data" / "poems_dataset_clean.csv"
MERGED_CSV = ROOT / "data" / "poems_dataset_merged.csv"


# ═══════════════════════════════════════════════════════
#  VALIDATION
# ═══════════════════════════════════════════════════════

def validate_poem(row):
    """Check if a scraped poem row is valid. Returns (is_valid, reason)."""
    content = str(row.get('content', ''))
    title = str(row.get('title', ''))
    author = str(row.get('author', ''))
    genre = str(row.get('genre', ''))

    # Must have content
    if not content or len(content) < 20:
        return False, "empty or too short"

    # Must contain Vietnamese characters
    if not any(ord(c) > 127 for c in content):
        return False, "no Vietnamese characters"

    # Content must have lines (at least 2)
    lines = content.split('<\n>')
    if len(lines) < 2:
        return False, f"only {len(lines)} line(s)"

    # Each line should be reasonable length
    for line in lines:
        words = line.split()
        if len(words) < 2:
            return False, f"line too short: '{line[:40]}'"

    # Title should not be empty
    if not title or len(title) < 2:
        return False, "empty title"

    return True, "ok"


def compute_hash(content):
    """Normalized content hash for fuzzy dedup: strip whitespace, lowercase."""
    text = str(content).strip().lower()
    text = re.sub(r'\s+', ' ', text)  # normalize whitespace
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def compute_title_hash(title):
    """Normalized title hash: lowercase, strip punctuation."""
    text = re.sub(r'[^\w\s]', '', str(title).lower().strip())
    text = re.sub(r'\s+', ' ', text)
    return hashlib.sha256(text.encode()).hexdigest()[:12]


# ═══════════════════════════════════════════════════════
#  MAIN LOGIC
# ═══════════════════════════════════════════════════════

def load_scraped():
    """Load all scraped poems from data_service/raw/."""
    raw_files = sorted(RAW_DIR.glob("*.csv"))
    if not raw_files:
        print("❌ No scraped files found in data_service/raw/")
        print("   Run: python data_service/scraper.py --all")
        return None

    print(f"\n📁  Loading {len(raw_files)} scraped files:")
    dfs = []
    for f in raw_files:
        df = pd.read_csv(f)
        dfs.append(df)
        print(f"    {f.name}: {len(df)} poems")

    scraped = pd.concat(dfs, ignore_index=True)
    scraped = scraped.drop_duplicates(subset=['url'])
    print(f"  Total (deduped by URL): {len(scraped)} poems")
    return scraped


def validate_all(scraped):
    """Validate all poems. Print report."""
    print(f"\n🔍  Validating {len(scraped)} poems...")
    results = []
    for _, row in scraped.iterrows():
        ok, reason = validate_poem(row)
        results.append((ok, reason))

    valid_mask = [r[0] for r in results]
    valid = scraped[valid_mask].copy()
    invalid = scraped[[not v for v in valid_mask]].copy()

    print(f"  ✅ Valid:   {len(valid)}")
    print(f"  ❌ Invalid: {len(invalid)}")
    if len(invalid) > 0:
        print(f"\n  Invalid poems:")
        for _, row in invalid.iterrows():
            ok, reason = validate_poem(row)
            print(f"    ❌ [{row.get('author', '?')}] {row.get('title', '?')[:50]} — {reason}")

    return valid, invalid


def check_duplicates(valid, existing_path):
    """Check new poems against existing dataset using normalized hashes."""
    print(f"\n📊  Checking against {existing_path.name}...")
    existing = pd.read_csv(existing_path, dtype={'content': str, 'title': str})
    print(f"  Existing: {len(existing):,} poems")

    # Build lookup: content hash + title hash for existing poems
    existing_content_hashes = set()
    existing_title_hashes = set()
    for _, row in existing.iterrows():
        existing_content_hashes.add(compute_hash(str(row['content'])))
        existing_title_hashes.add(compute_title_hash(str(row['title'])))

    dupes = []
    truly_new = []
    for _, row in valid.iterrows():
        ch = compute_hash(str(row['content']))
        th = compute_title_hash(str(row['title']))
        # Match by content hash OR by title hash (same poem, different formatting)
        if ch in existing_content_hashes or th in existing_title_hashes:
            dupes.append(row)
        else:
            truly_new.append(row)

    print(f"  New (unique):    {len(truly_new)}")
    print(f"  Duplicates:      {len(dupes)} (content hash or title match)")
    if len(dupes) > 0 and len(dupes) <= 10:
        print(f"\n  Duplicate poems:")
        for row in dupes:
            print(f"    ♻️  [{row['author']}] {str(row['title'])[:60]}")
    elif len(dupes) > 10:
        print(f"\n  Duplicate poems (first 10):")
        for row in dupes[:10]:
            print(f"    ♻️  [{row['author']}] {str(row['title'])[:60]}")
        print(f"    ... +{len(dupes)-10} more")

    return pd.DataFrame(truly_new) if truly_new else pd.DataFrame()


def stats_by_author(scraped, valid, new):
    """Print per-author statistics."""
    print(f"\n📈  By author:")

    def count(df, author):
        if df is None or len(df) == 0:
            return 0
        return df['author'].eq(author).sum()

    authors = scraped['author'].unique()
    for a in authors:
        total = count(scraped, a)
        v = count(valid, a)
        n = count(new, a) if new is not None else 0
        print(f"  {a:<20s}  scraped={total:3d}  valid={v:3d}  new={n:3d}  "
              f"{'✅' if v == total else '⚠️' if v > 0 else '❌'}")


def merge(new_df, existing_path, output_path):
    """Merge new poems into existing dataset."""
    print(f"\n📦  Merging...")
    existing = pd.read_csv(existing_path)

    # Ensure column alignment
    cols = ['content', 'title', 'url', 'genre', 'period', 'specific_genre', 'author']
    for c in cols:
        if c not in existing.columns:
            existing[c] = ""
        if c not in new_df.columns:
            new_df[c] = ""
    existing = existing[cols]
    new_df = new_df[cols]

    # Backup existing
    backup = existing_path.with_suffix('.csv.bak')
    import shutil
    shutil.copy2(existing_path, backup)
    print(f"  Backup: {backup}")

    merged = pd.concat([existing, new_df], ignore_index=True)
    merged.to_csv(output_path, index=False)
    print(f"  Existing: {len(existing):,}  +  New: {len(new_df):,}  →  {len(merged):,}")
    print(f"  Saved: {output_path}")

    # Genre stats
    lb = merged['genre'].eq('lục bát').sum()
    tn = merged['genre'].eq('bảy chữ').sum()
    other = len(merged) - lb - tn
    print(f"  Lục bát: {lb:,}  |  Bảy chữ: {tn:,}  |  Other: {other:,}")
    return merged


# ═══════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="Validate and merge scraped poems")
    p.add_argument("--commit", action="store_true",
                   help="Actually merge (without this flag, dry-run only)")
    p.add_argument("--validate", action="store_true",
                   help="Only validate, don't merge")
    args = p.parse_args()

    if not EXISTING_CSV.exists():
        print(f"❌ Existing dataset not found: {EXISTING_CSV}")
        sys.exit(1)

    # 1. Load scraped
    scraped = load_scraped()
    if scraped is None:
        return

    # 2. Validate
    valid, invalid = validate_all(scraped)

    # 3. Duplicate check
    new = check_duplicates(valid, EXISTING_CSV)

    # 4. Stats
    stats_by_author(scraped, valid, new)

    # 5. Merge (optional)
    if args.validate:
        print(f"\n✅ Validation complete. {len(new)} new poems ready to merge.")
        print(f"   Run with --commit to merge.")
        return

    if not args.commit:
        print(f"\n{'='*60}")
        print(f"🔍  DRY-RUN — no changes made.")
        print(f"   {len(new)} new poems would be merged.")
        print(f"   Run with --commit to actually merge.")
        print(f"{'='*60}")
        return

    if len(new) == 0:
        print(f"\n✅ No new poems to merge.")
        return

    merge(new, EXISTING_CSV, MERGED_CSV)
    print(f"\n✅ Merge complete! New dataset: {MERGED_CSV}")


if __name__ == "__main__":
    main()
