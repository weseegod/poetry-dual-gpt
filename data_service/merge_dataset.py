"""
Validate scraped poems, deduplicate, and merge into dataset.

═══ FULL PIPELINE ═══

  STEP 1 — Collect raw data into data_service/raw/
    python data_service/scraper.py --all                  # scrape poetry websites
    python data_service/extract_facebook.py --to-raw      # extract from Facebook JSON
    # (add any future source here → raw/*.csv)

  STEP 2 — Validate & dedup (dry-run first, safe to run anytime)
    python data_service/merge_dataset.py                  # show stats: how many valid/new/dupe

  STEP 3 — Extract genuinely new poems → data_service/new/
    python data_service/merge_dataset.py --extract-new    # deduped new poems by author

  STEP 4 — Review new/*.csv, then commit to merged dataset
    python data_service/merge_dataset.py --commit         # append → poems_dataset_merged.csv

  STEP 5 — Clean merged → final training dataset
    python src/clean_data.py --csv data/poems_dataset_merged.csv --output data/clean_data_vN.csv

  STEP 6 — Validate final output
    python src/clean_data.py --check-dupes data/clean_data_vN.csv

Files:
  data_service/raw/          ← append-only: all scraped sources land here
  data_service/new/          ← temporary: new deduped poems for review before commit
  data/poems_dataset_merged.csv  ← cumulative: all committed poems (append-only)
  data/clean_data_vN.csv     ← final: cleaned + deduped, ready for training

Usage:
  python data_service/merge_dataset.py                # dry-run: stats only
  python data_service/merge_dataset.py --extract-new   # extract new poems → new/*.csv
  python data_service/merge_dataset.py --commit        # merge from new/ into dataset
"""

import argparse, hashlib, os, re, shutil, sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
RAW_DIR = ROOT / "data_service" / "raw"
NEW_DIR = ROOT / "data_service" / "new"
EXISTING_CSV = ROOT / "data" / "poems_dataset_clean.csv"
MERGED_CSV = ROOT / "data" / "poems_dataset_merged.csv"

COLUMNS = ['content', 'title', 'url', 'genre', 'period', 'specific_genre', 'author']

# ═════════════════════════════════════════════════
#  HASHING
# ═════════════════════════════════════════════════

def content_hash(text):
    """Normalized hash: lowercase, collapse whitespace."""
    t = re.sub(r'\s+', ' ', str(text).strip().lower())
    return hashlib.sha256(t.encode()).hexdigest()[:16]

def title_hash(text):
    """Normalized title hash: lowercase, strip punctuation."""
    t = re.sub(r'[^\w\s]', '', str(text).lower().strip())
    t = re.sub(r'\s+', ' ', t)
    return hashlib.sha256(t.encode()).hexdigest()[:12]


# ═════════════════════════════════════════════════
#  VALIDATE
# ═════════════════════════════════════════════════

def validate_poem(row):
    content = str(row.get('content', ''))
    title = str(row.get('title', ''))
    if not content or len(content) < 20:
        return False, "empty or too short"
    if not any(ord(c) > 127 for c in content):
        return False, "no Vietnamese"
    lines = content.split('<\n>')
    if len(lines) < 2:
        return False, f"only {len(lines)} line(s)"
    if not title or len(title) < 2:
        return False, "empty title"
    return True, "ok"


# ═════════════════════════════════════════════════
#  LOAD & CLEAN
# ═════════════════════════════════════════════════

NOISE_PREFIXES = [
    'tập thơ', 'cách sử dụng', 'chuỗi tìm kiếm', 'qr code',
    'bạn không đủ sức', 'nếu bạn', 'mọt sách', 'thư viện',
    'mae west', 'they say', 'an institution',
]

def strip_noise_prefix(content, author_name='', title=''):
    """Remove leading garbage: site noise, author name, Tập thơ, embedded title."""
    sep = '<\n>'
    lines = content.split(sep)
    author_lower = author_name.lower().strip() if author_name else ''
    title_lower = title.lower().strip() if title else ''

    while lines:
        first = lines[0].strip().lower()
        if not first:
            lines = lines[1:]; continue
        words = first.split()
        # Noise keywords
        if any(n in first for n in NOISE_PREFIXES):
            lines = lines[1:]; continue
        # Author name alone
        if author_lower and first == author_lower:
            lines = lines[1:]
            if lines and ('tập thơ' in lines[0].lower()):
                lines = lines[1:]
            continue
        # Embedded metadata: title → author → "Tập thơ ..." (3 lines)
        if (title_lower and len(lines) >= 3 and
            first == title_lower and
            lines[1].strip().lower() == author_lower and
            'tập thơ' in lines[2].strip().lower()):
            lines = lines[3:]
            continue
        # "Tập thơ Author:" without preceding lines
        if 'tập thơ' in first and first.endswith(':'):
            lines = lines[1:]; continue
        # Non-Vietnamese or very short
        if len(words) <= 1:
            lines = lines[1:]; continue
        if len(words) <= 2 and not any(ord(c) > 127 for c in first):
            lines = lines[1:]; continue
        break
    return sep.join(lines)


def load_scraped():
    """Load + clean all raw/*.csv files."""
    files = sorted(RAW_DIR.glob("*.csv"))
    if not files:
        print("❌ No files in data_service/raw/")
        print("   Run: python data_service/scraper.py --all")
        return None

    print(f"\n📁  {len(files)} scraped files:")
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        dfs.append(df)
        print(f"    {f.name}: {len(df)} poems")

    scraped = pd.concat(dfs, ignore_index=True)
    scraped = scraped.drop_duplicates(subset=['url'])

    # Clean prefix garbage
    for idx, row in scraped.iterrows():
        scraped.at[idx, 'content'] = strip_noise_prefix(
            str(row['content']), str(row.get('author', '')), str(row.get('title', '')))

    print(f"  Total (deduped): {len(scraped)} poems")
    return scraped


# ═════════════════════════════════════════════════
#  EXTRACT NEW
# ═════════════════════════════════════════════════

def extract_new(scraped):
    """Validate scraped, check against existing, save new poems to new/*.csv."""
    # Validate
    valid_mask = [validate_poem(row)[0] for _, row in scraped.iterrows()]
    valid = scraped[valid_mask].copy()
    invalid = scraped[[not v for v in valid_mask]]

    print(f"\n🔍  Validate: {len(valid)} ✅  {len(invalid)} ❌")
    for _, row in invalid.iterrows():
        _, reason = validate_poem(row)
        print(f"    ❌ [{row.get('author','?')}] {str(row.get('title',''))[:50]} — {reason}")

    # Check against existing
    existing = pd.read_csv(EXISTING_CSV, dtype={'content': str, 'title': str})
    existing = existing.dropna(subset=['content'])
    print(f"\n📊  Existing dataset: {len(existing):,} poems")

    content_hashes = {content_hash(str(r['content'])) for _, r in existing.iterrows()}
    title_hashes = {title_hash(str(r['title'])) for _, r in existing.iterrows()}

    dupes, new_rows = [], []
    for _, row in valid.iterrows():
        ch = content_hash(str(row['content']))
        th = title_hash(str(row['title']))
        if ch in content_hashes or th in title_hashes:
            dupes.append(row)
        else:
            new_rows.append(row)

    new_df = pd.DataFrame(new_rows) if new_rows else pd.DataFrame()
    print(f"  New: {len(new_df)}  |  Duplicates: {len(dupes)}")

    # Per-author stats
    authors = valid['author'].unique()
    print(f"\n📈  By author:")
    for a in authors:
        t = valid['author'].eq(a).sum()
        n = new_df['author'].eq(a).sum() if len(new_df) > 0 else 0
        mark = '⚠️' if t > 0 and n == 0 else '✅'
        print(f"  {a:<20s}  scraped={t:3d}  new={n:3d}  {mark}")

    if len(new_df) == 0:
        print("\n✅ No new poems to extract.")
        return 0

    # Save new poems by author
    NEW_DIR.mkdir(parents=True, exist_ok=True)
    # Clean old new/ files
    for f in NEW_DIR.glob("*_new.csv"):
        f.unlink()

    total = 0
    for a in new_df['author'].unique():
        subset = new_df[new_df['author'] == a][COLUMNS]
        fname = NEW_DIR / f"{a.lower().replace(' ', '_')}_new.csv"
        subset.to_csv(fname, index=False)
        print(f"  💾  {fname.name}: {len(subset)} poems")
        total += len(subset)

    print(f"\n✅ Extracted {total} new poems to data_service/new/")
    print(f"   Review files, then run: python data_service/merge_dataset.py --commit")
    return total


# ═════════════════════════════════════════════════
#  MERGE
# ═════════════════════════════════════════════════

def merge():
    """Merge all new/*.csv files into dataset."""
    files = sorted(NEW_DIR.glob("*_new.csv"))
    if not files:
        print("❌ No files in data_service/new/")
        print("   Run: python data_service/merge_dataset.py --extract-new")
        return

    print(f"\n📦  Merging {len(files)} new files:")
    new_parts = []
    for f in files:
        df = pd.read_csv(f)[COLUMNS]
        new_parts.append(df)
        print(f"    {f.name}: {len(df)} poems")

    new_df = pd.concat(new_parts, ignore_index=True)
    existing = pd.read_csv(EXISTING_CSV)

    # Align columns
    for c in COLUMNS:
        if c not in existing.columns:
            existing[c] = ""
    existing = existing[COLUMNS]

    # Backup
    backup = EXISTING_CSV.with_suffix('.csv.bak')
    shutil.copy2(EXISTING_CSV, backup)

    merged = pd.concat([existing, new_df], ignore_index=True)
    merged.to_csv(MERGED_CSV, index=False)

    print(f"\n  Existing: {len(existing):,}  +  New: {len(new_df):,}  →  {len(merged):,}")
    print(f"  Backup: {backup}")
    print(f"  Output: {MERGED_CSV}")

    # Genre breakdown
    lb = merged['genre'].eq('lục bát').sum()
    tn = merged['genre'].eq('bảy chữ').sum()
    other = len(merged) - lb - tn
    print(f"  Lục bát: {lb:,}  |  Bảy chữ: {tn:,}  |  Other: {other:,}")
    print(f"\n✅ Done!")


# ═════════════════════════════════════════════════
#  CLI
# ═════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="Validate and merge scraped poems")
    p.add_argument("--extract-new", action="store_true",
                   help="Extract new poems to data_service/new/*.csv for review")
    p.add_argument("--commit", action="store_true",
                   help="Merge poems from data_service/new/ into dataset")
    args = p.parse_args()

    if args.commit:
        merge()
        return

    if not EXISTING_CSV.exists():
        print(f"❌ Dataset not found: {EXISTING_CSV}")
        sys.exit(1)

    scraped = load_scraped()
    if scraped is None:
        return

    if args.extract_new:
        extract_new(scraped)
        return

    # Default: dry-run stats
    valid_mask = [validate_poem(row)[0] for _, row in scraped.iterrows()]
    valid = scraped[valid_mask]

    existing = pd.read_csv(EXISTING_CSV, dtype={'content': str, 'title': str})
    existing = existing.dropna(subset=['content'])
    content_hashes = {content_hash(str(r['content'])) for _, r in existing.iterrows()}
    title_hashes = {title_hash(str(r['title'])) for _, r in existing.iterrows()}

    new_count = 0
    for _, row in valid.iterrows():
        ch = content_hash(str(row['content']))
        th = title_hash(str(row['title']))
        if ch not in content_hashes and th not in title_hashes:
            new_count += 1

    print(f"\n📊  Scraped: {len(scraped)}  |  Valid: {len(valid)}  |  New: {new_count}")
    print(f"   Existing dataset: {len(existing):,} poems")
    print(f"\n   Next: python data_service/merge_dataset.py --extract-new")
    print(f"         python data_service/merge_dataset.py --commit")


if __name__ == "__main__":
    main()
