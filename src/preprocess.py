"""
preprocess.py — Convert raw Vietnamese poetry CSV into training pairs.

Phase 1: Lục Bát only (6-syllable prompt → 8-syllable response).

Input:  data/poems_dataset.csv (198K poems, 7 columns)
Output: data/poetry_corpus.txt (one pair per line with control tokens)

Format: <|start|> [LUC_BAT] six-syllable-line, <|reply|> eight-syllable-line <|end|>
"""

import argparse
import re
import pandas as pd
from pathlib import Path


# Paths
ROOT = Path(__file__).parent.parent
CSV_PATH = ROOT / "data" / "poems_dataset.csv"
OUTPUT_PATH = ROOT / "data" / "poetry_corpus.txt"

# Control tokens (must match train_bpe.py)
START_TOKEN = "<|start|>"
REPLY_TOKEN = "<|reply|>"
END_TOKEN = "<|end|>"
GENRE_TAG = "[LUC_BAT]"


def count_syllables(line: str) -> int:
    """
    Count Vietnamese syllables in a line.
    A syllable = one word (Vietnamese is monosyllabic in writing).
    Splits on whitespace, ignores empty strings.
    """
    return len(line.strip().split())


def clean_line(line: str) -> str:
    """Remove HTML artifacts and extra whitespace from a poem line."""
    line = re.sub(r"<[^>]+>", "", line)      # remove HTML tags
    line = re.sub(r"\s+", " ", line)         # collapse whitespace
    line = line.strip(" ,.-;:!?")
    return line


def parse_poem_lines(content: str) -> list[str]:
    """
    Split poem content into individual lines.

    The CSV stores poems with ' <\\n> ' as line separator
    (literally: space, less-than, backslash, n, greater-than, space).
    """
    # Split on the exact separator found in the CSV
    lines = content.split(" <\n> ")
    # Clean each line
    lines = [clean_line(l) for l in lines]
    # Remove empty lines
    lines = [l for l in lines if l and len(l.split()) >= 2]
    return lines


def make_luc_bat_pairs(lines: list[str]) -> list[str]:
    """
    Convert Lục Bát lines into (6-syllable → 8-syllable) pairs.

    Lục Bát pattern: 6-8-6-8-6-8...
    We pair each 6-syllable line with the next 8-syllable line,
    and the 8-syllable line becomes a 6-syllable prompt for the NEXT line.
    """
    pairs = []
    i = 0
    while i < len(lines) - 1:
        prompt = lines[i]
        reply = lines[i + 1]

        prompt_syl = count_syllables(prompt)
        reply_syl = count_syllables(reply)

        # Accept approximate matches (5-7 for "6", 7-9 for "8")
        # Some lines have extra particles or missing syllables in the data
        prompt_ok = 5 <= prompt_syl <= 7
        reply_ok = 7 <= reply_syl <= 9

        if prompt_ok and reply_ok and prompt and reply:
            pair = f"{START_TOKEN} {GENRE_TAG} {prompt}, {REPLY_TOKEN} {reply} {END_TOKEN}"
            pairs.append(pair)

        i += 1  # step by 1 so 8-syllable line can serve as prompt for next

    return pairs


def preprocess(csv_path=None, output_path=None, max_poems=None):
    """Main preprocessing pipeline for Phase 1 (Lục Bát only)."""
    csv_path = Path(csv_path) if csv_path else CSV_PATH
    output_path = Path(output_path) if output_path else OUTPUT_PATH

    print(f"Reading: {csv_path}")
    df = pd.read_csv(csv_path)

    # Phase 1: Lục Bát only
    df_lb = df[df["genre"] == "lục bát"]
    print(f"Filtered lục bát: {len(df_lb):,} poems (from {len(df):,} total)")

    if max_poems:
        df_lb = df_lb.head(max_poems)
        print(f"Using first {max_poems:,} poems for quick test")

    all_pairs = []
    skipped = 0
    empty = 0

    for idx, row in df_lb.iterrows():
        content = row["content"]
        if pd.isna(content) or not content.strip():
            empty += 1
            continue

        lines = parse_poem_lines(content)
        if len(lines) < 2:
            skipped += 1
            continue

        pairs = make_luc_bat_pairs(lines)
        all_pairs.extend(pairs)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for pair in all_pairs:
            f.write(pair + "\n")

    print(f"\nResults:")
    print(f"  Poems processed: {len(df_lb):,}")
    print(f"  Empty poems:     {empty:,}")
    print(f"  Too-short poems: {skipped:,}")
    print(f"  Training pairs:  {len(all_pairs):,}")
    print(f"  Saved to:        {output_path}")

    return all_pairs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess poetry CSV for Phase 1 (Lục Bát)")
    parser.add_argument("--csv", type=str, default=None, help="Path to poems CSV")
    parser.add_argument("--output", type=str, default=None, help="Output path")
    parser.add_argument("--max", type=int, default=None, help="Limit number of poems (for testing)")
    args = parser.parse_args()

    preprocess(csv_path=args.csv, output_path=args.output, max_poems=args.max)
