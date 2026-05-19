"""
Convert poems_dataset.csv → training pairs for causal LM.
Phase 1: Lục Bát only (6-syllable prompt → 8-syllable response).

Output format per line:
  <|start|> [LUC_BAT] sáu chữ, <|reply|> tám chữ nối theo. <|end|>
"""

import argparse
import re
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).parent.parent
CSV_PATH = ROOT / "data" / "poems_dataset.csv"
OUTPUT_PATH = ROOT / "data" / "poetry_corpus.txt"

# Control tokens — must match train_bpe.py SPECIAL_TOKENS
START = "<|start|>"
REPLY = "<|reply|>"
END = "<|end|>"
TAG = "[LUC_BAT]"


def count_syllables(line: str) -> int:
    """Vietnamese words = syllables. Split on whitespace."""
    return len(line.strip().split())


def clean_line(line: str) -> str:
    """Remove HTML artifacts, collapse whitespace, strip punctuation."""
    line = re.sub(r"<[^>]+>", "", line)
    line = re.sub(r"\s+", " ", line)
    return line.strip(" ,.-;:!?")


def parse_poem(content: str) -> list[str]:
    """
    Split poem text into individual lines.
    CSV stores poems with ' <\\n> ' as line separator.
    """
    lines = content.split(" <\n> ")
    lines = [clean_line(l) for l in lines]
    return [l for l in lines if l and len(l.split()) >= 2]


def make_pairs(lines: list[str]) -> list[str]:
    """
    Convert Lục Bát lines into (prompt, reply) pairs.
    Pattern: 6-8-6-8... Pair each 6-syl line with next 8-syl line.
    Accepts ±1 syllable tolerance for noisy data.
    """
    pairs = []
    for i in range(len(lines) - 1):
        prompt, reply = lines[i], lines[i + 1]
        p_ok = 5 <= count_syllables(prompt) <= 7
        r_ok = 7 <= count_syllables(reply) <= 9
        if p_ok and r_ok and prompt and reply:
            pairs.append(f"{START} {TAG} {prompt}, {REPLY} {reply} {END}")
    return pairs


def preprocess(csv_path=None, output_path=None, max_poems=None):
    """Main: read CSV → filter lục bát → create pairs → save."""
    csv_path = Path(csv_path or CSV_PATH)
    output_path = Path(output_path or OUTPUT_PATH)

    # Load and filter
    df = pd.read_csv(csv_path)
    df = df[df["genre"] == "lục bát"]
    print(f"Lục bát poems: {len(df):,}")

    if max_poems:
        df = df.head(max_poems)

    # Process each poem into training pairs
    all_pairs, skipped, empty = [], 0, 0
    for _, row in df.iterrows():
        content = row["content"]
        if pd.isna(content) or not content.strip():
            empty += 1
            continue
        lines = parse_poem(content)
        if len(lines) < 2:
            skipped += 1
            continue
        all_pairs.extend(make_pairs(lines))

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for pair in all_pairs:
            f.write(pair + "\n")

    print(f"Empty: {empty} | Too short: {skipped} | Pairs: {len(all_pairs):,}")
    print(f"Saved → {output_path}")
    return all_pairs


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Preprocess poetry CSV → training pairs")
    p.add_argument("--csv", type=str, default=None)
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--max", type=int, default=None, help="Limit poems for testing")
    args = p.parse_args()
    preprocess(args.csv, args.output, args.max)
