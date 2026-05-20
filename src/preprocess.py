"""
Convert poems_dataset_clean.csv → training pairs for causal LM.
Supports: Lục Bát (6→8) + Thất Ngôn (7→7).

Output format per line:
  <|start|> [LUC_BAT] prompt_six_syl reply_eight_syl <|end|>
  <|start|> [THAT_NGON] prompt_seven_syl reply_seven_syl <|end|>
"""

import argparse
import re
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).parent.parent
CSV_PATH = ROOT / "data" / "poems_dataset_clean.csv"
OUTPUT_PATH = ROOT / "data" / "poetry_corpus.txt"

# Control tokens — must match train_bpe.py SPECIAL_TOKENS
START = "<|start|>"
REPLY = "<|reply|>"
END = "<|end|>"

# Genre rules: tag + strict syllable counts
GENRE_RULES = {
    "lục bát":  {"tag": "[LUC_BAT]",   "prompt_syl": 6, "reply_syl": 8},
    "bảy chữ":  {"tag": "[THAT_NGON]", "prompt_syl": 7, "reply_syl": 7},
}


def count_syllables(line: str) -> int:
    """Vietnamese words = syllables. Split on whitespace."""
    return len(line.strip().split())


def clean_line(line: str) -> str:
    """Collapse whitespace, strip punctuation."""
    line = re.sub(r"\s+", " ", line)
    return line.strip(" ,.-;:!?")


def parse_poem(content: str) -> list[str]:
    """
    Split poem text into individual lines.
    CSV stores poems with '<\n>' as line separator (literal newline).
    """
    lines = content.split("<\n>")
    lines = [clean_line(l) for l in lines]
    return [l for l in lines if l and len(l.split()) >= 2]


def make_pairs(lines: list[str], genre: str) -> list[str]:
    """
    Convert poem lines into (prompt, reply) pairs.
    Strict syllable matching per genre rule.
    Pairs every adjacent line pair: 0→1, 2→3, 4→5...
    """
    rule = GENRE_RULES[genre]
    tag = rule["tag"]
    p_syl_target = rule["prompt_syl"]
    r_syl_target = rule["reply_syl"]

    pairs = []
    for i in range(0, len(lines) - 1, 2):
        if i + 1 >= len(lines):
            break
        prompt, reply = lines[i], lines[i + 1]

        # Strict syllable match
        p_ok = count_syllables(prompt) == p_syl_target
        r_ok = count_syllables(reply) == r_syl_target

        if p_ok and r_ok and prompt and reply:
            pairs.append(f"{START} {tag} {prompt} {REPLY} {reply} {END}")
    return pairs


def preprocess(csv_path=None, output_path=None, max_poems=None):
    """Main: read clean CSV → create pairs for all genres."""
    csv_path = Path(csv_path or CSV_PATH)
    output_path = Path(output_path or OUTPUT_PATH)

    df = pd.read_csv(csv_path)
    print(f"Loaded: {len(df):,} poems")
    print(f"  Lục bát: {df['genre'].eq('lục bát').sum():,}")
    print(f"  Bảy chữ: {df['genre'].eq('bảy chữ').sum():,}")

    if max_poems:
        df = df.head(max_poems)

    # Process each poem into training pairs
    all_pairs, skipped, empty = [], 0, 0
    for _, row in df.iterrows():
        genre = row["genre"]
        content = row["content"]

        if pd.isna(content) or not content.strip():
            empty += 1
            continue

        lines = parse_poem(content)
        if len(lines) < 2:
            skipped += 1
            continue

        all_pairs.extend(make_pairs(lines, genre))

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for pair in all_pairs:
            f.write(pair + "\n")

    print(f"  Empty: {empty} | Too short: {skipped} | Pairs: {len(all_pairs):,}")
    print(f"  Saved → {output_path}")
    return all_pairs


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Preprocess poetry CSV → training pairs")
    p.add_argument("--csv", type=str, default=None)
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--max", type=int, default=None, help="Limit poems for testing")
    args = p.parse_args()
    preprocess(args.csv, args.output, args.max)
