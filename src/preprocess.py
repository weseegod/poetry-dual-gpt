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
from tones import get_luc_bat_tags, get_that_ngon_tags

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
            extras = ""
            if genre == "lục bát":
                rhyme, tone, trambong = get_luc_bat_tags(prompt + " " + reply)
                parts = [t for t in [rhyme, tone, trambong] if t]
                extras = " ".join(parts)
            elif genre == "bảy chữ":
                link2, doi_am = get_that_ngon_tags(prompt)
                extras_parts = [t for t in [link2, doi_am] if t]
                extras = " ".join(extras_parts)
            tag_part = f"{tag} {extras}".strip() if extras else tag
            pairs.append(f"{START} {tag_part} {prompt} {REPLY} {reply} {END}")
    return pairs


def make_pairs_song_that(lines: list[str]) -> list[str]:
    """
    Song thất lục bát pattern: 7-7-6-8 repeating.
    Extract 7→7 as [THAT_NGON], 6→8 as [LUC_BAT].
    """
    pairs = []
    i = 0
    while i + 3 < len(lines):
        # Lines i and i+1 should be 7-syllable each (thất ngôn couplet)
        l1, l2 = lines[i], lines[i + 1]
        if count_syllables(l1) == 7 and count_syllables(l2) == 7:
            link2, doi_am = get_that_ngon_tags(l1)
            extras_parts = [t for t in [link2, doi_am] if t]
            tag_part = f"[THAT_NGON] {' '.join(extras_parts)}".strip() if extras_parts else "[THAT_NGON]"
            pairs.append(f"{START} {tag_part} {l1} {REPLY} {l2} {END}")

        # Lines i+2 and i+3 should be 6→8 (lục bát couplet)
        l3, l4 = lines[i + 2], lines[i + 3]
        if count_syllables(l3) == 6 and count_syllables(l4) == 8:
            rhyme, tone, trambong = get_luc_bat_tags(l3 + " " + l4)
            parts = [t for t in [rhyme, tone, trambong] if t]
            extras = " ".join(parts)
            tag_part = f"[LUC_BAT] {extras}" if extras else "[LUC_BAT]"
            pairs.append(f"{START} {tag_part} {l3} {REPLY} {l4} {END}")

        i += 4  # jump to next stanza
    return pairs


def is_song_that(row) -> bool:
    """Check if poem is song thất lục bát (from specific_genre column)."""
    sg = str(row.get("specific_genre", ""))
    return "song thất" in sg.lower()


def preprocess(csv_path=None, output_path=None, max_poems=None, curriculum=False):
    """Main: read clean CSV → create pairs for all genres."""
    csv_path = Path(csv_path or CSV_PATH)
    output_path = Path(output_path or OUTPUT_PATH)

    df = pd.read_csv(csv_path)
    print(f"Loaded: {len(df):,} poems")
    print(f"  Lục bát: {df['genre'].eq('lục bát').sum():,}")
    print(f"  Bảy chữ: {df['genre'].eq('bảy chữ').sum():,}")
    st_count = df.apply(is_song_that, axis=1).sum()
    if st_count:
        print(f"    → song thất lục bát: {st_count} (split 7-7→[THAT_NGON] + 6-8→[LUC_BAT])")

    if max_poems:
        df = df.head(max_poems)

    # Process each poem into training pairs
    all_pairs, skipped, empty, st_pairs = [], 0, 0, 0
    for _, row in df.iterrows():
        content = row["content"]

        if pd.isna(content) or not content.strip():
            empty += 1
            continue

        lines = parse_poem(content)
        if len(lines) < 2:
            skipped += 1
            continue

        if is_song_that(row):
            all_pairs.extend(make_pairs_song_that(lines))
            st_pairs += 1
        else:
            genre = row["genre"]
            all_pairs.extend(make_pairs(lines, genre))

    # Curriculum: sort pairs by token count (short → long)
    if curriculum:
        all_pairs.sort(key=len)  # approximate token count via char length
        output_path = output_path.parent / "poetry_corpus_curriculum.txt"
        print(f"  Curriculum: sorted {len(all_pairs):,} pairs by length (short→long)")

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for pair in all_pairs:
            f.write(pair + "\n")

    # Count by genre tag
    lb = sum(1 for p in all_pairs if "[LUC_BAT]" in p)
    tn = sum(1 for p in all_pairs if "[THAT_NGON]" in p)
    print(f"  [LUC_BAT]: {lb:,} pairs  |  [THAT_NGON]: {tn:,}  |  Total: {len(all_pairs):,}")
    print(f"  Empty: {empty} | Too short: {skipped} | Song thất poems: {st_pairs}")
    print(f"  Saved → {output_path}")
    return all_pairs


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Preprocess poetry CSV → training pairs")
    p.add_argument("--csv", type=str, default=None)
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--max", type=int, default=None, help="Limit poems for testing")
    p.add_argument("--curriculum", action="store_true", help="Sort pairs by length (short→long) for curriculum learning")
    args = p.parse_args()
    preprocess(args.csv, args.output, args.max, args.curriculum)
