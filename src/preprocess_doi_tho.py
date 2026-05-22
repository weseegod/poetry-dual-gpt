"""
Generate couplet-to-couplet đối thơ training data from poems_dataset_clean.csv.

Strategy (sliding pair windows):
  For each poem, generate pairs:
    window=1: couplet_k → couplet_{k+1}
    window=2: couplet_k + couplet_{k+1} → couplet_{k+2}

Output format (coexists with single-couplet format in same corpus):
  <|start|> [DOI_THO] [RHYME:X] [TONE:XXXXXX]
    line6 <|linebreak|> line8 [<|linebreak|> line6_prev <|linebreak|> line8_prev] <|reply|>
    line6_out <|linebreak|> line8_out <|end|>

Tags always extracted from the LAST couplet of input:
  [RHYME:X] — from position 8 of the 8-syllable line (chain rhyme)
  [TONE:XXXXXX] — tone pattern of the 6-syllable line

Usage:
  python src/preprocess_doi_tho.py                   # generate both window sizes
  python src/preprocess_doi_tho.py --window 1        # window=1 only
  python src/preprocess_doi_tho.py --max 1000        # limit poems for testing
"""

import argparse
import re
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).parent.parent
CSV_PATH = ROOT / "data" / "poems_dataset_clean.csv"
OUTPUT_PATH = ROOT / "data" / "doi_tho_corpus.txt"

# Control tokens
START = "<|start|>"
REPLY = "<|reply|>"
END = "<|end|>"
LB = "<|linebreak|>"
DOI_THO = "[DOI_THO]"

# Import tones utilities from same package
import sys
sys.path.insert(0, str(ROOT / "src"))
from tones import get_tone, get_rhyme_group, get_tone_sequence


def count_syllables(line: str) -> int:
    return len(line.strip().split())


def clean_line(line: str) -> str:
    line = re.sub(r"\s+", " ", line)
    return line.strip(" ,.-;:!?")


def parse_poem(content: str) -> list[str]:
    """Split poem text into individual clean lines."""
    lines = content.split("<\n>")
    lines = [clean_line(l) for l in lines]
    return [l for l in lines if l and len(l.split()) >= 2]


def extract_couplets(lines: list[str]) -> list[tuple[str, str]]:
    """
    Extract valid Lục Bát couplets (6-syl + 8-syl pairs) from poem lines.
    Returns list of (line6, line8) tuples.
    """
    couplets = []
    i = 0
    while i + 1 < len(lines):
        s1 = count_syllables(lines[i])
        s2 = count_syllables(lines[i + 1])
        if s1 == 6 and s2 == 8:
            couplets.append((lines[i], lines[i + 1]))
            i += 2
        else:
            i += 1
    return couplets


def get_doi_tho_tags(six_line: str, eight_line: str) -> tuple[str, str]:
    """
    Extract [RHYME:X] and [TONE:XXXXXX] for đối thơ.
    
    [RHYME:X] — from position 8 of the 8-syllable line (chain rhyme)
    [TONE:XXXXXX] — tone pattern of the 6-syllable line
    
    Returns (rhyme_tag, tone_tag).
    """
    rhyme_tag = ""
    tone_tag = ""
    
    syls_8 = eight_line.strip().split()
    if len(syls_8) >= 8:
        rhyme = get_rhyme_group(syls_8[7])  # pos 8
        rhyme_tag = f"[RHYME:{rhyme}]"
    
    syls_6 = six_line.strip().split()
    if len(syls_6) >= 6:
        seq = get_tone_sequence(six_line)
        tone_tag = f"[TONE:{seq[:6]}]"
    
    return rhyme_tag, tone_tag


def make_doi_tho_pairs(couplets: list[tuple[str, str]], window: int = 2) -> list[str]:
    """
    Generate đối thơ training pairs using sliding windows.
    
    Window=1: couplet_k → couplet_{k+1}
    Window=2: couplet_k + couplet_{k+1} → couplet_{k+2}
    
    Returns list of formatted training lines.
    """
    pairs = []
    max_window = min(window, 2)  # default max window = 2
    
    for w in range(1, max_window + 1):
        for k in range(len(couplets) - w):
            # Input: couplets k ... k+w-1 (last w couplets)
            input_couplets = couplets[k:k + w]
            # Output: couplet k+w
            output_couplet = couplets[k + w]
            
            # Extract tags from LAST couplet of input
            last_6, last_8 = input_couplets[-1]
            rhyme_tag, tone_tag = get_doi_tho_tags(last_6, last_8)
            
            # Build input lines
            input_lines = []
            for six, eight in input_couplets:
                input_lines.append(six)
                input_lines.append(eight)
            input_str = f" {LB} ".join(input_lines)
            
            # Build output lines
            out_6, out_8 = output_couplet
            output_str = f"{out_6} {LB} {out_8}"
            
            # Combine
            tags = f"{rhyme_tag} {tone_tag}".strip()
            tag_part = f"{DOI_THO} {tags}" if tags else DOI_THO
            pairs.append(f"{START} {tag_part} {input_str} {REPLY} {output_str} {END}")
    
    return pairs


def preprocess(csv_path=None, output_path=None, max_poems=None, window=2):
    """Main: read clean CSV → create đối thơ training pairs."""
    csv_path = Path(csv_path or CSV_PATH)
    output_path = Path(output_path or OUTPUT_PATH)
    
    df = pd.read_csv(csv_path)
    print(f"Loaded: {len(df):,} poems")
    
    # Focus on Lục Bát for v2.0 (Thất Ngôn đối thơ is v3.0)
    df = df[df["genre"] == "lục bát"]
    print(f"  Lục bát: {len(df):,}")
    
    if max_poems:
        df = df.head(max_poems)
    
    all_pairs = []
    skipped_empty = 0
    skipped_short = 0
    total_couplets = 0
    
    for _, row in df.iterrows():
        content = row["content"]
        if pd.isna(content) or not content.strip():
            skipped_empty += 1
            continue
        
        lines = parse_poem(content)
        couplets = extract_couplets(lines)
        
        if len(couplets) < 2:
            skipped_short += 1
            continue
        
        total_couplets += len(couplets)
        
        # Generate sliding window pairs
        pairs = make_doi_tho_pairs(couplets, window=window)
        all_pairs.extend(pairs)
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for pair in all_pairs:
            f.write(pair + "\n")
    
    # Stats: count window sizes
    # w=1: 2 linebreaks (1 between input lines + 1 between output lines)
    # w=2: 4 linebreaks (3 between input lines + 1 between output lines)
    w1_count = sum(1 for p in all_pairs if p.count(LB) == 2)
    w2_count = sum(1 for p in all_pairs if p.count(LB) == 4)
    
    print(f"\n📊  Results:")
    print(f"  Poems processed: {len(df):,}")
    print(f"  Skipped (empty): {skipped_empty}")
    print(f"  Skipped (< 2 couplets): {skipped_short}")
    print(f"  Total couplets: {total_couplets:,}")
    print(f"  Training pairs: {len(all_pairs):,}")
    print(f"    window=1: {w1_count:,}")
    print(f"    window=2: {w2_count:,}")
    print(f"  Saved → {output_path}")
    
    return all_pairs


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Preprocess poems → đối thơ training pairs")
    p.add_argument("--csv", type=str, default=None)
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--max", type=int, default=None, help="Limit poems for testing")
    p.add_argument("--window", type=int, default=2, help="Max context window (1 or 2)")
    args = p.parse_args()
    preprocess(args.csv, args.output, args.max, args.window)
