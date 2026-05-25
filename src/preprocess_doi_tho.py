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


def extract_couplets(lines: list[str], syl_pair: tuple[int, int] = (6, 8)) -> list[tuple[str, str]]:
    """
    Extract valid couplets from poem lines.
    
    Args:
        lines: cleaned poem lines
        syl_pair: target syllable pair — (6,8) for Lục Bát, (7,7) for Thất Ngôn
    
    Returns list of (line_a, line_b) tuples.
    """
    s1_target, s2_target = syl_pair
    couplets = []
    i = 0
    while i + 1 < len(lines):
        s1 = count_syllables(lines[i])
        s2 = count_syllables(lines[i + 1])
        if s1 == s1_target and s2 == s2_target:
            couplets.append((lines[i], lines[i + 1]))
            i += 2
        else:
            i += 1
    return couplets


def get_doi_tho_tags_lb(six_line: str, eight_line: str) -> tuple[str, str]:
    """
    Extract [RHYME:X] and [TONE:XXXXXX] for Lục Bát đối thơ.
    
    [RHYME:X] — from position 8 of the 8-syllable line (chain rhyme)
    [TONE:XXXXXX] — tone pattern of the 6-syllable line (6 chars)
    
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
        tone_tag = f"[TONE:{seq[:6]}]"  # 6-char tone sequence
    
    return rhyme_tag, tone_tag


def get_doi_tho_tags_tn(seven_line: str) -> tuple[str, str]:
    """
    Extract [RHYME:X] and [TONE:YYYYYYY] for Thất Ngôn đối thơ.
    
    [RHYME:X] — from position 7 (last syllable) of the input 7-syl line
    [TONE:YYYYYYY] — tone pattern of the 7-syl line (7 chars)
    
    Returns (rhyme_tag, tone_tag).
    """
    rhyme_tag = ""
    tone_tag = ""
    
    syls = seven_line.strip().split()
    if len(syls) >= 7:
        rhyme = get_rhyme_group(syls[6])  # pos 7 (0-indexed: 6)
        rhyme_tag = f"[RHYME:{rhyme}]"
        seq = get_tone_sequence(seven_line)
        tone_tag = f"[TONE:{seq[:7]}]"  # 7-char tone sequence
    
    return rhyme_tag, tone_tag


# Alias for backward compat
get_doi_tho_tags = get_doi_tho_tags_lb


def make_doi_tho_pairs_multi(couplets: list[tuple[str, str]], tag_fn, window: int = 1,
                             genre_token: str = "") -> list[str]:
    """
    Generate đối thơ training pairs using sliding windows.
    
    Args:
        couplets: list of (line_a, line_b) tuples
        tag_fn: function(line_a, line_b) → (rhyme_tag, tone_tag)
        window: how many input couplets (1 or 2)
        genre_token: explicit genre tag like [LUC_BAT] or [THAT_NGON]
    
    Window=1: couplet_k → couplet_{k+1}
    Window=2: couplet_k + couplet_{k+1} → couplet_{k+2}
    
    Returns list of formatted training lines.
    """
    pairs = []
    max_window = min(window, 2)
    
    for w in range(1, max_window + 1):
        for k in range(len(couplets) - w):
            input_couplets = couplets[k:k + w]
            output_couplet = couplets[k + w]
            
            # Extract tags from LAST couplet of input
            last_a, last_b = input_couplets[-1]
            rhyme_tag, tone_tag = tag_fn(last_a, last_b)
            
            # Build input lines
            input_lines = []
            for a, b in input_couplets:
                input_lines.append(a)
                input_lines.append(b)
            input_str = f" {LB} ".join(input_lines)
            
            # Build output lines
            out_a, out_b = output_couplet
            output_str = f"{out_a} {LB} {out_b}"
            
            # Combine: [GENRE] [RHYME:X] [TONE:XXXXXX]
            genre_part = f"{genre_token} " if genre_token else ""
            tags = f"{rhyme_tag} {tone_tag}".strip()
            tag_part = f"{genre_part}{tags}" if (genre_part or tags) else ""
            pairs.append(f"{START} {tag_part} {input_str} {REPLY} {output_str} {END}")
    
    return pairs


# Backward compat
def make_doi_tho_pairs(couplets: list[tuple[str, str]], window: int = 2) -> list[str]:
    """Legacy wrapper — Lục Bát only."""
    return make_doi_tho_pairs_multi(couplets, tag_fn=get_doi_tho_tags_lb, window=window)


GENRE_CONFIG = {
    "lục bát": {
        "syl_pair": (6, 8),
        "tag_fn": get_doi_tho_tags_lb,
        "label": "Lục Bát",
        "genre_token": "[LUC_BAT]",
    },
    "bảy chữ": {
        "syl_pair": (7, 7),
        "tag_fn": lambda a, b: get_doi_tho_tags_tn(a),  # b is unused for TN
        "label": "Thất Ngôn",
        "genre_token": "[THAT_NGON]",
    },
}


def preprocess(csv_path=None, output_path=None, max_poems=None, window=2):
    """Main: read clean CSV → create đối thơ training pairs for ALL genres."""
    csv_path = Path(csv_path or CSV_PATH)
    output_path = Path(output_path or OUTPUT_PATH)
    
    df = pd.read_csv(csv_path)
    print(f"Loaded: {len(df):,} poems")
    
    # Keep only supported genres
    df = df[df["genre"].isin(GENRE_CONFIG.keys())]
    print(f"  Lục bát:  {df['genre'].eq('lục bát').sum():,}")
    print(f"  Bảy chữ:  {df['genre'].eq('bảy chữ').sum():,}")
    
    if max_poems:
        df = df.head(max_poems)
    
    all_pairs = []
    skipped_empty = 0
    skipped_short = 0
    total_couplets = {"lục bát": 0, "bảy chữ": 0}
    genre_pairs = {"lục bát": 0, "bảy chữ": 0}
    
    for _, row in df.iterrows():
        content = row["content"]
        genre = row["genre"]
        cfg = GENRE_CONFIG[genre]
        
        if pd.isna(content) or not content.strip():
            skipped_empty += 1
            continue
        
        lines = parse_poem(content)
        couplets = extract_couplets(lines, syl_pair=cfg["syl_pair"])
        
        if len(couplets) < 2:
            skipped_short += 1
            continue
        
        total_couplets[genre] += len(couplets)
        
        # Generate sliding window pairs with genre-specific tags + genre token
        pairs = make_doi_tho_pairs_multi(couplets, tag_fn=cfg["tag_fn"], window=window,
                                         genre_token=cfg["genre_token"])
        all_pairs.extend(pairs)
        genre_pairs[genre] += len(pairs)
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for pair in all_pairs:
            f.write(pair + "\n")
    
    # Stats: count window sizes
    w1_count = sum(1 for p in all_pairs if p.count(LB) == 2)
    w2_count = sum(1 for p in all_pairs if p.count(LB) == 4)
    
    print(f"\n📊  Results:")
    print(f"  Poems processed: {len(df):,}")
    print(f"  Skipped (empty): {skipped_empty}")
    print(f"  Skipped (< 2 couplets): {skipped_short}")
    print(f"  Lục Bát:   {genre_pairs['lục bát']:,} pairs | {total_couplets['lục bát']:,} couplets")
    print(f"  Thất Ngôn: {genre_pairs['bảy chữ']:,} pairs | {total_couplets['bảy chữ']:,} couplets")
    print(f"  Total training pairs: {len(all_pairs):,}")
    print(f"    window=1: {w1_count:,}")
    print(f"    window=2: {w2_count:,}")
    print(f"  Saved → {output_path}")
    
    return all_pairs


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Preprocess poems → đối thơ training pairs")
    p.add_argument("--csv", type=str, default=None)
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--max", type=int, default=None, help="Limit poems for testing")
    p.add_argument("--window", type=int, default=1, help="Max context window (1 or 2)")
    args = p.parse_args()
    preprocess(args.csv, args.output, args.max, args.window)
