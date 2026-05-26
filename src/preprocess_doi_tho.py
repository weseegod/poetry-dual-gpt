"""
Generate couplet-to-couplet đối thơ training data from poems_dataset_clean.csv.

v4.1: Lục Bát only. Thất Ngôn moved to v5. Trầm-Bổng rule added.

Strategy (sliding pair windows):
  For each poem, generate pairs:
    window=1: couplet_k → couplet_{k+1}

Output format:
  <|start|> [LUC_BAT] [RHYME:X] [TONE:BBBBBB] [TRAMBONG:NH]
    line6 <|linebreak|> line8 <|reply|> line6_out <|linebreak|> line8_out <|end|>

Tags:
  [RHYME:X] — from position 8 of the input's 8-syllable line (chain rhyme)
  [TONE:XXXXXX] — tone pattern of the input's 6-syllable line
  [TRAMBONG:NH/HN] — from output's 8-syllable line (teacher forcing)

Usage:
  python src/preprocess_doi_tho.py --window 1
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
    Generate đối thơ training pairs: couplet_k → couplet_{k+1}.
    
    This is a poetry DUEL model — input is always a full couplet.
    
    Format: <|start|> [LUC_BAT] [RHYME:X] [TONE:BBBBBB] [TRAMBONG:NH]
              6-syl <|linebreak|> 8-syl <|reply|> 6-syl <|linebreak|> 8-syl <|end|>
    
    Args:
        couplets: list of (line_a, line_b) tuples
        tag_fn: function(line_a, line_b) → (rhyme_tag, tone_tag)
        window: how many input couplets (1 or 2)
        genre_token: explicit genre tag like [LUC_BAT]
    """
    from tones import get_tram_bong_tag as _get_tram_bong_tag
    
    pairs = []
    max_window = min(window, 2)
    
    for w in range(1, max_window + 1):
        for k in range(len(couplets) - w):
            input_couplets = couplets[k:k + w]
            output_couplet = couplets[k + w]
            
            last_a, last_b = input_couplets[-1]
            rhyme_tag, tone_tag = tag_fn(last_a, last_b)
            
            out_a, out_b = output_couplet
            trambong_tag = _get_tram_bong_tag(out_b)
            
            genre_part = f"{genre_token} " if genre_token else ""
            tags = f"{genre_part}{rhyme_tag} {tone_tag} {trambong_tag}".strip()
            
            # Build input (full couplet) and output
            input_lines = []
            for a, b in input_couplets:
                input_lines.append(a)
                input_lines.append(b)
            input_str = f" {LB} ".join(input_lines)
            output_str = f"{out_a} {LB} {out_b}"
            
            pairs.append(f"{START} {tags} {input_str} {REPLY} {output_str} {END}")
    
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
    # v4.1: Thất Ngôn removed → moved to v5
}


def preprocess(csv_path=None, output_path=None, max_poems=None, window=2):
    """v4.1: Read clean CSV → create Lục Bát-only đối thơ training pairs."""
    csv_path = Path(csv_path or CSV_PATH)
    output_path = Path(output_path or OUTPUT_PATH)
    
    df = pd.read_csv(csv_path)
    print(f"Loaded: {len(df):,} poems")
    
    # v4.1: Lục Bát only
    df = df[df["genre"] == "lục bát"]
    print(f"  Lục bát:  {len(df):,}")
    
    if max_poems:
        df = df.head(max_poems)
    
    all_pairs = []
    skipped_empty = 0
    skipped_short = 0
    total_couplets = 0
    
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
        
        total_couplets += len(couplets)
        
        # Generate sliding window pairs with genre + rhyme + tone + trầm-bổng tags
        pairs = make_doi_tho_pairs_multi(couplets, tag_fn=cfg["tag_fn"], window=window,
                                         genre_token=cfg["genre_token"])
        all_pairs.extend(pairs)
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for pair in all_pairs:
            f.write(pair + "\n")
    
    # Stats
    w1_count = sum(1 for p in all_pairs if p.count(LB) == 2)
    tb_nh = sum(1 for p in all_pairs if "[TRAMBONG:NH]" in p)
    tb_hn = sum(1 for p in all_pairs if "[TRAMBONG:HN]" in p)
    
    print(f"\n📊  Results:")
    print(f"  Poems processed: {len(df):,}")
    print(f"  Skipped (empty): {skipped_empty}")
    print(f"  Skipped (< 2 couplets): {skipped_short}")
    print(f"  Lục Bát couplets: {total_couplets:,}")
    print(f"  Training pairs (couplet→couplet): {len(all_pairs):,}")
    print(f"  Trầm-Bổng: NH={tb_nh:,} HN={tb_hn:,}")
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
