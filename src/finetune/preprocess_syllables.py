#!/usr/bin/env python3
"""
Extract all unique Vietnamese syllables from the poetry corpus,
filter artifacts, and save the clean syllable vocabulary.

Used by train.py to add single-token syllable representations to Qwen's tokenizer.
This prevents BPE fragmentation that causes garbled output like "trắngồngọc".

Usage:
  python src/finetune/preprocess_syllables.py
  python src/finetune/preprocess_syllables.py --corpus src/finetune/corpus/corpus_luc_bat.txt
"""

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent


# Valid Vietnamese syllable pattern:
# - Optional leading punctuation (only at start)
# - Consonant cluster (optional): ch, gh, kh, ng, ngh, nh, ph, th, tr, gi, qu
#   or single consonant: b, c, d, đ, g, h, k, l, m, n, p, r, s, t, v, x
# - Vowel cluster with diacritics
# - Ending consonant (optional): c, ch, m, n, ng, nh, p, t
# - Optional trailing punctuation
#
# Simplified: any sequence of Vietnamese letters with diacritics

VIETNAMESE_CHARS = set(
    "aăâbcdđeêghiklmnoôơpqrstuưvxy"  # base
    "AĂÂBCDĐEÊGHIKLMNOÔƠPQRSTUƯVXY"
    "àáảãạằắẳẵặầấẩẫậèéẻẽẹềếểễệìíỉĩịòóỏõọồốổỗộờớởỡợùúủũụừứửữựỳýỷỹỵ"  # lowercase toned
    "ÀÁẢÃẠẰẮẲẴẶẦẤẨẪẬÈÉẺẼẸỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌỒỐỔỖỘỜỚỞỠỢÙÚỦŨỤỪỨỬỮỰỲÝỶỸỴ"  # uppercase toned
)

# Common punctuation to strip from syllable boundaries
PUNCT = set(".,!?;:-\"'()[]/" + "…" + "—" + "“" + "”" + "‘" + "’" + "«" + "»")


def is_valid_syllable(s: str) -> bool:
    """Check if a token is a valid Vietnamese syllable (not an artifact)."""
    # Reject empty or single-char punctuation
    if not s or s in PUNCT:
        return False
    
    # Strip leading/trailing punctuation
    clean = s.strip(''.join(PUNCT))
    if not clean:
        return False
    
    # Reject purely numeric tokens
    if clean.isdigit():
        return False
    
    # Reject tokens with mixed alphanumeric like "3a", "a1"
    if re.search(r'\d', clean):
        return False
    
    # Must contain at least one Vietnamese letter
    if not any(c in VIETNAMESE_CHARS for c in clean):
        return False
    
    # Reject tokens that are mostly punctuation/non-Vietnamese
    vn_count = sum(1 for c in clean if c in VIETNAMESE_CHARS)
    total = len(clean)
    if vn_count / max(total, 1) < 0.5:
        return False
    
    # Reject single letters that aren't real words (BPE artifact fragments)
    # Exception: common Vietnamese single-letter words like "ở", "ý", "ả"
    if len(clean) == 1 and clean not in VIETNAMESE_CHARS:
        return False
    
    return True


def extract_syllables(corpus_path: Path, min_freq: int = 1) -> dict:
    """
    Extract all unique Vietnamese syllables from the corpus.
    Returns {syllable: count} dict sorted by frequency.
    """
    syllable_counts = {}
    total_syllables = 0
    
    print(f"📖  Reading: {corpus_path}")
    with open(corpus_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Extract content portion: between <|reply|> and <|end|>
            # Also extract content before <|reply|> (after stripping tags)
            parts = line.split('<|reply|>')
            
            for part in parts:
                # Strip special tokens
                clean = re.sub(r'<\|[^>]+\|>', ' ', part)
                clean = re.sub(r'\[[^\]]+\]', ' ', clean)
                
                for token in clean.split():
                    token = token.strip()
                    if token:
                        syllable_counts[token] = syllable_counts.get(token, 0) + 1
                        total_syllables += 1
    
    # Filter valid syllables
    valid = {s: c for s, c in syllable_counts.items() 
             if is_valid_syllable(s) and c >= min_freq}
    
    # Sort by frequency (descending)
    valid = dict(sorted(valid.items(), key=lambda x: -x[1]))
    
    print(f"   Total syllable tokens: {total_syllables:,}")
    print(f"   Unique raw tokens: {len(syllable_counts):,}")
    print(f"   Valid Vietnamese syllables: {len(valid):,}")
    print(f"   Filtered out: {len(syllable_counts) - len(valid):,} artifacts")
    
    # Show examples of filtered-out tokens
    filtered = {s: c for s, c in syllable_counts.items() if not is_valid_syllable(s)}
    if filtered:
        worst = sorted(filtered.items(), key=lambda x: -x[1])[:20]
        print(f"   Top filtered artifacts: {worst}")
    
    return valid


def save_syllable_vocab(corpus_path: Path, output_path: Path = None, min_freq: int = 1):
    """Extract and save syllable vocabulary."""
    if output_path is None:
        output_path = corpus_path.parent / "syllable_vocab.json"
    
    syllables = extract_syllables(corpus_path, min_freq=min_freq)
    
    # Save as JSON: list of [syllable, count] pairs
    vocab_list = [[s, c] for s, c in syllables.items()]
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(vocab_list, f, ensure_ascii=False, indent=2)
    
    print(f"💾  Saved: {output_path} ({len(vocab_list):,} syllables)")
    
    # Also save just the syllable strings (one per line) for easy reading
    txt_path = output_path.with_suffix('.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        for s in syllables:
            f.write(f"{s}\n")
    print(f"💾  Saved: {txt_path}")
    
    return syllables


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract Vietnamese syllable vocabulary")
    parser.add_argument("--corpus", type=str, 
                        default="src/finetune/corpus/corpus_luc_bat.txt",
                        help="Path to corpus file")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path")
    parser.add_argument("--min-freq", type=int, default=1,
                        help="Minimum frequency to include syllable")
    args = parser.parse_args()
    
    corpus_path = ROOT / args.corpus if not Path(args.corpus).is_absolute() else Path(args.corpus)
    output_path = Path(args.output) if args.output else None
    
    save_syllable_vocab(corpus_path, output_path, min_freq=args.min_freq)
