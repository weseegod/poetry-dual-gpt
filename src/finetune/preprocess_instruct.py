#!/usr/bin/env python3
"""
Convert Lục Bát corpus to instruction-tuning JSONL for Qwen2.5-Instruct.

Input:  corpus_luc_bat.txt (618K lines)
        <|start|> [LUC_BAT] [RHYME:X] [TONE:XXXXXX] [TRAMBONG:NH]
          luc_line <|reply|> bat_line <|end|>

Output: instruct_train.jsonl + instruct_val.jsonl
        {"messages": [
          {"role": "system", "content": "Bạn là nhà thơ Lục Bát..."},
          {"role": "user", "content": "luc_line"},
          {"role": "assistant", "content": "bat_line"}
        ]}

Usage:
  python src/finetune/preprocess_instruct.py
  python src/finetune/preprocess_instruct.py --corpus corpus_luc_bat.txt --output-dir data/
"""

import argparse
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.tones import get_tone


# ═══════════════════════════════════════════════════════════════
#  SYSTEM PROMPT TEMPLATES
# ═══════════════════════════════════════════════════════════════

# Multiple templates for robustness — model sees varied phrasing
SYSTEM_TEMPLATES = [
    # Template 1: Full constraint listing
    (
        "Bạn là nhà thơ Lục Bát chuyên nghiệp. "
        "Cho dòng Lục (6 chữ), hãy viết dòng Bát (8 chữ) đúng luật thơ Lục Bát:\n"
        "- Vần: chữ thứ 6 của dòng Bát phải vần với \"{rhyme}\"\n"
        "- Thanh điệu dòng Bát (vị trí 2-4-6-8): {tone_pattern}\n"
        "- Trầm-Bổng: {trambong_rule}\n"
        "Chỉ trả lời 8 chữ của dòng Bát, không thêm gì khác."
    ),
    # Template 2: Concise
    (
        "Viết dòng Bát (8 chữ) cho dòng Lục sau, tuân thủ:\n"
        "Vần \"{rhyme}\" ở chữ thứ 6; thanh {tone_pattern}; {trambong_rule}.\n"
        "Chỉ đưa ra 8 chữ, không giải thích."
    ),
    # Template 3: Poetic instruction
    (
        "Làm thơ Lục Bát: từ dòng Lục đã cho, sáng tác dòng Bát 8 chữ liền mạch.\n"
        "Quy tắc: chữ thứ 6 vần \"{rhyme}\", thanh điệu {tone_pattern}, "
        "{trambong_rule}.\n"
        "Trả lời: 8 chữ dòng Bát."
    ),
    # Template 4: Rule-first
    (
        "Luật thơ Lục Bát cho dòng Bát:\n"
        "1. Chữ thứ 6 vần với \"{rhyme}\"\n"
        "2. Thanh điệu: {tone_pattern} (B=Bằng, T=Trắc)\n"
        "3. {trambong_rule}\n"
        "Dòng Lục: {{user_input}}\n"
        "Dòng Bát (8 chữ):"
    ),
    # Template 5: Minimal
    (
        "Dòng Lục → dòng Bát (8 chữ). "
        "Vần \"{rhyme}\", thanh {tone_pattern}, {trambong_rule}."
    ),
]

# Trầm-Bổng descriptions
TRAMBONG_DESCRIPTIONS = {
    "NH": "chữ thứ 6 thanh Ngang, chữ thứ 8 thanh Huyền (Ngang≠Huyền)",
    "HN": "chữ thứ 6 thanh Huyền, chữ thứ 8 thanh Ngang (Huyền≠Ngang)",
    "XX": "chữ thứ 6 và 8 khác dấu (Ngang≠Huyền)",  # fallback
}


# ═══════════════════════════════════════════════════════════════
#  CORPUS PARSING
# ═══════════════════════════════════════════════════════════════

def parse_corpus_line(line: str) -> dict | None:
    """
    Parse one corpus line into structured fields.
    
    Format: <|start|> [LUC_BAT] [RHYME:X] [TONE:XXXXXX] [TRAMBONG:NH]
             luc_line <|reply|> bat_line <|end|>
    
    Returns dict with keys: luc, bat, rhyme, tone_luc, tone_bat, trambong
    Returns None if line is malformed.
    """
    line = line.strip()
    if not line:
        return None
    
    # Extract control tags
    rhyme_match = re.search(r'\[RHYME:(\w+)\]', line)
    tone_match = re.search(r'\[TONE:([BT]+)\]', line)
    trambong_match = re.search(r'\[TRAMBONG:(NH|HN|XX)\]', line)
    
    if not rhyme_match or not tone_match:
        return None
    
    rhyme = rhyme_match.group(1)
    tone_luc = tone_match.group(1)
    trambong = trambong_match.group(1) if trambong_match else "NH"
    
    # Split on <|reply|> and <|end|>
    parts = line.split('<|reply|>')
    if len(parts) != 2:
        return None
    
    # Extract Lục line: everything between last ']' and <|reply|>
    before_reply = parts[0]
    # Find the last control token ']'
    last_bracket = before_reply.rfind(']')
    if last_bracket < 0:
        return None
    luc_line = before_reply[last_bracket + 1:].strip()
    
    # Extract Bát line: everything between <|reply|> and <|end|>
    bat_line = parts[1].split('<|end|>')[0].strip()
    
    # Remove <|start|> if present
    luc_line = luc_line.replace('<|start|>', '').strip()
    
    # Validate syllable counts
    luc_syls = luc_line.split()
    bat_syls = bat_line.split()
    
    if len(luc_syls) != 6:
        return None
    if len(bat_syls) < 7 or len(bat_syls) > 10:
        # Allow minor variation (punctuation can inflate count)
        # But reject extreme outliers
        return None
    
    # Compute Bát line tone pattern (positions 2-4-6-8, 0-indexed: 1,3,5,7)
    bat_tones = []
    for idx in [1, 3, 5, 7]:
        if idx < len(bat_syls):
            bat_tones.append(get_tone(bat_syls[idx]))
        else:
            bat_tones.append('B')  # fallback
    tone_bat = '-'.join(bat_tones)  # "B-T-B-B"
    
    return {
        'luc': luc_line,
        'bat': bat_line,
        'rhyme': rhyme,
        'tone_luc': tone_luc,
        'tone_bat': tone_bat,
        'trambong': trambong,
    }


# ═══════════════════════════════════════════════════════════════
#  CHAT FORMAT GENERATION
# ═══════════════════════════════════════════════════════════════

def build_chat_example(parsed: dict, template_idx: int) -> dict:
    """
    Build a chat-format training example.
    
    Returns: {"messages": [{"role": "system", ...}, {"role": "user", ...}, 
                           {"role": "assistant", ...}]}
    """
    rhyme = parsed['rhyme']
    tone_bat = parsed['tone_bat']
    trambong = parsed['trambong']
    
    # Build Trầm-Bổng description
    trambong_desc = TRAMBONG_DESCRIPTIONS.get(trambong, TRAMBONG_DESCRIPTIONS["XX"])
    
    # Select template
    template = SYSTEM_TEMPLATES[template_idx % len(SYSTEM_TEMPLATES)]
    
    # Template 4 has a special format with {user_input} placeholder
    if template_idx % len(SYSTEM_TEMPLATES) == 3:  # Template 4
        system_content = template.format(
            rhyme=rhyme,
            tone_pattern=tone_bat,
            trambong_rule=trambong_desc,
        ).replace('{user_input}', parsed['luc'])
        
        return {
            "messages": [
                {"role": "system", "content": system_content},
                {"role": "assistant", "content": parsed['bat']},
            ]
        }
    
    # Standard templates
    system_content = template.format(
        rhyme=rhyme,
        tone_pattern=tone_bat,
        trambong_rule=trambong_desc,
    )
    
    return {
        "messages": [
            {"role": "system", "content": system_content},
            {"role": "user", "content": parsed['luc']},
            {"role": "assistant", "content": parsed['bat']},
        ]
    }


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

def preprocess(corpus_path: Path, output_dir: Path, 
               val_frac: float = 0.10, seed: int = 42):
    """Convert corpus to instruction-tuning JSONL files."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📖  Reading: {corpus_path}")
    print(f"📁  Output: {output_dir}")
    
    # Parse all lines
    examples = []
    skipped = 0
    total = 0
    
    with open(corpus_path, encoding='utf-8') as f:
        for line in f:
            total += 1
            parsed = parse_corpus_line(line)
            if parsed:
                examples.append(parsed)
            else:
                skipped += 1
    
    print(f"   Total lines: {total:,}")
    print(f"   Valid: {len(examples):,} ({len(examples)/max(total,1)*100:.1f}%)")
    print(f"   Skipped: {skipped:,}")
    
    if not examples:
        print("❌  No valid examples found!")
        return
    
    # Shuffle
    random.seed(seed)
    random.shuffle(examples)
    
    # Split
    split = int(len(examples) * val_frac)
    train_examples = examples[:-split] if split > 0 else examples
    val_examples = examples[-split:] if split > 0 else []
    
    # Generate chat format
    datasets = {
        'train': (train_examples, output_dir / 'instruct_train.jsonl'),
        'val': (val_examples, output_dir / 'instruct_val.jsonl'),
    }
    
    for name, (exs, path) in datasets.items():
        if not exs:
            print(f"   ⚠️  No {name} examples — skipping")
            continue
        
        with open(path, 'w', encoding='utf-8') as f:
            for i, ex in enumerate(exs):
                # Cycle through templates
                chat = build_chat_example(ex, template_idx=i)
                f.write(json.dumps(chat, ensure_ascii=False) + '\n')
        
        # Verify
        with open(path, encoding='utf-8') as f:
            count = sum(1 for _ in f)
        print(f"   ✅ {name}: {count:,} examples → {path.name}")
    
    # ── Print samples ──
    print(f"\n📝  Sample training examples:")
    train_path = output_dir / 'instruct_train.jsonl'
    with open(train_path, encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 3:
                break
            ex = json.loads(line)
            for msg in ex['messages']:
                role = msg['role']
                content = msg['content'][:100]
                print(f"   [{role}] {content}...")
            print()
    
    # ── Statistics ──
    print(f"📊  Statistics:")
    
    # System prompt length
    lens = []
    with open(train_path, encoding='utf-8') as f:
        for line in f:
            ex = json.loads(line)
            sys_msg = ex['messages'][0]['content']
            lens.append(len(sys_msg.split()))
    print(f"   System prompt: {min(lens)}-{max(lens)} words (avg {sum(lens)/len(lens):.0f})")
    
    # Rhyme group distribution
    rhyme_counts = {}
    for ex in train_examples:
        r = ex['rhyme']
        rhyme_counts[r] = rhyme_counts.get(r, 0) + 1
    top_rhymes = sorted(rhyme_counts.items(), key=lambda x: -x[1])[:10]
    print(f"   Top rhyme groups: {[(r, c) for r, c in top_rhymes]}")
    print(f"   Unique rhyme groups: {len(rhyme_counts)}")
    
    print(f"\n✅  Done! Ready for instruction fine-tuning.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert Lục Bát corpus to instruction-tuning JSONL")
    parser.add_argument("--corpus", type=str,
                        default="src/finetune/corpus/corpus_luc_bat.txt",
                        help="Path to corpus_luc_bat.txt")
    parser.add_argument("--output-dir", type=str, default="data",
                        help="Output directory for JSONL files")
    parser.add_argument("--val-frac", type=float, default=0.10,
                        help="Validation fraction")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    args = parser.parse_args()
    
    corpus_path = Path(args.corpus)
    if not corpus_path.is_absolute():
        corpus_path = ROOT / corpus_path
    
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    
    preprocess(corpus_path, output_dir, args.val_frac, args.seed)
