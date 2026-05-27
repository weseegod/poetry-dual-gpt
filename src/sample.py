"""
Autoregressive generation + Vietnamese Lục Bát rule checking.

Usage:
  python src/sample.py --checkpoint checkpoints/best.pt
  python src/sample.py --interactive
"""

import argparse
from pathlib import Path
import re
import torch
from tokenizers import Tokenizer
from model import PoetryDuelGPT
from tones import get_luc_bat_tags

# v4.2: canonical generation module
from generation import (
    build_prompt, generate, generate_best, decode_response,
    score_candidate, is_bpe_artifact, count_bpe_artifacts,
)

ROOT = Path(__file__).parent.parent


# ═══════════════════════════════════════════════════════════════
#  VIETNAMESE TONE CLASSIFICATION (Bằng vs Trắc)
# ═══════════════════════════════════════════════════════════════
# Bằng = level tones (ngang, huyền)   Trắc = sharp (sắc, nặng, hỏi, ngã)

BANG = set("aăâeêioôơuưyAĂÂEÊIOÔƠUƯY"          # ngang
           "àằầèềìòồờùừỳÀẰẦÈỀÌÒỒỜÙỪỲ")         # huyền

TRAC = set("áắấéếíóốớúứýÁẮẤÉẾÍÓỐỚÚỨÝ"          # sắc
            "ạặậẹệịọộợụựỵẠẶẬẸỆỊỌỘỢỤỰỴ"          # nặng
            "ảẳẩẻểỉỏổởủửỷẢẲẨẺỂỈỎỔỞỦỬỶ"          # hỏi
            "ãẵẫẽễĩõỗỡũữỹÃẴẪẼỄĨÕỖỠŨỮỸ")          # ngã


def get_tone(syl):
    """Scan syllable for tone-marked vowels. Default: bằng."""
    for ch in syl:
        if ch in TRAC: return "trắc"
        if ch in BANG: return "bằng"
    return "bằng"


def count_syllables(text):
    return len(text.strip().split())


# ═══════════════════════════════════════════════════════════════
#  LỤC BÁT RULE CHECKS (v4.1: 5 rules from luc_bat.md)
# ═══════════════════════════════════════════════════════════════
# Rule 1 (Vần lưng):  tiếng 6 câu Lục vần với tiếng 6 câu Bát
# Rule 2 (Bằng-Trắc): BTB (Lục) + BTBB (Bát)
# Rule 3 (Syllable):  6+8 exact
# Rule 4 (Trầm-Bổng): tiếng 6 & 8 dòng Bát khác dấu (Ngang≠Huyền)
# Rule 5 (Nhịp điệu): 2/2/2 (Lục) + 2/2/2/2 or 4/4 (Bát)

def check_syllables(prompt, response):
    p, r = count_syllables(prompt), count_syllables(response)
    ok = (5 <= p <= 7) and (7 <= r <= 9)
    return ok, f"prompt={p} response={r} (expected 6→8)"


def check_tones(line, pattern):
    """Check tone at positions 2,4,6(,8) against B/T pattern."""
    syls = line.strip().split()
    results = []
    for i, want in enumerate(pattern):
        pos = i * 2 + 2  # positions: 2, 4, 6, 8
        if pos > len(syls): break
        got = get_tone(syls[pos - 1])
        mark = "✓" if got[0].upper() == want else "✗"
        results.append(f"pos{pos}={got[0].upper()}({want}){mark}")
    all_ok = all("✓" in r for r in results)
    return all_ok, " | ".join(results)


def check_tram_bong(eight_line):
    """
    R4: Trầm-Bổng — tiếng 6 & tiếng 8 của dòng Bát must have opposite dấu.
    Ngang ≠ Huyền (both are Bằng, but different register).
    """
    syls = eight_line.strip().split()
    if len(syls) < 8:
        return False, "too_short"
    # Use tones module's get_diacritic
    from tones import get_diacritic
    d6 = get_diacritic(syls[5])
    d8 = get_diacritic(syls[7])
    ok = d6 in ("ngang", "huyen") and d8 in ("ngang", "huyen") and d6 != d8
    detail = f"pos6={d6} pos8={d8} → {'PASS' if ok else 'FAIL (must differ)'}"
    return ok, detail


def check_rhythm(line, is_luc=True):
    """
    R5: Nhịp điệu — approximate check for chẵn rhythm.
    Lục: 6 syllables → can split 2/2/2
    Bát: 8 syllables → can split 2/2/2/2 or 4/4
    """
    n = len(line.strip().split())
    if is_luc:
        ok = n == 6
        return ok, f"{n}syl → {'2/2/2 ✓' if ok else 'not 6 ✗'}"
    else:
        ok = n == 8
        return ok, f"{n}syl → {'2/2/2/2 ✓' if ok else 'not 8 ✗'}"


def evaluate(prompt, response):
    return {
        "syl": check_syllables(prompt, response),
        "tone_p": check_tones(prompt, "BTB"),
        "tone_r": check_tones(response, "BTBB"),
        "tram_bong": check_tram_bong(response),
        "rhythm_p": check_rhythm(prompt, is_luc=True),
        "rhythm_r": check_rhythm(response, is_luc=False),
    }


# ═══════════════════════════════════════════════════════════════
#  MODEL LOADING
# ═══════════════════════════════════════════════════════════════

def load_model(ckpt_path, device="cpu"):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    m = PoetryDuelGPT(
        vocab_size=ckpt["vocab_size"],
        n_embd=ckpt["model_config"]["n_embd"],
        n_head=ckpt["model_config"]["n_head"],
        n_layer=ckpt["model_config"]["n_layer"],
        block_size=ckpt["model_config"]["block_size"],
        dropout=ckpt["model_config"].get("dropout", 0.1),
    )
    m.load_state_dict(ckpt["model_state_dict"])
    m.to(device).eval()
    print(f"Loaded checkpoint (step {ckpt['step']})")
    return m


# ═══════════════════════════════════════════════════════════════
#  GENERATION — delegates to src/generation.py (v4.2 unified)
# ═══════════════════════════════════════════════════════════════

# All generation, prompt building, and decoding functions are now imported
# from src.generation at the top of this file.


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/final.pt")
    p.add_argument("--tokenizer", default="tokenizer/poetry_bpe.model")
    p.add_argument("--prompt", default="Thân em như chẽn lúa đòng")
    p.add_argument("--temperature", type=float, default=0.75)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--top_p", type=float, default=None)
    p.add_argument("--max_tokens", type=int, default=64)
    p.add_argument("--num_samples", type=int, default=3)
    p.add_argument("--interactive", action="store_true")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    dev = args.device if torch.cuda.is_available() else "cpu"
    print(f"Device: {dev}")

    # Load
    tok = Tokenizer.from_file(str(ROOT / args.tokenizer))
    print(f"Tokenizer: vocab={tok.get_vocab_size():,}")

    ckpt = ROOT / args.checkpoint
    if not ckpt.exists():
        print(f"No checkpoint at {ckpt}. Train first: python src/train.py")
        exit(1)
    model = load_model(str(ckpt), dev)

    # Interactive mode
    if args.interactive:
        print("\n🎭  Interactive Poetry Duel  [v4.2 — Lục Bát + Soft Rhyme]")
        print("    Single line → [LUC_BAT] single couplet")
        print("    Multi-line (use '|' between lines) → couplet duel")
        print("    Example: kiều nhi phận mỏng như tờ | một lời đã lỗi tóc tơ với chàng!")
        print("    'quit' to exit.\n")
        while True:
            u = input("You: ").strip()
            if u.lower() == "quit": break
            if not u: continue
            
            # Support pipe as line separator for multi-line input
            if "|" in u:
                u = u.replace("|", "\n")
            
            # v4.2: single build_prompt call handles all formats
            prompt = build_prompt(u)
            ids, text = generate(model, tok, prompt, max_new=args.max_tokens,
                                 temperature=args.temperature, top_k=args.top_k,
                                 top_p=args.top_p, device=dev,
                                 rhyme_mode="soft")
            out_lines = decode_response(tok, ids)
            if len(out_lines) >= 2:
                print(f"Bot: {out_lines[0]}")
                print(f"     {out_lines[1]}\n")
            elif len(out_lines) == 1:
                print(f"Bot: {out_lines[0]}\n")
            else:
                print(f"Bot: {text.strip()}\n")

    # Batch generation
    else:
        # Support pipe as line separator in --prompt
        raw_prompt = args.prompt.replace("|", "\n")
        prompt = build_prompt(raw_prompt)
        is_doi_tho = "\n" in raw_prompt

        for i in range(args.num_samples):
            print(f"\n{'='*60}\nSample {i+1}/{args.num_samples}\n{'='*60}")
            print(f"Prompt:   {prompt}")

            # v4.2: use unified generate with soft rhyme
            ids, text = generate(model, tok, prompt, max_new=args.max_tokens,
                                 temperature=args.temperature, top_k=args.top_k,
                                 top_p=args.top_p, device=dev,
                                 rhyme_mode="soft")
            
            # Display with proper linebreak handling
            out_lines = decode_response(tok, ids)
            if len(out_lines) >= 2:
                print(f"Response: {out_lines[0]}")
                print(f"          {out_lines[1]}")
            elif len(out_lines) == 1:
                print(f"Response: {out_lines[0]}")
            else:
                print(f"Response: {text.strip()}")
            response_only = "\n".join(out_lines) if out_lines else text.strip()

            # v4.2: add semantic quality score
            quality = score_candidate(response_only)
            bpe = count_bpe_artifacts(response_only.split())
            print(f"\n  Quality: {quality:+.2f}  |  BPE artifacts: {bpe}  |  Lexical div: {len(set(response_only.split()))/max(len(response_only.split()),1):.2f}")

            # Rule check — strip control tokens properly with regex
            prompt_part = re.sub(r'\[(?:RHYME|TONE|LINK2|TRAMBONG):[^\]]+\]', '', prompt)
            prompt_part = prompt_part.replace('[LUC_BAT]', '').replace('[DOI_THO]', '').replace('[THAT_NGON]', '')
            prompt_part = prompt_part.replace('<|start|>', '').replace('<|reply|>', '')
            prompt_part = ' '.join(prompt_part.split()).strip().rstrip(',')
            resp_clean = response_only.rstrip(",. ")
            is_luc_bat = "[LUC_BAT]" in prompt

            if is_luc_bat:
                r = evaluate(prompt_part, resp_clean)
                print(f"\n{'─'*60}\n📏  Lục Bát (6→8) — 5-Rule Check (v4.2)\n{'─'*60}")
                print(f"  R1 Vần lưng:   (prompt pos6 → response pos6 rhyme match)")
                print(f"  R2 Bằng-Trắc:  BTB / BTBB tone pattern")
                print(f"  R3 Syllable:   6+8 exact")
                print(f"  R4 Trầm-Bổng:  tiếng 6 & 8 dòng Bát khác dấu Ngang≠Huyền")
                print(f"  R5 Nhịp điệu:  2/2/2 (Lục) + 2/2/2/2 (Bát)")
                print(f"{'─'*60}")
                print(f"  R3 Syllable:    {r['syl'][1]} → {'✅' if r['syl'][0] else '❌'}")
                print(f"  R2 Prompt tone:  {r['tone_p'][1]}")
                print(f"  R2 Resp tone:    {r['tone_r'][1]}")
                print(f"  R4 Trầm-Bổng:    {r['tram_bong'][1]}")
                print(f"  R5 Rhythm (P):   {r['rhythm_p'][1]}")
                print(f"  R5 Rhythm (R):   {r['rhythm_r'][1]}")
                print(f"{'─'*60}")
            elif is_doi_tho:
                if len(out_lines) >= 2:
                    r6 = out_lines[0].strip()
                    r8 = out_lines[1].strip() if len(out_lines) > 1 else ""
                    print(f"\n{'─'*60}\n📏  Đối Thơ (couplet→couplet) Rule Check\n{'─'*60}")
                    print(f"  Output: {len(r6.split())}syl → {len(r8.split()) if r8 else '?'}syl")
