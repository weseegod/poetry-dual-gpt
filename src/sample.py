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
import torch.nn.functional as F
from tokenizers import Tokenizer
from model import PoetryDuelGPT
from tones import get_luc_bat_tags, get_that_ngon_tags, get_doi_tho_tags, get_doi_tho_tags_tn, get_rhyme_group

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
#  GENERATION LOOP
# ═══════════════════════════════════════════════════════════════

def auto_tag(prompt):
    """
    v4.1: Auto-wrap Lục Bát with control tokens + <|reply|> for single-line prompts.
    Uses training format so model knows where output starts.
    """
    p = prompt.strip()
    # Already tagged with genre
    if p.startswith("[LUC_BAT]"):
        inner = p.replace("[LUC_BAT]", "").strip()
        rhyme, tone, trambong = get_luc_bat_tags(inner)
        tags = ' '.join(t for t in [rhyme, tone, trambong] if t)
        if tags:
            p = f"[LUC_BAT] {tags} {inner}"
        # Ensure <|reply|> is present for single-line prompts
        if "<|reply|>" not in p:
            p = f"{p} <|reply|>"
        return p

    # Default: Lục Bát single-line → add <|reply|>
    rhyme, tone, trambong = get_luc_bat_tags(p)
    tags = ' '.join(t for t in [rhyme, tone, trambong] if t)
    return f"[LUC_BAT] {tags} {p} <|reply|>" if tags else f"[LUC_BAT] {p} <|reply|>"


def auto_tag_doi_tho(user_input: str, max_context_couplets: int = 1) -> str:
    """
    v4.1: Detect multi-line input → wrap as Lục Bát đối thơ format.
    Adds [RHYME:X] [TONE:BBBBBB] [TRAMBONG:NH] tags.
    Thất Ngôn support removed (moved to v5).
    
    User input can be:  
      - "line6\\nline8" (1 couplet)  
      - "line6\\nline8\\nline6\\nline8" (2 couplets)  
    
    Returns formatted prompt with <|linebreak|> separators and <|reply|>.
    """
    user_input = user_input.lower()
    lines = [l.strip() for l in user_input.strip().split('\n') if l.strip()]
    
    # Single line -> delegate to auto_tag
    if len(lines) == 1:
        return auto_tag(lines[0])
    
    # Group into couplets (Lục Bát: 6+8)
    couplets = []
    i = 0
    while i + 1 < len(lines):
        s1 = len(lines[i].split())
        s2 = len(lines[i+1].split())
        if s1 == 6 and s2 == 8:
            couplets.append((lines[i], lines[i+1]))
            i += 2
        else:
            i += 1
    
    if not couplets:
        return auto_tag(lines[-1])
    
    couplets = couplets[-max_context_couplets:]
    
    # Extract tags from last input couplet
    last_a, last_b = couplets[-1]
    rhyme_tag, tone_tag = get_doi_tho_tags(last_a, last_b)
    # For inference, use NH (most common Trầm-Bổng pattern) as default
    trambong_tag = "[TRAMBONG:NH]"
    
    # Build input lines
    input_lines = []
    for a, b in couplets:
        input_lines.append(a)
        input_lines.append(b)
    input_str = " <|linebreak|> ".join(input_lines)
    
    # Build tag: [LUC_BAT] [RHYME:X] [TONE:BBBBBB] [TRAMBONG:NH]
    tags = f"{rhyme_tag} {tone_tag} {trambong_tag}".strip()
    
    return f"<|start|> [LUC_BAT] {tags} {input_str} <|reply|>"


def decode_doi_tho(tokenizer, new_token_ids, enforce_syllables=False, max_lines=2, is_tn=False):
    """
    Decode generated tokens, handling <|linebreak|> which decodes to empty
    string in ByteLevel BPE. Splits on linebreak token positions.
    Returns list of line strings.
    
    v4.1: enforce_syllables defaults to False (metrics should reflect raw quality).
    T2a TN re-split removed (TN moved to v5).
    """
    lb_id = tokenizer.token_to_id("<|linebreak|>")
    lines = []
    chunk = []
    for t in new_token_ids:
        if t == lb_id:
            if chunk:
                lines.append(tokenizer.decode(chunk).strip())
            chunk = []
        else:
            chunk.append(t)
    if chunk:
        lines.append(tokenizer.decode(chunk).strip())
    
    # Detect genre from first line syllable count
    if lines:
        s1 = len(lines[0].split())
        targets = (6, 8) if s1 <= 7 else (7, 7)
    else:
        targets = (6, 8)
    
    # P3: Optional syllable enforcement (off by default — metrics should show raw quality)
    if enforce_syllables:
        for i, line in enumerate(lines):
            words = line.split()
            target = targets[i % 2]
            if len(words) > target:
                words = words[:target]
            lines[i] = ' '.join(words)
    
    # P2: Fix overgeneration — if too many lines, find first valid couplet
    if len(lines) > max_lines:
        t1, t2 = targets
        for i in range(len(lines) - 1):
            s1 = len(lines[i].split())
            s2 = len(lines[i+1].split())
            if s1 == t1 and s2 == t2:
                lines = lines[i:i+2]
                break
        else:
            lines = lines[:max_lines]
    
    return lines


@torch.no_grad()
def generate(model, tokenizer, prompt, max_new=64, temperature=0.75,
              top_k=50, top_p=None, device="cpu", max_syllables=None,
              rhyme_constraint=True):
    """
    1. Encode prompt → 2. Loop: forward → sample next token → append
    3. Stop on <|end|> or max tokens → 4. Decode new tokens only
    
    P1 (rhyme_constraint): When generating the rhyme-position syllable
    in the second output line, mask out tokens whose rhyme group doesn't
    match the target [RHYME:X] from the prompt.
    """
    end_id = tokenizer.token_to_id("<|end|>")
    pad_id = tokenizer.token_to_id("<|pad|>")
    lb_id = tokenizer.token_to_id("<|linebreak|>")
    is_tn = '[THAT_NGON]' in prompt

    # Parse target rhyme from prompt
    target_rhyme = None
    rhyme_syl_idx = None
    rhyme_match = re.search(r'\[RHYME:([^\]]+)\]', prompt)
    if rhyme_match:
        target_rhyme = rhyme_match.group(1)
        if is_tn:
            rhyme_syl_idx = 6  # Thất Ngôn: 7th syllable of 2nd 7-syl line
        else:
            rhyme_syl_idx = 5  # Lục Bát: 6th syllable of 2nd 8-syl line

    # Encode prompt as token IDs
    ids = tokenizer.encode(prompt).ids
    idx = torch.tensor([ids], dtype=torch.long, device=device)

    new_tokens = []
    for _ in range(max_new):
        # Forward pass on cropped context
        logits, _ = model(idx[:, -model.block_size:])
        logits = logits[:, -1, :] / temperature

        # Suppress <|pad|> — never sample it
        logits[:, pad_id] = float("-inf")

        # P2: Repetition penalty — penalize tokens from recent output
        for prev in new_tokens[-16:]:
            logits[:, prev] -= 1.2

        # P1: Rhyme constraint — force rhyme at target position in 2nd line
        if rhyme_constraint and target_rhyme is not None and rhyme_syl_idx is not None:
            # Count syllables generated in the current (2nd) line after last <|linebreak|>
            if lb_id in new_tokens:
                last_lb = max(i for i, t in enumerate(new_tokens) if t == lb_id)
                after_lb = new_tokens[last_lb + 1:]
                decoded_after = tokenizer.decode(after_lb)
                current_syl_count = len(decoded_after.strip().split()) if decoded_after.strip() else 0
            else:
                current_syl_count = 0

            # If we're about to generate the rhyme-position syllable, enforce rhyme
            if current_syl_count == rhyme_syl_idx:
                # Examine candidate tokens — mask those with wrong rhyme group
                # Use top-k*2 candidates to have enough to check
                candidate_k = min(top_k * 2 if top_k else 100, logits.size(-1))
                _, topk_idx = torch.topk(logits, candidate_k)
                matching = []
                non_matching = []
                for tid in topk_idx[0]:
                    tid_i = tid.item()
                    # Skip special tokens
                    if tid_i in (end_id, pad_id, lb_id):
                        continue
                    decoded = tokenizer.decode([tid_i]).strip()
                    if not decoded:
                        continue
                    # Get rhyme group of the candidate token
                    rg = get_rhyme_group(decoded)
                    if rg == target_rhyme:
                        matching.append(tid_i)
                    else:
                        non_matching.append(tid_i)
                # Only mask if at least one matching candidate exists
                # (avoids all-masked edge case leading to random selection)
                if matching:
                    for tid_i in non_matching:
                        logits[:, tid_i] = float("-inf")

        # Top-k filtering
        if top_k:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, -1:]] = float("-inf")

        # Top-p (nucleus) filtering — apply after top-k
        if top_p is not None:
            probs = F.softmax(logits, dim=-1)
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cumsum = torch.cumsum(sorted_probs, dim=-1)
            # Keep first token past the threshold
            mask = cumsum > top_p
            mask[..., 1:] = mask[..., :-1].clone()
            mask[..., 0] = False
            logits[:, sorted_idx[mask]] = float("-inf")

        # Sample from softmax
        next_id = torch.multinomial(F.softmax(logits, dim=-1), 1).item()

        if next_id == end_id: break        # stop token

        new_tokens.append(next_id)
        idx = torch.cat((idx, torch.tensor([[next_id]], device=device)), dim=1)

    # Truncate to target syllable count if specified (R3 fix)
    if max_syllables:
        decoded = tokenizer.decode(ids + new_tokens)
        words = decoded.split()
        if len(words) > max_syllables:
            decoded = ' '.join(words[:max_syllables])
        # Re-encode truncated text back to token ids for return
        return decoded, new_tokens[:max_syllables] if len(new_tokens) > max_syllables else new_tokens
    
    return tokenizer.decode(ids + new_tokens), new_tokens


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
        print("\n🎭  Interactive Poetry Duel  [v4.1 — Lục Bát]")
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
            
            if "\n" in u:
                # Multi-line → đối thơ
                prompt = auto_tag_doi_tho(u)
                _, ids = generate(model, tok, prompt, args.max_tokens, args.temperature, args.top_k, args.top_p, dev)
                out_lines = decode_doi_tho(tok, ids)
                print(f"Bot: {out_lines[0] if len(out_lines)>0 else '?'}")
                print(f"     {out_lines[1] if len(out_lines)>1 else '?'}\n")
            elif not u.startswith("["):
                prompt = auto_tag(u)
                _, ids = generate(model, tok, prompt, args.max_tokens, args.temperature, args.top_k, args.top_p, dev)
                response = tok.decode(ids).replace("<|end|>", "").strip()
                print(f"Bot: {response}\n")
            else:
                prompt = u
                _, ids = generate(model, tok, prompt, args.max_tokens, args.temperature, args.top_k, args.top_p, dev)
                response = tok.decode(ids).replace("<|end|>", "").strip()
                if "<|linebreak|>" in response:
                    out_lines = response.replace("<|linebreak|>", "\n    ")
                    print(f"Bot: {out_lines}\n")
                else:
                    print(f"Bot: {response}\n")

    # Batch generation
    else:
        # Support pipe as line separator in --prompt
        raw_prompt = args.prompt.replace("|", "\n")
        is_doi_tho = "\n" in raw_prompt
        
        if is_doi_tho:
            prompt = auto_tag_doi_tho(raw_prompt)
        else:
            prompt = raw_prompt
            if not prompt.startswith('['):
                prompt = auto_tag(prompt)
            elif '[LUC_BAT]' in prompt and '[RHYME:' not in prompt:
                inner = prompt.replace('[LUC_BAT]', '').strip()
                prompt = auto_tag(inner)

        for i in range(args.num_samples):
            print(f"\n{'='*60}\nSample {i+1}/{args.num_samples}\n{'='*60}")
            print(f"Prompt:   {prompt}")

            _, ids = generate(model, tok, prompt, args.max_tokens,
                              args.temperature, args.top_k, args.top_p, dev)
            response_only = tok.decode(ids).replace("<|end|>", "").strip()
            
            # Display: use decode_doi_tho for proper linebreak handling
            if is_doi_tho:
                out_lines = decode_doi_tho(tok, ids)
                print(f"Response: {out_lines[0] if len(out_lines)>0 else '?'}")
                print(f"          {out_lines[1] if len(out_lines)>1 else '?'}")
                response_only = "\n".join(out_lines)

            # Rule check — strip control tokens properly with regex
            prompt_part = re.sub(r'\[(?:RHYME|TONE|LINK2|TRAMBONG):[^\]]+\]', '', prompt)
            prompt_part = prompt_part.replace('[LUC_BAT]', '').replace('[DOI_THO]', '').replace('[THAT_NGON]', '')
            prompt_part = ' '.join(prompt_part.split()).strip().rstrip(',')
            resp_clean = response_only.rstrip(",. ")
            is_luc_bat = "[LUC_BAT]" in prompt

            if is_luc_bat:
                r = evaluate(prompt_part, resp_clean)
                print(f"\n{'─'*60}\n📏  Lục Bát (6→8) — 5-Rule Check (v4.1)\n{'─'*60}")
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
