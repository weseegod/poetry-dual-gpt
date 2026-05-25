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
#  LỤC BÁT RULE CHECKS
# ═══════════════════════════════════════════════════════════════
# Line 1 (6 syl): pos 2,4,6 = B-T-B
# Line 2 (8 syl): pos 2,4,6,8 = B-T-B-B

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


def evaluate(prompt, response):
    return {
        "syl": check_syllables(prompt, response),
        "tone_p": check_tones(prompt, "BTB"),
        "tone_r": check_tones(response, "BTBB"),
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
    """Auto-wrap genre tag + rhyme/tone based on syllable count."""
    p = prompt.strip()
    # Already tagged with genre
    if any(p.startswith(t) for t in ["[LUC_BAT]", "[THAT_NGON]"]):
        # Inject rhyme/tone if missing
        if "[LUC_BAT]" in p and "[RHYME:" not in p:
            line = p.replace("[LUC_BAT]", "").strip()
            rhyme, tone = get_luc_bat_tags(line)
            extras = f"{rhyme} {tone}".strip()
            if extras:
                p = p.replace("[LUC_BAT]", f"[LUC_BAT] {extras}")
        if "[THAT_NGON]" in p and "[DOIAM:" not in p:
            line = p.replace("[THAT_NGON]", "").strip()
            link2, doi_am = get_that_ngon_tags(line)
            extras_parts = [t for t in [link2, doi_am] if t]
            if extras_parts:
                p = p.replace("[THAT_NGON]", f"[THAT_NGON] {' '.join(extras_parts)}")
        return p

    syl = len(p.split())
    if syl == 7:
        link2, doi_am = get_that_ngon_tags(p)
        extras_parts = [t for t in [link2, doi_am] if t]
        tag = f"[THAT_NGON] {' '.join(extras_parts)}" if extras_parts else "[THAT_NGON]"
        return f"{tag} {p}"

    # Default: Lục Bát
    rhyme, tone = get_luc_bat_tags(p)
    extras = f"{rhyme} {tone}".strip()
    tag = f"[LUC_BAT] {extras}" if extras else "[LUC_BAT]"
    return f"{tag} {p}"


def auto_tag_doi_tho(user_input: str, max_context_couplets: int = 1) -> str:
    """
    Detect multi-line input → wrap as [DOI_THO] đối thơ format.
    
    User input can be:  
      - "line6\\nline8" (1 couplet)  
      - "line6\\nline8\\nline6\\nline8" (2 couplets)  
    
    Returns formatted prompt with <|linebreak|> separators and <|reply|>.
    """
    user_input = user_input.lower()
    lines = [l.strip() for l in user_input.strip().split('\n') if l.strip()]
    
    # Single line -> delegate to existing auto_tag
    if len(lines) == 1:
        line = lines[0]
        syls = line.split()
        if len(syls) == 7:
            rhyme_tag, tone_tag = get_doi_tho_tags_tn(line)
            genre_token = "[THAT_NGON]"
        else:
            rhyme_tag, tone_tag = get_doi_tho_tags(line, line)
            genre_token = "[LUC_BAT]"
        tags = f"{rhyme_tag} {tone_tag}".strip()
        tag_part = f"{genre_token}"
        if tags:
            tag_part += f" {tags}"
        return f"<|start|> {tag_part} {line} <|reply|>"
    
    # Group into couplets: support both Lục Bát (6+8) and Thất Ngôn (7+7)
    couplets = []
    i = 0
    while i + 1 < len(lines):
        s1 = len(lines[i].split())
        s2 = len(lines[i+1].split())
        if (s1 == 6 and s2 == 8) or (s1 == 7 and s2 == 7):
            couplets.append((lines[i], lines[i+1]))
            i += 2
        else:
            i += 1
    
    if not couplets:
        # No valid couplets found → fall back to last line as single prompt
        return auto_tag(lines[-1])
    
    # Keep at most max_context_couplets recent couplets
    couplets = couplets[-max_context_couplets:]
    
    # Detect genre from last couplet for tag extraction
    last_a, last_b = couplets[-1]
    s1_last = len(last_a.split())
    if s1_last == 7:
        # Thất Ngôn: rhyme from last syllable of 7-syl line
        rhyme_tag, tone_tag = get_doi_tho_tags_tn(last_a)
        genre_token = "[THAT_NGON]"
    else:
        # Lục Bát: rhyme from pos 8 of 8-syl line
        rhyme_tag, tone_tag = get_doi_tho_tags(last_a, last_b)
        genre_token = "[LUC_BAT]"
    
    # Build input lines with <|linebreak|> separators
    input_lines = []
    for a, b in couplets:
        input_lines.append(a)
        input_lines.append(b)
    input_str = " <|linebreak|> ".join(input_lines)
    
    # Build tag: [LUC_BAT] [RHYME:X] [TONE:XXXXXX]
    tags = f"{rhyme_tag} {tone_tag}".strip()
    tag_part = f"{genre_token}"
    if tags:
        tag_part += f" {tags}"
    
    return f"<|start|> {tag_part} {input_str} <|reply|>"


def decode_doi_tho(tokenizer, new_token_ids, enforce_syllables=True, max_lines=2):
    """
    Decode generated tokens, handling <|linebreak|> which decodes to empty
    string in ByteLevel BPE. Splits on linebreak token positions.
    Returns list of line strings. Optionally enforces syllable pattern (P3)
    and fixes overgeneration (P2: > max_lines → find first valid couplet).
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
    
    # P3: Enforce syllable count per line
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
    
    # P1 (rhyme_constraint): When generating the rhyme-position syllable
    in the second output line, mask out tokens whose rhyme group doesn't
    match the target [RHYME:X] from the prompt.
    
    T2a: For Thất Ngôn, suppress <|linebreak|> until 7+ syllables
    are generated in the first output line.
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

        # T2a: For Thất Ngôn, suppress premature <|linebreak|> in 1st output line
        if is_tn and lb_id not in new_tokens:
            # Still in first output line (no linebreak emitted yet)
            decoded = tokenizer.decode(new_tokens)
            syl_count = len(decoded.strip().split()) if decoded.strip() else 0
            if syl_count < 7:
                logits[:, lb_id] = float("-inf")

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
        print("\n🎭  Interactive Poetry Duel")
        print("    Single line → [LUC_BAT] single couplet")
        print("    Multi-line (use '|' between lines) → [DOI_THO] couplet duel")
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
            elif '[THAT_NGON]' in prompt and '[DOIAM:' not in prompt:
                inner = prompt.replace('[THAT_NGON]', '').strip()
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
            prompt_part = re.sub(r'\[(?:RHYME|TONE|LINK2):[^\]]+\]', '', prompt)
            prompt_part = prompt_part.replace('[LUC_BAT]', '').replace('[DOI_THO]', '').replace('[THAT_NGON]', '')
            prompt_part = ' '.join(prompt_part.split()).strip().rstrip(',')
            resp_clean = response_only.rstrip(",. ")
            is_luc_bat = "[LUC_BAT]" in prompt
            is_that_ngon = "[THAT_NGON]" in prompt

            if is_luc_bat or is_that_ngon:
                tag = "Lục Bát (6→8)" if is_luc_bat else "Thất Ngôn (7→7)"
                r = evaluate(prompt_part, resp_clean)
                print(f"\n{'─'*60}\n📏  {tag} Rule Check\n{'─'*60}")
                print(f"  Syllables: {r['syl'][1]} → {'PASS' if r['syl'][0] else 'FAIL'}")
                print(f"  Prompt tone:  {r['tone_p'][1]}")
                print(f"  Response tone: {r['tone_r'][1]}")
                print(f"{'─'*60}")
            elif is_doi_tho:
                if len(out_lines) >= 2:
                    r6 = out_lines[0].strip()
                    r8 = out_lines[1].strip() if len(out_lines) > 1 else ""
                    print(f"\n{'─'*60}\n📏  Đối Thơ (couplet→couplet) Rule Check\n{'─'*60}")
                    print(f"  Output: {len(r6.split())}syl → {len(r8.split()) if r8 else '?'}syl")
