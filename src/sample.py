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
from tones import get_luc_bat_tags, get_that_ngon_tags, get_doi_tho_tags

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
    
    # Single line → delegate to existing auto_tag
    if len(lines) == 1:
        line = lines[0]
        syls = line.split()
        if len(syls) >= 6:
            rhyme_tag, tone_tag = get_doi_tho_tags(line, line)
        else:
            rhyme_tag, tone_tag = "", ""
        tags = f"{rhyme_tag} {tone_tag}".strip()
        tag_part = f"[DOI_THO] {tags}" if tags else "[DOI_THO]"
        return f"<|start|> {tag_part} {line} <|reply|>"
    
    # Group into (6-syl, 8-syl) pairs
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
        # No valid couplets found → fall back to last line as single prompt
        return auto_tag(lines[-1])
    
    # Keep at most max_context_couplets recent couplets
    couplets = couplets[-max_context_couplets:]
    
    # Extract tags from LAST couplet
    last_6, last_8 = couplets[-1]
    rhyme_tag, tone_tag = get_doi_tho_tags(last_6, last_8)
    
    # Build input lines with <|linebreak|> separators
    input_lines = []
    for six, eight in couplets:
        input_lines.append(six)
        input_lines.append(eight)
    input_str = " <|linebreak|> ".join(input_lines)
    
    # Build tag
    tags = f"{rhyme_tag} {tone_tag}".strip()
    tag_part = f"[DOI_THO] {tags}" if tags else "[DOI_THO]"
    
    return f"<|start|> {tag_part} {input_str} <|reply|>"


def decode_doi_tho(tokenizer, new_token_ids, enforce_syllables=True):
    """
    Decode generated tokens, handling <|linebreak|> which decodes to empty
    string in ByteLevel BPE. Splits on linebreak token positions.
    Returns list of line strings. Optionally enforces 6/8 syllable pattern (P3).
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
    
    # P3: Enforce 6/8 syllable pattern
    if enforce_syllables:
        targets = [6, 8]
        for i, line in enumerate(lines):
            words = line.split()
            target = targets[i % 2]
            if len(words) > target:
                words = words[:target]
            lines[i] = ' '.join(words)
    
    return lines


@torch.no_grad()
def generate(model, tokenizer, prompt, max_new=64, temperature=0.75,
              top_k=50, top_p=None, device="cpu", max_syllables=None):
    """
    1. Encode prompt → 2. Loop: forward → sample next token → append
    3. Stop on <|end|> or max tokens → 4. Decode new tokens only
    """
    end_id = tokenizer.token_to_id("<|end|>")
    pad_id = tokenizer.token_to_id("<|pad|>")

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
