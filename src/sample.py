"""
Autoregressive generation + Vietnamese Lục Bát rule checking.

Usage:
  python src/sample.py --checkpoint checkpoints/final.pt
  python src/sample.py --interactive
"""

import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from model import PoetryDuelGPT

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

@torch.no_grad()
def generate(model, tokenizer, prompt, max_new=64, temperature=0.75,
              top_k=50, top_p=None, device="cpu"):
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
            logits[sorted_idx[mask]] = float("-inf")

        # Sample from softmax
        next_id = torch.multinomial(F.softmax(logits, dim=-1), 1).item()

        if next_id == end_id: break        # stop token

        new_tokens.append(next_id)
        idx = torch.cat((idx, torch.tensor([[next_id]], device=device)), dim=1)

    return tokenizer.decode(ids + new_tokens), new_tokens


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/final.pt")
    p.add_argument("--tokenizer", default="tokenizer/poetry_bpe.model")
    p.add_argument("--prompt", default="[LUC_BAT] Thân em như chẽn lúa đòng đòng")
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
        print("    Type a Lục Bát line. 'quit' to exit.\n")
        while True:
            u = input("You: ").strip()
            if u.lower() == "quit": break
            if not u: continue
            if not u.startswith("[LUC_BAT]"): u = f"[LUC_BAT] {u}"
            _, ids = generate(model, tok, u, args.max_tokens, args.temperature, args.top_k, args.top_p, dev)
            print(f"Bot: {tok.decode(ids).replace('<|end|>','').strip()}\n")

    # Batch generation
    else:
        for i in range(args.num_samples):
            print(f"\n{'='*60}\nSample {i+1}/{args.num_samples}\n{'='*60}")
            print(f"Prompt:   {args.prompt}")

            _, ids = generate(model, tok, args.prompt, args.max_tokens,
                              args.temperature, args.top_k, args.top_p, dev)
            response_only = tok.decode(ids).replace("<|end|>", "").strip()
            print(f"Response: {response_only}")

            # Rule check for Lục Bát
            if "[LUC_BAT]" in args.prompt:
                prompt_part = args.prompt.replace("[LUC_BAT]", "").strip().rstrip(",")
                r = evaluate(prompt_part, response_only.rstrip(",. "))
                print(f"\n{'─'*60}\n📏  Lục Bát Rule Check\n{'─'*60}")
                print(f"  Syllables: {r['syl'][1]} → {'PASS' if r['syl'][0] else 'FAIL'}")
                print(f"  Prompt tone:  {r['tone_p'][1]}")
                print(f"  Response tone: {r['tone_r'][1]}")
                print(f"{'─'*60}")
