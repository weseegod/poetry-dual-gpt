"""
sample.py — Autoregressive generation & Vietnamese poetic rule checking.

Phase 1: Lục Bát evaluation (6-syllable prompt → 8-syllable response,
         B-T-B tone pattern verification).
"""

import argparse
import re
from pathlib import Path

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from model import PoetryDuelGPT


ROOT = Path(__file__).parent.parent


# =========================================================================
# Vietnamese tone classification (Bằng vs Trắc)
# =========================================================================

# Bằng (level): ngang (no mark), huyền (grave)
# Trắc (sharp): sắc (acute), nặng (dot), hỏi (hook), ngã (tilde)

# Common Vietnamese vowels with Bằng tones
BANG_CHARS = set("aăâeêioôơuưyAĂÂEÊIOÔƠUƯY"  # ngang (no diacritic)
                  "àằầèềìòồờùừỳÀẰẦÈỀÌÒỒỜÙỪỲ")  # huyền (grave)

# Common Vietnamese vowels with Trắc tones
TRAC_CHARS = set("áắấéếíóốớúứýÁẮẤÉẾÍÓỐỚÚỨÝ"   # sắc (acute)
                  "ạặậẹệịọộợụựỵẠẶẬẸỆỊỌỘỢỤỰỴ"   # nặng (dot)
                  "ảẳẩẻểỉỏổởủửỷẢẲẨẺỂỈỎỔỞỦỬỶ"   # hỏi (hook)
                  "ãẵẫẽễĩõỗỡũữỹÃẴẪẼỄĨÕỖỠŨỮỸ")   # ngã (tilde)


def get_tone_type(syllable: str) -> str:
    """
    Determine if a Vietnamese syllable has Bằng (level) or Trắc (sharp) tone.

    Scans characters for tone-marked vowels. If any Trắc vowel found → trắc.
    Defaults to bằng (most unmarked syllables are ngang = bằng).
    """
    for ch in syllable:
        if ch in TRAC_CHARS:
            return "trắc"
        if ch in BANG_CHARS:
            return "bằng"
    return "bằng"  # default for unrecognized syllables


def count_syllables(text: str) -> int:
    """Count Vietnamese syllables (words) in a line."""
    return len(text.strip().split())


def check_syllable_count(prompt: str, response: str) -> tuple[bool, str]:
    """
    Verify Lục Bát syllable counts: prompt=6, response=8.
    Allows ±1 tolerance for noisy data.
    """
    p = count_syllables(prompt)
    r = count_syllables(response)
    ok = (5 <= p <= 7) and (7 <= r <= 9)
    msg = f"prompt={p} syllables, response={r} syllables (expected 6→8)"
    return ok, msg


def check_tone_alignment(line: str, expected_pattern: str) -> tuple[bool, str]:
    """
    Check tone pattern at positions 2,4,6 (for 6-syllable) or 2,4,6,8 (for 8-syllable).

    Lục Bát rule:
      Line 1 (6 syllables): positions 2,4,6 = B-T-B
      Line 2 (8 syllables): positions 2,4,6,8 = B-T-B-B
    """
    syllables = line.strip().split()
    results = []

    for i, expected in enumerate(expected_pattern):
        pos = i * 2 + 2  # positions: 2, 4, 6, (8)
        if pos > len(syllables):
            break
        actual = get_tone_type(syllables[pos - 1])
        symbol = actual[0].upper()  # 'B' or 'T'
        match = "✓" if symbol == expected else "✗"
        results.append(f"pos{pos}={symbol}({expected}){match}")

    all_match = all("✓" in r for r in results)
    return all_match, " | ".join(results)


def evaluate_luc_bat(prompt: str, response: str) -> dict:
    """Full Lục Bát evaluation: syllable count + tone alignment."""
    results = {}

    # Syllable count
    ok, msg = check_syllable_count(prompt, response)
    results["syllable_count"] = {"pass": ok, "detail": msg}

    # Tone alignment for prompt (B-T-B)
    ok_p, detail_p = check_tone_alignment(prompt, "BTB")
    results["tone_prompt"] = {"pass": ok_p, "detail": detail_p}

    # Tone alignment for response (B-T-B-B)
    ok_r, detail_r = check_tone_alignment(response, "BTBB")
    results["tone_response"] = {"pass": ok_r, "detail": detail_r}

    return results


# =========================================================================
# Model loading
# =========================================================================

def load_model(checkpoint_path, device="cpu"):
    """Load trained PoetryDuelGPT from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = PoetryDuelGPT(
        vocab_size=ckpt["vocab_size"],
        n_embd=ckpt["model_config"]["n_embd"],
        n_head=ckpt["model_config"]["n_head"],
        n_layer=ckpt["model_config"]["n_layer"],
        block_size=ckpt["model_config"]["block_size"],
        dropout=ckpt["model_config"].get("dropout", 0.1),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    print(f"Loaded model from {checkpoint_path} (step {ckpt['step']})")
    return model


# =========================================================================
# Generation
# =========================================================================

@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens=64, temperature=0.75, top_k=50, device="cpu"):
    """
    Generate a poetic response from a prompt.

    Steps:
      1. Encode prompt → token IDs
      2. Loop: forward pass → sample next token → append
      3. Stop on <|end|> or max tokens
      4. Decode and return
    """
    end_id = tokenizer.token_to_id("<|end|>")
    pad_id = tokenizer.token_to_id("<|pad|>")

    # Encode prompt
    ids = tokenizer.encode(prompt).ids
    idx = torch.tensor([ids], dtype=torch.long, device=device)

    generated_ids = []
    for _ in range(max_new_tokens):
        # Crop to block_size
        idx_cond = idx[:, -model.block_size:]

        # Forward
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature

        # Top-k filtering
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, -1:]] = float("-inf")

        # Sample
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)

        token_id = idx_next.item()

        # Stop conditions
        if token_id == end_id:
            break
        if token_id == pad_id:
            continue  # skip padding

        generated_ids.append(token_id)
        idx = torch.cat((idx, idx_next), dim=1)

    # Decode
    full_text = tokenizer.decode(ids + generated_ids)

    return full_text, generated_ids


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description="PoetryDuelGPT Inference")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/final.pt")
    parser.add_argument("--tokenizer", type=str, default="tokenizer/poetry_bpe.model")
    parser.add_argument("--prompt", type=str, default="[LUC_BAT] Thân em như chẽn lúa đòng đòng,")
    parser.add_argument("--temperature", type=float, default=0.75)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--max_tokens", type=int, default=64)
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_samples", type=int, default=1)
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    # Load tokenizer
    tok_path = ROOT / args.tokenizer
    tokenizer = Tokenizer.from_file(str(tok_path))
    print(f"Tokenizer: vocab={tokenizer.get_vocab_size():,}")

    # Load model
    ckpt_path = ROOT / args.checkpoint
    if ckpt_path.exists():
        model = load_model(str(ckpt_path), device)
    else:
        print(f"No checkpoint at {ckpt_path}")
        print("Run training first: python src/train.py")
        return

    if args.interactive:
        print("\n🎭  Interactive Poetry Duel Mode 🎭")
        print("    Type a Lục Bát line and the model will respond!")
        print("    Format: Your 6-syllable line,  (or paste a full prompt)")
        print("    Type 'quit' to exit.\n")

        while True:
            user_input = input("You: ").strip()
            if user_input.lower() == "quit":
                break
            if not user_input:
                continue

            # Format as Lục Bát prompt
            if not user_input.startswith("[LUC_BAT]"):
                user_input = f"[LUC_BAT] {user_input}"

            output, ids = generate(model, tokenizer, user_input,
                                   max_new_tokens=args.max_tokens,
                                   temperature=args.temperature,
                                   top_k=args.top_k,
                                   device=device)
            print(f"Bot: {output}\n")
    else:
        for i in range(args.num_samples):
            print(f"\n{'='*60}")
            print(f"Sample {i+1}/{args.num_samples}")
            print(f"{'='*60}")
            print(f"Prompt:   {args.prompt}")

            output, ids = generate(model, tokenizer, args.prompt,
                                   max_new_tokens=args.max_tokens,
                                   temperature=args.temperature,
                                   top_k=args.top_k,
                                   device=device)
            print(f"Response: {output}")

            # Evaluate if it's Lục Bát
            if "[LUC_BAT]" in args.prompt:
                print(f"\n{'─'*60}")
                print("📏  Lục Bát Rule Check")
                print(f"{'─'*60}")

                # Extract just the response part
                if "<|reply|>" in output:
                    response_part = output.split("<|reply|>")[-1].replace("<|end|>", "").strip()
                else:
                    # Try to find the comma-separated boundary
                    parts = output.split(",")
                    if len(parts) >= 2:
                        response_part = parts[-1].strip().replace("<|end|>", "")
                    else:
                        response_part = output.replace(args.prompt, "").strip()

                # Extract prompt part
                prompt_part = args.prompt.replace("[LUC_BAT]", "").strip().rstrip(",")

                results = evaluate_luc_bat(prompt_part, response_part)
                print(f"  Syllables: {results['syllable_count']['detail']} "
                      f"→ {'PASS' if results['syllable_count']['pass'] else 'FAIL'}")
                print(f"  Prompt tone (2-4-6 B-T-B): {results['tone_prompt']['detail']}")
                print(f"  Response tone (2-4-6-8 B-T-B-B): {results['tone_response']['detail']}")
                print(f"{'─'*60}")


if __name__ == "__main__":
    main()
