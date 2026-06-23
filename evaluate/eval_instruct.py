#!/usr/bin/env python3
"""
Evaluate Qwen2.5-Instruct QLoRA on 5 Lục Bát rules using chat template.

Key differences from failed eval_qwen.py:
  - Uses Qwen's native chat template (no control tokens)
  - Extracts assistant response cleanly
  - Post-processes BPE fragments
  - Optional soft rhyme constraint at inference
  - Same 5-rule evaluation logic as v4.2.3

Usage:
  HF_TOKEN=hf_xxx python evaluate/eval_instruct.py
  HF_TOKEN=hf_xxx python evaluate/eval_instruct.py --checkpoint checkpoints/instruct_final
  HF_TOKEN=hf_xxx python evaluate/eval_instruct.py --checkpoint checkpoints/instruct_final --num 20
"""

import argparse, json, os, re, sys, time
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.tones import get_tone, get_rhyme_group, get_luc_bat_tags, get_diacritic
from evaluate.prompts import COUPLET_PROMPTS

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

# ═══════════════════════════════════════════
# GENERATION CONFIG
# ═══════════════════════════════════════════

TEMPERATURE = 0.75
TOP_K = 50
TOP_P = 0.92
MAX_NEW_TOKENS = 32
RHYME_LOGIT_BOOST = 2.0       # soft rhyme boost
RHYME_SAFETY_THRESHOLD = 0.05  # fallback to hard mask when uncertain

# ═══════════════════════════════════════════
# SYSTEM PROMPT
# ═══════════════════════════════════════════

SYSTEM_PROMPT = (
    "Bạn là nhà thơ Lục Bát chuyên nghiệp. "
    "Cho dòng Lục (6 chữ), hãy viết dòng Bát (8 chữ) đúng luật thơ Lục Bát:\n"
    "- Vần: chữ thứ 6 của dòng Bát phải vần với \"{rhyme}\"\n"
    "- Thanh điệu dòng Bát (vị trí 2-4-6-8): {tone_pattern}\n"
    "- Trầm-Bổng: {trambong_rule}\n"
    "Chỉ trả lời 8 chữ của dòng Bát, không thêm gì khác."
)

TRAMBONG_DESC = {
    "NH": "chữ thứ 6 thanh Ngang, chữ thứ 8 thanh Huyền (Ngang≠Huyền)",
    "HN": "chữ thứ 6 thanh Huyền, chữ thứ 8 thanh Ngang (Huyền≠Ngang)",
    "XX": "chữ thứ 6 và 8 khác dấu (Ngang≠Huyền)",
}


# ═══════════════════════════════════════════
# BPE FRAGMENT FIXING
# ═══════════════════════════════════════════

def join_bpe_fragments(text: str) -> str:
    """Join BPE sub-token fragments that lack Vietnamese vowels."""
    if not text:
        return text
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()
    if len(words) <= 1:
        return text

    VN_VOWELS = set('aăâeêioôơuưyAĂÂEÊIOÔƠUƯY'
                     'àáảãạằắẳẵặầấẩẫậèéẻẽẹềếểễệìíỉĩịòóỏõọồốổỗộờớởỡợùúủũụừứửữựỳýỷỹỵ'
                     'ÀÁẢÃẠẰẮẲẴẶẦẤẨẪẬÈÉẺẼẸỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌỒỐỔỖỘỜỚỞỠỢÙÚỦŨỤỪỨỬỮỰỲÝỶỸỴ')
    def has_vowel(w):
        return any(c in VN_VOWELS for c in w)

    i = len(words) - 1
    while i > 0:
        if not has_vowel(words[i]) and has_vowel(words[i-1]):
            words[i-1] = words[i-1] + words[i]
            words.pop(i)
        i -= 1
    i = 0
    while i < len(words) - 1:
        if not has_vowel(words[i]) and has_vowel(words[i+1]):
            words[i+1] = words[i] + words[i+1]
            words.pop(i)
        else:
            i += 1
    return ' '.join(words)


# ═══════════════════════════════════════════
# MODEL LOADING
# ═══════════════════════════════════════════

def load_model(checkpoint_path: str, cache_dir: str = None):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise RuntimeError("Set HF_TOKEN environment variable")
    if cache_dir is None:
        cache_dir = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

    has_gpu = torch.cuda.is_available()
    dev = "cuda" if has_gpu else "cpu"
    print(f"📦  Loading {MODEL_ID} ({'GPU' if has_gpu else 'CPU'})...")

    kwargs = {"trust_remote_code": True, "token": hf_token, "cache_dir": cache_dir}

    if has_gpu:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        kwargs["device_map"] = "auto"
    else:
        kwargs["torch_dtype"] = torch.float32

    # Load base tokenizer/model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, **kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **kwargs)

    if not has_gpu:
        model = model.to(dev)

    # Load LoRA adapter
    print(f"   🎯  Loading LoRA adapter: {checkpoint_path}")
    model = PeftModel.from_pretrained(model, checkpoint_path)
    model.eval()

    print(f"   Vocab: {len(tokenizer):,} (native — no new tokens)")
    return model, tokenizer, dev


# ═══════════════════════════════════════════
# PROMPT BUILDING
# ═══════════════════════════════════════════

def build_constraints_from_couplet(line6: str, line8: str) -> dict:
    """Extract rhyme, tone pattern, and Trầm-Bổng from a ground-truth couplet."""
    couplet = f"{line6} {line8}"
    luc_syls = line6.split()
    bat_syls = line8.split()

    # Rhyme: last syllable of Lục line → rhyme group for Bát[pos6]
    rhyme = get_rhyme_group(luc_syls[5]) if len(luc_syls) >= 6 else "a"

    # Tone pattern for Bát line: positions 2-4-6-8 (BTBB)
    bat_tones = []
    for idx in [1, 3, 5, 7]:
        if idx < len(bat_syls):
            bat_tones.append(get_tone(bat_syls[idx]))
        else:
            bat_tones.append('B')
    tone_pattern = '-'.join(bat_tones)  # "B-T-B-B"

    # Trầm-Bổng: diacritic at pos6 vs pos8 of Bát
    trambong = "NH"
    if len(bat_syls) >= 8:
        d6 = get_diacritic(bat_syls[5])
        d8 = get_diacritic(bat_syls[7])
        if d6 == "ngang" and d8 == "huyen":
            trambong = "NH"
        elif d6 == "huyen" and d8 == "ngang":
            trambong = "HN"

    return {
        "rhyme": rhyme,
        "tone_pattern": tone_pattern,
        "trambong": trambong,
        "trambong_desc": TRAMBONG_DESC.get(trambong, TRAMBONG_DESC["XX"]),
    }


def build_chat_prompt(line6: str, line8: str) -> tuple[str, str]:
    """Build system + user messages from a couplet. Returns (system, user)."""
    c = build_constraints_from_couplet(line6, line8)
    system = SYSTEM_PROMPT.format(
        rhyme=c["rhyme"],
        tone_pattern=c["tone_pattern"],
        trambong_rule=c["trambong_desc"],
    )
    user = line6
    return system, user, c["rhyme"]


# ═══════════════════════════════════════════
# GENERATION
# ═══════════════════════════════════════════

def _get_rhyme_token_ids(tokenizer, rhyme_group: str) -> set:
    """Find token IDs whose decoded text ends with the given rhyme group."""
    if not rhyme_group:
        return set()
    matching = set()
    vocab = tokenizer.get_vocab()
    for token_str, token_id in vocab.items():
        clean = token_str.lstrip("Ġ▁ ")
        if clean and get_rhyme_group(clean) == rhyme_group:
            matching.add(token_id)
    return matching


@torch.no_grad()
def generate_bat_line(model, tokenizer, line6: str, line8: str,
                       temperature: float = TEMPERATURE,
                       top_k: int = TOP_K, top_p: float = TOP_P,
                       rhyme_boost: float = RHYME_LOGIT_BOOST) -> str:
    """Generate Bát line using chat template with optional rhyme constraint."""
    dev = next(model.parameters()).device

    system, user, rhyme_group = build_chat_prompt(line6, line8)

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = tokenizer(text, return_tensors="pt").to(dev)
    input_len = inputs["input_ids"].shape[1]
    
    # EOS tokens for stopping
    eos_ids = [tokenizer.eos_token_id]
    end_tokens = tokenizer.encode("<|im_end|>", add_special_tokens=False)
    if end_tokens:
        eos_ids.extend(end_tokens)
    
    # Pre-compute rhyme tokens
    rhyme_ids = _get_rhyme_token_ids(tokenizer, rhyme_group) if rhyme_boost > 0 else set()
    
    # Generate
    generated_ids = inputs["input_ids"]
    rhyme_applied = False
    
    for _ in range(MAX_NEW_TOKENS):
        with torch.autocast(device_type=str(dev), dtype=torch.bfloat16):
            outputs = model(generated_ids)
        logits = outputs.logits[0, -1, :]

        # Temperature
        logits = logits / temperature

        # Top-k
        if top_k > 0:
            topk_vals, topk_idx = torch.topk(logits, min(top_k, len(logits)))
            mask = torch.full_like(logits, float('-inf'))
            mask[topk_idx] = topk_vals
            logits = mask

        # Soft rhyme at output position 6 (after generating 5 content tokens post-prompt)
        gen_len = generated_ids.shape[1] - input_len
        if not rhyme_applied and gen_len == 5 and rhyme_ids:
            probs = F.softmax(logits, dim=-1)
            if probs.max().item() < RHYME_SAFETY_THRESHOLD:
                # Hard mask fallback
                non_matching = torch.ones(len(logits), dtype=torch.bool, device=dev)
                for tid in rhyme_ids:
                    non_matching[tid] = False
                logits[non_matching] = float('-inf')
            else:
                # Soft boost
                for tid in rhyme_ids:
                    logits[tid] += rhyme_boost
            rhyme_applied = True

        # Top-p
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumprobs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            remove = cumprobs > top_p
            remove[1:] = remove[:-1].clone()
            remove[0] = False
            logits[sorted_indices[remove]] = float('-inf')

        # Sample
        probs = F.softmax(logits, dim=-1)
        probs = probs / probs.sum()
        next_id = torch.multinomial(probs, 1).item()

        if next_id in eos_ids:
            break

        generated_ids = torch.cat(
            [generated_ids, torch.tensor([[next_id]], device=dev)], dim=1
        )

    # Decode only the newly generated tokens
    new_ids = generated_ids[0, input_len:]
    text = tokenizer.decode(new_ids, skip_special_tokens=True).strip()

    # Strip <|im_end|> if present
    text = text.replace("<|im_end|>", "").strip()

    # Fix BPE fragments
    text = join_bpe_fragments(text)

    return text


# ═══════════════════════════════════════════
# EVALUATION
# ═══════════════════════════════════════════

def evaluate_couplet(line6, line8, response):
    """Score all 5 Lục Bát rules."""
    in_syls = line6.split()
    out_syls = response.split()
    r_len = len(out_syls)

    r1 = False
    if len(in_syls) >= 6 and len(out_syls) >= 6:
        r1 = get_rhyme_group(in_syls[5]) == get_rhyme_group(out_syls[5])

    r2_ok, r2_total = 0, 0
    for idx, want in [(1, 'B'), (3, 'T'), (5, 'B'), (7, 'B')]:
        if idx < len(out_syls):
            r2_total += 1
            if get_tone(out_syls[idx]) == want:
                r2_ok += 1

    r3 = (len(out_syls) == 8)
    r4 = False
    if len(out_syls) >= 8:
        d6 = get_diacritic(out_syls[5])
        d8 = get_diacritic(out_syls[7])
        r4 = d6 in ('ngang', 'huyen') and d8 in ('ngang', 'huyen') and d6 != d8

    r5 = (len(out_syls) == 8)
    lex = len(set(out_syls)) / max(len(out_syls), 1)

    VIET_CHARS = set("aăâeêioôơuưyAĂÂEÊIOÔƠUƯY"
                      "àáảãạằắẳẵặầấẩẫậèéẻẽẹềếểễệìíỉĩịòóỏõọồốổỗộờớởỡợùúủũụừứửữựỳýỷỹỵ"
                      "ÀÁẢÃẠẰẮẲẴẶẦẤẨẪẬÈÉẺẼẸỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌỒỐỔỖỘỜỚỞỠỢÙÚỦŨỤỪỨỬỮỰỲÝỶỸỴ")
    bpe = sum(1 for s in out_syls if len(s) < 2 or not any(c in VIET_CHARS for c in s))

    return {
        'prompt': f'{line6} / {line8[:20]}',
        'response': response,
        'r_len': r_len,
        'R1_ok': r1, 'R2_r_ok': r2_ok, 'R2_r_total': r2_total,
        'R3_exact': r3, 'R4_ok': r4, 'R5_ok': r5,
        'all_5': r1 and r3 and r2_ok == r2_total and r4,
        'lex_div': lex, 'bpe_artifacts': bpe,
    }


def summarize(results):
    n = len(results)
    return {
        'n': n,
        'R1': sum(r['R1_ok'] for r in results) / n * 100,
        'R2': sum(r['R2_r_ok'] for r in results) / max(sum(r['R2_r_total'] for r in results), 1) * 100,
        'R3': sum(r['R3_exact'] for r in results) / n * 100,
        'R4': sum(r['R4_ok'] for r in results) / n * 100,
        'R5': sum(r['R5_ok'] for r in results) / n * 100,
        'all5': sum(r['all_5'] for r in results) / n * 100,
        'avg_len': sum(r['r_len'] for r in results) / n,
        'avg_lex': sum(r['lex_div'] for r in results) / n,
        'avg_bpe': sum(r['bpe_artifacts'] for r in results) / n,
        'empty': sum(1 for r in results if r['r_len'] == 0) / n * 100,
    }


# ═══════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════

def main():
    global TEMPERATURE, TOP_K, TOP_P, RHYME_LOGIT_BOOST

    parser = argparse.ArgumentParser(description="Evaluate Instruct QLoRA on Lục Bát")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/instruct_final")
    parser.add_argument("--num", type=int, default=0)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--no-rhyme", action="store_true")
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--top-p", type=float, default=TOP_P)
    args = parser.parse_args()

    TEMPERATURE = args.temperature
    TOP_K = args.top_k
    TOP_P = args.top_p
    if args.no_rhyme:
        RHYME_LOGIT_BOOST = 0.0

    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        print(f"❌  Checkpoint not found: {ckpt}")
        sys.exit(1)

    prompts = COUPLET_PROMPTS[:args.num] if args.num else COUPLET_PROMPTS

    model, tokenizer, dev = load_model(str(ckpt), cache_dir=args.cache_dir)
    print(f"   Device: {dev} | Prompts: {len(prompts)}")
    print(f"   Temp={TEMPERATURE} TopK={TOP_K} TopP={TOP_P}")
    print(f"   RhymeBoost={RHYME_LOGIT_BOOST}")

    print(f"\n{'='*60}")
    print(f"Qwen2.5-Instruct QLoRA — 5-Rule Lục Bát Evaluation")
    print(f"Checkpoint: {ckpt.name}")
    print(f"{'='*60}\n")

    t0 = time.time()
    results = []

    for i, (l6, l8) in enumerate(tqdm(prompts, desc="Evaluating", unit="prompt")):
        response = generate_bat_line(
            model, tokenizer, l6, l8,
            temperature=TEMPERATURE, top_k=TOP_K, top_p=TOP_P,
            rhyme_boost=RHYME_LOGIT_BOOST,
        )
        results.append(evaluate_couplet(l6, l8, response))

        if (i + 1) % 30 == 0:
            s = summarize(results)
            print(f"  {i+1}/{len(prompts)} | R1:{s['R1']:.0f}% R2:{s['R2']:.0f}% "
                  f"R3:{s['R3']:.0f}% R4:{s['R4']:.0f}% | All5:{s['all5']:.0f}% "
                  f"Lex:{s['avg_lex']:.2f} BPE:{s['avg_bpe']:.1f}")

    s = summarize(results)
    elapsed = time.time() - t0

    # ── Report ──
    TARGETS = {
        'R1': 85, 'R2': 90, 'R3': 90, 'R4': 85, 'R5': 90,
        'all5': 70, 'avg_lex': 0.85, 'avg_bpe': 5.0, 'empty': 10,
    }

    print(f"\n{'='*60}")
    print(f"📊  RESULTS — Phase 1 Targets (roadmap_v5.md)")
    print(f"{'='*60}")
    print(f"{'Metric':<30} {'Value':>8} {'Target':>8} {'Status':>8}")
    print(f"{'-'*30} {'-'*8} {'-'*8} {'-'*8}")

    all_pass = True
    for key, label in [
        ('R1', 'R1 Rhyme (vần lưng)'),
        ('R2', 'R2 Tone (BTBB)'),
        ('R3', 'R3 Syllable (exact 8)'),
        ('R4', 'R4 Trầm-Bổng'),
        ('R5', 'R5 Nhịp điệu (exact 8)'),
        ('all5', 'All 5 rules pass'),
    ]:
        v = s[key]; t = TARGETS[key]
        ok = v >= t
        if not ok: all_pass = False
        print(f"{label:<30} {v:>7.1f}% {t:>7.0f}% {'✅' if ok else '❌':>8}")

    print(f"\n{'─'*30} {'─'*8} {'─'*8} {'─'*8}")
    for key, label in [
        ('avg_lex', 'Lexical diversity'),
        ('avg_bpe', 'BPE artifact rate'),
        ('empty', 'Empty response rate'),
    ]:
        v = s[key]; t = TARGETS[key]
        ok = v >= t if key == 'avg_lex' else v <= t
        if not ok: all_pass = False
        print(f"{label:<30} {v:>7.1f} {'≥' if key == 'avg_lex' else '≤'}{t:>7.0f} {'✅' if ok else '❌':>8}")

    print(f"\n{'='*60}")
    print(f"{'🎉 ALL TARGETS MET!' if all_pass else '⚠️  Some targets not met'}")
    print(f"   Elapsed: {elapsed:.0f}s | Prompts: {len(prompts)}")
    print(f"{'='*60}")

    # ── Samples ──
    print(f"\n📝  Sample outputs:")
    for i, r in enumerate(results[:10]):
        emoji = lambda ok: '✅' if ok else '❌'
        l6 = r['prompt'].split(' / ')[0]
        print(f"\n  [{i+1}] Lục: {l6}")
        print(f"     → Bát: {r['response'][:80]}")
        print(f"       {emoji(r['R1_ok'])}R1 {emoji(r['R2_r_ok']==r['R2_r_total'])}R2 "
              f"{emoji(r['R3_exact'])}R3 {emoji(r['R4_ok'])}R4")

    # ── Save ──
    out_path = Path(args.output or ROOT / "evaluate" / "instruct_evaluation.json")
    json.dump({
        'version': 'v5-instruct',
        'checkpoint': str(ckpt),
        'model': MODEL_ID,
        'n_prompts': len(prompts),
        'summary': s,
        'targets': TARGETS,
        'all_pass': all_pass,
        'elapsed': elapsed,
        'results': results,
    }, open(out_path, 'w'), indent=2, ensure_ascii=False)
    print(f"\n📄  JSON saved: {out_path}")


if __name__ == "__main__":
    main()
