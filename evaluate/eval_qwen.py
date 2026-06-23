#!/usr/bin/env python3
"""
Qwen2.5-1.5B QLoRA — 5-rule Lục Bát evaluation (v5 FIXED).

Key improvements over v5 original:
  - Soft rhyme constraint (P2): logit bias instead of hard masking
  - Repetition penalty: -1.2 for recent 16 tokens
  - Syllable-aware decoding: handles single-token syllable representations
  - Top-k + Top-p filtering

Matches roadmap_v5.md Phase 1 targets:
  All-5-pass ≥ 90% | R1 Rhyme ≥ 90% | R2 Tone ≥ 95%
  R4 Trầm-Bổng ≥ 90% | Lexical diversity ≥ 0.90

Usage:
  HF_TOKEN=hf_xxx python evaluate/eval_qwen.py
  HF_TOKEN=hf_xxx python evaluate/eval_qwen.py --checkpoint checkpoints/qwen_stage1_best
  HF_TOKEN=hf_xxx python evaluate/eval_qwen.py --checkpoint checkpoints/qwen_stage1_best --num 10
"""

import argparse, json, os, re, sys, time
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.tones import (get_tone, get_rhyme_group, get_luc_bat_tags,
                        get_diacritic)
from evaluate.prompts import COUPLET_PROMPTS

MODEL_ID = "Qwen/Qwen2.5-1.5B"

# ═══════════════════════════════════════════════════════════════
# GENERATION CONFIG
# ═══════════════════════════════════════════════════════════════

TEMPERATURE = 0.75
TOP_K = 50
TOP_P = 0.92
MAX_NEW_TOKENS = 32
REPETITION_PENALTY = 1.2
REPETITION_WINDOW = 16

# v5 FIXED: Soft rhyme constraint
RHYME_LOGIT_BOOST = 2.0       # +2.0 logit boost for matching rhyme candidates
RHYME_SAFETY_THRESHOLD = 0.05  # fall back to hard masking if model uncertain


# ═══════════════════════════════════════════════════════════════
# MODEL LOADING
# ═══════════════════════════════════════════════════════════════

def load_qwen_model(checkpoint_path: str, cache_dir: str = None):
    """Load Qwen base + LoRA adapter. Falls back to CPU if no GPU."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise RuntimeError("Set HF_TOKEN environment variable")
    if cache_dir is None:
        cache_dir = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

    has_gpu = torch.cuda.is_available()
    dev = "cuda" if has_gpu else "cpu"
    print(f"📦  Loading {MODEL_ID} ({'GPU' if has_gpu else 'CPU — will be slow'})...")
    print(f"   Cache: {cache_dir}")

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

    # ── Load checkpoint tokenizer first (has correct token→ID mapping) ──
    print(f"   🔤  Loading tokenizer from checkpoint: {checkpoint_path}")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    ckpt_vocab_size = len(tokenizer)
    print(f"   Checkpoint vocab: {ckpt_vocab_size:,}")

    # ── Load base model ──
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **kwargs)
    base_vocab_size = model.config.vocab_size
    print(f"   Base Qwen vocab: {base_vocab_size:,}")

    # ── Resize embeddings to match checkpoint (must happen BEFORE loading LoRA) ──
    if ckpt_vocab_size != base_vocab_size:
        model.resize_token_embeddings(ckpt_vocab_size)
        print(f"   Resized embeddings: {base_vocab_size:,} → {ckpt_vocab_size:,}")

    if not has_gpu:
        model = model.to(dev)

    print(f"   🎯  Loading LoRA adapter: {checkpoint_path}")
    model = PeftModel.from_pretrained(model, checkpoint_path)
    model.eval()

    return model, tokenizer, dev


# ═══════════════════════════════════════════════════════════════
# v5 FIXED: GENERATION WITH SOFT RHYME + REPETITION PENALTY
# ═══════════════════════════════════════════════════════════════

def build_qwen_prompt(line6: str, line8: str = None) -> str:
    """
    Build prompt in v5 training format:
    <|start|> [LUC_BAT] [RHYME:X] [TONE:BBTBBT] [TRAMBONG:NH] line6 <|reply|>
    
    Training: model takes a Lục line + control tags → generates matching Bát line.
    """
    if line8:
        couplet = f"{line6} {line8}"
        rhyme, tone, trambong = get_luc_bat_tags(couplet)
    else:
        rhyme, tone, trambong = get_luc_bat_tags(line6)
    tags = " ".join(t for t in [rhyme, tone, trambong] if t)
    return f"<|start|> [LUC_BAT] {tags} {line6} <|reply|>"


def _get_rhyme_token_ids(tokenizer, rhyme_group: str) -> list:
    """
    Find all token IDs in the vocab that belong to the given rhyme group.
    Matches tokens that end with the rhyme pattern (e.g., 'ong' matches 'trong', 'song', etc.)
    
    With syllable-level tokenization (v5 FIXED), most syllables are single tokens.
    We check each token to see if it matches the desired rhyme group.
    """
    if not rhyme_group:
        return []
    
    matching = []
    vocab = tokenizer.get_vocab()
    
    for token_str, token_id in vocab.items():
        # Skip special tokens and control tokens
        if token_id < 215:  # control token range
            continue
        if token_str.startswith("<") or token_str.startswith("["):
            continue
        
        # Check if this token rhymes with the target rhyme group
        # Strip leading space prefix if present (Ġ or ▁)
        clean = token_str.lstrip("Ġ▁ ")
        if clean and get_rhyme_group(clean) == rhyme_group:
            matching.append(token_id)
    
    return matching


@torch.no_grad()
def generate_response(model, tokenizer, prompt: str, max_new: int = 32) -> str:
    """
    Generate Bát line from Qwen + LoRA model with soft rhyme constraint.
    
    v5 FIXED improvements:
      - Soft rhyme constraint: logit boost for matching rhyme tokens at position 6
      - Fallback to hard masking only when model is uncertain
      - Repetition penalty on recent tokens
      - Top-k + Top-p filtering
    """
    dev = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(dev)
    input_ids = inputs["input_ids"]
    input_len = input_ids.shape[1]
    
    # Extract target rhyme group from prompt
    rhyme_match = re.search(r'\[RHYME:(\w+)\]', prompt)
    target_rhyme = rhyme_match.group(1) if rhyme_match else None
    
    # Pre-compute matching rhyme token IDs for soft constraint
    rhyme_matching_ids = set()
    rhyme_safety_ids = set()
    if target_rhyme:
        rhyme_matching_ids = set(_get_rhyme_token_ids(tokenizer, target_rhyme))
        # Also add tokens whose decode starts with known rhyming syllables
        if not rhyme_matching_ids:
            print(f"   ⚠️  No rhyme tokens found for group '{target_rhyme}'")
    
    # Get end token ID
    end_id = tokenizer.encode("<|end|>", add_special_tokens=False)
    end_id = end_id[0] if end_id else tokenizer.eos_token_id
    
    # ── Autoregressive generation with rhyme constraint at output position 6 ──
    generated_ids = []
    rhyme_applied = False
    
    for step in range(max_new):
        # Forward pass
        with torch.autocast(device_type=dev, dtype=torch.bfloat16):
            outputs = model(input_ids)
        
        logits = outputs.logits[0, -1, :]  # (V,)
        
        # ── Repetition penalty ──
        if len(generated_ids) > 0:
            recent = generated_ids[-REPETITION_WINDOW:]
            for tid in set(recent):
                if logits[tid] > 0:
                    logits[tid] /= REPETITION_PENALTY
                else:
                    logits[tid] *= REPETITION_PENALTY
        
        # ── Top-k filtering ──
        if TOP_K > 0:
            topk_vals, topk_idx = torch.topk(logits, min(TOP_K, len(logits)))
            mask = torch.full_like(logits, float('-inf'))
            mask[topk_idx] = topk_vals
            logits = mask
        
        # ── Soft rhyme constraint at output position 6 (0-indexed: 5) ──
        if not rhyme_applied and len(generated_ids) == 5 and rhyme_matching_ids:
            probs = F.softmax(logits, dim=-1)
            max_prob = probs.max().item()
            
            if max_prob < RHYME_SAFETY_THRESHOLD:
                # Model is very uncertain — fall back to hard masking
                non_matching_mask = torch.ones(len(logits), dtype=torch.bool, device=dev)
                for tid in rhyme_matching_ids:
                    non_matching_mask[tid] = False
                logits[non_matching_mask] = float('-inf')
            else:
                # Soft boost: add logit bias for matching rhyme candidates
                rhyme_mask = torch.full((len(logits),), 0.0, device=dev)
                for tid in rhyme_matching_ids:
                    rhyme_mask[tid] = RHYME_LOGIT_BOOST
                logits = logits + rhyme_mask
            
            rhyme_applied = True
        
        # ── Top-p (nucleus) filtering ──
        if TOP_P < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > TOP_P
            sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
            sorted_indices_to_remove[0] = False
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = float('-inf')
        
        # ── Sample ──
        probs = F.softmax(logits, dim=-1)
        probs = probs / probs.sum()  # renormalize
        next_id = torch.multinomial(probs, 1).item()
        
        # Stop on end token
        if next_id == end_id:
            break
        
        generated_ids.append(next_id)
        input_ids = torch.cat([input_ids, torch.tensor([[next_id]], device=dev)], dim=1)
    
    # Decode (skip special tokens to avoid control tokens in output)
    text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    
    # Stop at <|end|> if present
    if "<|end|>" in text:
        text = text.split("<|end|>")[0].strip()
    
    # ── v5 FIXED: Post-process BPE fragments ──
    # Qwen's tokenizer may split Vietnamese syllables into sub-tokens.
    # Join fragments without space prefix back to the previous word.
    text = join_bpe_fragments(text)
    
    return text


def join_bpe_fragments(text: str) -> str:
    """
    Fix BPE fragmentation from Qwen's tokenizer.
    
    Qwen splits rare Vietnamese syllables into sub-tokens (e.g., trắng → tr + ắng).
    Sub-tokens without Vietnamese vowels get joined to adjacent words.
    
    Two-pass approach:
    1. Join left: fragments without vowels join to previous word
    2. Join right: remaining fragments without vowels join to next word
    """
    if not text:
        return text
    
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()
    if len(words) <= 1:
        return text
    
    # Vietnamese vowels (any character in a syllable can be a vowel)
    VN_VOWELS = set('aăâeêioôơuưyAĂÂEÊIOÔƠUƯY'
                     'àáảãạằắẳẵặầấẩẫậèéẻẽẹềếểễệìíỉĩịòóỏõọồốổỗộờớởỡợùúủũụừứửữựỳýỷỹỵ'
                     'ÀÁẢÃẠẰẮẲẴẶẦẤẨẪẬÈÉẺẼẸỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌỒỐỔỖỘỜỚỞỠỢÙÚỦŨỤỪỨỬỮỰỲÝỶỸỴ')
    
    def has_vowel(w):
        return any(c in VN_VOWELS for c in w)
    
    # Pass 1: join non-vowel fragments to PREVIOUS word (right-to-left)
    i = len(words) - 1
    while i > 0:
        if not has_vowel(words[i]) and has_vowel(words[i-1]):
            words[i-1] = words[i-1] + words[i]
            words.pop(i)
        i -= 1
    
    # Pass 2: join non-vowel fragments to NEXT word (left-to-right)  
    i = 0
    while i < len(words) - 1:
        if not has_vowel(words[i]) and has_vowel(words[i+1]):
            words[i+1] = words[i] + words[i+1]
            words.pop(i)
        else:
            i += 1
    
    return ' '.join(words)


# ═══════════════════════════════════════════════════════════════
# EVALUATION (same logic as eval_rules.py)
# ═══════════════════════════════════════════════════════════════

def evaluate_couplet(line6_in, line8_in, response_text):
    """Score all 5 Lục Bát rules on line→line generation."""
    in_luc_syls = line6_in.split()
    out_bat_syls = response_text.split()
    r_len = len(out_bat_syls)

    # R1: Vần lưng — input Lục[pos6] vs output Bát[pos6]
    r1_ok = False
    if len(in_luc_syls) >= 6 and len(out_bat_syls) >= 6:
        r1_ok = get_rhyme_group(in_luc_syls[5]) == get_rhyme_group(out_bat_syls[5])

    # R2: Bằng-Trắc on output Bát line (even positions: BTBB)
    r2_ok = 0; r2_total = 0
    for idx, want in [(1, 'B'), (3, 'T'), (5, 'B'), (7, 'B')]:
        if idx < len(out_bat_syls):
            r2_total += 1
            if get_tone(out_bat_syls[idx]) == want:
                r2_ok += 1

    # R3: Syllable count — output must be exactly 8 syllables
    r3_exact = (len(out_bat_syls) == 8)

    # R4: Trầm-Bổng — pos6 and pos8 of Bát must have opposite diacritics
    r4_ok = False
    if len(out_bat_syls) >= 8:
        d6 = get_diacritic(out_bat_syls[5])
        d8 = get_diacritic(out_bat_syls[7])
        r4_ok = d6 in ('ngang', 'huyen') and d8 in ('ngang', 'huyen') and d6 != d8

    # R5: Nhịp điệu — correct syllable count
    r5_ok = (len(out_bat_syls) == 8)

    # Quality
    lex_div = len(set(out_bat_syls)) / max(len(out_bat_syls), 1)
    all_5 = r1_ok and r3_exact and r2_ok == r2_total and r4_ok

    # BPE artifacts — with syllable tokens, count fragments that aren't valid syllables
    VIET_CHARS = set("aăâeêioôơuưyAĂÂEÊIOÔƠUƯY"
                      "àáảãạằắẳẵặầấẩẫậèéẻẽẹềếểễệìíỉĩịòóỏõọồốổỗộờớởỡợùúủũụừứửữựỳýỷỹỵ"
                      "ÀÁẢÃẠẰẮẲẴẶẦẤẨẪẬÈÉẺẼẸỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌỒỐỔỖỘỜỚỞỠỢÙÚỦŨỤỪỨỬỮỰỲÝỶỸỴ")
    bpe_count = sum(1 for s in out_bat_syls if len(s) < 2 or not any(c in VIET_CHARS for c in s))

    return {
        'prompt': f'{line6_in} / {line8_in[:20]}',
        'response': response_text,
        'out_bat': response_text,
        'r_len': r_len,
        'R1_ok': r1_ok, 'R2_r_ok': r2_ok, 'R2_r_total': r2_total,
        'R3_exact': r3_exact, 'R4_ok': r4_ok, 'R5_ok': r5_ok,
        'all_5': all_5, 'lex_div': lex_div, 'bpe_artifacts': bpe_count,
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


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    global TEMPERATURE, TOP_K, TOP_P, RHYME_LOGIT_BOOST
    
    parser = argparse.ArgumentParser(description="Evaluate Qwen QLoRA on Lục Bát rules (v5 FIXED)")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/qwen_stage1_best",
                        help="Path to LoRA adapter directory")
    parser.add_argument("--num", type=int, default=0,
                        help="Number of prompts to evaluate (0=all)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path")
    parser.add_argument("--cache-dir", type=str, default=None,
                        help="HuggingFace cache directory")
    parser.add_argument("--no-rhyme", action="store_true",
                        help="Disable soft rhyme constraint")
    parser.add_argument("--temperature", type=float, default=TEMPERATURE,
                        help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=TOP_K,
                        help="Top-k filtering")
    parser.add_argument("--top-p", type=float, default=TOP_P,
                        help="Top-p (nucleus) filtering")
    args = parser.parse_args()

    # Override globals from CLI
    TEMPERATURE = args.temperature
    TOP_K = args.top_k
    TOP_P = args.top_p
    
    if args.no_rhyme:
        RHYME_LOGIT_BOOST = 0.0

    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        print(f"❌  Checkpoint not found: {ckpt}")
        print("   Looking for: adapter_config.json in", ckpt.resolve())
        sys.exit(1)

    prompts = COUPLET_PROMPTS[:args.num] if args.num else COUPLET_PROMPTS

    model, tokenizer, dev = load_qwen_model(str(ckpt), cache_dir=args.cache_dir)
    print(f"   Device: {dev}")
    if dev == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Prompts: {len(prompts)}")
    print(f"   Temp={TEMPERATURE} TopK={TOP_K} TopP={TOP_P}")
    print(f"   Rhyme boost={RHYME_LOGIT_BOOST} SafetyThreshold={RHYME_SAFETY_THRESHOLD}")
    print(f"   RepPenalty={REPETITION_PENALTY} RepWindow={REPETITION_WINDOW}")

    # ── Evaluate ──
    print(f"\n{'='*60}")
    print(f"Qwen2.5-1.5B QLoRA — 5-Rule Lục Bát Evaluation (v5 FIXED)")
    print(f"Checkpoint: {ckpt.name}")
    print(f"{'='*60}\n")

    t0 = time.time()
    results = []

    for i, (l6, l8) in enumerate(tqdm(prompts, desc="Evaluating", unit="prompt")):
        prompt = build_qwen_prompt(l6, l8)
        response = generate_response(model, tokenizer, prompt)
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
        'R1': 90, 'R2': 95, 'R3': 100, 'R4': 90, 'R5': 100,
        'all5': 90, 'avg_lex': 0.90, 'avg_bpe': 2.0, 'empty': 5,
        'avg_len': 8,
    }

    print(f"\n{'='*60}")
    print(f"📊  RESULTS — Phase 1 Targets (roadmap_v5.md)")
    print(f"{'='*60}")
    print(f"{'Metric':<30} {'Value':>8} {'Target':>8} {'Status':>8}")
    print(f"{'-'*30} {'-'*8} {'-'*8} {'-'*8}")

    all_pass = True
    for key, label in [
        ('R1', 'R1 Rhyme (vần lưng)'),
        ('R2', 'R2 Tone (Bằng-Trắc)'),
        ('R3', 'R3 Syllable (exact 8)'),
        ('R4', 'R4 Trầm-Bổng'),
        ('R5', 'R5 Nhịp điệu (exact 8)'),
        ('all5', 'All 5 rules pass'),
    ]:
        v = s[key]
        t = TARGETS[key]
        ok = v >= t
        if not ok: all_pass = False
        tag = '✅' if ok else '❌'
        print(f"{label:<30} {v:>7.1f}% {t:>7.0f}% {tag:>8}")

    print(f"\n{'─'*30} {'─'*8} {'─'*8} {'─'*8}")
    for key, label in [
        ('avg_len', 'Average Bát length'),
        ('avg_lex', 'Lexical diversity'),
        ('avg_bpe', 'BPE artifact rate'),
        ('empty', 'Empty response rate'),
    ]:
        v = s[key]
        t = TARGETS[key]
        if key == 'avg_bpe':
            ok = v <= t
            sign = '<='
        elif key == 'empty':
            ok = v < t
            sign = '<'
        else:
            ok = v >= t
            sign = '>='
        if not ok: all_pass = False
        tag = '✅' if ok else '❌'
        print(f"{label:<30} {v:>7.1f} {sign}{t:>7.0f} {tag:>8}")

    print(f"\n{'='*60}")
    if all_pass:
        print("🎉  ALL TARGETS MET — v5 Phase 1 SUCCESS!")
    else:
        print("⚠️  Some targets not met — see ❌ above")
    print(f"   Elapsed: {elapsed:.0f}s | Prompts: {len(prompts)}")
    print(f"{'='*60}")

    # ── Samples ──
    print(f"\n📝  Sample outputs:")
    for i, r in enumerate(results[:10]):
        emoji = lambda ok: '✅' if ok else '❌'
        line6_in = r['prompt'].split(' / ')[0]
        print(f"\n  [{i+1}] Lục: {line6_in[:50]}")
        print(f"     → Bát: {r['out_bat'][:60]}")
        print(f"       {emoji(r['R1_ok'])}R1 {emoji(r['R2_r_ok']==r['R2_r_total'])}R2 "
              f"{emoji(r['R3_exact'])}R3 {emoji(r['R4_ok'])}R4")
    if len(results) > 10:
        print(f"\n  ... and {len(results)-10} more")

    # ── Save JSON ──
    out_path = Path(args.output or ROOT / "evaluate" / "qwen_evaluation.json")
    json.dump({
        'version': 'v5-fixed-stage1',
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
