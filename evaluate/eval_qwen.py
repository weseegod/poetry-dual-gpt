#!/usr/bin/env python3
"""
Qwen2.5-1.5B QLoRA — 5-rule Lục Bát evaluation.

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
from tqdm import tqdm

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.tones import (get_tone, get_rhyme_group, get_luc_bat_tags,
                        get_diacritic)
from evaluate.prompts import COUPLET_PROMPTS

MODEL_ID = "Qwen/Qwen2.5-1.5B"

# ═══════════════════════════════════════════════
# MODEL LOADING
# ═══════════════════════════════════════════════

def load_qwen_model(checkpoint_path: str):
    """Load Qwen base + LoRA adapter. Falls back to CPU if no GPU."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise RuntimeError("Set HF_TOKEN environment variable")

    has_gpu = torch.cuda.is_available()
    dev = "cuda" if has_gpu else "cpu"
    print(f"📦  Loading {MODEL_ID} ({'GPU' if has_gpu else 'CPU — will be slow'})...")

    kwargs = {"trust_remote_code": True, "token": hf_token}

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

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if not has_gpu:
        model = model.to(dev)

    print(f"   🎯  Loading LoRA adapter: {checkpoint_path}")
    model = PeftModel.from_pretrained(model, checkpoint_path)
    model.eval()

    return model, tokenizer, dev


# ═══════════════════════════════════════════════
# GENERATION
# ═══════════════════════════════════════════════

def build_qwen_prompt(line6: str, line8: str) -> str:
    """
    Build prompt in v5 training format:
    <|start|> [LUC_BAT] [RHYME:X] [TONE:BBTBBT] [TRAMBONG:NH]
      line6 line8 <|reply|>
    """
    couplet = f"{line6} {line8}"
    rhyme, tone, trambong = get_luc_bat_tags(couplet)
    tags = " ".join(t for t in [rhyme, tone, trambong] if t)
    return f"<|start|> [LUC_BAT] {tags} {couplet} <|reply|>"


@torch.no_grad()
def generate_response(model, tokenizer, prompt: str, max_new: int = 32) -> str:
    """Generate from Qwen + LoRA model."""
    dev = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(dev)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new,
        temperature=0.75,
        top_p=0.92,
        top_k=50,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.encode("<|end|>", add_special_tokens=False)[0]
            if "<|end|>" in tokenizer.get_vocab() else tokenizer.eos_token_id,
    )

    generated = outputs[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(generated, skip_special_tokens=True).strip()

    # Stop at <|end|> if present
    if "<|end|>" in text:
        text = text.split("<|end|>")[0].strip()

    # Split into two lines (6 + 8 syllables)
    parts = text.split()
    if len(parts) >= 14:
        return "  ".join([" ".join(parts[:6]), " ".join(parts[6:14])])
    elif len(parts) >= 8:
        return "  ".join([" ".join(parts[:6]), " ".join(parts[6:])])
    elif len(parts) >= 6:
        return "  ".join([" ".join(parts[:6]), " ".join(parts[6:])])
    return text


# ═══════════════════════════════════════════════
# EVALUATION (same logic as eval_rules.py)
# ═══════════════════════════════════════════════

def evaluate_couplet(line6_in, line8_in, response_text):
    """Score all 5 Lục Bát rules on couplet→couplet generation."""
    parts = response_text.split('  ')
    out_luc = parts[0].strip() if len(parts) > 0 else ''
    out_bat = parts[1].strip() if len(parts) > 1 else ''
    out_luc_syls = out_luc.split()
    out_bat_syls = out_bat.split()
    in_bat_syls = line8_in.split()
    r_len = len(out_luc_syls) + len(out_bat_syls)

    # R1: Vần lưng — input Bát[pos8] vs output Lục[pos6]
    r1_ok = False
    if len(in_bat_syls) >= 8 and len(out_luc_syls) >= 6:
        r1_ok = get_rhyme_group(in_bat_syls[7]) == get_rhyme_group(out_luc_syls[5])

    # R2: Bằng-Trắc on output Bát line
    r2_ok = 0; r2_total = 0
    for idx, want in [(1, 'B'), (3, 'T'), (5, 'B'), (7, 'B')]:
        if idx < len(out_bat_syls):
            r2_total += 1
            if get_tone(out_bat_syls[idx]) == want:
                r2_ok += 1

    # R3: Syllable count
    r3_exact = (len(out_luc_syls) == 6 and len(out_bat_syls) == 8)

    # R4: Trầm-Bổng
    r4_ok = False
    if len(out_bat_syls) >= 8:
        d6 = get_diacritic(out_bat_syls[5])
        d8 = get_diacritic(out_bat_syls[7])
        r4_ok = d6 in ('ngang', 'huyen') and d8 in ('ngang', 'huyen') and d6 != d8

    # R5: Nhịp điệu
    r5_ok = (len(out_luc_syls) == 6 and len(out_bat_syls) == 8)

    # Quality
    all_syls = out_luc_syls + out_bat_syls
    lex_div = len(set(all_syls)) / max(len(all_syls), 1)
    all_5 = r3_exact and r1_ok and r2_ok == r2_total and r4_ok

    # BPE artifacts (check for subword fragments in Qwen output)
    bpe_count = sum(1 for s in all_syls if len(s) < 2 or not any(
        c in "aăâeêioôơuưyAĂÂEÊIOÔƠUƯY" for c in s))

    return {
        'prompt': f'{line6_in} / {line8_in[:20]}',
        'response': response_text,
        'out_luc': out_luc, 'out_bat': out_bat,
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


# ═══════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Evaluate Qwen QLoRA on Lục Bát rules")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/qwen_stage1_best",
                        help="Path to LoRA adapter directory")
    parser.add_argument("--num", type=int, default=0,
                        help="Number of prompts to evaluate (0=all)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path")
    args = parser.parse_args()

    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        print(f"❌  Checkpoint not found: {ckpt}")
        print("   Looking for: adapter_config.json in", ckpt.resolve())
        sys.exit(1)

    prompts = COUPLET_PROMPTS[:args.num] if args.num else COUPLET_PROMPTS

    model, tokenizer, dev = load_qwen_model(str(ckpt))
    print(f"   Device: {dev}")
    if dev == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Prompts: {len(prompts)}")

    # ── Evaluate ──
    print(f"\n{'='*60}")
    print(f"Qwen2.5-1.5B QLoRA — 5-Rule Lục Bát Evaluation")
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
                  f"R3:{s['R3']:.0f}% R4:{s['R4']:.0f}% | All5:{s['all5']:.0f}%")

    s = summarize(results)
    elapsed = time.time() - t0

    # ── Report ──
    TARGETS = {
        'R1': 90, 'R2': 95, 'R3': 100, 'R4': 90, 'R5': 100,
        'all5': 90, 'avg_lex': 0.90, 'avg_bpe': 2.0, 'empty': 5,
    }

    print(f"\n{'='*60}")
    print(f"📊  RESULTS — Phase 1 Targets (roadmap_v5.md)")
    print(f"{'='*60}")
    print(f"{'Metric':<30} {'Value':>8} {'Target':>8} {'Status':>8}")
    print(f"{'-'*30} {'-'*8} {'-'*8} {'-'*8}")

    all_pass = True
    for key, label, fmt in [
        ('R1', 'R1 Rhyme (vần lưng)', '.0f%'),
        ('R2', 'R2 Tone (Bằng-Trắc)', '.0f%'),
        ('R3', 'R3 Syllable (6+8)', '.0f%'),
        ('R4', 'R4 Trầm-Bổng', '.0f%'),
        ('R5', 'R5 Nhịp điệu', '.0f%'),
        ('all5', 'All 5 rules pass', '.0f%'),
    ]:
        v = s[key]
        t = TARGETS[key]
        ok = v >= t
        if not ok: all_pass = False
        tag = '✅' if ok else '❌'
        print(f"{label:<30} {v:>7.1f}% {t:>7.0f}% {tag:>8}")

    print(f"\n{'─'*30} {'─'*8} {'─'*8} {'─'*8}")
    for key, label, fmt in [
        ('avg_lex', 'Lexical diversity', '.3f'),
        ('avg_bpe', 'BPE artifact rate', '.1f'),
        ('empty', 'Empty response rate', '.1f%'),
    ]:
        v = s[key]
        t = TARGETS[key]
        sign = '<=' if key == 'avg_bpe' else '<' if key == 'empty' else '>='
        if key == 'avg_bpe':
            ok = v <= t
        elif key == 'empty':
            ok = v < t
        else:
            ok = v >= t
        if not ok: all_pass = False
        tag = '✅' if ok else '❌'
        print(f"{label:<30} {v:>7}{fmt.split('f')[1].replace('>','')} {sign}{t:>7}{fmt.split('f')[1].replace('>','')} {tag:>8}")

    print(f"\n{'='*60}")
    if all_pass:
        print("🎉  ALL TARGETS MET — v5 Phase 1 SUCCESS!")
    else:
        print("⚠️  Some targets not met — see ❌ above")
    print(f"   Elapsed: {elapsed:.0f}s | Prompts: {len(prompts)}")
    print(f"{'='*60}")

    # ── Samples ──
    print(f"\n📝  Sample outputs:")
    for i, r in enumerate(results[:8]):
        emoji = lambda ok: '✅' if ok else '❌'
        print(f"\n  [{i+1}] {r['prompt'][:50]}")
        print(f"     → Lục: {r['out_luc'][:40]}")
        print(f"       Bát: {r['out_bat'][:40]}")
        print(f"       {emoji(r['R1_ok'])}R1 {emoji(r['R2_r_ok']==r['R2_r_total'])}R2 "
              f"{emoji(r['R3_exact'])}R3 {emoji(r['R4_ok'])}R4")
    if len(results) > 8:
        print(f"\n  ... and {len(results)-8} more")

    # ── Save JSON ──
    out_path = Path(args.output or ROOT / "evaluate" / "qwen_evaluation.json")
    json.dump({
        'version': 'v5-qwen-stage1',
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
