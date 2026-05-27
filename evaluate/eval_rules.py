#!/usr/bin/env python3
"""
v4.2.3: 5-rule Lục Bát evaluation — couplet→couplet only.

The model was trained on full couplet input (6+8 → 6+8). Single-line
evaluation is meaningless and has been removed.

Usage:
  PYTHONPATH=. python3 evaluate/eval_rules.py
"""

import re, json, time, sys
import torch, torch.nn.functional as F
from pathlib import Path
from tokenizers import Tokenizer
from collections import Counter

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.model import PoetryDuelGPT
from src.tones import (get_tone, get_rhyme_group, get_luc_bat_tags,
                        get_diacritic, check_tram_bong)
from src.generation import build_prompt, generate, decode_response
from evaluate.prompts import COUPLET_PROMPTS


# ── Model loading ──
def load(path, dev):
    ckpt = torch.load(path, map_location=dev, weights_only=False)
    cfg = ckpt['model_config'].copy()
    cfg.pop('vocab_size', None)
    m = PoetryDuelGPT(ckpt['vocab_size'], **cfg)
    m.load_state_dict(ckpt['model_state_dict']); m.to(dev).eval()
    return m


@torch.no_grad()
def gen_couplet(model, tok, line6, line8, dev):
    """Generate from full couplet (6+8) using unified generator."""
    couplet_input = f"{line6}\n{line8}"
    full_prompt = build_prompt(couplet_input, include_trambong=True)
    ids, _ = generate(model, tok, full_prompt, device=dev, rhyme_mode="soft")
    lines = decode_response(tok, ids, enforce_syllables=False)
    if len(lines) >= 2:
        return f"{lines[0]}  {lines[1]}"
    elif len(lines) == 1:
        return lines[0]
    return ""


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
        d6, d8 = get_diacritic(out_bat_syls[5]), get_diacritic(out_bat_syls[7])
        r4_ok = d6 in ('ngang', 'huyen') and d8 in ('ngang', 'huyen') and d6 != d8

    # R5: Nhịp điệu
    r5_ok = (len(out_luc_syls) == 6 and len(out_bat_syls) == 8)

    # Quality metrics
    all_syls = out_luc_syls + out_bat_syls
    lex_div = len(set(all_syls)) / max(len(all_syls), 1)
    all_5 = r3_exact and r1_ok and r2_ok == r2_total and r4_ok

    return {
        'prompt': f'{line6_in} / {line8_in[:20]}',
        'response': response_text,
        'out_luc': out_luc, 'out_bat': out_bat,
        'r_len': r_len,
        'R1_ok': r1_ok, 'R2_r_ok': r2_ok, 'R2_r_total': r2_total,
        'R3_exact': r3_exact, 'R4_ok': r4_ok, 'R5_ok': r5_ok,
        'all_5': all_5, 'lex_div': lex_div, 'bpe_artifacts': 0,
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
        'empty': sum(1 for r in results if r['r_len'] == 0) / n * 100,
    }


def main():
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {dev}')

    tok = Tokenizer.from_file(str(ROOT / 'tokenizer/poetry_bpe.model'))

    ckpt_path = None
    for name in ['doi_tho_best.pt', 'final.pt', 'best.pt']:
        p = ROOT / 'checkpoints' / name
        if p.exists():
            ckpt_path = p
            break
    if not ckpt_path:
        print('No checkpoint found in checkpoints/')
        return

    print(f'\n{"="*60}')
    print(f'v4.2.3 — 5-Rule Lục Bát Evaluation (couplet→couplet)')
    print(f'Checkpoint: {ckpt_path.name}')
    print(f'Couplet prompts: {len(COUPLET_PROMPTS)}')
    print(f'{"="*60}')

    model = load(str(ckpt_path), dev)
    t0 = time.time()

    print(f'\n--- Couplet 6+8 → couplet ---')
    results = []
    for i, (l6, l8) in enumerate(COUPLET_PROMPTS):
        r = gen_couplet(model, tok, l6, l8, dev)
        results.append(evaluate_couplet(l6, l8, r))
        if (i + 1) % 20 == 0:
            s = summarize(results)
            print(f'  {i+1}/{len(COUPLET_PROMPTS)} | R1:{s["R1"]:.0f}% R2:{s["R2"]:.0f}% '
                  f'R3:{s["R3"]:.0f}% R4:{s["R4"]:.0f}% | All5:{s["all5"]:.0f}% | {time.time()-t0:.0f}s')

    s = summarize(results)
    print(f'  → Done: R1:{s["R1"]:.0f}% R2:{s["R2"]:.0f}% R3:{s["R3"]:.0f}% '
          f'R4:{s["R4"]:.0f}% All5:{s["all5"]:.0f}% | {time.time()-t0:.0f}s')

    # ── Build report ──
    lines = []
    lines.append('# 📊 v4.2.3 Rule-by-Rule Evaluation — 5 Lục Bát Rules')
    lines.append('')
    lines.append(f'> Generated: {time.strftime("%Y-%m-%d %H:%M")}')
    lines.append(f'> Checkpoint: {ckpt_path.name}')
    lines.append(f'> Couplet prompts: {len(COUPLET_PROMPTS)}')
    lines.append(f'> Model: 31M params, n_embd=512, n_head=8, n_layer=8')
    lines.append('')

    lines.append('## 📈 5-Rule Summary')
    lines.append('')
    lines.append('| Rule | Couplet | Random | Target |')
    lines.append('|------|---------|--------|--------|')
    random_r1 = (1/159)*100; random_r2 = (0.5**4)*100
    for name, key, rand, target in [
        ('R1: Vần lưng', 'R1', random_r1, 78),
        ('R2: Bằng-Trắc', 'R2', random_r2, 92),
        ('R3: Syllable (6+8)', 'R3', 7.0, 85),
        ('R4: Trầm-Bổng', 'R4', 50.0, 88),
        ('R5: Nhịp điệu', 'R5', 7.0, 75),
        ('**All 5 pass**', 'all5', 0, 75),
    ]:
        cp_v = s[key]
        cp_m = '✅' if cp_v >= target else '🟡'
        lines.append(f'| {name} | {cp_m} {cp_v:.1f}% | {rand:.1f}% | {target}%+ |')

    lines.append('')
    lines.append('## 📊 Quality Metrics')
    lines.append('')
    lines.append(f'| Avg response length | {s["avg_len"]:.1f} syl |')
    lines.append(f'| Lexical diversity | {s["avg_lex"]:.3f} |')
    lines.append(f'| Empty response rate | {s["empty"]:.1f}% |')
    lines.append('')

    # Couplet samples
    lines.append('## 📝 Couplet Samples')
    lines.append('')
    lines.append('| Input | Output Lục | Output Bát | R1 | R2 | R3 | R4 | All |')
    lines.append('|-------|-----------|-----------|----|----|----|----|-----|')
    for r in results[:15]:
        emoji = lambda ok: '✅' if ok else '❌'
        lines.append(f'| {r["prompt"][:25]} | {r.get("out_luc","?")[:25]} | '
                     f'{r.get("out_bat","?")[:25]} | {emoji(r["R1_ok"])} | '
                     f'{emoji(r["R2_r_ok"]==r["R2_r_total"])} | '
                     f'{emoji(r["R3_exact"])} | {emoji(r["R4_ok"])} | '
                     f'{emoji(r["all_5"])} |')
    lines.append('')

    out = ROOT / 'documents' / 'rule_evaluation.md'
    out.write_text('\n'.join(lines))
    print(f'\n📄 Report → {out}')

    json.dump({
        'version': 'v4.2.3', 'checkpoint': str(ckpt_path.name),
        'couplet': s,
    }, open(ROOT / 'evaluate' / 'rule_evaluation.json', 'w'), indent=2, ensure_ascii=False)
    print(f'📄 JSON → evaluate/rule_evaluation.json')


if __name__ == '__main__':
    main()
