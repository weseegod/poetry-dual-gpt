#!/usr/bin/env python3
"""
v4.2 P9: Semantic Quality Evaluation Suite

Measures what structural metrics miss — lexical diversity, BPE artifacts,
repetition patterns, syllable validity. Complements eval_rules.py.

Usage:
  PYTHONPATH=. python3 evaluate/eval_quality.py [--checkpoint checkpoints/doi_tho_best.pt]
"""

import re, json, time, sys
import torch
from pathlib import Path
from tokenizers import Tokenizer

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from src.model import PoetryDuelGPT
from src.generation import (
    build_prompt, generate, decode_response,
    score_candidate, is_bpe_artifact, count_bpe_artifacts,
    get_valid_syllables,
)
from src.tones import get_tone, get_rhyme_group


# ── 20 diverse Lục Bát prompts for semantic evaluation ──
EVAL_PROMPTS = [
    ("thân em như chẽn lúa đòng", "phất phơ dưới ngọn nắng hồng ban mai"),
    ("công cha như núi thái sơn", "nghĩa mẹ như nước trong nguồn chảy ra"),
    ("một lòng thờ mẹ kính cha", "cho tròn chữ hiếu mới là đạo con"),
    ("trèo lên cây khế nửa ngày", "ai mang theo nắng đi đâu mất rồi"),
    ("ru con con ngủ cho lâu", "để mẹ đi cấy đồng sâu chưa về"),
    ("con cò bay lả bay la", "bay từ cửa phủ bay ra cánh đồng"),
    ("mẹ già như chuối ba hương", "như xôi nếp mật như đường mía lau"),
    ("cày đồng đang buổi ban trưa", "mồ hôi thánh thót như mưa ruộng cày"),
    ("ai về tôi gửi buồng cau", "buồng cau non mẹ để già lâu năm"),
    ("cây đa bến nước sân đình", "qua đình ngả nón trông đình xa xa"),
    ("nhiễu điều phủ lấy giá gương", "người trong một nước phải thương nhau cùng"),
    ("bầu ơi thương lấy bí cùng", "tuy rằng khác giống nhưng chung một giàn"),
    ("đất lành chim đậu về đây", "người hiền thì lại gặp may mắn nhiều"),
    ("đường vô xứ nghệ quanh quanh", "non xanh nước biếc như tranh họa đồ"),
    ("sen tàn cúc lại nở hoa", "sầu dài ngày ngắn sang đông lạnh lùng"),
    ("xa quê nhớ mẹ nhớ cha", "nhớ hàng cau trước sân nhà ngày xưa"),
    ("lời ru của mẹ ngày xưa", "theo con suốt cả chặng đường hôm mai"),
    ("thương người như thể thương thân", "ở hiền thì lại gặp lành ở hiền"),
    ("trâu ơi ta bảo trâu này", "trâu ra ngoài ruộng trâu cày với ta"),
    ("mưa từ xa tới mưa mau", "trời mưa trời gió đùng đùng sấm vang"),
]


def load_model(ckpt_path, dev):
    ckpt = torch.load(ckpt_path, map_location=dev, weights_only=False)
    cfg = ckpt['model_config'].copy()
    m = PoetryDuelGPT(ckpt['vocab_size'], **cfg)
    # Remap old checkpoint keys
    old = ckpt['model_state_dict']
    new_s = {}
    for k, v in old.items():
        nk = k.replace('qkv_proj', 'qkv').replace('out_proj', 'out')
        nk = nk.replace('causal_mask', 'mask')
        nk = nk.replace('.ffn.fc1.', '.ffn.net.0.').replace('.ffn.fc2.', '.ffn.net.2.')
        nk = {'token_embedding.weight': 'tok_emb.weight',
              'position_embedding.weight': 'pos_emb.weight',
              'ln_final.weight': 'ln_f.weight', 'ln_final.bias': 'ln_f.bias',
              'lm_head.weight': 'head.weight'}.get(nk, nk)
        new_s[nk] = v
    m.load_state_dict(new_s, strict=False)
    m.to(dev).eval()
    return m


def evaluate_semantic(model, tok, dev, prompts):
    """Run all 9 semantic quality metrics on the given prompts."""
    results = []
    for line6, line8 in prompts:
        couplet_input = f"{line6}\n{line8}"
        prompt = build_prompt(couplet_input, include_trambong=True)
        ids, text = generate(model, tok, prompt, device=dev, rhyme_mode="soft",
                            max_new=64, temperature=0.75, top_k=50, top_p=0.92)
        lines = decode_response(tok, ids, enforce_syllables=False)
        
        # Combine output lines
        if len(lines) >= 2:
            output = f"{lines[0]} {lines[1]}"
            out_luc = lines[0]
            out_bat = lines[1]
        elif len(lines) == 1:
            output = lines[0]
            out_luc = lines[0]
            out_bat = ""
        else:
            output = ""
            out_luc = ""
            out_bat = ""
        
        syls = output.split()
        n = len(syls)
        
        # Metrics
        r = {
            'input_luc': line6,
            'input_bat': line8,
            'output_luc': out_luc,
            'output_bat': out_bat,
            'total_syls': n,
        }
        
        # 1. Lexical diversity
        r['lex_div'] = len(set(syls)) / max(n, 1) if n > 0 else 0.0
        
        # 2. Adjacent repeat rate
        if n >= 2:
            r['adj_repeats'] = sum(1 for i in range(n-1) if syls[i] == syls[i+1])
        else:
            r['adj_repeats'] = 0
        
        # 3. Bigram novelty: % of bigrams not repeated in recent window
        if n >= 4:
            recent_bigrams = set()
            for i in range(max(0, n - 8), n - 1):
                recent_bigrams.add((syls[i], syls[i + 1]))
            novel = sum(1 for i in range(n - 1)
                       if (syls[i], syls[i + 1]) not in recent_bigrams)
            r['bigram_novelty'] = novel / max(n - 1, 1)
        else:
            r['bigram_novelty'] = 1.0
        
        # 4. BPE artifact rate
        r['bpe_artifacts'] = count_bpe_artifacts(syls)
        
        # 5. Syllable validity
        valid_set = get_valid_syllables()
        if n > 0:
            r['syl_validity'] = sum(1 for s in syls if s.lower() in valid_set) / n
        else:
            r['syl_validity'] = 0.0
        
        # 6. Output completeness (expect 14±2 syllables for 6+8 output)
        r['is_complete'] = 12 <= n <= 16
        
        # 7. Is empty/short (<4 syllables)?
        r['is_short'] = n < 4
        
        # 8. Candidate quality score (from generation.py scorer)
        r['quality_score'] = score_candidate(output)
        
        # 9. Structural rules (quick check) — passes if R3 (syllable count) is plausible
        out_luc_syls = len(out_luc.split())
        out_bat_syls = len(out_bat.split())
        r['syl_luc'] = out_luc_syls
        r['syl_bat'] = out_bat_syls
        r['syl_ok'] = (out_luc_syls == 6 and out_bat_syls == 8)
        
        results.append(r)
    
    return results


def summarize(results):
    """Compute summary statistics across all results."""
    n = len(results)
    avg = lambda key: sum(r[key] for r in results) / n
    
    return {
        'n_samples': n,
        'avg_lex_div': avg('lex_div'),
        'avg_adj_repeats': avg('adj_repeats'),
        'adj_repeat_rate': sum(r['adj_repeats'] for r in results) / max(sum(r['total_syls'] for r in results), 1) * 100,
        'avg_bigram_novelty': avg('bigram_novelty'),
        'total_bpe_artifacts': sum(r['bpe_artifacts'] for r in results),
        'bpe_artifact_rate': sum(r['bpe_artifacts'] for r in results) / max(sum(r['total_syls'] for r in results), 1) * 100,
        'avg_syl_validity': avg('syl_validity'),
        'completeness': sum(r['is_complete'] for r in results) / n * 100,
        'short_rate': sum(r['is_short'] for r in results) / n * 100,
        'avg_quality_score': avg('quality_score'),
        'syl_accuracy': sum(r['syl_ok'] for r in results) / n * 100,
    }


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    
    dev = args.device
    print(f"Device: {dev}")
    
    tok = Tokenizer.from_file(str(ROOT / "tokenizer" / "poetry_bpe.model"))
    
    # Find checkpoint
    ckpt_path = None
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
    else:
        for name in ['doi_tho_best.pt', 'final.pt', 'best.pt']:
            p = ROOT / "checkpoints" / name
            if p.exists():
                ckpt_path = p
                break
    
    if not ckpt_path or not ckpt_path.exists():
        print("No checkpoint found. Use --checkpoint PATH")
        return
    
    print(f"\n{'='*60}")
    print(f"v4.2 — Semantic Quality Evaluation (P9)")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Prompts: {len(EVAL_PROMPTS)}")
    print(f"{'='*60}\n")
    
    model = load_model(str(ckpt_path), dev)
    t0 = time.time()
    
    results = evaluate_semantic(model, tok, dev, EVAL_PROMPTS)
    summary = summarize(results)
    
    elapsed = time.time() - t0
    
    # ── Print report ──
    print(f"{'─'*60}")
    print(f"📊 Semantic Quality Summary")
    print(f"{'─'*60}")
    print(f"  Lexical diversity:      {summary['avg_lex_div']:.3f}  (>0.75 = good)")
    print(f"  Adjacent repeat rate:   {summary['adj_repeat_rate']:.1f}%  (<3% = good)")
    print(f"  Bigram novelty:         {summary['avg_bigram_novelty']:.3f}  (>0.80 = good)")
    print(f"  BPE artifact rate:      {summary['bpe_artifact_rate']:.1f}%  (<5% = good)")
    print(f"  Syllable validity:      {summary['avg_syl_validity']:.3f}  (>0.95 = good)")
    print(f"  Output completeness:    {summary['completeness']:.0f}%  (>90% = good)")
    print(f"  Empty/short rate:       {summary['short_rate']:.0f}%  (0% = good)")
    print(f"  Syllable accuracy:      {summary['syl_accuracy']:.0f}%  (>90% = good)")
    print(f"  Quality score (avg):    {summary['avg_quality_score']:+.2f}")
    print(f"{'─'*60}")
    
    # ── Per-sample output ──
    print(f"\n{'─'*60}")
    print(f"📝 Per-Sample Output")
    print(f"{'─'*60}")
    for i, r in enumerate(results):
        qs = r['quality_score']
        emoji = "✅" if qs > 1.0 else "⚠️" if qs > -1.0 else "❌"
        print(f"\n  [{i+1:2d}] {emoji} QS={qs:+.2f} | lex={r['lex_div']:.2f} | "
              f"BPE={r['bpe_artifacts']} | rep={r['adj_repeats']} | "
              f"syl={r['syl_luc']}+{r['syl_bat']}")
        print(f"       In:  {r['input_luc']} / {r['input_bat'][:30]}...")
        print(f"       Out: {r['output_luc']}")
        print(f"            {r['output_bat']}")
    
    print(f"\n{'─'*60}")
    print(f"⏱️  Completed in {elapsed:.0f}s")
    print(f"{'─'*60}")
    
    # ── Save JSON ──
    out_path = ROOT / "evaluate" / "quality_evaluation.json"
    json.dump({
        'checkpoint': str(ckpt_path.name),
        'summary': {k: (round(v, 4) if isinstance(v, float) else v) for k, v in summary.items()},
        'samples': [{k: v for k, v in r.items() if k not in ('input_luc', 'input_bat')} for r in results],
    }, open(out_path, 'w'), indent=2, ensure_ascii=False)
    print(f"📄 JSON → {out_path}")


if __name__ == '__main__':
    main()
