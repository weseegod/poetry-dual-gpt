"""
Evaluate Stage 1 vs Stage 2 models on Lục Bát poetry generation.
Measures: syllable accuracy, tone correctness, rhyme accuracy.
Saves report to documents/stage_comparison.md
"""

import re, json, time, sys
import torch
from pathlib import Path
from tokenizers import Tokenizer

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.model import PoetryDuelGPT
from src.tones import get_tone, get_rhyme_group

ROOT = Path(__file__).parent.parent

PROMPTS = [
    "Trăm năm trong cõi người ta",
    "Thân em như chẽn lúa đòng",
    "Gió đưa cành trúc la đà",
    "Anh đi anh nhớ quê nhà",
    "Trèo lên cây bưởi hái hoa",
    "Đêm qua em những mơ màng",
    "Chiều chiều ra đứng ngõ sau",
    "Ai về ai có nhớ không",
    "Núi cao chi lắm núi ơi",
    "Con cò bay lả bay la",
    "Qua cầu ngả nón trông cầu",
    "Hỡi cô tát nước bên đàng",
    "Người về em những trông theo",
    "Bây giờ mận mới hỏi đào",
    "Trúc xinh trúc mọc đầu đình",
    "Đất Quảng Nam chưa mưa đà thấm",
    "Công anh bắt cá dưới ao",
    "Đêm nằm lưng chẳng tới giường",
    "Lên non mới biết non cao",
    "Ngó lên nuộc lạt mái nhà",
]


def load_model(ckpt_path, device="cpu"):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["model_config"].copy()
    cfg.pop("vocab_size", None)
    m = PoetryDuelGPT(ckpt["vocab_size"], **cfg)
    m.load_state_dict(ckpt["model_state_dict"])
    m.to(device).eval()
    return m


def auto_tag(prompt):
    from src.tones import get_luc_bat_tags
    p = prompt.strip()
    if p.startswith("["):
        return p
    rhyme, tone = get_luc_bat_tags(p)
    extras = f"{rhyme} {tone}".strip()
    return f"[LUC_BAT] {extras} {p}" if extras else f"[LUC_BAT] {p}"


@torch.no_grad()
def generate(model, tokenizer, prompt, max_new=64, temperature=0.75, top_k=50, device="cpu"):
    end_id = tokenizer.token_to_id("<|end|>")
    pad_id = tokenizer.token_to_id("<|pad|>")
    ids = tokenizer.encode(prompt).ids
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    new_tokens = []
    for _ in range(max_new):
        logits, _ = model(idx[:, -model.block_size:])
        logits = logits[:, -1, :] / temperature
        logits[:, pad_id] = float("-inf")
        if top_k:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, -1:]] = float("-inf")
        next_id = torch.multinomial(torch.nn.functional.softmax(logits, dim=-1), 1).item()
        if next_id == end_id:
            break
        new_tokens.append(next_id)
        idx = torch.cat((idx, torch.tensor([[next_id]], device=device)), dim=1)
    return tokenizer.decode(new_tokens)


def clean(text):
    return text.replace("<|end|>", "").replace("<|reply|>", "").strip(",.-;:!? ")


def eval_one(prompt_text, response_text):
    """Score one generation."""
    p_syls = prompt_text.split()
    r_syls_all = response_text.split()
    r_syls = r_syls_all[:8]  # Analyze first 8 syllables
    p_len = len(p_syls)
    r_len = len(r_syls_all)

    # Syllable count
    syl_exact = r_len == 8
    syl_close = 6 <= r_len <= 10

    # Prompt tones: pos 2=B, 4=T, 6=B
    p_tone_ok = 0
    for i, w in [(1, "B"), (3, "T"), (5, "B")]:
        if i < p_len and get_tone(p_syls[i]) == w:
            p_tone_ok += 1

    # Response tones (first 8): pos 2=B, 4=T, 6=B, 8=B
    r_tone_ok = 0
    r_tone_possible = 0
    for i, w in [(1, "B"), (3, "T"), (5, "B"), (7, "B")]:
        if i < len(r_syls):
            r_tone_possible += 1
            if get_tone(r_syls[i]) == w:
                r_tone_ok += 1

    # Rhyme: response pos 6 matches prompt pos 6
    rhyme_ok = False
    p_rhyme = None
    r_rhyme = None
    if p_len >= 6 and len(r_syls) >= 6:
        p_rhyme = get_rhyme_group(p_syls[5])
        r_rhyme = get_rhyme_group(r_syls[5])
        rhyme_ok = p_rhyme == r_rhyme

    return {
        "prompt": prompt_text,
        "response": response_text,
        "r_len": r_len,
        "syl_exact": syl_exact,
        "syl_close": syl_close,
        "p_tone_ok": p_tone_ok,
        "r_tone_ok": r_tone_ok,
        "r_tone_possible": r_tone_possible,
        "rhyme_ok": rhyme_ok,
        "p_rhyme": p_rhyme,
        "r_rhyme": r_rhyme,
    }


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {dev}")

    tok = Tokenizer.from_file(str(ROOT / "tokenizer/poetry_bpe.model"))
    s1_model = load_model(str(ROOT / "checkpoints/stage1_best.pt"), dev)
    s2_model = load_model(str(ROOT / "checkpoints/final.pt"), dev)

    def run_eval(model, name):
        results = []
        print(f"\n=== {name} ===")
        for p in PROMPTS:
            tagged = auto_tag(p)
            for _ in range(3):
                out = generate(model, tok, tagged, device=dev)
                resp = clean(out)
                r = eval_one(p, resp)
                results.append(r)
            avg_len = sum(r["r_len"] for r in results[-3:]) / 3
            rhyme_pct = sum(r["rhyme_ok"] for r in results[-3:]) / 3 * 100
            tone_pct = sum(r["r_tone_ok"] for r in results[-3:]) / max(sum(r["r_tone_possible"] for r in results[-3:]), 1) * 100
            print(f"  {p[:30]:30s} → avg {avg_len:.1f}syl  rhyme={rhyme_pct:.0f}%  tone={tone_pct:.0f}%  \"{results[-1]['response'][:40]}\"")
        return results

    s1 = run_eval(s1_model, "Stage 1 (all genres)")
    s2 = run_eval(s2_model, "Stage 2 (Lục Bát)")

    # ── Aggregates ──
    def agg(r):
        n = len(r)
        return {
            "n": n,
            "syl_exact_8": sum(x["syl_exact"] for x in r) / n * 100,
            "syl_close_6_10": sum(x["syl_close"] for x in r) / n * 100,
            "avg_len": sum(x["r_len"] for x in r) / n,
            "prompt_tone": sum(x["p_tone_ok"] for x in r) / (n * 3) * 100,
            "resp_tone": sum(x["r_tone_ok"] for x in r) / max(sum(x["r_tone_possible"] for x in r), 1) * 100,
            "rhyme": sum(x["rhyme_ok"] for x in r) / n * 100,
        }

    a1 = agg(s1)
    a2 = agg(s2)

    # ── Print summary ──
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    metrics = [
        ("Syllable: exact 8", "syl_exact_8", "higher"),
        ("Syllable: 6-10 range", "syl_close_6_10", "higher"),
        ("Avg response length", "avg_len", "closer_to_8"),
        ("Prompt tone (B-T-B)", "prompt_tone", "higher"),
        ("Response tone (B-T-B-B)", "resp_tone", "higher"),
        ("Rhyme (vần lưng)", "rhyme", "higher"),
    ]
    print(f"{'Metric':<35s} {'Stage 1':>8s} {'Stage 2':>8s} {'Winner':>12s}")
    print("-" * 70)
    for name, key, better in metrics:
        v1 = a1[key]; v2 = a2[key]
        if better == "higher":
            w = "Stage 1 🥇" if v1 > v2 else "Stage 2 🥇" if v2 > v1 else "Tie 🤝"
        else:
            d1, d2 = abs(v1 - 8), abs(v2 - 8)
            w = "Stage 1 🥇" if d1 < d2 else "Stage 2 🥇" if d2 < d1 else "Tie 🤝"
        print(f"{name:<35s} {v1:7.1f}% {v2:7.1f}% {w:>12s}")

    # ── Generate report ──
    lines = []
    lines.append("# 📊 Stage 1 vs Stage 2 — Poetry Evaluation Report")
    lines.append("")
    lines.append(f"> Generated: {time.strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"> 20 prompts × 3 samples = 60 per model")
    lines.append(f"> Model: 30.9M params, n_embd=512, n_head=8, n_layer=8")
    lines.append("")
    lines.append("## 📈 Summary")
    lines.append("")
    lines.append(f"| Metric | Stage 1 | Stage 2 | Winner |")
    lines.append(f"|--------|---------|---------|--------|")
    for name, key, better in metrics:
        v1 = a1[key]; v2 = a2[key]
        if better == "higher":
            w = "Stage 1" if v1 > v2 else "Stage 2" if v2 > v1 else "Tie"
        else:
            d1, d2 = abs(v1 - 8), abs(v2 - 8)
            w = "Stage 1" if d1 < d2 else "Stage 2" if d2 < d1 else "Tie"
        lines.append(f"| {name} | {v1:.1f}% | {v2:.1f}% | {w} |")

    lines.append("")
    lines.append("## 🎭 Per-Prompt Comparison")
    lines.append("")
    lines.append("| Prompt | S1 Syll | S2 Syll | S1 Rhyme | S2 Rhyme | S1 Tone | S2 Tone | Best S1 | Best S2 |")
    lines.append("|--------|---------|---------|----------|----------|---------|---------|---------|---------|")

    for i, p in enumerate(PROMPTS):
        s1r = [r for r in s1 if r["prompt"] == p]
        s2r = [r for r in s2 if r["prompt"] == p]
        s1_syl = sum(r["syl_exact"] for r in s1r) / 3 * 100
        s2_syl = sum(r["syl_exact"] for r in s2r) / 3 * 100
        s1_rh = sum(r["rhyme_ok"] for r in s1r) / 3 * 100
        s2_rh = sum(r["rhyme_ok"] for r in s2r) / 3 * 100
        s1_to = sum(r["r_tone_ok"] for r in s1r) / max(sum(r["r_tone_possible"] for r in s1r), 1) * 100
        s2_to = sum(r["r_tone_ok"] for r in s2r) / max(sum(r["r_tone_possible"] for r in s2r), 1) * 100
        best_s1 = min(s1r, key=lambda x: abs(x["r_len"] - 8))["response"][:50]
        best_s2 = min(s2r, key=lambda x: abs(x["r_len"] - 8))["response"][:50]
        lines.append(f"| {p[:30]} | {s1_syl:.0f}% | {s2_syl:.0f}% | {s1_rh:.0f}% | {s2_rh:.0f}% | {s1_to:.0f}% | {s2_to:.0f}% | {best_s1} | {best_s2} |")

    lines.append("")
    lines.append("## 🧪 Methodology")
    lines.append("")
    lines.append("- 20 Lục Bát prompts × 3 samples = 60 generations per model")
    lines.append("- temperature=0.75, top_k=50")
    lines.append("- Syllable: response first 8 syllables checked for exact 8-syl count")
    lines.append("- Tone: prompt pos 2,4,6 = B-T-B; response pos 2,4,6,8 = B-T-B-B")
    lines.append("- Rhyme: response syllable 6 rhyme group matches prompt syllable 6")
    lines.append("- Note: model often generates >8 syllables (multi-line). Analysis focuses on first 8.")
    lines.append("")

    out = ROOT / "documents" / "stage_comparison.md"
    out.write_text("\n".join(lines))
    print(f"\n📄 Report → {out}")

    # JSON
    json_out = ROOT / "documents" / "stage_comparison.json"
    json_out.write_text(json.dumps({"stage1": a1, "stage2": a2, "s1_samples": [{"p": r["prompt"], "r": r["response"]} for r in s1], "s2_samples": [{"p": r["prompt"], "r": r["response"]} for r in s2]}, indent=2, ensure_ascii=False))
    print(f"📄 JSON → {json_out}")


if __name__ == "__main__":
    main()
