"""
Stress-test đối thơ across all input lengths: 1-line, 2-line, 4-line, 6-line.
Tests the [DOI_THO] checkpoint for correctness at every input size.

Usage:
  python evaluate/eval_doi_tho.py
  python evaluate/eval_doi_tho.py --checkpoint checkpoints/doi_tho_best.pt
  python evaluate/eval_doi_tho.py --device cuda
"""

import sys, time, argparse
import torch, torch.nn.functional as F
from pathlib import Path
from tokenizers import Tokenizer

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.model import PoetryDuelGPT
from src.tones import get_rhyme_group, get_tone_sequence, get_doi_tho_tags


# ── Test prompts at every input length ──
PROMPTS_1LINE = [
    "Thân em như chẽn lúa đòng",
    "Trèo lên cây khế nửa ngày",
    "Đêm khuya thắp ngọn đèn dầu",
    "Công cha như núi thái sơn",
    "Gió đưa cành trúc la đà",
]

PROMPTS_2LINE = [
    ("Thân em như chẽn lúa đòng", "Phất phơ dưới ngọn nắng hồng ban mai"),
    ("Trèo lên cây khế nửa ngày", "Ai làm cho khế rụng đầy vườn ai"),
    ("Qua đình ngả nón trông đình", "Đình bao nhiêu ngói thương mình bấy nhiêu"),
    ("Cày đồng đang buổi ban trưa", "Mồ hôi thánh thót như mưa ruộng cày"),
    ("Sông sâu còn có kẻ đò", "Đường xa còn có người qua đón chờ"),
]

PROMPTS_4LINE = [
    ("Trăm năm trong cõi người ta", "Chữ tài chữ mệnh khéo là ghét nhau",
     "Trải qua một cuộc bể dâu", "Những điều trông thấy mà đau đớn lòng"),
    ("Gió đưa cành trúc la đà", "Tiếng chuông Trấn Vũ canh gà Thọ Xương",
     "Mịt mù khói tỏa ngàn sương", "Nhịp chày Yên Thái mặt gương Tây Hồ"),
    ("Đường vô xứ nghệ quanh quanh", "Non xanh nước biếc như tranh họa đồ",
     "Ai vô xứ nghệ thì vô", "Đường vô xứ nghệ quanh co khúc khều"),
]

PROMPTS_6LINE = [
    ("Trăm năm trong cõi người ta", "Chữ tài chữ mệnh khéo là ghét nhau",
     "Trải qua một cuộc bể dâu", "Những điều trông thấy mà đau đớn lòng",
     "Lạ gì bỉ sắc tư phong", "Trời xanh quen thói má hồng đánh ghen"),
]


def decode_doi_tho(tokenizer, token_ids):
    """Split token list at <|linebreak|> positions (id=9 decodes to empty)."""
    lb_id = tokenizer.token_to_id("<|linebreak|>")
    lines = []
    chunk = []
    for t in token_ids:
        if t == lb_id:
            if chunk:
                lines.append(tokenizer.decode(chunk).strip())
            chunk = []
        else:
            chunk.append(t)
    if chunk:
        lines.append(tokenizer.decode(chunk).strip())
    return lines


def load_model(ckpt_path, device="cpu"):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["model_config"]
    m = PoetryDuelGPT(
        ckpt["vocab_size"],
        n_embd=cfg["n_embd"],
        n_head=cfg["n_head"],
        n_layer=cfg["n_layer"],
        block_size=cfg["block_size"],
        dropout=cfg.get("dropout", 0.1),
    )
    m.load_state_dict(ckpt["model_state_dict"])
    m.to(device).eval()
    return m, ckpt["step"]


def auto_tag_doi_tho(user_input: str, max_context: int = 1) -> str:
    """Wrap input as [DOI_THO] format. Handles 1-N lines."""
    lines = [l.strip() for l in user_input.strip().split('\n') if l.strip()]
    
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
    
    couplets = []
    i = 0
    while i + 1 < len(lines):
        s1, s2 = len(lines[i].split()), len(lines[i+1].split())
        if s1 == 6 and s2 == 8:
            couplets.append((lines[i], lines[i+1]))
            i += 2
        else:
            i += 1
    
    if not couplets:
        return auto_tag_doi_tho(lines[-1])
    
    couplets = couplets[-max_context:]
    last_6, last_8 = couplets[-1]
    rhyme_tag, tone_tag = get_doi_tho_tags(last_6, last_8)
    
    input_lines = []
    for six, eight in couplets:
        input_lines.append(six)
        input_lines.append(eight)
    input_str = " <|linebreak|> ".join(input_lines)
    
    tags = f"{rhyme_tag} {tone_tag}".strip()
    tag_part = f"[DOI_THO] {tags}" if tags else "[DOI_THO]"
    return f"<|start|> {tag_part} {input_str} <|reply|>"


@torch.no_grad()
def generate(model, tokenizer, prompt, device="cpu", max_new=80):
    end_id = tokenizer.token_to_id("<|end|>")
    pad_id = tokenizer.token_to_id("<|pad|>")
    
    ids = tokenizer.encode(prompt).ids
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    
    new_tokens = []
    for _ in range(max_new):
        logits, _ = model(idx[:, -model.block_size:])
        logits = logits[:, -1, :] / 0.75
        logits[:, pad_id] = float("-inf")
        v, _ = torch.topk(logits, min(50, logits.size(-1)))
        logits[logits < v[:, -1:]] = float("-inf")
        next_id = torch.multinomial(F.softmax(logits, dim=-1), 1).item()
        if next_id == end_id:
            break
        new_tokens.append(next_id)
        idx = torch.cat([idx, torch.tensor([[next_id]], device=device)], dim=1)
    
    return new_tokens


def score_output(out_lines, max_expected_lines):
    """Score generated output for correctness."""
    if not out_lines:
        return {"valid": False, "issue": "empty"}
    
    # Count garbled tokens (ByteLevel BPE artifacts)
    garbled = 0
    for l in out_lines:
        for ch in l:
            if ord(ch) < 32 and ch not in '\n\t':
                garbled += 1
    
    if garbled > 0:
        return {"valid": False, "issue": f"garbled ({garbled} control chars)", "lines": out_lines}
    
    # Check each line has reasonable syllable count (1-12)
    for l in out_lines:
        syls = len(l.split())
        if syls < 1 or syls > 15:
            return {"valid": False, "issue": f"bad syllable count ({syls})", "lines": out_lines}
    
    # Check output isn't too long (> 2x expected)
    if len(out_lines) > max_expected_lines * 2:
        return {"valid": False, "issue": f"too many lines ({len(out_lines)} > {max_expected_lines*2})", "lines": out_lines}
    
    return {"valid": True, "lines": out_lines}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/doi_tho_best.pt")
    p.add_argument("--tokenizer", default="tokenizer/poetry_bpe.model")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    
    dev = args.device
    tok = Tokenizer.from_file(str(ROOT / args.tokenizer))
    model, step = load_model(str(ROOT / args.checkpoint), dev)
    print(f"Model: step {step}, vocab={tok.get_vocab_size():,}\n")
    
    all_results = {}
    t0 = time.time()
    
    # ── 1-line tests ──
    print("=" * 60)
    print("📝  1-LINE INPUT (single line)")
    print("=" * 60)
    results_1 = []
    for text in PROMPTS_1LINE:
        prompt = auto_tag_doi_tho(text)
        tokens = generate(model, tok, prompt, dev)
        out_lines = decode_doi_tho(tok, tokens)
        score = score_output(out_lines, max_expected_lines=1)
        results_1.append(score)
        icon = "✅" if score["valid"] else "❌"
        print(f"  {icon} {text[:40]}...")
        if score["valid"]:
            for l in out_lines:
                print(f"     → {l}")
        else:
            print(f"     {score['issue']}: {out_lines}")
    all_results["1-line"] = results_1
    
    # ── 2-line tests ──
    print(f"\n{'='*60}")
    print("📝  2-LINE INPUT (1 couplet)")
    print("=" * 60)
    results_2 = []
    for in6, in8 in PROMPTS_2LINE:
        prompt = auto_tag_doi_tho(f"{in6}\n{in8}")
        tokens = generate(model, tok, prompt, dev)
        out_lines = decode_doi_tho(tok, tokens)
        score = score_output(out_lines, max_expected_lines=2)
        results_2.append(score)
        icon = "✅" if score["valid"] else "❌"
        print(f"  {icon} {in6[:40]}...")
        if score["valid"]:
            for l in out_lines:
                print(f"     → {l}")
        else:
            print(f"     {score['issue']}: {out_lines}")
    all_results["2-line"] = results_2
    
    # ── 4-line tests ──
    print(f"\n{'='*60}")
    print("📝  4-LINE INPUT (2 couplets — truncated to last 1)")
    print("=" * 60)
    results_4 = []
    for lines in PROMPTS_4LINE:
        text = "\n".join(lines)
        prompt = auto_tag_doi_tho(text, max_context=1)
        tokens = generate(model, tok, prompt, dev)
        out_lines = decode_doi_tho(tok, tokens)
        score = score_output(out_lines, max_expected_lines=2)
        results_4.append(score)
        icon = "✅" if score["valid"] else "❌"
        print(f"  {icon} ...{lines[-2][:30]} / {lines[-1][:30]}")
        if score["valid"]:
            for l in out_lines:
                print(f"     → {l}")
        else:
            print(f"     {score['issue']}: {out_lines}")
    all_results["4-line"] = results_4
    
    # ── 6-line tests ──
    print(f"\n{'='*60}")
    print("📝  6-LINE INPUT (3 couplets — truncated to last 1)")
    print("=" * 60)
    results_6 = []
    for lines in PROMPTS_6LINE:
        text = "\n".join(lines)
        prompt = auto_tag_doi_tho(text, max_context=1)
        tokens = generate(model, tok, prompt, dev)
        out_lines = decode_doi_tho(tok, tokens)
        score = score_output(out_lines, max_expected_lines=2)
        results_6.append(score)
        icon = "✅" if score["valid"] else "❌"
        print(f"  {icon} ...{lines[-2][:30]} / {lines[-1][:30]}")
        if score["valid"]:
            for l in out_lines:
                print(f"     → {l}")
        else:
            print(f"     {score['issue']}: {out_lines}")
    all_results["6-line"] = results_6
    
    # ── Summary ──
    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"📊  STRESS TEST SUMMARY — step {step} ({elapsed:.0f}s)")
    print(f"{'='*60}")
    
    for label, results in all_results.items():
        valid = sum(1 for r in results if r["valid"])
        total = len(results)
        bar = "█" * (valid * 20 // max(total, 1))
        print(f"  {label:12s}: {valid}/{total} valid  {bar}")
    
    total_valid = sum(sum(1 for r in results if r["valid"]) for results in all_results.values())
    total_all = sum(len(results) for results in all_results.values())
    print(f"  {'TOTAL':12s}: {total_valid}/{total_all} valid ({total_valid/total_all*100:.0f}%)")


if __name__ == "__main__":
    main()
