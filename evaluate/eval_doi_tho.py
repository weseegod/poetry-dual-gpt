"""
Đối thơ evaluation — chain rhyme, internal rhyme, tone, syllable.
Tests the [DOI_THO] checkpoint against 20+ novel couplets.

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

# ── 25 test couplets (ca dao, folk poetry, Truyện Kiều — verified novel) ──
TEST_COUPLETS = [
    ("Thân em như tấm lụa đào", "Phất phơ giữa chợ biết vào tay ai"),
    ("Trèo lên cây khế nửa ngày", "Ai làm cho khế rụng đầy vườn ai"),
    ("Đêm khuya thắp ngọn đèn dầu", "Lòng em thương nhớ anh lâu lắm rồi"),
    ("Công cha như núi thái sơn", "Nghĩa mẹ như nước trong nguồn chảy ra"),
    ("Gió đưa cành trúc la đà", "Tiếng chuông Trấn Vũ canh gà Thọ Xương"),
    ("Cày đồng đang buổi ban trưa", "Mồ hôi thánh thót như mưa ruộng cày"),
    ("Qua đình ngả nón trông đình", "Đình bao nhiêu ngói thương mình bấy nhiêu"),
    ("Đường vô xứ nghệ quanh quanh", "Non xanh nước biếc như tranh họa đồ"),
    ("Sông sâu còn có kẻ đò", "Đường xa còn có người qua đón chờ"),
    ("Cây khô chưa dễ mọc chồi", "Người khôn chưa dễ nói lời thị phi"),
    ("Mẹ già như chuối ba hương", "Như xôi nếp một như đường mía lau"),
    ("Dẻo thơm một hạt đắng cay", "Muôn phần đắng chát cũng vay ngọt bùi"),
    ("Ru con con ngủ cho lâu", "Để mẹ đi cấy đồng sâu ruộng cày"),
    ("Rủ nhau xuống biển mò cua", "Về nhà nấu cháo nấu cua ăn cùng"),
    ("Thuyền ơi có nhớ bến không", "Bến thì một dạ khăng khăng đợi thuyền"),
    ("Trời mưa trời gió đùng đùng", "Đèn nhà ai nấy sáng trưng góc trời"),
    ("Sen tàn cúc lại nở hoa", "Sầu dài ngày ngắn sang đông lạnh lùng"),
    ("Một cây làm chẳng nên non", "Ba cây chụm lại nên hòn núi cao"),
    ("Gần mực thì đen gần đèn", "Gần người hiền trí thì nên thân mình"),
    ("Nước chảy đá mòn theo năm", "Người thương nhớ mãi xa xăm phương trời"),
    ("Học thầy không tày học bạn", "Đi một ngày đàng học một sàng khôn"),
    ("Xa mặt nhưng chẳng cách lòng", "Gần nhau càng thấy nhớ mong từng giờ"),
    ("Đói cho sạch rách cho thơm", "Khôn ngoan đá đáp người ngoài gà con"),
    ("Thương người như thể thương thân", "Nhiễu điều phủ lấy giá gương soi chung"),
    ("Mưa xuân lất phất vườn đào", "Nụ tầm xuân nở ra chào đón xuân"),
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


@torch.no_grad()
def generate_doi_tho(model, tokenizer, six_line, eight_line, device="cpu"):
    rhyme_tag, tone_tag = get_doi_tho_tags(six_line, eight_line)
    tags = f"[DOI_THO] {rhyme_tag} {tone_tag}".strip()
    prompt = f"{tags} {six_line} <|linebreak|> {eight_line} <|reply|>"
    
    end_id = tokenizer.token_to_id("<|end|>")
    pad_id = tokenizer.token_to_id("<|pad|>")
    
    ids = tokenizer.encode(prompt).ids
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    
    new_tokens = []
    for _ in range(80):
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
    
    return decode_doi_tho(tokenizer, new_tokens)


def evaluate_one(in6, in8, out_lines):
    """Score a single đối thơ generation."""
    if len(out_lines) < 2:
        return {"valid": False}
    
    out6, out8 = out_lines[0], out_lines[1]
    s6, s8 = out6.split(), out8.split()
    
    result = {"valid": True, "out6": out6, "out8": out8}
    
    # Syllable
    result["syl_ok"] = len(s6) == 6 and len(s8) == 8
    result["s6_len"] = len(s6)
    result["s8_len"] = len(s8)
    
    # Chain rhyme: pos 6 of out6 vs pos 8 of input 8-syl line
    expected_rhyme = get_rhyme_group(in8.split()[7]) if len(in8.split()) >= 8 else ""
    actual_chain = get_rhyme_group(s6[5]) if len(s6) >= 6 else ""
    result["chain_rhyme_ok"] = actual_chain == expected_rhyme
    result["chain_expected"] = expected_rhyme
    result["chain_actual"] = actual_chain
    
    # Internal rhyme: pos 6 of out6 vs pos 6 of out8
    if len(s6) >= 6 and len(s8) >= 6:
        r6 = get_rhyme_group(s6[5])
        r8 = get_rhyme_group(s8[5])
        result["internal_rhyme_ok"] = r6 == r8
        result["int_r6"] = r6
        result["int_r8"] = r8
    else:
        result["internal_rhyme_ok"] = False
    
    # Tone: B-T-B for 6-syl, B-T-B-B for 8-syl
    t6 = get_tone_sequence(out6)
    t8 = get_tone_sequence(out8)
    if len(t6) >= 6:
        result["tone6_ok"] = t6[1] == 'B' and t6[3] == 'T' and t6[5] == 'B'
    else:
        result["tone6_ok"] = False
    if len(t8) >= 8:
        result["tone8_ok"] = t8[1] == 'B' and t8[3] == 'T' and t8[5] == 'B' and t8[7] == 'B'
    else:
        result["tone8_ok"] = False
    
    result["tone6"] = t6[:6] if len(t6) >= 6 else t6
    result["tone8"] = t8[:8] if len(t8) >= 8 else t8
    
    return result


def main():
    p = argparse.ArgumentParser(description="Đối thơ evaluation")
    p.add_argument("--checkpoint", default="checkpoints/doi_tho_best.pt")
    p.add_argument("--tokenizer", default="tokenizer/poetry_bpe.model")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num_couplets", type=int, default=None)
    args = p.parse_args()
    
    dev = args.device
    print(f"Device: {dev}")
    
    tok = Tokenizer.from_file(str(ROOT / args.tokenizer))
    model, step = load_model(str(ROOT / args.checkpoint), dev)
    print(f"Model: step {step}, vocab={tok.get_vocab_size():,}\n")
    
    couplets = TEST_COUPLETS[:args.num_couplets] if args.num_couplets else TEST_COUPLETS
    
    results = []
    t0 = time.time()
    
    for i, (in6, in8) in enumerate(couplets):
        out_lines = generate_doi_tho(model, tok, in6, in8, dev)
        r = evaluate_one(in6, in8, out_lines)
        results.append(r)
        
        if r["valid"]:
            icon = "✅" if r["syl_ok"] else "❌"
            print(f"  {icon} {r['out6']}")
            print(f"     {r['out8']}")
            print(f"     syl={r['s6_len']}+{r['s8_len']} "
                  f"chain={r['chain_actual']}/{r['chain_expected']} "
                  f"int={r.get('int_r6','?')}/{r.get('int_r8','?')} "
                  f"t6={r['tone6']} t8={r['tone8']}")
        else:
            print(f"  ❌  {in6[:40]}... → EMPTY/SHORT")
    
    elapsed = time.time() - t0
    
    # Summary
    n = sum(1 for r in results if r["valid"])
    n_total = len(results)
    
    syl_ok = sum(1 for r in results if r.get("syl_ok"))
    chain_ok = sum(1 for r in results if r.get("chain_rhyme_ok"))
    int_ok = sum(1 for r in results if r.get("internal_rhyme_ok"))
    tone6_ok = sum(1 for r in results if r.get("tone6_ok"))
    tone8_ok = sum(1 for r in results if r.get("tone8_ok"))
    
    print(f"\n{'='*60}")
    print(f"📊  ĐỐI THƠ EVALUATION — {n}/{n_total} valid, step {step} ({elapsed:.0f}s)")
    print(f"{'='*60}")
    print(f"  Valid output:         {n}/{n_total} ({n/n_total*100:.0f}%)")
    if n > 0:
        print(f"  Syllable (6+8):       {syl_ok}/{n} = {syl_ok/n*100:.0f}%")
        print(f"  Chain rhyme:          {chain_ok}/{n} = {chain_ok/n*100:.0f}%")
        print(f"  Internal rhyme:       {int_ok}/{n} = {int_ok/n*100:.0f}%")
        print(f"  Tone 6 (B-T-B):       {tone6_ok}/{n} = {tone6_ok/n*100:.0f}%")
        print(f"  Tone 8 (B-T-B-B):     {tone8_ok}/{n} = {tone8_ok/n*100:.0f}%")
    print()
    print(f"  v1 baseline (single-couplet, 15K steps):")
    print(f"    Syllable: 71% | Rhyme: 50% | Tone: 88%")
    print(f"  v2.1 (doi_tho, single-stage, {step} steps):")
    print(f"    Syllable: {syl_ok/n*100:.0f}% | Chain: {chain_ok/n*100:.0f}% | "
          f"Internal: {int_ok/n*100:.0f}% | Tone: {tone6_ok/n*100:.0f}%/{tone8_ok/n*100:.0f}%")


if __name__ == "__main__":
    main()
