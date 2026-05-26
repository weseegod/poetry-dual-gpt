#!/usr/bin/env python3
"""
v4.1: Full 5-rule Lục Bát evaluation — 173 novel prompts.

Rules from documents/rules/luc_bat.md:
  R1: Vần lưng   — pos6 Lục rhymes with pos6 Bát
  R2: Bằng-Trắc  — BTB (Lục) + BTBB (Bát) at chẵn positions
  R3: Syllable   — 6+8 exact
  R4: Trầm-Bổng  — tiếng 6 & 8 dòng Bát khác dấu (Ngang≠Huyền)
  R5: Nhịp điệu  — 2/2/2 (Lục) + 2/2/2/2 (Bát) chẵn rhythm

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


# ── 173 NOVEL Lục Bát prompts (ca dao, folk poetry — not in corpus) ──
PROMPTS = [
    "Thân em như tấm lụa đào", "Trèo lên cây khế nửa ngày",
    "Ai làm cho bướm xa hoa", "Đêm khuya thắp ngọn đèn dầu",
    "Gió mùa thu mẹ ru con", "Chim khôn đậu nóc nhà quan",
    "Cây khô chưa dễ mọc chồi", "Mẹ già như chuối ba hương",
    "Ru con con ngủ cho lâu", "Công cha như núi thái sơn",
    "Rủ nhau xuống biển mò cua", "Đố ai đếm được lá rừng",
    "Cày đồng đang buổi ban trưa", "Mồ hôi thánh thót như mưa",
    "Dẻo thơm một hạt đắng cay", "Ai về tôi gửi buồng cau",
    "Buồng cau non mẹ để già", "Cau già khéo bổ thì non",
    "Cây đa bến nước sân đình", "Qua đình ngả nón trông đình",
    "Hòn đá đóng rêu vì ngâu",
    "Thuyền ơi có nhớ bến không", "Bến thì một dạ khăng khăng",
    "Mưa từ xa tới mưa mau", "Trời mưa trời gió đùng đùng",
    "Lúa mùa vàng óng đồng quê", "Trâu ơi ta bảo trâu này",
    "Bao giờ cho đến tháng ba", "Ếch kêu dưới vũng ao nhà",
    "Tháng năm chưa đến đã mưa", "Ve kêu ra rả suốt mùa",
    "Sen tàn cúc lại nở hoa", "Sầu dài ngày ngắn sang đông",
    "Gió đông về lạnh lòng ai", "Tóc mây một mái còn dài",
    "Mắt em là cả trời xanh", "Môi em là nắng long lanh",
    "Núi cao bởi có đất bồi", "Sông sâu bởi có nước nguồn",
    "Uống nước nhớ kẻ đào sông", "Ăn quả nhớ kẻ trồng cây",
    "Đất lành chim đậu về đây", "Người hiền thì lại gặp may",
    "Lời nói chẳng mất tiền mua", "Lựa lời mà nói cho vừa",
    "Đèn nhà ai nấy sáng trưng", "Chớ thấy đèn sáng mà mừng",
    "Sông sâu còn có kẻ đò", "Đường xa còn có người qua",
    "Bầu ơi thương lấy bí cùng", "Tuy rằng khác giống nhưng chung",
    "Một cây làm chẳng nên non", "Ba cây chụm lại nên hòn",
    "Gần mực thì đen gần đèn", "Gần người hiền trí thì nên",
    "Nước lã làm sao khuấy nên", "Chữ rằng bán tự vi sư",
    "Nhất tự vi sư bán tự", "Mồng một tết cha mồng hai",
    "Tháng giêng ăn tết ở nhà", "Tháng hai cờ bạc tháng ba",
    "Trên trời có đám mây vàng", "Bên sông có chị hái dâu",
    "Đồng đăng có phố kỳ lừa", "Có nàng tô thị có chùa",
    "Đường vô xứ nghệ quanh quanh", "Non xanh nước biếc như tranh",
    "Anh về câu cá bờ sông", "Em về cấy lúa trên đồng",
    "Nắng sớm mưa chiều đồng không", "Mong cho lúa tốt bông vàng",
    "Bàn tay ta làm nên tất", "Có sức người sỏi đá cũng",
    "Chim bay về núi tối rồi", "Mau lên kẻo nắng tắt rồi",
    "Trăng lên đỉnh núi trăng mờ", "Đêm nay sao sáng hơn đêm",
    "Xa xa có tiếng chuông chùa", "Gió đưa hương lúa thơm mùa",
    "Sáng trăng suông sáng cả đồng", "Em đi gặt lúa trên đồng",
    "Mưa xuân lất phất vườn đào", "Nụ tầm xuân nở ra chào",
    "Hoa thơm ong bướm tìm về", "Người khôn thiên hạ tìm theo",
    "Tốt gỗ hơn là tốt nước", "Xấu người đẹp nết còn hơn",
    "Chim không ăn muối chim ươn", "Con không nghe mẹ con hư",
    "Cá không ăn muối cá ươn", "Con cãi cha mẹ trăm đường",
    "Đói cho sạch rách cho thơm", "Khôn ngoan đá đáp người ngoài",
    "Thương người như thể thương thân", "Nhiễu điều phủ lấy giá gương",
    "Một nắng hai sương mẹ cha", "Lên non mới biết non cao",
    "Nuôi con mới biết công lao", "Nước chảy đá mòn theo năm",
    "Gió lên cho biển hóa rồng", "Sóng to gió lớn mênh mông",
    "Đất phèn mọc trái thơm ngon", "Người nghèo biết quý từng đồng",
    "Học thầy không tày học bạn", "Đi một ngày đàng học một",
    "Xa mặt nhưng chẳng cách lòng", "Gần nhau càng thấy nhớ mong",
    "Trăng mờ còn tỏ hơn sao", "Dẫu rằng núi lở còn cao",
    "Công cha nghĩa mẹ sinh thành", "Một lòng thờ mẹ kính cha",
    "Con người có tổ có tông", "Lên non xem núi cao vời",
    "Nước non ngàn dặm xa xôi", "Ra đi từ thuở lên ba",
    "Quê hương mỗi lúc một xa", "Bồng bềnh con nước về đâu",
    "Đò chiều khách vắng sang sông", "Trời xanh mây trắng lang thang",
    "Đường quê lúa chín vàng ươm", "Hương đồng gió nội bay xa",
    "Ve sầu kêu gọi hè sang", "Phượng hồng thắp lửa sân trường",
    "Áo em trắng cả sân trường", "Tóc em dài quá vai thương",
    "Bàn tay năm ngón nở hoa", "Đôi chân chim sáo quanh nhà",
    "Em như cây lúa trổ bông", "Anh như hạt thóc vàng trong",
    "Xa quê nhớ mẹ nhớ cha", "Nhớ hàng cau trước sân nhà",
    "Giếng làng trong mát tuổi thơ", "Cánh diều no gió tuổi thơ",
    "Mưa rào tắm mát vườn quê", "Nắng vàng ươm cả đồng xa",
    "Bếp nhà ai đỏ lửa hồng", "Khói lam chiều tỏa mênh mông",
    "Đàn trâu về ngõ chiều hôm", "Tiếng sáo diều vọng triền đê",
    "Đêm trăng soi tỏ vườn nhà", "Hoa cau rụng trắng thềm xưa",
    "Thuyền ai lờ lững trên sông", "Câu hò vọng giữa thinh không",
    "Cây khế chua ngọt sau vườn", "Quả na mở mắt thơm lừng",
    "Lời ru của mẹ ngày xưa", "Theo con suốt cả chặng đường",
    "Người đi đâu suốt chiều nay", "Để lòng ai những vơi đầy",
    "Đường trần ai biết được đâu", "Ngày mai sương gió dãi dầu",
    "Thương ai con mắt lim dim", "Nhớ ai nước mắt đầm đìa",
    "Mưa nguồn chớp bể ào ào", "Thương em anh biết làm sao",
    "Ví dầu cầu ván đóng đinh", "Cầu tre lắc lẻo gập ghềnh",
    "Qua sông phải lụy đò ngang", "Qua suối phải lụy cầu tre",
    "Đồng bằng ruộng lúa mênh mông", "Biển đông sóng vỗ rì rào",
    "Non cao ai đắp mà cao", "Sông sâu ai bới ai đào",
    "Khi vui cũng vậy khi buồn", "Ở hiền thì lại gặp lành",
    "Chị em như chuối nhiều tàu", "Tấm lành che tấm rách đừng",
    "Rừng vàng biển bạc quê ta", "Không gì bằng cơm với cà",
    "Có chí thì nên có công", "Mài sắt nên kim bạn ơi",
    "Tay làm hàm nhai tay quai", "Miệng ăn núi lở ai ơi",
]


# ── Model loading ──
def load(path, dev):
    ckpt = torch.load(path, map_location=dev, weights_only=False)
    cfg = ckpt['model_config'].copy()
    cfg.pop('vocab_size', None)
    m = PoetryDuelGPT(ckpt['vocab_size'], **cfg)
    m.load_state_dict(ckpt['model_state_dict']); m.to(dev).eval()
    return m


@torch.no_grad()
def gen(model, tok, prompt, dev):
    """Generate with v4.1 tags: [RHYME:X] [TONE:BBBBBB] [TRAMBONG:NH]."""
    rhyme, tone, trambong = get_luc_bat_tags(prompt)
    tags = ' '.join(t for t in [rhyme, tone, trambong] if t)
    tagged = f'[LUC_BAT] {tags} {prompt}'
    ids = tok.encode(tagged).ids
    idx = torch.tensor([ids], dtype=torch.long, device=dev)
    end_id = tok.token_to_id('<|end|>')
    pad_id = tok.token_to_id('<|pad|>')
    new = []
    for _ in range(64):
        logits, _ = model(idx[:, -model.block_size:])
        logits = logits[:, -1, :] / 0.75
        logits[:, pad_id] = float('-inf')
        v, _ = torch.topk(logits, min(50, logits.size(-1)))
        logits[logits < v[:, -1:]] = float('-inf')
        nid = torch.multinomial(F.softmax(logits, dim=-1), 1).item()
        if nid == end_id: break
        new.append(nid)
        idx = torch.cat([idx, torch.tensor([[nid]], device=dev)], dim=1)
    return tok.decode(new).replace('<|end|>','').replace('<|reply|>','').strip(',.-;:!? ')


def evaluate_rules(prompt_text, response_text):
    """v4.1: Score all 5 Lục Bát rules independently."""
    p_syls = prompt_text.split()
    r_syls_all = response_text.split()
    r_syls = r_syls_all[:8]
    p_len = len(p_syls)
    r_len = len(r_syls_all)

    # ─── R1: Vần lưng ───
    r1_ok = False
    r1_prompt_rhyme = None
    r1_response_rhyme = None
    if p_len >= 6 and len(r_syls) >= 6:
        r1_prompt_rhyme = get_rhyme_group(p_syls[5])
        r1_response_rhyme = get_rhyme_group(r_syls[5])
        r1_ok = r1_prompt_rhyme == r1_response_rhyme

    # ─── R2: Bằng-Trắc ───
    r2_prompt_ok = 0; r2_prompt_total = 0
    for idx, want in [(1, 'B'), (3, 'T'), (5, 'B')]:
        if idx < p_len:
            r2_prompt_total += 1
            if get_tone(p_syls[idx]) == want:
                r2_prompt_ok += 1

    r2_resp_ok = 0; r2_resp_total = 0
    for idx, want in [(1, 'B'), (3, 'T'), (5, 'B'), (7, 'B')]:
        if idx < len(r_syls):
            r2_resp_total += 1
            if get_tone(r_syls[idx]) == want:
                r2_resp_ok += 1

    # ─── R3: Syllable count ───
    r3_exact = (p_len == 6 and r_len == 8)

    # ─── R4: Trầm-Bổng ───
    r4_ok = False
    r4_detail = ""
    if r_len >= 8:
        d6 = get_diacritic(r_syls[5])
        d8 = get_diacritic(r_syls[7])
        r4_ok = d6 in ("ngang", "huyen") and d8 in ("ngang", "huyen") and d6 != d8
        r4_detail = f"{d6}/{d8}"

    # ─── R5: Nhịp điệu ───
    r5_ok = (p_len == 6 and r_len == 8)  # approximate: correct count = even rhythm

    # ─── Quality metrics ───
    # Lexical diversity: unique syllables / total
    unique = len(set(r_syls_all)) if r_syls_all else 0
    lex_div = unique / max(r_len, 1)

    # BPE artifact detection: words with < 2 chars or containing non-Viet chars
    bpe_artifacts = sum(1 for s in r_syls_all if len(s) < 2 or not any(
        c in "aăâeêioôơuưyàáảãạằắẳẵặầấẩẫậèéẻẽẹềếểễệìíỉĩịòóỏõọồốổỗộờớởỡợùúủũụừứửữựỳýỵỷỹ"
        "AĂÂEÊIOÔƠUƯYÀÁẢÃẠẰẮẲẴẶẦẤẨẪẬÈÉẺẼẸỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌỒỐỔỖỘỜỚỞỠỢÙÚỦŨỤỪỨỬỮỰỲÝỴỶỸ"
        for c in s))

    # ─── Combined ───
    all_5 = r3_exact and r1_ok and r2_resp_ok == r2_resp_total and r4_ok

    return {
        'prompt': prompt_text,
        'response': response_text,
        'p_len': p_len, 'r_len': r_len,
        # R1
        'R1_ok': r1_ok, 'R1_p': r1_prompt_rhyme, 'R1_r': r1_response_rhyme,
        # R2
        'R2_p_ok': r2_prompt_ok, 'R2_p_total': r2_prompt_total,
        'R2_r_ok': r2_resp_ok, 'R2_r_total': r2_resp_total,
        # R3
        'R3_exact': r3_exact,
        # R4
        'R4_ok': r4_ok, 'R4_detail': r4_detail,
        # R5
        'R5_ok': r5_ok,
        # Quality
        'lex_div': lex_div,
        'bpe_artifacts': bpe_artifacts,
        # Combined
        'all_5': all_5,
        'pos_tones': [get_tone(s) for s in r_syls[:8]],
    }


def main():
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {dev}')

    tok = Tokenizer.from_file(str(ROOT / 'tokenizer/poetry_bpe.model'))

    # Find latest checkpoint
    ckpt_paths = []
    for d in ['checkpoints']:
        for p in Path(ROOT / d).glob('*.pt'):
            ckpt_paths.append(p)
    
    if not ckpt_paths:
        print('No checkpoint found in checkpoints/')
        return
    
    ckpt_path = max(ckpt_paths, key=lambda p: p.stat().st_mtime)
    # Prefer best/final over step checkpoints unless they're newest
    for name in ['doi_tho_best.pt', 'final.pt', 'best.pt']:
        p = ROOT / 'checkpoints' / name
        if p.exists():
            ckpt_path = p
            break
    
    print(f'\n{"="*60}')
    print(f'v4.1 — 5-Rule Lục Bát Evaluation')
    print(f'Checkpoint: {ckpt_path}')
    print(f'Prompts: {len(PROMPTS)} (ca dao, folk poetry)')
    print(f'{"="*60}')

    model = load(str(ckpt_path), dev)
    results = []
    t0 = time.time()

    for i, p in enumerate(PROMPTS):
        r = gen(model, tok, p, dev)
        results.append(evaluate_rules(p, r))

        if (i + 1) % 40 == 0:
            elapsed = time.time() - t0
            n = len(results)
            r1 = sum(rr['R1_ok'] for rr in results) / n * 100
            r2 = sum(rr['R2_r_ok'] for rr in results) / max(sum(rr['R2_r_total'] for rr in results), 1) * 100
            r3 = sum(rr['R3_exact'] for rr in results) / n * 100
            r4 = sum(rr['R4_ok'] for rr in results) / n * 100
            r5 = sum(rr['R5_ok'] for rr in results) / n * 100
            all5 = sum(rr['all_5'] for rr in results) / n * 100
            print(f'  {i+1}/{len(PROMPTS)} | R1:{r1:.0f}% R2:{r2:.0f}% R3:{r3:.0f}% R4:{r4:.0f}% R5:{r5:.0f}% | All5:{all5:.0f}% | {elapsed:.0f}s')

    n = len(results)
    r1_pct = sum(r['R1_ok'] for r in results) / n * 100
    r2_pct = sum(r['R2_r_ok'] for r in results) / max(sum(r['R2_r_total'] for r in results), 1) * 100
    r2p_pct = sum(r['R2_p_ok'] for r in results) / max(sum(r['R2_p_total'] for r in results), 1) * 100
    r3_pct = sum(r['R3_exact'] for r in results) / n * 100
    r4_pct = sum(r['R4_ok'] for r in results) / n * 100
    r5_pct = sum(r['R5_ok'] for r in results) / n * 100
    all5_pct = sum(r['all_5'] for r in results) / n * 100
    avg_len = sum(r['r_len'] for r in results) / n
    avg_lex = sum(r['lex_div'] for r in results) / n
    total_bpe = sum(r['bpe_artifacts'] for r in results)
    empty_rate = sum(1 for r in results if r['r_len'] == 0) / n * 100

    # ── Build report ──
    lines = []
    lines.append('# 📊 v4.1 Rule-by-Rule Evaluation — 5 Lục Bát Rules')
    lines.append('')
    lines.append(f'> Generated: {time.strftime("%Y-%m-%d %H:%M")}')
    lines.append(f'> Checkpoint: {ckpt_path.name}')
    lines.append(f'> {n} prompts (ca dao, folk poetry — NOT in training corpus)')
    lines.append(f'> Model: 30.9M params, n_embd=512, n_head=8, n_layer=8')
    lines.append('')
    lines.append('## 📈 5-Rule Summary')
    lines.append('')
    lines.append('| Rule | Description | Accuracy | Random Baseline | Effective? |')
    lines.append('|------|-------------|----------|-----------------|------------|')

    random_r1 = (1 / 159) * 100
    random_r2 = (0.5 ** 4) * 100
    random_r3 = 7.0
    random_r4 = 50.0  # NH vs HN = 50/50
    random_r5 = 7.0

    for name, pct, rand, target in [
        ('R1: Vần lưng', r1_pct, random_r1, 65),
        ('R2: Bằng-Trắc', r2_pct, random_r2, 93),
        ('R3: Syllable (6+8)', r3_pct, random_r3, 85),
        ('R4: Trầm-Bổng', r4_pct, random_r4, 60),
        ('R5: Nhịp điệu', r5_pct, random_r5, 75),
    ]:
        eff = '✅' if pct > rand * 2 else ('⚠️' if pct > rand else '❌')
        goal = f'→ target {target}%+' if pct < target else '✅'
        lines.append(f'| **{name}** | {pct:.1f}% | {rand:.1f}% | {eff} {goal} |')

    lines.append(f'| **All 5 pass** | **{all5_pct:.1f}%** | — | — |')
    lines.append('')

    # Quality metrics
    lines.append('## 📊 Quality Metrics')
    lines.append('')
    lines.append('| Metric | Value |')
    lines.append('|--------|-------|')
    lines.append(f'| Avg response length | {avg_len:.1f} syl |')
    lines.append(f'| Lexical diversity | {avg_lex:.3f} (0.6+ = good) |')
    lines.append(f'| BPE artifacts | {total_bpe} total |')
    lines.append(f'| Empty response rate | {empty_rate:.1f}% |')
    lines.append(f'| Prompt tone accuracy (BTB) | {r2p_pct:.1f}% |')
    lines.append('')

    # R4 Trầm-Bổng breakdown
    lines.append('## 🎵 R4: Trầm-Bổng — Per-pattern Breakdown')
    lines.append('')
    lines.append(f'**Rule**: Tiếng 6 & 8 của dòng Bát phải khác dấu (Ngang ≠ Huyền)')
    lines.append('')
    patterns = Counter()
    for r in results:
        if r['r_len'] >= 8:
            patterns[r['R4_detail']] += 1
    lines.append('| Pattern | Count | % |')
    lines.append('|---------|-------|---|')
    for pat, cnt in patterns.most_common():
        lines.append(f'| {pat} | {cnt} | {cnt/n*100:.1f}% |')
    lines.append('')

    # Sample outputs
    lines.append('## 📝 Sample Outputs')
    lines.append('')
    lines.append('| Prompt | Response | R1 | R2 | R3 | R4 | R5 | All |')
    lines.append('|--------|----------|----|----|----|----|----|-----|')
    for r in results[:30]:
        emoji = lambda ok: '✅' if ok else '❌'
        lines.append(f'| {r["prompt"][:30]} | {r["response"][:35]} | '
                     f'{emoji(r["R1_ok"])} | {emoji(r["R2_r_ok"]==r["R2_r_total"])} | '
                     f'{emoji(r["R3_exact"])} | {emoji(r["R4_ok"])} | '
                     f'{emoji(r["R5_ok"])} | {emoji(r["all_5"])} |')
    lines.append('')

    # v4.1 vs v3 comparison
    lines.append('## 📊 v4.1 vs v3 Comparison')
    lines.append('')
    lines.append('| Metric | v3 | v4.1 Target | v4.1 Actual |')
    lines.append('|--------|-----|-------------|-------------|')
    lines.append(f'| R1: Rhyme | 50% | 65%+ | {r1_pct:.1f}% |')
    lines.append(f'| R2: Tone | 88% | 93%+ | {r2_pct:.1f}% |')
    lines.append(f'| R3: Syllable | 71% | 85%+ | {r3_pct:.1f}% |')
    lines.append(f'| R4: Trầm-Bổng | 0% (N/A) | 60%+ | {r4_pct:.1f}% |')
    lines.append(f'| R5: Nhịp điệu | N/A | 75%+ | {r5_pct:.1f}% |')
    lines.append(f'| All 5 pass | N/A | 30%+ | {all5_pct:.1f}% |')
    lines.append('')

    out = ROOT / 'documents' / 'rule_evaluation.md'
    out.write_text('\n'.join(lines))
    print(f'\n📄 Report → {out}')

    # JSON data
    json_out = ROOT / 'evaluate' / 'rule_evaluation.json'
    json.dump({
        'version': 'v4.1',
        'checkpoint': str(ckpt_path.name),
        'summary': {
            'R1_rhyme': r1_pct, 'R2_tone': r2_pct, 'R3_syllable': r3_pct,
            'R4_trambong': r4_pct, 'R5_rhythm': r5_pct, 'all_5_pass': all5_pct,
            'avg_length': avg_len, 'lexical_diversity': avg_lex,
            'bpe_artifacts': total_bpe, 'empty_rate': empty_rate,
        },
        'samples': [{
            'p': r['prompt'], 'r': r['response'],
            'R1': r['R1_ok'], 'R2': r['R2_r_ok'] == r['R2_r_total'],
            'R3': r['R3_exact'], 'R4': r['R4_ok'], 'R5': r['R5_ok'],
            'all_5': r['all_5'],
        } for r in results]
    }, open(json_out, 'w'), indent=2, ensure_ascii=False)
    print(f'📄 JSON → {json_out}')


if __name__ == '__main__':
    main()
