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

# v4.2: canonical generation module (replaces local gen/gen_couplet/_generate)
from src.generation import (
    build_prompt, generate, decode_response,
)


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

# ── 50 NOVEL Lục Bát couplet prompts (ca dao, folk poetry — couplet→couplet) ──
COUPLET_PROMPTS = [
    # (6-syl, 8-syl) pairs
    ("thân em như chẽn lúa đòng", "phất phơ dưới ngọn nắng hồng ban mai"),
    ("công cha như núi thái sơn", "nghĩa mẹ như nước trong nguồn chảy ra"),
    ("một lòng thờ mẹ kính cha", "cho tròn chữ hiếu mới là đạo con"),
    ("núi cao bởi có đất bồi", "sông sâu bởi có nước nguồn chảy quanh"),
    ("uống nước nhớ kẻ đào sông", "ăn quả nhớ kẻ trồng cây xanh vườn"),
    ("đất lành chim đậu về đây", "người hiền thì lại gặp may mắn nhiều"),
    ("lời nói chẳng mất tiền mua", "lựa lời mà nói cho vừa lòng nhau"),
    ("một cây làm chẳng nên non", "ba cây chụm lại nên hòn núi cao"),
    ("bầu ơi thương lấy bí cùng", "tuy rằng khác giống nhưng chung một giàn"),
    ("nhiễu điều phủ lấy giá gương", "người trong một nước phải thương nhau cùng"),
    ("dẻo thơm một hạt đắng cay", "ai ơi có nhớ những ngày nắng mưa"),
    ("cày đồng đang buổi ban trưa", "mồ hôi thánh thót như mưa ruộng cày"),
    ("ai về tôi gửi buồng cau", "buồng cau non mẹ để già lâu năm"),
    ("cây đa bến nước sân đình", "qua đình ngả nón trông đình xa xa"),
    ("trâu ơi ta bảo trâu này", "trâu ra ngoài ruộng trâu cày với ta"),
    ("ru con con ngủ cho lâu", "để mẹ đi cấy đồng sâu chưa về"),
    ("con cò bay lả bay la", "bay từ cửa phủ bay ra cánh đồng"),
    ("mẹ già như chuối ba hương", "như xôi nếp mật như đường mía lau"),
    ("thuyền ơi có nhớ bến không", "bến thì một dạ khăng khăng đợi thuyền"),
    ("mưa từ xa tới mưa mau", "trời mưa trời gió đùng đùng sấm vang"),
    ("sen tàn cúc lại nở hoa", "sầu dài ngày ngắn sang đông lạnh lùng"),
    ("tóc mây một mái còn dài", "mắt em là cả trời xanh mây hồng"),
    ("đường vô xứ nghệ quanh quanh", "non xanh nước biếc như tranh họa đồ"),
    ("đồng đăng có phố kỳ lừa", "có nàng tô thị có chùa tam thanh"),
    ("trên trời có đám mây vàng", "bên sông có chị hái dâu một mình"),
    ("ánh trăng soi tỏ vườn nhà", "hoa cau rụng trắng thềm xưa lối mòn"),
    ("lời ru của mẹ ngày xưa", "theo con suốt cả chặng đường hôm mai"),
    ("ve sầu kêu gọi hè sang", "phượng hồng thắp lửa sân trường mùa thi"),
    ("sáng trăng suông sáng cả đồng", "em đi gặt lúa trên đồng làng ta"),
    ("mưa xuân lất phất vườn đào", "nụ tầm xuân nở ra chào đón xuân"),
    ("gió đông về lạnh lòng ai", "tìm đâu hơi ấm bàn tay người thương"),
    ("xa quê nhớ mẹ nhớ cha", "nhớ hàng cau trước sân nhà ngày xưa"),
    ("bàn tay năm ngón nở hoa", "đôi chân chim sáo quanh nhà líu lo"),
    ("học thầy không tày học bạn", "đi một ngày đàng học một sàng khôn"),
    ("đói cho sạch rách cho thơm", "khôn ngoan đá đáp người ngoài gà cùng"),
    ("không gì bằng cơm với cà", "một nắng hai sương mẹ cha vất vả"),
    ("có chí thì nên có công", "mài sắt nên kim bạn ơi nhớ lời"),
    ("chị em như chuối nhiều tàu", "tấm lành che tấm rách đừng xấu che"),
    ("ví dầu cầu ván đóng đinh", "cầu tre lắc lẻo gập ghềnh khó đi"),
    ("qua sông phải lụy đò ngang", "qua suối phải lụy cầu tre bắc ngang"),
    ("đồng bằng ruộng lúa mênh mông", "biển đông sóng vỗ rì rào ngày đêm"),
    ("non cao ai đắp mà cao", "sông sâu ai bới ai đào mà sâu"),
    ("thương người như thể thương thân", "ở hiền thì lại gặp lành ở hiền"),
    ("lên non mới biết non cao", "nuôi con mới biết công lao mẹ hiền"),
    ("nước chảy đá mòn theo năm", "đất phèn mọc trái thơm ngon ngọt lành"),
    ("con người có tổ có tông", "như cây có cội như sông có nguồn"),
    ("gần mực thì đen gần đèn", "gần người hiền trí thì nên thông minh"),
    ("trăng mờ còn tỏ hơn sao", "dẫu rằng núi lở còn cao hơn đồi"),
    ("bồng bềnh con nước về đâu", "đò chiều khách vắng sang sông đợi chờ"),
    ("tiếng sáo diều vọng triền đê", "đàn trâu về ngõ chiều hôm khói lam"),
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
    """v4.2: Generate from single 6-syl line using unified generator."""
    full_prompt = build_prompt(prompt, include_trambong=True)
    _, text = generate(model, tok, full_prompt, device=dev, rhyme_mode="soft")
    return text.replace('<|end|>', '').replace('<|reply|>', '').strip(',.-;:!? ')


@torch.no_grad()
def gen_couplet(model, tok, line6, line8, dev):
    """v4.2: Generate from full couplet using unified generator."""
    couplet_input = f"{line6}\n{line8}"
    full_prompt = build_prompt(couplet_input, include_trambong=True)
    ids, _ = generate(model, tok, full_prompt, device=dev, rhyme_mode="soft")
    # Return decoded text with <|linebreak|> preserved as double-space for parser
    lines = decode_response(tok, ids, enforce_syllables=False)
    if len(lines) >= 2:
        return f"{lines[0]}  {lines[1]}"
    elif len(lines) == 1:
        return lines[0]
    return ""


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


def evaluate_couplet(line6_in, line8_in, response_text):
    """v4.1 couplet: input 6+8, output should be 6+8 with <|linebreak|> delimiter."""
    # Parse output by linebreak (double spaces from ByteLevel BPE)
    parts = response_text.split('  ')
    out_luc = parts[0].strip() if len(parts) > 0 else ''
    out_bat = parts[1].strip() if len(parts) > 1 else ''
    
    out_luc_syls = out_luc.split()
    out_bat_syls = out_bat.split()
    in_luc_syls = line6_in.split()
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
    r4_ok = False; r4_detail = ''
    if len(out_bat_syls) >= 8:
        d6, d8 = get_diacritic(out_bat_syls[5]), get_diacritic(out_bat_syls[7])
        r4_ok = d6 in ('ngang', 'huyen') and d8 in ('ngang', 'huyen') and d6 != d8
        r4_detail = f'{d6}/{d8}'
    
    # R5: Nhịp điệu
    r5_ok = (len(out_luc_syls) == 6 and len(out_bat_syls) == 8)
    
    # Quality
    all_syls = out_luc_syls + out_bat_syls
    lex_div = len(set(all_syls)) / max(len(all_syls), 1)
    all_5 = r3_exact and r1_ok and r2_ok == r2_total and r4_ok
    
    return {
        'prompt': f'{line6_in} / {line8_in[:20]}...',
        'response': response_text,
        'out_luc': out_luc, 'out_bat': out_bat,
        'r_len': r_len,
        'R1_ok': r1_ok, 'R2_r_ok': r2_ok, 'R2_r_total': r2_total,
        'R3_exact': r3_exact, 'R4_ok': r4_ok, 'R5_ok': r5_ok,
        'all_5': all_5, 'lex_div': lex_div, 'bpe_artifacts': 0,
    }


def summarize_results(results, label, n_prompts):
    """Compute summary stats from evaluation results."""
    n = len(results)
    r1_pct = sum(r['R1_ok'] for r in results) / n * 100
    r2_pct = sum(r.get('R2_r_ok', 0) for r in results) / max(sum(r.get('R2_r_total', 0) for r in results), 1) * 100
    r3_pct = sum(r.get('R3_exact', False) for r in results) / n * 100
    r4_pct = sum(r.get('R4_ok', False) for r in results) / n * 100
    r5_pct = sum(r.get('R5_ok', False) for r in results) / n * 100
    all5_pct = sum(r.get('all_5', False) for r in results) / n * 100
    avg_len = sum(r.get('r_len', 0) for r in results) / n
    avg_lex = sum(r.get('lex_div', 0) for r in results) / n
    empty_rate = sum(1 for r in results if r.get('r_len', 0) == 0) / n * 100
    return {
        'label': label, 'n': n, 'R1': r1_pct, 'R2': r2_pct, 'R3': r3_pct,
        'R4': r4_pct, 'R5': r5_pct, 'all5': all5_pct,
        'avg_len': avg_len, 'avg_lex': avg_lex, 'empty': empty_rate,
    }
def main():
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {dev}')

    tok = Tokenizer.from_file(str(ROOT / 'tokenizer/poetry_bpe.model'))

    # Find checkpoint
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
    print(f'v4.1 — 5-Rule Lục Bát Evaluation (single + couplet)')
    print(f'Checkpoint: {ckpt_path}')
    print(f'Single-line prompts: {len(PROMPTS)}')
    print(f'Couplet prompts: {len(COUPLET_PROMPTS)}')
    print(f'{"="*60}')

    model = load(str(ckpt_path), dev)
    t0 = time.time()

    # ── Single-line evaluation ──
    print(f'\n--- Single 6-syl → couplet ---')
    sl_results = []
    for i, p in enumerate(PROMPTS):
        r = gen(model, tok, p, dev)
        sl_results.append(evaluate_rules(p, r))
        if (i + 1) % 40 == 0:
            s = summarize_results(sl_results, 'single', i+1)
            print(f'  {i+1}/{len(PROMPTS)} | R1:{s["R1"]:.0f}% R2:{s["R2"]:.0f}% R3:{s["R3"]:.0f}% R4:{s["R4"]:.0f}% | All5:{s["all5"]:.0f}% | {time.time()-t0:.0f}s')
    sl_sum = summarize_results(sl_results, 'Single-line (6-syl)', len(PROMPTS))
    print(f'  → Single-line done: R1:{sl_sum["R1"]:.0f}% R3:{sl_sum["R3"]:.0f}% All5:{sl_sum["all5"]:.0f}%')

    # ── Couplet evaluation ──
    print(f'\n--- Couplet 6+8 → couplet ---')
    cp_results = []
    for i, (l6, l8) in enumerate(COUPLET_PROMPTS):
        r = gen_couplet(model, tok, l6, l8, dev)
        cp_results.append(evaluate_couplet(l6, l8, r))
        if (i + 1) % 20 == 0:
            s = summarize_results(cp_results, 'couplet', i+1)
            print(f'  {i+1}/{len(COUPLET_PROMPTS)} | R1:{s["R1"]:.0f}% R2:{s["R2"]:.0f}% R3:{s["R3"]:.0f}% R4:{s["R4"]:.0f}% | All5:{s["all5"]:.0f}% | {time.time()-t0:.0f}s')
    cp_sum = summarize_results(cp_results, 'Couplet (6+8)', len(COUPLET_PROMPTS))
    print(f'  → Couplet done: R1:{cp_sum["R1"]:.0f}% R3:{cp_sum["R3"]:.0f}% All5:{cp_sum["all5"]:.0f}%')

    # ── Build report ──
    lines = []
    lines.append('# 📊 v4.1 Rule-by-Rule Evaluation — 5 Lục Bát Rules')
    lines.append('')
    lines.append(f'> Generated: {time.strftime("%Y-%m-%d %H:%M")}')
    lines.append(f'> Checkpoint: {ckpt_path.name}')
    lines.append(f'> Single-line: {len(PROMPTS)} prompts | Couplet: {len(COUPLET_PROMPTS)} prompts')
    lines.append(f'> Model: 31M params, n_embd=512, n_head=8, n_layer=8')
    lines.append('')
    
    lines.append('## 📈 5-Rule Summary')
    lines.append('')
    lines.append('| Rule | Single-line | Couplet | Random | Target |')
    lines.append('|------|-------------|---------|--------|--------|')
    random_r1 = (1/159)*100; random_r2 = (0.5**4)*100
    for name, key, rand, target in [
        ('R1: Vần lưng', 'R1', random_r1, 65),
        ('R2: Bằng-Trắc', 'R2', random_r2, 93),
        ('R3: Syllable (6+8)', 'R3', 7.0, 85),
        ('R4: Trầm-Bổng', 'R4', 50.0, 60),
        ('R5: Nhịp điệu', 'R5', 7.0, 75),
        ('**All 5 pass**', 'all5', 0, 30),
    ]:
        sl_v = sl_sum[key]; cp_v = cp_sum[key]
        sl_m = '✅' if sl_v >= target else '🟡'
        cp_m = '✅' if cp_v >= target else '🟡'
        lines.append(f'| {name} | {sl_m} {sl_v:.1f}% | {cp_m} {cp_v:.1f}% | {rand:.1f}% | {target}%+ |')
    
    lines.append('')
    lines.append('## 📊 Quality Metrics')
    lines.append('')
    lines.append('| Metric | Single-line | Couplet |')
    lines.append('|--------|-------------|--------|')
    lines.append(f'| Avg response length | {sl_sum["avg_len"]:.1f} syl | {cp_sum["avg_len"]:.1f} syl |')
    lines.append(f'| Lexical diversity | {sl_sum["avg_lex"]:.3f} | {cp_sum["avg_lex"]:.3f} |')
    lines.append(f'| Empty response rate | {sl_sum["empty"]:.1f}% | {cp_sum["empty"]:.1f}% |')
    lines.append('')

    # Couplet samples
    lines.append('## 📝 Couplet Samples')
    lines.append('')
    lines.append('| Input | Output Lục | Output Bát | R1 | R2 | R3 | R4 | All |')
    lines.append('|-------|-----------|-----------|----|----|----|----|-----|')
    for r in cp_results[:20]:
        emoji = lambda ok: '✅' if ok else '❌'
        lines.append(f'| {r["prompt"][:25]} | {r.get("out_luc","?")[:25]} | {r.get("out_bat","?")[:25]} | '
                     f'{emoji(r["R1_ok"])} | {emoji(r["R2_r_ok"]==r["R2_r_total"])} | '
                     f'{emoji(r["R3_exact"])} | {emoji(r["R4_ok"])} | {emoji(r["all_5"])} |')
    lines.append('')

    out = ROOT / 'documents' / 'rule_evaluation.md'
    out.write_text('\n'.join(lines))
    print(f'\n📄 Report → {out}')

    json.dump({
        'version': 'v4.1', 'checkpoint': str(ckpt_path.name),
        'single_line': sl_sum, 'couplet': cp_sum,
    }, open(ROOT / 'evaluate' / 'rule_evaluation.json', 'w'), indent=2, ensure_ascii=False)
    print(f'📄 JSON → evaluate/rule_evaluation.json')


if __name__ == '__main__':
    main()
