"""Evaluate 100 NOVEL Lục Bát prompts (not in training corpus)."""
import re, json, time
import torch, torch.nn.functional as F
from pathlib import Path
from tokenizers import Tokenizer
from src.model import PoetryDuelGPT
from src.tones import get_tone, get_rhyme_group, get_luc_bat_tags

ROOT = Path(__file__).parent.parent

# 100 classic Vietnamese folk poetry (ca dao) — NOT in training corpus
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
    "Hòn đá đóng rêu vì ngâu", "Đứng bên ni đồng ngó bên tê",
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
]

# Verify NONE are in corpus
corpus = open(ROOT / 'data/poetry_corpus.txt').read()
in_corpus = sum(1 for p in PROMPTS if p in corpus)
print(f'Novel prompts: {100-in_corpus}/{len(PROMPTS)} (in corpus: {in_corpus})')

def load(path, dev):
    ckpt = torch.load(path, map_location=dev, weights_only=False)
    m = PoetryDuelGPT(ckpt['vocab_size'], **ckpt['model_config'])
    m.load_state_dict(ckpt['model_state_dict']); m.to(dev).eval()
    return m

@torch.no_grad()
def gen(model, tok, prompt, dev):
    rhyme, tone = get_luc_bat_tags(prompt)
    tagged = f'[LUC_BAT] {rhyme} {tone} {prompt}'
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

def score(p, r):
    ps = p.split(); rs = r.split(); rs8 = rs[:8]
    rlen = len(rs)
    exact = rlen == 8; close = 6 <= rlen <= 10
    pt_ok = sum(1 for i,w in [(1,'B'),(3,'T'),(5,'B')] if i < len(ps) and get_tone(ps[i]) == w)
    rt_ok = sum(1 for i,w in [(1,'B'),(3,'T'),(5,'B'),(7,'B')] if i < len(rs8) and get_tone(rs8[i]) == w)
    rt_total = min(len(rs8), 4)
    rh_ok = False; prh = rrh = None
    if len(ps) >= 6 and len(rs8) >= 6:
        prh = get_rhyme_group(ps[5]); rrh = get_rhyme_group(rs8[5])
        rh_ok = prh == rrh
    return {'r_len': rlen, 'syl_exact': exact, 'syl_close': close,
            'p_tone_ok': pt_ok, 'r_tone_ok': rt_ok, 'r_tone_total': rt_total,
            'rhyme_ok': rh_ok, 'p_rhyme': prh, 'r_rhyme': rrh, 'response': r}

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
tok = Tokenizer.from_file(str(ROOT / 'tokenizer/poetry_bpe.model'))

print('Loading Stage 2...')
s2 = load(str(ROOT / 'checkpoints/final.pt'), dev)

print(f'Evaluating {len(PROMPTS)} novel prompts...')
t0 = time.time()
results = []
for i, p in enumerate(PROMPTS):
    r = gen(s2, tok, p, dev)
    results.append(score(p, r))
    if (i+1) % 25 == 0:
        elapsed = time.time() - t0
        exact = sum(rr['syl_exact'] for rr in results) / len(results) * 100
        rhyme = sum(rr['rhyme_ok'] for rr in results) / len(results) * 100
        print(f'  {i+1}/{len(PROMPTS)} | exact8={exact:.1f}% rhyme={rhyme:.1f}% | {elapsed:.0f}s')

from collections import Counter
n = len(results)
agg = {
    'n': n,
    'syl_exact_8': sum(r['syl_exact'] for r in results) / n * 100,
    'syl_close_6_10': sum(r['syl_close'] for r in results) / n * 100,
    'avg_len': sum(r['r_len'] for r in results) / n,
    'prompt_tone': sum(r['p_tone_ok'] for r in results) / (n * 3) * 100,
    'resp_tone': sum(r['r_tone_ok'] for r in results) / max(sum(r['r_tone_total'] for r in results), 1) * 100,
    'rhyme': sum(r['rhyme_ok'] for r in results) / n * 100,
}
len_dist = Counter(r['r_len'] for r in results)

print(f'\n{"="*60}')
print(f'RESULTS — 100 NOVEL prompts')
print(f'{"="*60}')
print(f'Exact 8 syl:  {agg["syl_exact_8"]:.1f}%')
print(f'Close 6-10:   {agg["syl_close_6_10"]:.1f}%')
print(f'Avg length:    {agg["avg_len"]:.1f} syl')
print(f'Prompt tone:   {agg["prompt_tone"]:.1f}%')
print(f'Response tone: {agg["resp_tone"]:.1f}%')
print(f'Rhyme (vần lưng): {agg["rhyme"]:.1f}%')
print(f'\nLength distribution:')
for length, count in sorted(len_dist.items()):
    bar = '█' * int(count / n * 40)
    print(f'  {length:3d} syl: {count:3d} ({count/n*100:5.1f}%) {bar}')

json.dump({'aggregate': agg, 'length_distribution': dict(len_dist),
           'samples': [{'p': PROMPTS[i], 'r': results[i]['response'], 'len': results[i]['r_len'],
                        'rhyme_ok': results[i]['rhyme_ok']} for i in range(n)]},
          open(ROOT / 'documents/eval_100_novel.json', 'w'), indent=2, ensure_ascii=False)
print(f'\nSaved → documents/eval_100_novel.json')
