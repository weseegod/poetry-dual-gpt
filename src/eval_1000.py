"""Batch evaluate 1000 Lục Bát prompts on Stage 1 and Stage 2."""
import re, json, time
import torch, torch.nn.functional as F
from pathlib import Path
from tokenizers import Tokenizer
from src.model import PoetryDuelGPT
from src.tones import get_tone, get_rhyme_group, get_luc_bat_tags

ROOT = Path(__file__).parent.parent

# Load 1000 prompts
prompts = []
with open(ROOT / 'data/poetry_corpus.txt') as f:
    for line in f:
        if '[LUC_BAT]' in line and '<|reply|>' in line:
            parts = line.split('<|reply|>')
            before = parts[0]
            clean = re.sub(r'<\|start\|>', '', before)
            clean = re.sub(r'\[LUC_BAT\]', '', clean)
            clean = re.sub(r'\[RHYME:[^\]]+\]', '', clean)
            clean = re.sub(r'\[TONE:[^\]]+\]', '', clean)
            clean = clean.strip()
            syls = clean.split()
            if len(syls) == 6 and all(any(c.isalpha() for c in w) for w in syls):
                if not re.search(r'[\(\)0-9]', clean):
                    prompts.append(clean)
        if len(prompts) >= 1000: break

print(f'Loaded {len(prompts)} prompts')

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
    exact = rlen == 8
    close = 6 <= rlen <= 10
    
    # Prompt tone
    pt_ok = sum(1 for i,w in [(1,'B'),(3,'T'),(5,'B')] if i < len(ps) and get_tone(ps[i]) == w)
    
    # Response tone (first 8)
    rt_ok = sum(1 for i,w in [(1,'B'),(3,'T'),(5,'B'),(7,'B')] if i < len(rs8) and get_tone(rs8[i]) == w)
    rt_total = min(len(rs8), 4)
    
    # Rhyme
    rh_ok = False; prh = rrh = None
    if len(ps) >= 6 and len(rs8) >= 6:
        prh = get_rhyme_group(ps[5])
        rrh = get_rhyme_group(rs8[5])
        rh_ok = prh == rrh
    
    return {
        'r_len': rlen, 'syl_exact': exact, 'syl_close': close,
        'p_tone_ok': pt_ok, 'r_tone_ok': rt_ok, 'r_tone_total': rt_total,
        'rhyme_ok': rh_ok, 'p_rhyme': prh, 'r_rhyme': rrh,
        'response': r,
    }

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
tok = Tokenizer.from_file(str(ROOT / 'tokenizer/poetry_bpe.model'))

# Only Stage 2 — faster, and it's the model we care about
print('Loading Stage 2...')
s2 = load(str(ROOT / 'checkpoints/final.pt'), dev)

print(f'Evaluating on {len(prompts)} prompts...')
t0 = time.time()
results = []
for i, p in enumerate(prompts):
    r = gen(s2, tok, p, dev)
    results.append(score(p, r))
    if (i+1) % 100 == 0:
        elapsed = time.time() - t0
        eta = elapsed / (i+1) * (len(prompts) - i - 1)
        exact = sum(rr['syl_exact'] for rr in results) / len(results) * 100
        close = sum(rr['syl_close'] for rr in results) / len(results) * 100
        rhyme = sum(rr['rhyme_ok'] for rr in results) / len(results) * 100
        print(f'  {i+1}/{len(prompts)} | exact8={exact:.1f}% close={close:.1f}% rhyme={rhyme:.1f}% | ETA {eta:.0f}s')

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

# Distribution of lengths
from collections import Counter
len_dist = Counter(r['r_len'] for r in results)

print(f'\n{"="*60}')
print(f'RESULTS ({n} prompts)')
print(f'{"="*60}')
print(f'Exact 8 syl:  {agg["syl_exact_8"]:.1f}%')
print(f'Close 6-10:   {agg["syl_close_6_10"]:.1f}%')
print(f'Avg length:    {agg["avg_len"]:.1f} syl')
print(f'Prompt tone:   {agg["prompt_tone"]:.1f}%')
print(f'Response tone: {agg["resp_tone"]:.1f}%')
print(f'Rhyme (vần lưng): {agg["rhyme"]:.1f}%')
print(f'\nLength distribution:')
for length, count in sorted(len_dist.items()):
    bar = '█' * int(count / n * 50)
    print(f'  {length:3d} syl: {count:4d} ({count/n*100:5.1f}%) {bar}')

# Save
json.dump({'aggregate': agg, 'length_distribution': dict(len_dist), 
           'samples': [{'p': prompts[i], 'r': results[i]['response'], 'len': results[i]['r_len'],
                        'rhyme_ok': results[i]['rhyme_ok'], 'p_rhyme': results[i]['p_rhyme'],
                        'r_rhyme': results[i]['r_rhyme']} for i in range(min(50, n))]},
          open(ROOT / 'documents/eval_1000.json', 'w'), indent=2, ensure_ascii=False)
print(f'\nSaved → documents/eval_1000.json')
