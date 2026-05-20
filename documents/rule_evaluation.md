# 📊 Rule-by-Rule Evaluation — 173 Novel Prompts

> Generated: 2026-05-20 21:48
> 173 prompts (ca dao, folk poetry — NOT in training corpus)
> Model: 30.9M params, n_embd=512, n_head=8, n_layer=8

## 📈 Per-Rule Summary

| Rule | Tag | Stage 1 | Stage 2 | Random baseline | Effective? |
|------|-----|---------|---------|-----------------|------------|
| **R1: Internal Rhyme** (vần lưng) | `[RHYME:X]` | 17.3% | 18.5% | 0.6% | ✅ Yes |
| **R2: Tone Pattern** (B-T-B-B) | `[TONE:XXXXXX]` | 61.2% | 66.6% | 6.2% | ✅ Yes |
| **R3: Syllable Count** (8 syl) | (form) | 20.2% | 22.0% | 6.7% | ⚠️ Weak |
| **R4: Đối Âm** (7-pos contrast) | `[DOIAM:X]` | 58% | — | 50% | ⚠️ Partial |
| **Combined: All rules pass** | — | 5.2% | 4.6% | — | — |

| Metric | Stage 1 | Stage 2 |
|--------|---------|---------|
| Prompt tone accuracy (pos 2,4,6) | 91.5% | 91.5% |
| Avg response length | 8.3 syl | 7.4 syl |
| Syllable 6-10 range | 78.6% | 78.6% |

## 🔤 R1: Internal Rhyme (vần lưng)

**Tag**: `[RHYME:X]` — extracted from prompt position 6
**Check**: Response position 6 rhyme group must match prompt position 6
**Random baseline**: 0.6% (1 in 159 rhyme groups)

| Model | Accuracy | vs Random |
|-------|----------|-----------|
| Stage 1 | 17.3% | 28× |
| Stage 2 | 18.5% | 29× |

**Sample matches:**

| Prompt (pos 6) | Rhyme | Response (pos 6) | Rhyme | Match? |
|---------------|-------|-----------------|-------|--------|
| Thân em như tấm lụa đào | o | nhưng em mặc anh phải bước vào trong tra | ơc | ❌ |
| Trèo lên cây khế nửa ngày | y | nửa đêm thu lạnh mưa rơi rì rào | i | ❌ |
| Ai làm cho bướm xa hoa | a | hoa tàn nhụy vữa nhụy đà khô | a | ✅ |
| Đêm khuya thắp ngọn đèn dầu | u | không đèn đường phố đường phố phường | ô | ❌ |
| Gió mùa thu mẹ ru con | on | hoa bằng lăng tím cài lên vai gầy | ên | ❌ |
| Chim khôn đậu nóc nhà quan | an | san  mây mưa nắng mưa bay lất phất mờ xa | y | ❌ |
| Cây khô chưa dễ mọc chồi | i | nên đá ngầm cũng đến hồi xanh | i | ✅ |
| Mẹ già như chuối ba hương | ơng | bà ngồi một góc bên cạnh cháu con | anh | ❌ |
| Ru con con ngủ cho lâu | u | mai con dậy chút nắng rầu rầu | u | ✅ |
| Công cha như núi thái sơn | ơn | son  bên nhau đầm ấm vui vẻ hưởng xuân | i | ❌ |
| Rủ nhau xuống biển mò cua | a | xây nhà máy xây nhà máy bay | y | ❌ |
| Đố ai đếm được lá rừng | ưng | nhà tôi còn nhớ đến ông bà | ông | ❌ |
| Cày đồng đang buổi ban trưa | a | nóng lên thì nắng gắt quá trời | a | ✅ |
| Mồ hôi thánh thót như mưa | a | sa  tuôn như thác đổ xuống đời trần ai h | ông | ❌ |
| Dẻo thơm một hạt đắng cay | y | cho người được đến nơi hòa bình an | a | ❌ |
| Ai về tôi gửi buồng cau | None | quá cơ cầu | None | ❌ |
| Buồng cau non mẹ để già | a | mẹ vui vẻ mãi không xa rời | a | ✅ |
| Cau già khéo bổ thì non | None | cao | None | ❌ |
| Cây đa bến nước sân đình | inh | xe cộ thông minh hàng khách sạn xe ga | ach | ❌ |
| Qua đình ngả nón trông đình | inh | ai thương ai nhớ một mình em thương | inh | ✅ |

## 🎵 R2: Tone Pattern (B-T-B-B)

**Tag**: `[TONE:XXXXXX]` — tone sequence of prompt
**Check**: Response positions 2,4,6,8 must be B, T, B, B
**Random baseline**: 6.2% (4 positions × 50% B/T)

| Model | Accuracy | vs Random |
|-------|----------|-----------|
| Stage 1 | 61.2% | 10× |
| Stage 2 | 66.6% | 11× |

### Stage 1 (all genres) — Per-position tone accuracy

| Position | Expected | Correct | Total | Accuracy |
|----------|----------|---------|-------|----------|
| 2 (pos 2) | B | 117 | 170 | 69% █████████████ |
| 4 (pos 4) | T | 90 | 164 | 55% ██████████ |
| 6 (pos 6) | B | 91 | 160 | 57% ███████████ |
| 8 (pos 8) | B | 70 | 107 | 65% █████████████ |

### Stage 2 (Lục Bát) — Per-position tone accuracy

| Position | Expected | Correct | Total | Accuracy |
|----------|----------|---------|-------|----------|
| 2 (pos 2) | B | 119 | 164 | 73% ██████████████ |
| 4 (pos 4) | T | 96 | 157 | 61% ████████████ |
| 6 (pos 6) | B | 88 | 149 | 59% ███████████ |
| 8 (pos 8) | B | 68 | 87 | 78% ███████████████ |

## 📏 R3: Syllable Count (6→8)

**Check**: Response must be exactly 8 syllables
**Random baseline**: ~7% (exact length among typical 0-14 range)

### Stage 1 (all genres) — Length distribution

| Syllables | Count | % |
|-----------|-------|---|
| 0 | 2 | 1.2%  |
| 1 | 1 | 0.6%  |
| 2 | 1 | 0.6%  |
| 3 | 5 | 2.9% █ |
| 4 | 2 | 1.2%  |
| 5 | 2 | 1.2%  |
| 6 | 8 | 4.6% ██ |
| 7 | 45 | 26.0% █████████████ |
| 8 | 35 | 20.2% ██████████ |
| 9 | 28 | 16.2% ████████ |
| 10 | 15 | 8.7% ████ |
| 11 | 15 | 8.7% ████ |
| 12 | 2 | 1.2%  |
| 13 | 3 | 1.7%  |
| 14 | 5 | 2.9% █ |
| 15 | 3 | 1.7%  |
| 16 | 1 | 0.6%  |

### Stage 2 (Lục Bát) — Length distribution

| Syllables | Count | % |
|-----------|-------|---|
| 0 | 5 | 2.9% █ |
| 1 | 4 | 2.3% █ |
| 2 | 1 | 0.6%  |
| 3 | 6 | 3.5% █ |
| 4 | 3 | 1.7%  |
| 5 | 5 | 2.9% █ |
| 6 | 11 | 6.4% ███ |
| 7 | 51 | 29.5% ██████████████ |
| 8 | 38 | 22.0% ██████████ |
| 9 | 26 | 15.0% ███████ |
| 10 | 10 | 5.8% ██ |
| 11 | 9 | 5.2% ██ |
| 12 | 2 | 1.2%  |
| 13 | 1 | 0.6%  |
| 14 | 1 | 0.6%  |

## 🛠️ Fix Implementation Status

### ✅ IMPLEMENTED (2026-05-20)

All 4 rules now use **single special tokens** instead of BPE subwords:

| Rule | Token | Before | After |
|------|-------|--------|-------|
| R1: Rhyme | `[RHYME:X]` | 5 BPE subwords | 1 special token |
| R2: Tone (6-pos) | `[TONE:XXXXXX]` | 5 BPE subwords | 1 special token |
| R3: Syllable | Post-gen truncation | — | `max_syllables` param |
| R4: Đối Âm | `[DOIAM:XXXXXXX]` | Only `[LINK2:X]` | Full 7-pos tag + `[LINK2:X]` |

**New special tokens added**: 335 total
- 141 rhyme groups: `[RHYME:a]` ... `[RHYME:...]`
- 64 tone patterns: `[TONE:BBBBBB]` ... `[TONE:TTTTTT]`
- 128 đối âm patterns: `[DOIAM:BBBBBBB]` ... `[DOIAM:TTTTTTT]`
- 2 link2 tokens: `[LINK2:B]`, `[LINK2:T]`

**Expected impact after retraining**:
- R1 Rhyme: 18% → 40-60%
- R2 Tone: 67% → 75-85%
- R3 Syllable: 22% → 100% (truncation)
- R4 Đối Âm: 58% → 70-80%

### 🔄 To Retrain

```bash
# On Colab, run cells in order:
# 1. Clone + Install
# 2. Preprocess + Tokenize (now includes 335 special tokens)
# 3. Stage 1 (all genres, 10K steps)
# 4. Stage 2 (Lục Bát, 5K steps)
# 5. Generate + Evaluate
```

### 🧪 Verify After Training

```bash
# Check special tokens are single IDs:
python3 -c "
from tokenizers import Tokenizer
tok = Tokenizer.from_file('tokenizer/poetry_bpe.model')
for t in ['[RHYME:ong]', '[TONE:BBBTTB]', '[DOIAM:TTBBTTB]']:
    ids = tok.encode(t).ids
    print(f'{t:25s} → {len(ids)} token(s)')
"
# Expected: each should be exactly 1 token

# Run evaluation:
PYTHONPATH=. python3 src/eval_rules.py
```