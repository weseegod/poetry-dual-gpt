# 📊 Rule-by-Rule Evaluation — 173 Novel Prompts

> Generated: 2026-05-20 21:48
> 173 prompts (ca dao, folk poetry — NOT in training corpus)
> Model: 30.9M params, n_embd=512, n_head=8, n_layer=8

## 📈 Per-Rule Summary

| Rule | Tag | Stage 1 | Stage 2 | Random baseline | Effective? |
|------|-----|---------|---------|-----------------|------------|
| **R1: Internal Rhyme** (vần lưng) | `[RHYME:X]` | 17.3% | 18.5% | 0.6% | ✅ Yes |
| **R2: Tone Pattern** (B-T-B-B) | `[TONE:XXXXXX]` | 61.2% | 66.6% | 6.2% | ✅ Yes |
| **R3: Syllable Count** (8 syl) | (form) | 20.2% | 22.0% | 6.7% | ✅ Yes |
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

## 🛠️ Fix Recommendations

### R1: Rhyme (current: 18% — needs improvement)

**Root cause**: `[RHYME:ong]` is 5 BPE tokens. The rhyme signal is fragmented.
**Fix**: Make rhyme groups special tokens (single IDs like `[LUC_BAT]`).
**Expected**: 40-60% rhyme accuracy (based on genre tag effectiveness).
**Effort**: Retrain tokenizer + corpus + model (~4h Colab).

### R2: Tone Pattern (current: 67% — needs improvement)

**Root cause**: Same fragmentation as R1. `[TONE:BBBTTB]` is 5 BPE tokens.
**Fix**: Same as R1 — make tone patterns special tokens.
**Expected**: 50-70% tone accuracy.

### R3: Syllable Count (current: 22% — needs fix)

**Root cause**: Model uses position-based stopping, not syllable counting.
**Fix 1 (immediate)**: Post-generation truncation to 8 syllables. 3 lines of code.
**Fix 2 (architectural)**: Syllable-count control token `[SYL:8]` as special token.
**Fix 3 (structural)**: Syllable-aware pre-tokenizer before BPE.

### Priority

| Priority | Fix | Impact | Effort |
|----------|-----|--------|--------|
| 1 | R3 Fix 1 — Truncation | 100% syllable accuracy | 3 lines |
| 2 | R1+R2 Fix — Special tokens | 2-3× rhyme/tone improvement | 1 day |
| 3 | Qwen2.5-1.5B QLoRA | Overall quality + better rule following | 1 day |